"""Launch the GeoTAC recipe used for UniVTAC initialization."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

from dinov2.tacdino.file_records import file_record, write


ROOT = Path(__file__).resolve().parents[1]


def build_config(args):
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "dinov2/configs/ssl_default_config.yaml"),
        OmegaConf.load(ROOT / "configs/geotac.yaml"),
    )
    for path in (args.data_root, args.metadata_root, args.train_split):
        if ":" in str(path.resolve()):
            raise ValueError(f"Dataset paths cannot contain colons: {path}")
    cfg.train.dataset_path = (
        f"Scan3dTac:split=TRAIN:root={args.metadata_root.resolve()}:"
        f"extra={args.data_root.resolve()}:split_file={args.train_split.resolve()}:no3d=true"
    )
    cfg.student.pretrained_weights = str(args.dino_checkpoint.resolve())
    cfg.touch.pointbert_pretrained = str(args.pointbert_checkpoint.resolve())
    cfg.touch.tactile_geometry_manifest = str(args.geometry_manifest.resolve())
    cfg.touch.tactile_geometry_manifest_sha256 = file_record(args.geometry_manifest)["sha256"]
    cfg.touch.contact_metadata_path = str(args.output.resolve() / "contact_metadata.json")
    cfg.train.batch_size_per_gpu = args.batch_per_gpu
    cfg.train.num_workers = args.workers
    if args.workers == 0:
        cfg.train.prefetch_factor = None
    cfg.optim.base_lr = cfg.optim.peak_learning_rate / math.sqrt(args.batch_per_gpu * args.devices / 1024)
    for role in ("student", "teacher"):
        policies = cfg.compute_precision[role]
        for policy in policies.values():
            if "mixed_precision" in policy:
                policy.mixed_precision.param_dtype = "bf16"
                policy.mixed_precision.reduce_dtype = "fp32"
        for name in ("tactile_backbone", "depth_backbone", "point_backbone",
                     "clip_head", "rgb_projection", "depth_projection", "pc_projection"):
            policies[name] = copy.deepcopy(policies.backbone)
            if name not in ("tactile_backbone", "depth_backbone"):
                policies[name].mixed_precision.param_dtype = "fp32"
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Restored Touch3D object directory")
    parser.add_argument("--metadata-root", type=Path, required=True, help="Directory containing mat2idx.json and catemap")
    parser.add_argument("--train-split", type=Path, required=True, help="JSON list of paper TRAIN object IDs")
    parser.add_argument("--contact-filter", type=Path, required=True, help="JSON containing excluded_contact_ids")
    parser.add_argument("--geometry-manifest", type=Path, required=True, help="Prepared tactile geometry manifest.json")
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--pointbert-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=8)
    recipe = OmegaConf.load(ROOT / "configs/geotac.yaml")
    parser.add_argument("--batch-per-gpu", type=int, default=recipe.train.batch_size_per_gpu)
    parser.add_argument("--workers", type=int, default=recipe.train.num_workers)
    parser.add_argument("--stop-after", type=int, default=recipe.optim.total_steps,
                        help="Checkpoint at an earlier step without changing the schedule")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.devices < 1 or args.batch_per_gpu < 2 or args.workers < 0 or not 1 <= args.stop_after <= recipe.optim.total_steps:
        parser.error("invalid GPU, batch, worker or step count")
    for path in (args.train_split, args.contact_filter, args.geometry_manifest,
                 args.dino_checkpoint, args.pointbert_checkpoint,
                 args.metadata_root / "mat2idx.json", args.metadata_root / "catemap/label_to_idx.json"):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    instances = json.loads(args.train_split.read_text())
    if (not isinstance(instances, list) or not instances
            or any(not isinstance(instance, str) for instance in instances)
            or len(set(instances)) != len(instances)):
        parser.error("train split must be a nonempty list of unique category/object paths")
    for instance in instances:
        path = Path(instance)
        if path.is_absolute() or len(path.parts) != 2 or ".." in path.parts:
            parser.error(f"invalid relative object ID: {instance}")
        if not (args.data_root / path / "Annotation/tactile_material.json").is_file():
            parser.error(f"missing contact annotations for {instance}")
    contact_filter = json.loads(args.contact_filter.read_text())
    if not isinstance(contact_filter, dict) or not isinstance(contact_filter.get("excluded_contact_ids"), dict):
        parser.error("contact filter must contain an excluded_contact_ids mapping")

    from dinov2.tacdino.file_records import sha
    from dinov2.tacdino.pointbert_runtime import WEIGHT_SHA256
    from dinov2.tacdino.tactile_geometry import geometry_index
    if sha(args.pointbert_checkpoint) != WEIGHT_SHA256:
        parser.error("Point-BERT checkpoint does not match the published pretraining weights")
    index = geometry_index(str(args.geometry_manifest.resolve()))
    if Path(index["data_root"]).resolve() != args.data_root.resolve():
        parser.error("geometry manifest and Touch3D data root differ")

    cfg = build_config(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "launch.yaml"
    payload = OmegaConf.to_yaml(cfg)
    if config_file.exists() and config_file.read_text() != payload:
        parser.error("existing output uses another resolved configuration")
    config_file.write_text(payload)
    write(output / "contact_metadata.json", dict(data_root=str(args.data_root.resolve()), instances=instances))
    if args.prepare_only:
        print(config_file)
        return

    import torch
    if torch.cuda.device_count() < args.devices or not torch.cuda.is_bf16_supported():
        parser.error("GeoTAC training requires the requested number of BF16-capable NVIDIA GPUs")
    env = dict(os.environ, OMP_NUM_THREADS="1", WANDB_MODE="disabled",
        TACDINO_CONTACT_FILTER=str(args.contact_filter.resolve()),
        TACDINO_TRAIN_SPLIT_FILE=str(args.train_split.resolve()),
        TACDINO_TRAIN_SPLIT_SHA256=file_record(args.train_split)["sha256"],
        TACDINO_DATA_SCOPE="train_core_only",
        TACDINO_PRETRAINED_SHA256=file_record(args.dino_checkpoint)["sha256"],
        TACDINO_TACTILE_GEOMETRY_MANIFEST=str(args.geometry_manifest.resolve()),
        TACDINO_TACTILE_CACHE_ROOT=str(output / "frame_cache"),
        TACDINO_STOP_AFTER_STEPS=str(args.stop_after),
        TACDINO_MAX_CONSECUTIVE_SKIPPED_STEPS="3")
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone",
         f"--nproc_per_node={args.devices}", "--module", "dinov2.run.train.train_multiresource",
         "--config-file", str(config_file), "--output-dir", str(output / "training")],
        env=env, check=True,
    )


if __name__ == "__main__":
    main()
