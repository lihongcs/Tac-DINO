"""Train a GeoTAC-initialized Diffusion Policy on UniVTAC demonstrations."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from dinov2.tacdino.file_records import write as atomic_json
from .data import LEGACY_INPUT_PROTOCOL, sha_file


SOURCE_COMMIT = "e649af2"
TASKS = ("lift_can", "lift_bottle", "grasp_classify", "insert_tube")
SETTINGS = Path(__file__).resolve().parents[3] / "configs/univtac_geotac.json"


def emit(path, value):
    line = json.dumps(value, allow_nan=False)
    print(line, flush=True)
    with path.open("a") as stream:
        stream.write(line + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def code_sha():
    return SOURCE_COMMIT


def require_download_complete(cfg, task):
    folder = Path(cfg["data_root"]) / "isaac45" / task / "hdf5"
    expected = set(range(cfg["expected_episodes"]))
    observed = {int(p.stem) for p in folder.glob("*.hdf5") if p.stem.isdigit()}
    missing = expected - observed
    if missing:
        raise ValueError(f"{folder} is missing episodes: {sorted(missing)}")


def config(args):
    cfg = json.loads(SETTINGS.read_text())
    cfg.update(
        data_root=str(args.data_root.resolve()),
        output_root=str(args.output.resolve()),
        checkpoint=str(args.geotac_checkpoint.resolve()),
        geotac_initialization={"sha256": sha_file(args.geotac_checkpoint)},
        micro_batch_size=args.micro_batch_size,
        data_workers=args.workers,
    )
    if args.selection:
        selection = json.loads(args.selection.read_text())
        cfg["geotac_initialization"].update(
            selection_path=str(args.selection.resolve()), step=selection["selected_step"])
    if args.legacy_input_compat:
        cfg["input_protocol"] = LEGACY_INPUT_PROTOCOL
    if args.stage == "initial":
        cfg["training_steps"] = cfg["initial_training_steps"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train"))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--geotac-checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="Optional validation selection JSON for the supplied checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial", "final"), default="initial")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-training", action="store_true")
    parser.add_argument("--allow-batch-change", action="store_true")
    parser.add_argument("--legacy-input-compat", action="store_true")
    defaults = json.loads(SETTINGS.read_text())
    parser.add_argument("--micro-batch-size", type=int, default=defaults["micro_batch_size"])
    parser.add_argument("--workers", type=int, default=defaults["data_workers"])
    args = parser.parse_args()
    args.revise_training_budget = False
    if args.extend_training and (not args.resume or args.stage != "final"):
        parser.error("--extend-training requires --resume --stage final")
    if args.allow_batch_change and not args.resume:
        parser.error("--allow-batch-change requires --resume")
    if args.micro_batch_size < 1 or args.workers < 0:
        parser.error("invalid microbatch or worker count")
    if not args.geotac_checkpoint.is_file():
        parser.error(f"missing GeoTAC checkpoint: {args.geotac_checkpoint}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("UniVTAC training requires a BF16-capable NVIDIA GPU")
    cfg = config(args)
    from . import joint
    return {"preflight": joint.preflight, "train": joint.train}[args.command](args, cfg)


if __name__ == "__main__":
    main()
