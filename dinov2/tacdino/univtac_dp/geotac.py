"""Initialize the unchanged DP architecture from a VAL-selected GeoTAC pair.

Only the two ViT-L/14 backbones transfer. Representation projection, geometry,
temporal and DINO heads do not become policy inputs or auxiliary losses.
"""
import json
import os
from pathlib import Path
import re

import torch

from .data import sha_file


def backbone_state(teacher, prefix):
    result = {}
    prefix += "."
    for key, value in teacher.items():
        if not key.startswith(prefix):
            continue
        key = key[len(prefix):]
        # FSDP chunks contain global block indices (with Identity padding).
        key = re.sub(r"^blocks\.\d+\.(\d+)\.", r"blocks.\1.", key)
        if key in result:
            raise ValueError(f"duplicate unchunked backbone key: {key}")
        result[key] = value
    if not result:
        raise ValueError(f"missing teacher submodule: {prefix}")
    return result


def verify_selection(cfg):
    source = cfg["geotac_initialization"]
    selection_mode = "explicit_checkpoint"
    if source.get("selection_path"):
        selection = json.loads(Path(source["selection_path"]).read_text())
        if (not selection.get("closed") or not selection.get("selection_uses_VAL_only")
                or selection.get("TEST_used_for_selection") is not False):
            raise ValueError("GeoTAC initialization must be selected on closed VAL only")
        if (selection["selected_step"] != source["step"]
                or selection["selected_model_sha256"] != source["sha256"]):
            raise ValueError("GeoTAC VAL selection does not match the checkpoint")
        selection_mode = "closed_VAL_not_TEST"
    if sha_file(cfg["checkpoint"]) != source["sha256"]:
        raise ValueError("GeoTAC representation checkpoint SHA256 mismatch")
    return dict(source, checkpoint=cfg["checkpoint"],
                selection=selection_mode, state_key="teacher",
                vision_submodule="backbone", tactile_submodule="tactile_backbone",
                transferred_heads=[], policy_architecture="unchanged_DINOv2_ViT_L14_DP")


def load_pair(cfg, device, *, initialize=True):
    if not cfg["use_tactile"] or cfg["freeze_backbones"]:
        raise ValueError("GeoTAC policy requires two independently trainable backbones")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    from dinov2.hub.backbones import dinov2_vitl14
    states, provenance = {}, dict(cfg["geotac_initialization"])
    if initialize:
        provenance = verify_selection(cfg)
        saved = torch.load(cfg["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
        states = {name: backbone_state(saved["teacher"], prefix)
                  for name, prefix in (("vision", "backbone"), ("tactile", "tactile_backbone"))}
    models = {}
    for name in ("vision", "tactile"):
        with torch.device("meta"):
            model = dinov2_vitl14(pretrained=False, drop_path_rate=0.)
        if initialize:
            model.load_state_dict(states[name], strict=True, assign=True)
            model = model.to(device=device, dtype=torch.float32)
        model.mask_token.requires_grad_(False)
        models[name] = model
    # On rollout, ImageDP assigns the complete policy state before materializing.
    # Consequently the training-only representation file is not required there.
    return models["vision"], models["tactile"], provenance
