"""Episode-isolated, causal UniVTAC data protocol."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


FORMAL_INPUT_PROTOCOL = "isaac_rgb_v1"
LEGACY_INPUT_PROTOCOL = "legacy_pil_rgb_v0"
INPUT_PROTOCOLS = {FORMAL_INPUT_PROTOCOL, LEGACY_INPUT_PROTOCOL}
FORMAL_TACTILE_IMAGE = "rgb"


def tactile_image_field(cfg):
    """Return the formal marker-free GelSight image field, or fail closed.

    UniVTAC trajectories contain both ``rgb`` and ``rgb_marker`` streams.  The
    paper protocol uses only the raw marker-free stream for both offline
    training and closed-loop deployment.  Keeping this validation beside the
    RGB contract prevents a stale marker checkpoint from being silently reused.
    """
    value = cfg.get("tactile_image")
    if value != FORMAL_TACTILE_IMAGE:
        raise ValueError(
            "formal UniVTAC tactile input must be marker-free 'rgb'; "
            f"found {value!r}"
        )
    return value


def input_protocol(cfg, *, allow_legacy_default=False):
    """Resolve the RGB contract and fail closed for new training configs.

    Official UniVTAC HDF5 files were written by passing Isaac RGB arrays
    directly to OpenCV's BGR JPEG encoder. PIL therefore decodes their red and
    blue channels in the opposite order from live Isaac RGB arrays.
    """
    value = cfg.get("input_protocol")
    if value is None and allow_legacy_default:
        value = LEGACY_INPUT_PROTOCOL
    if value not in INPUT_PROTOCOLS:
        raise ValueError(
            f"unsupported or missing input_protocol {value!r}; expected one of {sorted(INPUT_PROTOCOLS)}"
        )
    return value


def _uint8_rgb(value):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC RGB observation, found {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and float(array.max()) <= 1.5 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def hdf5_image_chw(value, size, protocol):
    """Decode a legacy UniVTAC image into the checkpoint's RGB contract."""
    if protocol not in INPUT_PROTOCOLS:
        raise ValueError(f"unsupported input_protocol {protocol!r}")
    with Image.open(io.BytesIO(bytes(value))) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if protocol == FORMAL_INPUT_PROTOCOL:
        # Undo the historical RGB-to-OpenCV-JPEG channel inversion.
        array = array[..., ::-1]
    image = Image.fromarray(np.ascontiguousarray(array)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()


def live_image_chw(value, size, protocol):
    """Convert a live Isaac RGB array into the checkpoint's RGB contract."""
    if protocol not in INPUT_PROTOCOLS:
        raise ValueError(f"unsupported input_protocol {protocol!r}")
    array = _uint8_rgb(value)
    if protocol == LEGACY_INPUT_PROTOCOL:
        # Old checkpoints learned from the uncorrected PIL-decoded HDF5 bytes.
        array = array[..., ::-1]
    image = Image.fromarray(np.ascontiguousarray(array)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def episode_split(cfg):
    count = int(cfg.get("protocol_episode_count", cfg["expected_episodes"]))
    if cfg.get("episode_split_policy") == "univtac_act_v1":
        # Official ACT calls np.random.seed(1), permutes the first 50 released
        # episodes, and uses the first 80% for TRAIN and the remainder for VAL.
        ids = np.random.RandomState(cfg["split_seed"]).permutation(count)
        boundary = int(float(cfg.get("train_ratio", 0.8)) * count)
        train = sorted(ids[:boundary].tolist())
        val = sorted(ids[boundary:].tolist())
    else:
        ids = np.random.default_rng(cfg["split_seed"]).permutation(count)
        val = sorted(ids[:cfg["validation_episodes"]].tolist())
        train = sorted(ids[cfg["validation_episodes"]:].tolist())
    assert set(train).isdisjoint(val)
    return {"train": train, "val": val}


def causal_indices(t, length, observation_steps, horizon):
    """Observations end at t; actions are q[t+1:t+1+H], never cross episodes."""
    if not 0 <= t < length - 1:
        raise ValueError("observation must have at least one future target")
    obs = np.maximum(np.arange(t - observation_steps + 1, t + 1), 0)
    future = t + 1 + np.arange(horizon)
    valid = future < length
    return obs, np.minimum(future, length - 1), valid

