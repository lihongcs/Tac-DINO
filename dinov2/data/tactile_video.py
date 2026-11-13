"""Deterministic temporal decoding for GelSight contact videos.

The released TacDINO pipeline uses frame 10 as its canonical peak-contact
frame.  The helpers below keep that protocol, but expose the approach-to-
contact trajectory instead of silently reducing every video to one image.
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image


TACTILE_CLIP_CACHE_SCHEMA = "tacdino.tactile-clip-cache.v1"
LEGACY_CANONICAL_FRAME_INTERPOLATION = cv2.INTER_LINEAR


def _resize_interpolation_name(value: int) -> str:
    names = {
        cv2.INTER_AREA: "opencv_inter_area",
        cv2.INTER_LINEAR: "opencv_inter_linear",
    }
    if int(value) not in names:
        raise ValueError(f"unsupported tactile resize interpolation: {value}")
    return names[int(value)]


def tactile_clip_cache_path(
    cache_root: str,
    mp4_path: str,
    frame_indices: Iterable[int],
    image_size: int,
    resize_interpolation: int = cv2.INTER_AREA,
) -> Path:
    """Return the content-addressed cache path for one decoded tactile clip."""

    source = Path(mp4_path).resolve(strict=True)
    stat = source.stat()
    payload = {
        "schema_version": TACTILE_CLIP_CACHE_SCHEMA,
        "source_path": str(source),
        "source_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "frame_indices": [int(value) for value in frame_indices],
        "image_size": int(image_size),
        "pixel_format": "rgb_uint8_thwc",
        "resize_interpolation": _resize_interpolation_name(
            resize_interpolation
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Path(cache_root).resolve() / digest[:2] / f"{digest}.npy"


def _validate_cached_clip(
    array: np.ndarray,
    *,
    frame_count: int,
    image_size: int,
    path: Path,
) -> np.ndarray:
    expected = (int(frame_count), int(image_size), int(image_size), 3)
    if array.dtype != np.uint8 or tuple(array.shape) != expected:
        raise ValueError(
            f"invalid tactile cache entry {path}: dtype={array.dtype}, "
            f"shape={tuple(array.shape)}, expected=uint8{expected}"
        )
    return array


def _read_cached_clip(path: Path, *, frame_count: int, image_size: int):
    try:
        array = np.load(path, allow_pickle=False)
        return _validate_cached_clip(
            array,
            frame_count=frame_count,
            image_size=image_size,
            path=path,
        )
    except (OSError, ValueError, EOFError):
        # A worker may have been killed between create and replace on a prior
        # run. The cache is derived data; reject and regenerate a bad entry.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None


def _write_cached_clip(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(temporary, "xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _decode_tactile_clip(
    mp4_path: str,
    indices: List[int],
    image_size: int,
    max_retries: int,
    resize_interpolation: int,
) -> np.ndarray:
    """Decode exact frames once and return contiguous RGB uint8 THWC."""

    capture = None
    for attempt in range(max_retries):
        capture = cv2.VideoCapture(mp4_path)
        if capture.isOpened():
            break
        capture.release()
        capture = None
        time.sleep(0.1 * (attempt + 1))
    if capture is None:
        raise IOError(f"cannot open tactile video after {max_retries} retries: {mp4_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= indices[-1]:
            raise ValueError(
                f"tactile clip requires frame {indices[-1]}, but "
                f"frame_count={frame_count}: {mp4_path}"
            )

        requested = set(indices)
        decoded = {}
        frame_number = 0
        while frame_number <= indices[-1]:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise IOError(f"failed to decode tactile frame {frame_number}: {mp4_path}")
            if frame_number in requested:
                frame = cv2.resize(
                    frame,
                    (image_size, image_size),
                    interpolation=int(resize_interpolation),
                )
                decoded[frame_number] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_number += 1

        if set(decoded) != requested:
            missing = sorted(requested - set(decoded))
            raise IOError(f"failed to decode requested tactile frames {missing}: {mp4_path}")
        return np.stack([decoded[index] for index in indices], axis=0)
    finally:
        capture.release()


def load_tactile_clip_from_mp4(
    mp4_path: str,
    frame_indices: Iterable[int],
    image_size: int = 98,
    max_retries: int = 20,
    resize_interpolation: int = cv2.INTER_AREA,
    cache_root: str = None,
) -> List[Image.Image]:
    """Read exact frames from one tactile video and return RGB PIL images.

    Exact, phase-normalized indices make the temporal protocol reproducible
    across workers and runs.  A missing requested frame is a data error rather
    than an invitation to repeat the last frame, which would fabricate motion.
    """

    indices = [int(value) for value in frame_indices]
    if not indices or any(value < 0 for value in indices):
        raise ValueError(f"invalid tactile clip indices: {indices}")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError(f"tactile clip indices must be unique and increasing: {indices}")
    _resize_interpolation_name(resize_interpolation)
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"tactile video does not exist: {mp4_path}")

    cache_root = (os.environ.get("TACDINO_TACTILE_CACHE_ROOT", "") if cache_root is None else cache_root).strip()
    cache_required = os.environ.get("TACDINO_TACTILE_CACHE_REQUIRED", "0") == "1"
    cached = None
    cache_path = None
    if cache_root:
        try:
            cache_path = tactile_clip_cache_path(
                cache_root,
                mp4_path,
                indices,
                image_size,
                resize_interpolation=resize_interpolation,
            )
            if cache_path.is_file():
                cached = _read_cached_clip(
                    cache_path,
                    frame_count=len(indices),
                    image_size=image_size,
                )
        except OSError:
            if cache_required:
                raise

    if cached is None:
        decoded = _decode_tactile_clip(
            mp4_path,
            indices,
            image_size,
            max_retries,
            resize_interpolation,
        )
        if cache_path is not None:
            try:
                _write_cached_clip(cache_path, decoded)
                cached = decoded
            except OSError:
                if cache_required:
                    raise
        if cached is None:
            cached = decoded

    # PIL construction is intentionally after the cache boundary. The
    # downstream normalization receives exactly the same RGB pixels whether
    # the clip was decoded now or loaded from local cache.
    return [Image.fromarray(frame.copy(), mode="RGB") for frame in cached]
