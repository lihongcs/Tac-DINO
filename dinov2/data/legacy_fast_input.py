"""Loss-neutral canonical-frame input pipeline for registered paper baselines.

Only unaugmented RGB uint8 pixels are cached. Sampling, labels, frame 10,
OpenCV linear resize, transforms and model code remain unchanged.
"""
from collections.abc import Sequence
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from PIL import Image

from dinov2.data.tactile_video import (
    LEGACY_CANONICAL_FRAME_INTERPOLATION, load_tactile_clip_from_mp4,
)


@lru_cache(maxsize=2048)
def _canonical_pixels(path, source_bytes, source_mtime_ns, cache_root):
    # Source identity is part of the RAM and existing atomic disk-cache keys.
    image = load_tactile_clip_from_mp4(
        path, [10], image_size=98,
        resize_interpolation=LEGACY_CANONICAL_FRAME_INTERPOLATION,
    )[0]
    pixels = np.array(image, dtype=np.uint8, copy=True)
    pixels.setflags(write=False)
    return pixels


def canonical_frame(path):
    path = os.path.abspath(os.fspath(path))
    stat = os.stat(path)
    pixels = _canonical_pixels(
        path, stat.st_size, stat.st_mtime_ns,
        os.environ.get("TACDINO_TACTILE_CACHE_ROOT", ""),
    )
    # Never share mutable PIL storage with downstream augmentations.
    return Image.fromarray(pixels.copy(), mode="RGB")


class LazyCanonicalFrames(Sequence):
    """Keep aligned contact order, decoding only indices actually consumed."""
    def __init__(self, paths):
        self.paths = tuple(os.fspath(path) for path in paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [canonical_frame(path) for path in self.paths[index]]
        return canonical_frame(self.paths[index])

