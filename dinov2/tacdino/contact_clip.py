"""Direct local-resource <-> same-contact tactile CLIP correspondence.

No history/target split, future label, generated candidate, or regression loss.
Sampling progress is not a calibrated contact phase or a metric footprint.
"""
import hashlib
import math
import numpy as np

NAME = "tacdino.local-multiresource-dynamic-contact.v1"
FRAMES = 4


def protocol():
    return dict(name=NAME, frames=FRAMES, sampling="uniform_inclusive_recording_progress",
        query_modalities=["RGB", "depth", "pointcloud"],
        correspondence="same_object_same_contact_ordered_tactile_clip",
        directions="bidirectional", objective="symmetric_multi_positive_InfoNCE",
        generation=False, future_prediction=False, whole_object_geometry=False,
        crop_pixels=98, metric_footprint_calibrated=False,
        controls=["repeat_last", "reverse"],
        old_semantic_tasks="retained_separately_no3D",
        primary_metrics=["object_macro_top1", "object_macro_mrr"],
        galleries=["all_contacts", "within_object"],
        checkpoint_selection="VAL_only", objectfolder_real="TEST_only")


def frame_indices(count, frames=FRAMES):
    if count < frames or frames < 2:
        raise ValueError("require distinct frames, no duplicate padding")
    return tuple(i * (count-1) // (frames-1) for i in range(frames))


def recording(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid recording FPS: " + str(video))
    return frame_indices(n), n, fps


def contact_key(video):
    # Stable across views/ranks; same video is a multi-positive, not a negative.
    import os
    value = os.path.realpath(str(video)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63)-1)


def geometry_inputs(xyz, mask, camera, points=256):
    """Only visible points INSIDE the RGB crop; depth + explicit mask channel.

    Shape-derived camera Z is not measured sensor depth. Relative Z removes
    unverified absolute units. Invalid pixels stay zero, never inpainted.
    """
    valid = np.asarray(mask, bool)
    p = np.asarray(xyz, np.float32)[valid]
    z = np.asarray(camera, np.float32)[..., 2]
    if len(p) < 16 or not np.isfinite(p).all() or not np.isfinite(z[valid]).all() or (z[valid] <= 0).any():
        raise ValueError("invalid/insufficient local geometry")
    # Deterministic evenly spaced valid pixels; FPS grouping occurs in encoder.
    selected = np.arange(points) * len(p) // points
    cloud = p[selected]
    relative = np.zeros_like(z)
    relative[valid] = z[valid] / z[valid].mean() - 1.
    depth = np.stack([relative, relative, relative, valid.astype(np.float32)])
    return cloud.astype(np.float32), depth.astype(np.float32)


