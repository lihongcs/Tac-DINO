"""Contact alignment validation and augmentation RNG isolation."""
from contextlib import contextmanager
import random

import numpy as np
import torch

from dinov2.data.datasets.tacdino.contact_alignment import contact_index


def validate_contact_input_alignment(paths, correspondence, materials):
    """Fail before sampling/decoding if equal-length lists refer to other IDs.

    The dataset already joins by ID. Re-check at the augmentation boundary so
    a new loader/caller cannot silently turn an ordering bug into supervision.
    This guard never sorts or rewrites a caller's data.
    """
    ids = [contact_index(key) for key in correspondence]
    video_ids = [contact_index(path) for path in paths]
    if (not ids or any(i is None for i in ids) or len(set(ids)) != len(ids)
            or ids != video_ids or len(materials) != len(ids)):
        raise ValueError("contact IDs, videos and material rows are not aligned")
    return ids


@contextmanager
def preserve_data_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


