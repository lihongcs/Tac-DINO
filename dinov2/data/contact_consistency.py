"""Training contact metadata and contact-centered crops."""
from functools import lru_cache
from pathlib import Path
import torch
from dinov2.tacdino.file_records import read
from dinov2.tacdino.contact_clip import contact_key
from dinov2.data.datasets.tacdino.contact_alignment import contact_centered_crop_box, index_contact_mapping


@lru_cache(maxsize=4)
def training_metadata(metadata_path):
    metadata = read(metadata_path)
    data_root = Path(metadata["data_root"]).resolve()
    raw = {}
    for instance in sorted(metadata["instances"]):
        root = data_root / instance
        path = root / 'Annotation/tactile_material.json'
        for contact, value in index_contact_mapping(read(path), str(path)).items():
            values = value if isinstance(value, (list, tuple)) else [value]
            raw[str(root / 'Tactile' / f'tactile_{contact}.mp4')] = frozenset(
                str(material).strip().lower() for material in values if str(material).strip())
    labels = sorted({label for values in raw.values() for label in values})
    if not labels:
        raise ValueError('TRAIN contact materials are missing')
    output = {}
    for video, values in raw.items():
        if not values:
            raise ValueError(f'empty TRAIN contact material set: {video}')
        obj = contact_key(Path(video).parent.parent)
        output[video] = (obj, tuple(label in values for label in labels))
    return output


def contact_metadata(metadata_path, video):
    obj, labels = training_metadata(metadata_path)[str(Path(video).resolve())]
    return torch.tensor(obj, dtype=torch.long), torch.tensor(labels, dtype=torch.bool)


def matched_contact_box(coord_yx, image_size, size=98):
    return contact_centered_crop_box(coord_yx, image_size, size)
