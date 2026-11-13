"""Admissible contact candidates; ambiguous contacts are ignored, not positives."""
import torch


def admissible_contacts(ids, objects, materials, candidate_ids, candidate_objects, candidate_materials):
    if (materials.ndim != 2 or candidate_materials.ndim != 2
            or materials.shape[1] != candidate_materials.shape[1]
            or len(ids) != len(objects) or len(ids) != len(materials)
            or len(candidate_ids) != len(candidate_objects)
            or len(candidate_ids) != len(candidate_materials)):
        raise ValueError('contact metadata shapes differ')
    if not materials.any(1).all() or not candidate_materials.any(1).all():
        raise ValueError('empty material labels cannot establish a negative')
    same_contact = ids[:, None] == candidate_ids[None]
    different_object = objects[:, None] != candidate_objects[None]
    overlap = (materials.float() @ candidate_materials.float().T) > 0
    return same_contact | different_object | ~overlap
