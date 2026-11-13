"""Utilities for aligning Scan3dTac records by their explicit contact IDs."""

import os
import re
import json
import math

from PIL import Image


def contact_index(value):
    """Extract a contact ID without mistaking a numeric extension for the ID."""
    stem = os.path.splitext(os.path.basename(str(value)))[0]
    match = re.search(r"(?:tactile|contact)[_-](\d+)", stem, flags=re.IGNORECASE)
    if match is None:
        match = re.fullmatch(r"\d+", stem)
    if match is None:
        match = re.search(r"(\d+)$", stem)
    return int(match.group(1) if match.lastindex else match.group(0)) if match else None


def index_contact_mapping(mapping, source_path):
    """Return an ID-keyed mapping while accepting sparse, explicitly named IDs."""
    indexed = {}
    for key, value in mapping.items():
        index = contact_index(key)
        if index is None:
            raise ValueError(f"contact key without numeric ID in {source_path}: {key!r}")
        if index in indexed:
            raise ValueError(f"duplicate contact ID {index} in {source_path}")
        indexed[index] = value
    return dict(sorted(indexed.items()))


def tactile_video_mapping(tactile_dir):
    """Return contact ID -> video path, independent of filesystem order."""
    indexed = {}
    for name in os.listdir(tactile_dir):
        if not name.startswith("tactile") or not name.lower().endswith(".mp4"):
            continue
        index = contact_index(name)
        if index is None:
            raise ValueError(f"tactile video without numeric contact ID: {os.path.join(tactile_dir, name)}")
        if index in indexed:
            raise ValueError(f"duplicate tactile video contact ID {index} in {tactile_dir}")
        indexed[index] = os.path.join(tactile_dir, name)
    return dict(sorted(indexed.items()))


def is_visible_coordinate(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        row, column = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(row) and math.isfinite(column) and not (row == 0 and column == 0)


def contact_centered_crop_box(value, image_size, crop_size):
    """Return a deterministic square local crop containing one stored contact.

    Scan3dTac stores image coordinates as ``[row, column]`` while PIL uses
    ``[x, y]``.  This helper is shared by physical-belief evaluation and its
    renderer so the displayed local query is exactly the tensor region used by
    the evaluator.
    """

    if not is_visible_coordinate(value):
        raise ValueError(f"contact is not visible: {value!r}")
    width, height = (int(image_size[0]), int(image_size[1]))
    side = min(int(crop_size), width, height)
    if side <= 0:
        raise ValueError(f"invalid crop/image size: crop={crop_size}, image={image_size}")
    row, column = float(value[0]), float(value[1])
    if not (0 <= column < width and 0 <= row < height):
        raise ValueError(f"contact lies outside image {image_size}: {value!r}")
    left = min(max(int(round(column - side / 2)), 0), width - side)
    top = min(max(int(round(row - side / 2)), 0), height - side)
    return left, top, left + side, top + side


def align_contact_view(material_mapping, correspondence_mapping, tactile_dir, material_path, correspondence_path):
    """Align one rendered view to material/video records by explicit contact ID.

    Some released correspondence files omit contacts that are invisible in a
    particular rendered view.  Missing IDs are therefore represented as
    ``[0, 0]`` instead of shifting every later contact position.
    """
    materials = index_contact_mapping(material_mapping, material_path)
    correspondences = index_contact_mapping(correspondence_mapping, correspondence_path)
    videos = tactile_video_mapping(tactile_dir)
    usable_ids = sorted(set(materials) & set(videos))
    usable_id_set = set(usable_ids)
    if not usable_ids:
        raise ValueError(f"no material/video contact IDs overlap in {tactile_dir}")

    missing_visible = sorted(
        index
        for index, value in correspondences.items()
        if is_visible_coordinate(value) and index not in usable_id_set
    )
    if missing_visible:
        raise ValueError(
            f"visible correspondence IDs lack a material or video in {correspondence_path}: "
            f"{missing_visible[:20]}"
        )

    aligned_correspondences = {
        str(index): correspondences.get(index, [0, 0]) for index in usable_ids
    }
    aligned_materials = [materials[index] for index in usable_ids]
    aligned_videos = [videos[index] for index in usable_ids]
    return usable_ids, aligned_materials, aligned_correspondences, aligned_videos


def view_usable_contact_count(
    correspondence_mapping,
    available_contact_ids,
    source_path,
    image_size=None,
):
    correspondences = index_contact_mapping(correspondence_mapping, source_path)
    count = 0
    for index, value in correspondences.items():
        if index not in available_contact_ids or not is_visible_coordinate(value):
            continue
        if image_size is None:
            count += 1
            continue
        width, height = image_size
        row, column = float(value[0]), float(value[1])
        if 0 <= column < width and 0 <= row < height:
            count += 1
    return count


def view_has_usable_contact(correspondence_mapping, available_contact_ids, source_path, image_size=None):
    return (
        view_usable_contact_count(
            correspondence_mapping,
            available_contact_ids,
            source_path,
            image_size,
        )
        >= 1
    )


def usable_image_names(
    instance_root,
    image_names,
    excluded_contact_ids=None,
    minimum_visible_contacts=1,
):
    """Filter views by usable visual/tactile correspondences after QC."""
    minimum_visible_contacts = int(minimum_visible_contacts)
    if minimum_visible_contacts < 1:
        raise ValueError("minimum_visible_contacts must be positive")
    tactile_dir = os.path.join(instance_root, "Tactile")
    material_path = os.path.join(instance_root, "Annotation", "tactile_material.json")
    with open(material_path, "r") as stream:
        material_ids = set(index_contact_mapping(json.load(stream), material_path))
    available_ids = material_ids & set(tactile_video_mapping(tactile_dir))
    available_ids -= set(excluded_contact_ids or ())
    correspondence_dir = os.path.join(
        instance_root, "Preprocessed0111", "tacdino", "tactile_corr_label"
    )
    usable = []
    for image_name in image_names:
        image_path = os.path.join(
            instance_root, "Preprocessed0111", "tacdino", "rgb_images_processed", image_name
        )
        with Image.open(image_path) as image:
            image_size = image.size
        corr_path = os.path.join(
            correspondence_dir,
            f"{os.path.splitext(image_name)[0]}_tactile_correspondence.json",
        )
        with open(corr_path, "r") as stream:
            correspondence = json.load(stream)
        if (
            view_usable_contact_count(
                correspondence,
                available_ids,
                corr_path,
                image_size,
            )
            >= minimum_visible_contacts
        ):
            usable.append(image_name)
    return usable
