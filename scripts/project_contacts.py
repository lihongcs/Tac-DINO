"""Project manually labeled 3D Touch3D contacts into the processed RGB views."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import KDTree

from dinov2.data.datasets.tacdino.contact_alignment import index_contact_mapping


def project_object(object_root):
    render = object_root / "Preprocessed0111"
    views = render / "tacdino"
    annotations = json.loads((object_root / "Annotation/tactile_normalized_pc_location.json").read_text())
    contacts = index_contact_mapping(annotations, object_root)
    # This is the same +90° X rotation used by the visibility projection.
    points = np.load(render / "scan.npy")[:, :3]
    points = points[:, [0, 2, 1]] * [1, -1, 1]
    nearest = KDTree(points).query(list(contacts.values()))[1] if contacts else []
    output = views / "tactile_corr_label"
    output.mkdir(exist_ok=True)
    with np.load(views / "visible_pt_dict_processed.npz", allow_pickle=True) as visibility:
        for rgb in sorted((views / "rgb_images_processed").glob("*.png")):
            with Image.open(views / "object_masks_processed" / rgb.name) as image:
                mask = np.array(image.convert("L"))
            data = visibility[rgb.name].item()
            locations = dict(zip(data["vis_pt_idx"], data["proj_2D_loc"]))
            projected = {}
            for contact_id, point_id in zip(contacts, nearest):
                row, col = map(int, locations.get(point_id, (0, 0)))
                if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col] > 0):
                    row, col = 0, 0
                projected[contact_id] = [row, col]
            (output / f"{rgb.stem}_tactile_correspondence.json").write_text(json.dumps(projected, indent=4) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", "--root_dir", required=True, type=Path)
    args = parser.parse_args()
    scans = sorted(args.root_dir.glob("*/*/Preprocessed0111/scan.npy"))
    if not scans:
        parser.error("no category/object/Preprocessed0111/scan.npy files found")
    for scan in scans:
        project_object(scan.parents[1])


if __name__ == "__main__":
    main()
