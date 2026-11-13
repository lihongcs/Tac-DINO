"""Build the hard-exclusion contact filter consumed in GeoTAC training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2

from dinov2.data.datasets.tacdino.contact_alignment import index_contact_mapping, tactile_video_mapping
from dinov2.tacdino.file_records import write


FRAME_INDEX = 10


def audit_video(item):
    contact_id, path = item
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return contact_id, "video_open_failed"
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= FRAME_INDEX:
            return contact_id, "clip_frame_unavailable"
        for _ in range(count):
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                return contact_id, "video_decode_failed"
        return contact_id, None
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    instances = json.loads(args.train_split.read_text())
    if not instances:
        parser.error("empty train split")
    exclusions, audit = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for instance in instances:
            path = Path(instance)
            if path.is_absolute() or len(path.parts) != 2 or ".." in path.parts:
                parser.error(f"invalid object ID: {instance!r}")
            root = args.data_root / path
            material_path = root / "Annotation/tactile_material.json"
            materials = index_contact_mapping(json.loads(material_path.read_text()), material_path)
            videos = tactile_video_mapping(root / "Tactile")
            paired = sorted(materials.keys() & videos.keys())
            reasons = {contact: "contact_inventory_mismatch" for contact in materials.keys() ^ videos.keys()}
            reasons.update((contact, reason) for contact, reason in
                           pool.map(audit_video, ((contact, videos[contact]) for contact in paired)) if reason)
            if not set(paired).difference(reasons):
                raise ValueError(f"No usable tactile video for {instance}")
            if reasons:
                audit[instance] = reasons
                exclusions[instance] = sorted(reasons)
    write(args.output, dict(excluded_contact_ids=exclusions, exclusion_reasons=audit, frame_index=FRAME_INDEX))
    print(f"Audited {len(instances)} objects, excluded {sum(map(len, exclusions.values()))} contacts")


if __name__ == "__main__":
    main()
