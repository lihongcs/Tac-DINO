"""Index frame-10 tactile PLYs restored under the Touch3D data root."""
import argparse
import json
from pathlib import Path
import re

from dinov2.tacdino.file_records import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.data_root.glob("*/*/Tactile/tactile_3d_*/*.ply"))
    if not paths:
        parser.error("no tactile PLYs found under category/object/Tactile/tactile_3d_*")
    rows = []
    for path in paths:
        match = re.fullmatch(r"(?:contact|tactile)_(\d+)_(\d+)", path.stem)
        if match is None:
            parser.error(f"unrecognized geometry filename: {path.relative_to(args.data_root)}")
        contact_id = int(path.parent.name.rsplit("_", 1)[-1])
        if int(match.group(2)) == 10 or int(match.group(1)) != contact_id:
            rows.append(dict(path=path.relative_to(args.data_root).as_posix(), sha256=sha(path),
                             directory_filename_id_mismatch=int(match.group(1)) != contact_id))
    if not any(Path(row["path"]).stem.endswith("_10") for row in rows):
        parser.error("no frame-10 geometry found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
