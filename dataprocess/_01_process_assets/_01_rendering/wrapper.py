"""Render Touch3D scans with Blender using the registered camera positions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--hdr-dir", type=Path, required=True)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--object-list", type=Path, help="JSON list of category/object or category-object IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    hdri = sorted(p for p in args.hdr_dir.iterdir() if p.suffix.lower() in {".hdr", ".exr"})
    if not hdri:
        parser.error("HDR directory has no .hdr or .exr files")
    requested = set(json.loads(args.object_list.read_text())) if args.object_list else None
    renderer = Path(__file__).with_name("render_glb.py").resolve()
    meshes = []
    for mesh in sorted(args.data_path.glob("*/*/Preprocessed0111/scan.glb")):
        instance = mesh.parents[1].relative_to(args.data_path)
        if requested is not None and not ({instance.as_posix(), "-".join(instance.parts)} & requested):
            continue
        depth = mesh.parent / "render" / "depth_exr"
        if len(list(depth.glob("*.exr"))) != 19:
            meshes.append(mesh)
    rng = random.Random(args.seed)
    commands = [
        [args.blender, "--background", "--python-exit-code", "1", "--python", str(renderer),
         "--", str(mesh.resolve()), str(rng.choice(hdri).resolve()),
         str((mesh.parent / "render").resolve()), "2", "490", "490"]
        for mesh in meshes[args.start:args.end]
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda command: subprocess.run(command, check=True), commands))


if __name__ == "__main__":
    main()
