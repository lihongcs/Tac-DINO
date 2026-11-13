# Touch3D data preparation

The [Touch3D dataset on Hugging Face](https://huggingface.co/datasets/lihong-cs/Tac-DINO) provides two releases. Use `raw` to regenerate scan views from the original meshes. Use `Scan3dTac` to work with the published views or prepare the inputs for GeoTAC pretraining.

Run these commands from the Tac-DINO repository root. Install the dependencies with `python -m pip install -r requirements.txt -r requirements-preprocess.txt`.

## 1. Download the raw data

Start with one public object. The dry run reports the compressed download size and restored size. Selecting one object still requires downloading the shards that contain its files.

```sh
python -m scripts.download_touch3d --group raw --object wipes/obj_0000 \
  --output ./data/touch3d_raw --anonymous --dry-run
python -m scripts.download_touch3d --group raw --object wipes/obj_0000 \
  --output ./data/touch3d_raw --anonymous
```

Remove `--object` to restore the full raw release and check its disk requirements with `--dry-run` first. The downloader verifies file hashes and resumes when you repeat the same command. The restored tree uses `<category>/<object_id>/` and contains `3D_Scanning/`, `Annotation/`, and `Tactile/`. The downloader places label maps and public split lists in `./data/touch3d_raw_metadata/`.

## 2. Regenerate views and contact correspondences

Use Blender 4.x with Cycles and make it available as `blender`. The normalization script needs `joblib` in Blender's Python environment. Put at least one HDRI lighting file (`.hdr` or `.exr`) in `./data/hdri/`.

Run the conversion on a writable copy. Start with a fresh `./data/touch3d_work` directory. Each object needs `3D_Scanning/scan.ply` and `Annotation/tactile_pc_location.json` for mesh and contact normalization.

```sh
cp -R ./data/touch3d_raw ./data/touch3d_work
ASSETS=dataprocess/_01_process_assets

blender --background --python-exit-code 1 \
  --python "$ASSETS/_00_geomotry_process/clear_parents_and_normalize.py" \
  -- --object_dir ./data/touch3d_work
python "$ASSETS/_00_geomotry_process/extract_pointcloud.py" \
  --data_path ./data/touch3d_work
python "$ASSETS/_01_rendering/wrapper.py" \
  --data_path ./data/touch3d_work --hdr-dir ./data/hdri
python "$ASSETS/_02_postprocessing/convert_exr_depth.py" \
  --object_dir ./data/touch3d_work
python "$ASSETS/_02_postprocessing/find_visible_points.py" \
  --object_dir ./data/touch3d_work
python -m scripts.process_scan_views --root_dir ./data/touch3d_work
python -m scripts.project_contacts --root-dir ./data/touch3d_work
```

Under each object's `Preprocessed0111/`, `scan.npy` contains the sampled scan, `render/` contains rendered views, depth and camera parameters, and `tacdino/` contains cropped 224 × 224 RGB images, masks and projected contact labels. The projected coordinates are stored as `[row, column]`; `[0, 0]` marks a contact that is unavailable in that view.

For example, open one converted image and inspect the matching contact positions:

```python
import json
from pathlib import Path
from PIL import Image

views = Path("./data/touch3d_work/wipes/obj_0000/Preprocessed0111/tacdino")
image_path = sorted((views / "rgb_images_processed").glob("*.png"))[0]
with Image.open(image_path) as image:
    print(image.size)
contacts_path = views / "tactile_corr_label" / f"{image_path.stem}_tactile_correspondence.json"
contacts = json.loads(contacts_path.read_text())
visible_contacts = {
    contact_id: position
    for contact_id, position in contacts.items()
    if position != [0, 0]
}
print(visible_contacts)
```

To pair regenerated views with the published tactile PLYs, restore the `geometry` profile into the working copy using the same object selection as the raw download:

```sh
python -m scripts.download_touch3d --group Scan3dTac --profile geometry \
  --object wipes/obj_0000 --output ./data/touch3d_work --anonymous
```

The geometry conversion commands below also accept this working copy. Set `--data-root` to `./data/touch3d_work` and use a fresh directory for derived files. For the paper's training recipe, use the full published `Scan3dTac` release.

## 3. Prepare the published data for GeoTAC

The full `Scan3dTac` profile supplies the published rendered views, annotations, tactile MP4 clips, and organized tactile PLY reconstructions. Use it in a separate directory for the paper recipe. The raw release contains sensor recordings, including HDF5 depth files. This code release starts tactile geometry conversion from the published PLYs and does not include the original video-to-PLY reconstruction or sensor calibration.

```sh
python -m scripts.download_touch3d --group Scan3dTac --profile full \
  --output ./data/touch3d --anonymous --dry-run
python -m scripts.download_touch3d --group Scan3dTac --profile full \
  --output ./data/touch3d --anonymous
```

Use `--profile core` for published views, annotations and clips alone. The default training and test split lists refer to the full release. Use the same object selection across the inputs if you are preparing a subset for inspection.

Index the frame-10 PLYs, convert them into per-contact point arrays, relative depth and validity masks, then certify the result:

```sh
python -m scripts.index_tactile_geometry \
  --data-root ./data/touch3d --output ./data/derived/frame10.jsonl
python -m scripts.prepare_tactile_geometry \
  --data-root ./data/touch3d --source-manifest ./data/derived/frame10.jsonl \
  --output ./data/derived/tactile_geometry
python -m scripts.certify_tactile_geometry \
  --manifest ./data/derived/tactile_geometry/manifest.json \
  --output ./data/derived/tactile_geometry/certified.json \
  --report ./data/derived/tactile_geometry/report
```

Check `./data/derived/tactile_geometry/report/status.json` for usable contacts and reasons for unavailable geometry. Each usable entry in `certified.json` points to an `.npz` file relative to the manifest directory. Read it with `numpy.load(path, allow_pickle=False)` to access `xyz`, `height` and `valid`.

For pretraining on the full release, build the paper's object split and contact filter:

```sh
python -m scripts.build_touch3d_split \
  --source-train-split ./data/touch3d_metadata/split0111/train_split.json \
  --output-dir ./data/splits
python -m scripts.filter_touch3d_contacts \
  --data-root ./data/touch3d --train-split ./data/splits/train_core_split.json \
  --output ./data/splits/train_contact_filter.json
```

Pass the restored data root, sibling metadata directory, training split, contact filter and `certified.json` to the [GeoTAC training command](../README.md#geotac-pretraining). Add `--prepare-only` to validate inputs and write the training configuration before launching GPU training.

Keep generated data, manifests, logs and checkpoints local. Some generated files record the resolved paths of their inputs. Regenerate the geometry manifests if the dataset root moves.
