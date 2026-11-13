"""Read existing Scan3dTac shape-to-image correspondences, never tactile depth.

No rendering, writes to source data, future frames, colors or whole-shape tokens.
The result is a MASKED organized pointmap, not a claimed dense tactile map.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from dinov2.tacdino.file_records import file_record

GEOMETRY = dict(name="tacdino.shape-local-pointmap.v1", crop_size=98, grid=7,
    source="processed_visible_scan_indices", coordinates="local_centered_camera_xyz",
    scale="RMS_radius_of_visible_local_points", collision="nearest_positive_camera_z",
    invalid="zero_with_explicit_mask_no_fill", channels=["x", "y", "z", "coverage"],
    scan_to_annotation="x,y,z -> x,-z,y", colors_used=False, future_tactile_used=False)


def rasterize(xyz, pixels_yx, foreground, xy, size=98, *, return_camera=False):
    """Camera XYZ -> visible masked local map; z-buffer BEFORE normalization."""
    xyz = np.asarray(xyz, dtype=np.float64)
    uv = np.asarray(pixels_yx)
    mask = np.asarray(foreground, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (len(xyz), 2) or mask.ndim != 2:
        raise ValueError("expected N,3 camera points, N,2 pixels, and H,W mask")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all() or not np.equal(uv, np.round(uv)).all():
        raise ValueError("nonfinite coordinates or noninteger processed pixel indices")
    uv = uv.astype(np.int64)
    height, width = mask.shape
    if ((uv < 0).any() or (uv[:, 0] >= height).any() or (uv[:, 1] >= width).any()):
        raise ValueError("processed pixel indices out of bounds")
    left, top = int(round(float(xy[0])))-size//2, int(round(float(xy[1])))-size//2
    ok = ((xyz[:, 2] > 0) & mask[uv[:, 0], uv[:, 1]] &
          (uv[:, 0] >= top) & (uv[:, 0] < top+size) &
          (uv[:, 1] >= left) & (uv[:, 1] < left+size))
    pts, local = xyz[ok], uv[ok]-[top, left]
    if len(pts) < 3:
        raise ValueError("insufficient visible local shape points")
    flat = local[:, 0]*size+local[:, 1]
    order = np.lexsort((pts[:, 2], flat))
    _, first = np.unique(flat[order], return_index=True)
    keep = order[first]
    pts, local = pts[keep], local[keep]
    center = pts.mean(0)
    radius = float(np.sqrt(np.mean(np.sum((pts-center)**2, axis=1))))
    if not np.isfinite(radius) or radius <= 1e-9:
        raise ValueError("degenerate local shape")
    out = np.zeros((size, size, 3), dtype=np.float32)
    valid = np.zeros((size, size), dtype=bool)
    out[local[:, 0], local[:, 1]] = (pts-center)/radius
    valid[local[:, 0], local[:, 1]] = True
    expected = np.zeros_like(valid)
    y0, y1, x0, x1 = max(top, 0), min(top+size, height), max(left, 0), min(left+size, width)
    if y1 > y0 and x1 > x0:
        expected[y0-top:y1-top, x0-left:x1-left] = mask[y0:y1, x0:x1]
    stats = dict(crop_xyxy=[left, top, left+size, top+size], mapped_points=int(ok.sum()),
        valid_pixels=int(valid.sum()), foreground_pixels=int(expected.sum()),
        foreground_coverage=float(valid.sum()/max(int(expected.sum()), 1)),
        collision_count=int(ok.sum()-valid.sum()), has_tactile_pixel_calibration=False)
    if return_camera:
        camera = np.zeros_like(out)
        camera[local[:, 0], local[:, 1]] = pts
        return out, valid, stats, camera
    return out, valid, stats


def rasterize_box(xyz, pixels_yx, foreground, crop_xyxy, size=98, *, flip=False,
                  return_camera=False):
    """Rasterize the exact RGB crop used by the tactile-paired local branch.

    Unlike :func:`rasterize`, the source crop can have an arbitrary size and
    is resized to ``size`` exactly like RandomResizedCrop.  The optional
    horizontal flip is the same already sampled for the RGB crop.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    uv = np.asarray(pixels_yx)
    foreground = np.asarray(foreground, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or uv.shape != (len(xyz), 2):
        raise ValueError("expected N,3 camera points and N,2 pixels")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all() or not np.equal(uv, np.round(uv)).all():
        raise ValueError("nonfinite coordinates or noninteger processed pixel indices")
    left, top, right, bottom = map(int, crop_xyxy)
    if right <= left or bottom <= top or size < 1:
        raise ValueError("invalid local crop box")
    uv = uv.astype(np.int64)
    height, width = foreground.shape
    if ((uv < 0).any() or (uv[:, 0] >= height).any() or (uv[:, 1] >= width).any()):
        raise ValueError("processed pixel indices out of bounds")
    ok = ((xyz[:, 2] > 0) & foreground[uv[:, 0], uv[:, 1]] &
          (uv[:, 0] >= top) & (uv[:, 0] < bottom) &
          (uv[:, 1] >= left) & (uv[:, 1] < right))
    pts, source_uv = xyz[ok], uv[ok]
    if len(pts) < 3:
        raise ValueError("insufficient visible local shape points")
    # Pixel centers are mapped into the resized output. Clamp handles the
    # half-open source box without inventing observations outside the crop.
    local_y = np.floor((source_uv[:, 0] - top + .5) * size / (bottom - top)).astype(np.int64)
    local_x = np.floor((source_uv[:, 1] - left + .5) * size / (right - left)).astype(np.int64)
    local_y = np.clip(local_y, 0, size - 1)
    local_x = np.clip(local_x, 0, size - 1)
    if flip:
        local_x = size - 1 - local_x
    local = np.column_stack((local_y, local_x))
    flat = local[:, 0] * size + local[:, 1]
    order = np.lexsort((pts[:, 2], flat))
    _, first = np.unique(flat[order], return_index=True)
    keep = order[first]
    pts, local = pts[keep], local[keep]
    center = pts.mean(0)
    radius = float(np.sqrt(np.mean(np.sum((pts - center) ** 2, axis=1))))
    if not np.isfinite(radius) or radius <= 1e-9:
        raise ValueError("degenerate local shape")
    out = np.zeros((size, size, 3), dtype=np.float32)
    valid = np.zeros((size, size), dtype=bool)
    out[local[:, 0], local[:, 1]] = (pts - center) / radius
    valid[local[:, 0], local[:, 1]] = True
    camera = np.zeros_like(out)
    camera[local[:, 0], local[:, 1]] = pts
    stats = dict(crop_xyxy=[left, top, right, bottom], mapped_points=int(ok.sum()),
        valid_pixels=int(valid.sum()), collision_count=int(ok.sum() - valid.sum()),
        resized_to=int(size), horizontal_flip=bool(flip),
        has_tactile_pixel_calibration=False)
    if return_camera:
        return out, valid, stats, camera
    return out, valid, stats


def pool_pointmap(points, valid, grid=7):
    size = points.shape[0]
    if points.shape != (size, size, 3) or valid.shape != (size, size) or size % grid:
        raise ValueError("pointmap must form an evenly divisible square grid")
    if not np.isfinite(points).all() or np.any(points[~valid] != 0):
        raise ValueError("invalid geometry pixels must be zero AND masked")
    b = size//grid
    m = valid.reshape(grid, b, grid, b).transpose(0, 2, 1, 3)
    p = points.reshape(grid, b, grid, b, 3).transpose(0, 2, 1, 3, 4)
    n = m.sum((2, 3))
    mean = (p*m[..., None]).sum((2, 3))/np.maximum(n[..., None], 1)
    return np.concatenate((mean, (n/(b*b))[..., None]), -1).reshape(grid*grid, 4).astype(np.float32)


@lru_cache(maxsize=4)
def _shape_inputs(root):
    root = Path(root)
    scan_path = root/"scan.npy"
    mapping_path = root/"tacdino/visible_pt_dict_processed.npz"
    scan = np.load(scan_path, allow_pickle=False)
    if scan.ndim != 2 or scan.shape[1] < 3 or not np.isfinite(scan[:, :3]).all():
        raise ValueError("invalid scan XYZ")
    # This trusted dataset cache was created by our existing resize_image.py.
    # Do not use this object-NPZ loader for arbitrary user-supplied model files.
    with np.load(mapping_path, allow_pickle=True) as archive:
        mapping = {key: archive[key].reshape(-1)[0] for key in archive.files}
    xyz = np.column_stack((scan[:, 0], -scan[:, 2], scan[:, 1]))
    return xyz, mapping, [file_record(p) for p in (scan_path, mapping_path)]


def load_local_pointmap(image_path, xy, *, return_camera=False):
    image = Path(image_path)
    if image.parent.name != "rgb_images_processed" or image.parent.parent.name != "tacdino":
        raise ValueError("requires explicitly processed shape-view RGB, not an arbitrary photo")
    root = image.parent.parent.parent
    xyz, mappings, source = _shape_inputs(str(root.resolve()))
    rec = mappings[image.name]
    idx = np.asarray(rec["vis_pt_idx"])
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or ((idx < 0) | (idx >= len(xyz))).any():
        raise ValueError("scan indices out of bounds")
    pose_path, rt_path = root/"render/obj_pose"/(image.stem+".npy"), root/"render/RT"/(image.stem+".npy")
    mask_path = root/"tacdino/object_masks_processed"/image.name
    pose, rt = np.load(pose_path, allow_pickle=False), np.load(rt_path, allow_pickle=False)
    if pose.shape != (4, 4) or rt.shape != (3, 4) or not np.isfinite(pose).all() or not np.isfinite(rt).all():
        raise ValueError("invalid object/camera transform")
    cam = (xyz[idx] @ pose[:3, :3].T+pose[:3, 3]) @ rt[:, :3].T+rt[:, 3]
    with Image.open(mask_path) as im:
        foreground = np.asarray(im.convert("L")) > 127
    result = rasterize(cam, rec["proj_2D_loc"], foreground, xy, return_camera=return_camera)
    points, valid, stats = result[:3]
    stats.update(protocol=GEOMETRY, source_files=source+[file_record(p) for p in (pose_path, rt_path, mask_path)],
        image_path=str(image), query_xy=list(map(float, xy)))
    return (*result[:2], stats, *result[3:])


def load_local_pointmap_box(image_path, crop_xyxy, *, size=98, flip=False,
                            return_camera=False):
    """Load geometry for the exact sampled local RGB crop and augmentation."""
    image = Path(image_path)
    if image.parent.name != "rgb_images_processed" or image.parent.parent.name != "tacdino":
        raise ValueError("requires explicitly processed shape-view RGB, not an arbitrary photo")
    root = image.parent.parent.parent
    xyz, mappings, source = _shape_inputs(str(root.resolve()))
    rec = mappings[image.name]
    idx = np.asarray(rec["vis_pt_idx"])
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or ((idx < 0) | (idx >= len(xyz))).any():
        raise ValueError("scan indices out of bounds")
    pose_path, rt_path = root/"render/obj_pose"/(image.stem+".npy"), root/"render/RT"/(image.stem+".npy")
    mask_path = root/"tacdino/object_masks_processed"/image.name
    pose, rt = np.load(pose_path, allow_pickle=False), np.load(rt_path, allow_pickle=False)
    if pose.shape != (4, 4) or rt.shape != (3, 4) or not np.isfinite(pose).all() or not np.isfinite(rt).all():
        raise ValueError("invalid object/camera transform")
    cam = (xyz[idx] @ pose[:3, :3].T + pose[:3, 3]) @ rt[:, :3].T + rt[:, 3]
    with Image.open(mask_path) as im:
        foreground = np.asarray(im.convert("L")) > 127
    result = rasterize_box(cam, rec["proj_2D_loc"], foreground, crop_xyxy,
        size=size, flip=flip, return_camera=return_camera)
    stats = result[2]
    stats.update(protocol=GEOMETRY, source_files=source + [file_record(p) for p in (pose_path, rt_path, mask_path)],
        image_path=str(image), query_crop_xyxy=list(map(int, crop_xyxy)))
    return (*result[:2], stats, *result[3:])
