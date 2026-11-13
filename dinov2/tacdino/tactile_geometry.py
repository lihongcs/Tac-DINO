"""Read sensor-reconstructed PLY without confusing it with scanned geometry.

XYZ/RGB source files remain immutable. Rasterization requires a verified XY
grid; arbitrary point ordering is never treated as sensor pixel ordering.
"""
from pathlib import Path
from functools import lru_cache
import json
import os
import hashlib
import io
import numpy as np


PLY_TYPES = dict(char='i1', uchar='u1', short='<i2', ushort='<u2', int='<i4',
    uint='<u4', float='<f4', double='<f8', float32='<f4', float64='<f8')


class UnavailableGeometry(ValueError):
    """An observed reconstruction failure, not a parser/checksum failure."""


def preprocessing_identity():
    """Bind certification to the exact validator and unchanged model adapter."""
    paths = (Path(__file__), Path(__file__).with_name('pointbert_runtime.py'))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def preprocessing_certificate(checked, accepted):
    return dict(complete=True, all_usable_passed=True, checked=checked,
        accepted=accepted, implementation=preprocessing_identity())


def training_views(arrays):
    """One validator for offline certification and actual training inputs.

    Do not make exploding reconstruction heights acceptable by clipping or
    float64 normalization. Test the exact unchanged float32 Point-BERT path.
    Structural/checksum failures stay fatal; numerically unusable geometry is
    quarantined explicitly by the offline manifest builder.
    """
    from dinov2.tacdino.pointbert_runtime import prepare_local_pool, POOL_SIZE
    xyz, height, mask = arrays['xyz'], arrays['height'], arrays['valid']
    if xyz.shape != (98, 98, 3) or height.shape != (98, 98) or mask.shape != (98, 98):
        raise ValueError('tactile geometry cache shape mismatch')
    if mask.dtype != np.bool_ or not 32 <= mask.sum() <= 98*98:
        raise ValueError('tactile geometry cache validity mismatch')
    if not np.isfinite(xyz).all() or not np.isfinite(height).all():
        raise UnavailableGeometry('nonfinite tactile geometry cache')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            points = arrays['point_xyz'] if 'point_xyz' in arrays else xyz[mask]
            pc = prepare_local_pool(points)
    except FloatingPointError as exc:
        raise UnavailableGeometry('Point-BERT float32 preprocessing overflow: '+str(exc)) from exc
    except ValueError as exc:
        if str(exc) not in ('nonfinite local XYZ', 'degenerate local point cloud'):
            raise
        raise UnavailableGeometry('Point-BERT preprocessing rejected: '+str(exc)) from exc
    depth = np.stack([height, height, height, mask.astype(np.float32)])
    if pc.shape != (POOL_SIZE, 3) or not np.isfinite(pc).all() or not np.isfinite(depth).all():
        raise UnavailableGeometry('invalid model-ready tactile geometry')
    return depth, pc


def read_tactile_ply(path):
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('not a PLY')
        fields, count, vertex, binary = [], None, False, False
        for _ in range(256):
            words = stream.readline().decode('ascii').strip().split()
            if not words: raise ValueError('truncated PLY header')
            if words[0] == 'format': binary = words[1] == 'binary_little_endian'
            if words[0] == 'element':
                vertex = words[1] == 'vertex'
                if vertex: count = int(words[2])
            if words[0] == 'property' and vertex:
                if len(words) != 3 or words[1] not in PLY_TYPES:
                    raise ValueError('unsupported PLY vertex property')
                fields.append((words[2], PLY_TYPES[words[1]]))
            if words[0] == 'end_header': break
        else: raise ValueError('oversized PLY header')
        if not binary or count is None or not 32 <= count <= 2000000:
            raise ValueError('unsupported PLY encoding or count')
        dtype = np.dtype(fields)
        if not {'x', 'y', 'z', 'red', 'green', 'blue'} <= set(dtype.names):
            raise ValueError('XYZ and RGB are required')
        data = np.fromfile(stream, dtype=dtype, count=count)
    if len(data) != count: raise ValueError('truncated PLY vertex payload')
    xyz = np.stack([data[k] for k in ('x', 'y', 'z')], -1)
    rgb = np.stack([data[k] for k in ('red', 'green', 'blue')], -1)
    return xyz, rgb


def organized_grid(xyz, rgb, return_mask=False):
    # Some exported pixel coordinates have tiny floating point roundoff.
    # Snap only a proven integer raster in the derived view, never the PLY.
    finite = np.isfinite(xyz).all(1)
    xyz, rgb = xyz[finite].copy(), rgb[finite]
    if len(xyz) < 32: raise UnavailableGeometry('insufficient finite tactile points')
    rounded = np.rint(xyz[:, :2])
    if np.max(np.abs(rounded-xyz[:, :2])) >= 1e-5:
        raise ValueError('PLY is not an integer sensor XY raster')
    xyz[:, :2] = rounded
    ix, iy = rounded.astype(np.int64).T
    width, height = int(ix.max())+1, int(iy.max())+1
    if min(ix.min(), iy.min()) < 0 or not 16 <= min(width, height) or max(width, height) > 4096:
        raise ValueError('invalid sensor XY grid extent')
    address = iy*width+ix
    # Invalid reconstruction points can repeat (0,0). Do not arbitrarily
    # keep the last depth at a collided coordinate. Mark all collisions and
    # missing/nonfinite cells invalid; retain the remaining measured surface.
    counts = np.bincount(address, minlength=height*width)
    unique = counts[address] == 1
    mask = (counts == 1).reshape(height, width)
    if mask.sum() < 32 or mask.mean() < .5:
        raise ValueError('insufficient uniquely observed sensor grid coverage')
    points = np.zeros((height, width, 3), dtype=np.float32)
    colors = np.zeros((height, width, 3), dtype=np.uint8)
    points[iy[unique], ix[unique]] = xyz[unique]
    colors[iy[unique], ix[unique]] = rgb[unique]
    return (points, colors, mask) if return_mask else (points, colors)


def geometry_arrays(xyz, rgb, size=98):
    import cv2
    points, _, mask = organized_grid(xyz, rgb, return_mask=True)
    # Actual observed points only. Do not synthesize 3D samples by bilinear
    # interpolation or assume a calibrated sensor-to-object transformation.
    local = cv2.resize(points, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mask.sum() < 32: raise ValueError('insufficient downsampled sensor geometry')
    z = local[..., 2]
    center = np.median(z[mask])
    scale = max(float(np.quantile(np.abs(z[mask]-center), .95)), 1e-6)
    height = np.zeros_like(z); height[mask] = (z[mask]-center)/scale
    return dict(xyz=np.ascontiguousarray(local, dtype=np.float32),
        height=np.ascontiguousarray(height, dtype=np.float32), valid=mask)


@lru_cache(maxsize=2)
def geometry_index(path):
    value = json.loads(Path(path).read_text())
    if not value.get('complete') or value.get('frame') != 10:
        raise ValueError('complete certified tactile geometry index required')
    certificate = value.get('preprocessing_validation', {})
    accepted = sum(bool(r['usable']) for r in value['entries'].values())
    if (not certificate.get('complete') or not certificate.get('all_usable_passed')
            or certificate.get('accepted') != accepted
            or certificate.get('implementation') != preprocessing_identity()
            or any(r['usable'] and not r.get('cache_sha256') for r in value['entries'].values())):
        raise ValueError('full exact Point-BERT preprocessing certification required')
    return value


def training_geometry(video):
    """One same-contact geometric view; never replaces the dynamic RGB clip."""
    index = geometry_index(os.environ['TACDINO_TACTILE_GEOMETRY_MANIFEST'])
    key = str(Path(video).resolve().relative_to(Path(index['data_root']).resolve()))
    if key not in index['entries']:
        raise ValueError('contact absent from audited tactile geometry index: '+key)
    row = index['entries'][key]
    if not row['usable']:
        # Filled with the observed external geometry by the caller. The
        # accompanying observation-valid flag removes these views from CL.
        return None, None, False
    try:
        payload = (Path(os.environ['TACDINO_TACTILE_GEOMETRY_MANIFEST']).parent / row['cache']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['cache_sha256']:
            raise ValueError('certified tactile geometry cache checksum changed')
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            depth, pc = training_views(arrays)
    except Exception as exc:
        # An unexpected later failure identifies the exact contact. Do not
        # silently discard samples at runtime or conceal cache drift.
        raise ValueError(f'tactile geometry failed for {key} ({row["cache"]}): {exc}') from exc
    return depth, pc, True


def image_variants(colors):
    """Only for correspondence audit, never choose an orientation by labels."""
    for transpose in (False, True):
        image = colors.transpose(1, 0, 2) if transpose else colors
        for vertical in (False, True):
            for horizontal in (False, True):
                value = image[::-1] if vertical else image
                value = value[:, ::-1] if horizontal else value
                for bgr in (False, True):
                    yield dict(transpose=transpose, vertical=vertical, horizontal=horizontal, bgr=bgr), value[..., ::-1] if bgr else value


def video_frame(path, frame):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 <= frame < count: raise ValueError('geometry frame is outside video')
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok: raise ValueError('cannot decode geometry-matched video frame')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), count
    finally: cap.release()


def color_match(colors, image):
    import cv2
    target = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    scores = []
    for transform, value in image_variants(colors):
        candidate = cv2.resize(value, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)/255
        scores.append((float(np.mean((candidate-target)**2)), transform))
    return min(scores, key=lambda row: row[0])
