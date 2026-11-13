"""Canonical TRAIN/VAL/TEST-independent geometric conversion, without fitting.

Never renames source labels. Ambiguous directory/file contact IDs receive an
explicit unavailable observation, while their original RGB/tactile training
samples remain in the unchanged dataset. No TEST score enters this process.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from dinov2.tacdino.tactile_geometry import (read_tactile_ply, geometry_arrays, video_frame,
    UnavailableGeometry, training_views, preprocessing_certificate)
from dinov2.tacdino.file_records import write, file_record


def convert(item):
    root, cache, row = item
    path = root/row['path']
    contact = int(path.parent.name.split('_')[-1])
    video = path.parent.parent/f'tactile_{contact}.mp4'
    key = str(video.relative_to(root))
    result = dict(video=key, source=row['path'], source_sha256=row['sha256'], frame=10,
        usable=False, original_data_modified=False)
    if row.get('directory_filename_id_mismatch'):
        return dict(result, reason='unresolved_directory_filename_contact_mismatch')
    if not video.is_file(): return dict(result, reason='missing_video')
    try:
        _, count = video_frame(video, 10)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('source checksum differs from imported archive')
        xyz, rgb = read_tactile_ply(path)
        result.update(source_points=len(xyz), finite_points=int(np.isfinite(xyz).all(1).sum()))
        arrays = geometry_arrays(xyz, rgb)
        training_views(arrays)
        destination = cache/Path(key).with_suffix('.npz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open('xb') as stream: np.savez_compressed(stream, **arrays)
        with np.load(destination, allow_pickle=False) as saved:
            for name in arrays:
                if not np.array_equal(arrays[name], saved[name]): raise ValueError('cache roundtrip differs')
        result.update(usable=True, cache=str(destination.relative_to(cache.parent)), cache_sha256=file_record(destination)['sha256'], video_frames=count,
            valid_fraction=float(arrays['valid'].mean()),
            raster_height_range=[float(arrays['height'].min()), float(arrays['height'].max())],
            rgb_is_grayscale=bool(np.array_equal(rgb[:, 0], rgb[:, 1]) and np.array_equal(rgb[:, 1], rgb[:, 2])))
        return result
    except UnavailableGeometry as exc:
        return dict(result, reason='unavailable_reconstruction', reconstruction_error=str(exc))
    except Exception as exc:
        return dict(result, reason=type(exc).__name__+': '+str(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.output.exists() and not a.resume: raise ValueError('use a new derived cache directory')
    if a.resume and (a.output/'manifest.json').exists():
        if json.loads((a.output/'manifest.json').read_text())['complete']:
            raise ValueError('do not overwrite a published manifest')
        os.rename(a.output/'manifest.json', a.output/f'manifest_failed_{time.time_ns()}.json')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    rows = [json.loads(line) for line in a.source_manifest.read_text().splitlines()]
    ambiguous = {str(Path(r['path']).parent) for r in rows if r.get('directory_filename_id_mismatch')}
    rows = [dict(r, directory_filename_id_mismatch=str(Path(r['path']).parent) in ambiguous)
        for r in rows if Path(r['path']).stem.endswith('_10')]
    unique = {}
    for row in rows:
        path = Path(row['path']); key = str(path.parent)
        if key in unique:
            # Never select one of conflicting same-frame filenames as truth.
            unique[key]['directory_filename_id_mismatch'] = True
        else: unique[key] = row
    source_frame_files = len(rows)
    rows = list(unique.values())
    entries, last, start = {}, 0., time.time()
    def status(complete=False):
        write(a.output/'status.json', dict(complete=complete, processed=len(entries), total=len(rows),
            usable=sum(r['usable'] for r in entries.values()), elapsed_seconds=round(time.time()-start, 1),
            unavailable_reasons=dict(Counter(r['reason'] for r in entries.values() if not r['usable']))))
    with (a.output/f'conversion_{time.time_ns()}.jsonl').open('x') as log, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(convert, ((a.data_root.resolve(), a.output/'cache', r) for r in rows)):
            if row['video'] in entries: raise ValueError('duplicate contact conversion: '+row['video'])
            entries[row['video']] = row
            log.write(json.dumps(row)+'\n'); log.flush()
            if time.time()-last >= 10:
                status(); last = time.time(); print(f'{len(entries)}/{len(rows)}', flush=True)
    unexpected = [r for r in entries.values() if not r['usable'] and r['reason'] not in
        ('unresolved_directory_filename_contact_mismatch', 'missing_video', 'unavailable_reconstruction')]
    # A few quarantined naming problems must not hide corrupt arrays or a
    # broken converter affecting the full dataset.
    result = dict(complete=not unexpected, data_root=str(a.data_root.resolve()), frame=10,
        source_manifest=file_record(a.source_manifest), entries=entries,
        input_geometry='sensor_pixel_XY_and_native_reconstruction_Z', metric_units_verified=False,
        pc_preprocessing='one_isotropic_center_unit_sphere_on_native_XYZ_no_axis_unit_guess',
        depth_preprocessing='median_centered_height_divided_by_95pct_absolute_deviation',
        pixel_registration_to_scan=False, original_files_modified=False, training_sample_filter_changed=False,
        source_frame_files=source_frame_files,
        total=len(entries), usable=sum(r['usable'] for r in entries.values()),
        unresolved=[r for r in entries.values() if not r['usable']], unexpected_errors=unexpected)
    if not unexpected:
        result['preprocessing_validation'] = preprocessing_certificate(len(entries), result['usable'])
    write(a.output/'manifest.json', result); status(not unexpected)
    print(json.dumps({k:result[k] for k in ('complete', 'total', 'usable', 'unexpected_errors')}), flush=True)
    if unexpected: raise SystemExit(2)


if __name__ == '__main__': main()
