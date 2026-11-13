"""Certify all converted contacts using the exact training preprocessing.

Publish a new immutable manifest without copying/replacing any raw data or
cache. Only explicitly unusable reconstructed views are quarantined; original
RGB/tactile contacts, splits, losses and model normalization stay unchanged.
"""
import argparse
from functools import partial
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from dinov2.tacdino.tactile_geometry import training_views, UnavailableGeometry, preprocessing_certificate
from dinov2.tacdino.file_records import read, write, file_record


def certify_contact(item, root=Path(".")):
    key, source = item
    row = dict(source)
    if not row['usable']:
        return key, row, False
    try:
        payload = (root / row['cache']).read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if row.get('cache_sha256') and row['cache_sha256'] != checksum:
            raise ValueError('previously certified cache checksum changed')
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            training_views(arrays)
        row.update(cache_sha256=checksum, preprocessing_passed=True)
        return key, row, False
    except UnavailableGeometry as exc:
        row.update(usable=False, reason='pointbert_preprocessing_rejected',
            preprocessing_passed=False, preprocessing_error=str(exc), original_data_modified=False)
        return key, row, False
    except Exception as exc:
        row.update(usable=False, reason='unexpected_certification_failure',
            preprocessing_error=type(exc).__name__+': '+str(exc))
        return key, row, True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve() == a.manifest.resolve():
        raise ValueError('certification must publish a new manifest')
    if not 1 <= a.workers <= 32: raise ValueError('invalid worker count')
    source_record = file_record(a.manifest)
    original = read(a.manifest)
    if not original.get('complete') or original.get('frame') != 10:
        raise ValueError('complete original conversion manifest required')
    entries, unexpected, start, last = {}, [], time.time(), 0.
    a.report.mkdir(parents=True, exist_ok=True)
    def status(complete=False):
        value = dict(complete=complete, processed=len(entries), total=len(original['entries']),
            usable=sum(r['usable'] for r in entries.values()),
            elapsed_seconds=round(time.time()-start, 2),
            unavailable_reasons=dict(Counter(r['reason'] for r in entries.values() if not r['usable'])),
            unexpected_errors=unexpected)
        write(a.report/'status.json', value)
        return value
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for key, row, failed in pool.map(partial(certify_contact, root=a.manifest.resolve().parent), original['entries'].items()):
            if row.get('cache'):
                row['cache'] = os.path.relpath(a.manifest.resolve().parent / row['cache'], a.output.resolve().parent)
            entries[key] = row
            if failed: unexpected.append(dict(video=key, error=row['preprocessing_error']))
            if time.time()-last > 10:
                status(); last = time.time()
                print(f'certified {len(entries)}/{len(original["entries"])}', flush=True)
    if file_record(a.manifest) != source_record:
        raise ValueError('source manifest changed during certification')
    result = dict(original, entries=entries, certified_from=source_record,
        total=len(entries), usable=sum(r['usable'] for r in entries.values()),
        unresolved=[r for r in entries.values() if not r['usable']], unexpected_errors=unexpected,
        complete=not unexpected)
    if not unexpected:
        result['preprocessing_validation'] = preprocessing_certificate(
            sum(r['usable'] for r in original['entries'].values()), result['usable'])
    else:
        result.pop('preprocessing_validation', None)
    write(a.output, result)
    print(json.dumps(status(not unexpected)), flush=True)
    if unexpected: raise SystemExit(2)


if __name__ == '__main__': main()
