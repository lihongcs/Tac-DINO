#!/usr/bin/env python3
"""Download and restore Tac-DINO one verified archive at a time."""
import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath

REPO = 'lihong-cs/Tac-DINO'
BLOCK = 4 * 1024**2


def network_retry(operation):
    for attempt in range(5):
        try:
            return operation()
        except Exception as exc:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if (isinstance(exc, (ValueError, TypeError, ImportError, FileNotFoundError, PermissionError)) or
                    code not in {None, 429, 500, 502, 503, 504} or attempt == 4):
                raise
            delay = 2 ** (attempt + 1)
            print(f'Network retry {attempt + 1}/4 in {delay}s ({type(exc).__name__}).', flush=True)
            # Recreate failed proxy/TLS connections before the next attempt.
            try:
                from huggingface_hub import utils
                close = getattr(utils, 'close_session', None)
                if close is not None:
                    close()
            except ImportError:
                pass
            time.sleep(delay)


def download_http_ranges(url, target, entry, force=False, session=None, chunk_size=8 * 1024**2):
    """Bound each HTTP request and resume from the last complete range."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.' + entry['archive_sha256'] + '.incomplete')
    if force:
        target.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    if target.is_file():
        return target
    total = entry['archive_bytes']
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        offset = 0
    if offset != total:
        offset -= offset % chunk_size
    with partial.open('r+b' if partial.exists() else 'w+b') as stream:
        stream.truncate(offset)
    last_report = time.monotonic()
    while offset < total:
        start, end = offset, min(offset + chunk_size, total) - 1
        def transfer():
            current = session
            headers = {'Range': f'bytes={start}-{end}'}
            # Commit and checksum remain pinned; refresh signed redirects through caches.
            fresh_url = url + ('&' if '?' in url else '?') + f'tacdino_range={start}-{end}&tacdino_nonce={time.time_ns()}'
            if current is None:
                response_context = urllib.request.urlopen(urllib.request.Request(fresh_url, headers=headers), timeout=45)
            elif hasattr(current, 'stream'):
                response_context = current.stream('GET', fresh_url, headers=headers, follow_redirects=True, timeout=45)
            else:
                response_context = current.get(fresh_url, headers=headers, allow_redirects=True, stream=True, timeout=45)
            received = 0
            with response_context as response, partial.open('r+b') as out:
                if hasattr(response, 'raise_for_status'):
                    response.raise_for_status()
                status = getattr(response, 'status_code', getattr(response, 'status', None))
                if status != 206 or response.headers.get('content-range') != f'bytes {start}-{end}/{total}':
                    raise ValueError('Server did not honor the requested byte range. Retry with --use-xet or another endpoint.')
                out.seek(start)
                out.truncate(start)
                if current is None:
                    chunks = iter(lambda: response.read(1024**2), b'')
                else:
                    chunks = response.iter_bytes(chunk_size=1024**2) if hasattr(response, 'iter_bytes') else response.iter_content(chunk_size=1024**2)
                for block in chunks:
                    received += len(block)
                    if received > end - start + 1:
                        raise ValueError('HTTP response exceeded the requested range')
                    out.write(block)
            if received != end - start + 1:
                raise ConnectionError('Incomplete byte-range response')
        network_retry(transfer)
        offset = end + 1
        if time.monotonic() - last_report > 15 or offset == total:
            print(f'  Downloaded {offset / 1e6:.1f}/{total / 1e6:.1f} MB', flush=True)
            last_report = time.monotonic()
    os.replace(partial, target)
    return target


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if not name or '\\' in name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Invalid relative path: {name}')
    target = root.joinpath(*relative.parts)
    target.resolve().relative_to(root.resolve())
    if target.is_symlink():
        raise ValueError(f'Destination is a symbolic link: {target}')
    return target


def read_index(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def make_plan(index, group, profile='full', objects=(), categories=(), split_ids=None, patterns=()):
    plan = {}
    logical_files = logical_bytes = 0
    selected_objects = set()
    for row in read_index(index):
        parts = PurePosixPath(row['path']).parts
        if len(parts) < 3:
            raise ValueError(f'Invalid indexed path: {row["path"]}')
        uid = '/'.join(parts[:2])
        if objects and uid not in objects:
            continue
        if categories and parts[0] not in categories:
            continue
        if split_ids is not None and uid not in split_ids:
            continue
        if patterns and not any(fnmatch.fnmatchcase(row['path'], pattern) for pattern in patterns):
            continue
        if profile != 'full' and row['profile'] != profile:
            continue
        sha = row['sha256']
        if not re.fullmatch('[0-9a-f]{64}', sha) or row['blob'] != f'blobs/{sha[:2]}/{sha}':
            raise ValueError('Invalid indexed SHA-256/blob mapping')
        archive = row['archive']
        if not archive.startswith(group + '/') or not archive.endswith(('.tar.zst', '.tar.gz')):
            raise ValueError(f'Unexpected indexed archive: {archive}')
        safe_path(Path('/'), archive)
        blobs = plan.setdefault(archive, {})
        spec = blobs.setdefault(sha, {'bytes': row['bytes'], 'paths': []})
        if spec['bytes'] != row['bytes']:
            raise ValueError('Conflicting indexed payload sizes')
        spec['paths'].append(row['path'])
        logical_files += 1
        logical_bytes += row['bytes']
        selected_objects.add(uid)
    if not plan:
        raise ValueError('No files match this selection. Check the object/category, split and profile.')
    return plan, {'objects': len(selected_objects), 'logical_files': logical_files,
                  'logical_bytes': logical_bytes,
                  'unique_bytes': sum(spec['bytes'] for blobs in plan.values() for spec in blobs.values()),
                  'archives': len(plan)}


def fingerprint(path):
    info = path.stat()
    return [info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino]


def receipt_valid(receipt, blobs, output, mode, verify_existing=False):
    if not receipt.is_file():
        return False
    try:
        data = json.loads(receipt.read_text())
        if data.get('mode') != mode:
            return False
        records = data['files']
        for sha, spec in blobs.items():
            for name in spec['paths']:
                path = safe_path(output, name)
                entry = records.get(name)
                if not entry or entry['sha256'] != sha or entry['fingerprint'] != fingerprint(path):
                    return False
                if verify_existing and file_sha256(path) != sha:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_receipt(path, blobs, output, mode):
    records = {}
    if path.is_file():
        try:
            records = json.loads(path.read_text()).get('files', {})
        except (ValueError, OSError):
            pass
    for sha, spec in blobs.items():
        for name in spec['paths']:
            records[name] = {'sha256': sha, 'fingerprint': fingerprint(safe_path(output, name))}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.tmp')
    partial.write_text(json.dumps({'mode': mode, 'files': records}, ensure_ascii=False))
    os.replace(partial, path)


def restore_archive(path, blobs, output, mode='copy'):
    """Write directly to original paths; never retain an extra uncompressed blob tree."""
    seen = set()
    with path.open('rb') as raw:
        if path.name.endswith('.zst'):
            import zstandard
            decoded = zstandard.ZstdDecompressor().stream_reader(raw)
        else:
            decoded = gzip.GzipFile(fileobj=raw)
        with decoded:
            with tarfile.open(fileobj=decoded, mode='r|') as archive:
                for member in archive:
                    if not member.isfile() or not re.fullmatch(r'blobs/[0-9a-f]{2}/[0-9a-f]{64}', member.name):
                        raise ValueError(f'Unexpected archive member: {member.name}')
                    sha = PurePosixPath(member.name).name
                    if member.name != f'blobs/{sha[:2]}/{sha}':
                        raise ValueError('Invalid archive hash prefix')
                    if sha not in blobs:
                        continue
                    if sha in seen:
                        raise ValueError(f'Duplicate payload in archive: {sha}')
                    spec = blobs[sha]
                    if member.size != spec['bytes']:
                        raise ValueError(f'Payload size mismatch: {sha}')
                    targets = [safe_path(output, name) for name in spec['paths']]
                    for target in targets:
                        if target.exists() and (target.stat().st_size != spec['bytes'] or file_sha256(target) != sha):
                            raise FileExistsError(f'Existing file differs from the release: {target}. Use a new output directory.')
                    first = targets[0]
                    first.parent.mkdir(parents=True, exist_ok=True)
                    partial = first.with_name(first.name + '.tacdino-part')
                    needs_write = not first.exists()
                    out = partial.open('wb') if needs_write else None
                    digest = hashlib.sha256()
                    try:
                        with archive.extractfile(member) as stream:
                            for block in iter(lambda: stream.read(BLOCK), b''):
                                digest.update(block)
                                if out is not None:
                                    out.write(block)
                    finally:
                        if out is not None:
                            out.close()
                    if digest.hexdigest() != sha:
                        partial.unlink(missing_ok=True)
                        raise ValueError(f'Payload SHA-256 mismatch: {sha}')
                    if needs_write:
                        os.replace(partial, first)
                    elif mode == 'copy' and first.stat().st_nlink > 1:
                        shutil.copyfile(first, partial)
                        os.replace(partial, first)
                    for target in targets[1:]:
                        if target.exists():
                            if mode == 'copy' and target.stat().st_nlink > 1:
                                temporary = target.with_name(target.name + '.tacdino-part')
                                shutil.copyfile(first, temporary)
                                os.replace(temporary, target)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name(target.name + '.tacdino-part')
                        temporary.unlink(missing_ok=True)
                        if mode == 'hardlink':
                            try:
                                os.link(first, temporary)
                            except OSError:
                                shutil.copyfile(first, temporary)
                        else:
                            shutil.copyfile(first, temporary)
                        os.replace(temporary, target)
                    seen.add(sha)
    if seen != set(blobs):
        raise ValueError(f'Archive is missing {len(set(blobs) - seen)} required payloads')


def ensure_space(cache, output, compressed_bytes, restored_bytes):
    reserve = 1024**3
    if cache.stat().st_dev == output.stat().st_dev:
        if shutil.disk_usage(output).free < compressed_bytes + restored_bytes + reserve:
            raise OSError('Not enough disk space for the next archive and its restored files.')
    elif (shutil.disk_usage(cache).free < compressed_bytes + reserve or
          shutil.disk_usage(output).free < restored_bytes + reserve):
        raise OSError('Not enough cache/output disk space for the next archive.')


def process_plan(plan, checksums, output, cache, fetch, mode='copy', keep_archives=False,
                 offline_root=None, verify_existing=False, group='Scan3dTac'):
    """fetch(name, force) returns a managed local file; it is never called for completed shards."""
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256((str(output.resolve()) + '\n' + group).encode()).hexdigest()[:20]
    receipts = cache / 'state' / identity
    completed = resumed = 0
    for number, (name, blobs) in enumerate(sorted(plan.items()), 1):
        entry = checksums.get(name)
        key = hashlib.sha256(name.encode()).hexdigest()
        receipt = receipts / (key + '.json')
        managed = safe_path(cache / 'archives', name)
        if receipt_valid(receipt, blobs, output, mode, verify_existing):
            if offline_root is None and not keep_archives:
                managed.unlink(missing_ok=True)
            resumed += 1
            print(f'[{number}/{len(plan)}] Already restored: {name}', flush=True)
            continue
        restored_size = sum(spec['bytes'] * len(spec['paths']) for spec in blobs.values())
        ensure_space(cache, output, entry['archive_bytes'] if entry and offline_root is None else 0, restored_size)
        print(f'[{number}/{len(plan)}] Download/restore: {name}', flush=True)
        path = safe_path(offline_root, name) if offline_root else Path(fetch(name, False))
        if not path.is_file():
            raise FileNotFoundError(f'Archive is not available locally: {name}')
        if entry:
            valid = path.stat().st_size == entry['archive_bytes'] and file_sha256(path) == entry['archive_sha256']
            if not valid and offline_root is None:
                path = Path(fetch(name, True))
                valid = path.stat().st_size == entry['archive_bytes'] and file_sha256(path) == entry['archive_sha256']
            if not valid:
                raise ValueError(f'Compressed archive SHA-256 mismatch: {name}')
        restore_archive(path, blobs, output, mode)
        write_receipt(receipt, blobs, output, mode)
        completed += 1
        if offline_root is None and not keep_archives:
            # This is a file under our dedicated local_dir, not the global HF cache.
            path.absolute().relative_to((cache / 'archives').absolute())
            path.unlink()
        print(f'[{number}/{len(plan)}] Verified original files; archive cache released.', flush=True)
    return {'new_archives': completed, 'resumed_archives': resumed}


def save_metadata(get_meta, output, revision, args, plan):
    target = output.parent / (output.name + '_metadata')
    target.mkdir(parents=True, exist_ok=True)
    names = ['mat2idx.json', 'idx2mat.json', 'catemap/label_to_idx.json', 'catemap/idx_to_label.json',
             'split0111/train_split.json', 'split0111/test_split.json', 'objects.csv']
    for name in names:
        source = get_meta('Scan3dTac/metadata/' + name)
        destination = safe_path(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_sha256(destination) != file_sha256(source):
            raise FileExistsError(f'Metadata differs from this release: {destination}')
        if not destination.exists():
            shutil.copyfile(source, destination)
    ids = sorted({'/'.join(PurePosixPath(name).parts[:2]) for blobs in plan.values()
                  for spec in blobs.values() for name in spec['paths']})
    selection = {'repo_id': args.repo_id, 'revision': revision, 'group': args.group,
                 'profile': args.profile, 'objects': ids, 'split': args.split}
    key = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:16]
    (target / 'selections').mkdir(exist_ok=True)
    (target / 'selections' / (key + '.json')).write_text(json.dumps(selection, indent=2) + '\n')
    print(f'Label maps, official splits and selection record: {target}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['raw', 'Scan3dTac'], required=True)
    parser.add_argument('--profile', choices=['full', 'core', 'geometry'], default='full')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--repo-dir', type=Path, help='Use previously downloaded archives instead of accessing the Hub')
    parser.add_argument('--repo-id', default=REPO)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    parser.add_argument('--use-xet', action='store_true', help='Use native Xet downloads instead of the default resumable HTTP transport')
    parser.add_argument('--anonymous', action='store_true', help='Do not use a cached Hugging Face login')
    parser.add_argument('--http-chunk-mb', type=int, default=8, help='HTTP range size; use 1 on unstable connections')
    parser.add_argument('--object', action='append', default=[], dest='objects', help='category/object_id; repeatable')
    parser.add_argument('--category', action='append', default=[], dest='categories')
    parser.add_argument('--path', action='append', default=[], dest='patterns', help='Original-path glob; repeatable')
    parser.add_argument('--split', choices=['all', 'train', 'test'], default='all')
    parser.add_argument('--hardlinks', action='store_true', help='Share identical restored files; aliases then share edits')
    parser.add_argument('--keep-archives', action='store_true')
    parser.add_argument('--verify-existing', action='store_true', help='Rehash completed files when resuming')
    parser.add_argument('--dry-run', action='store_true', help='Show selection and publication status without fetching archives')
    args = parser.parse_args(argv)
    if args.group == 'raw' and args.profile != 'full':
        parser.error('Use --profile full for raw assets.')
    if args.http_chunk_mb < 1:
        parser.error('--http-chunk-mb must be positive.')
    output = args.output.expanduser().resolve()
    cache = (args.cache_dir or output.parent / ('.' + output.name + '.tacdino-download')).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_XET_CACHE', str(cache / 'xet'))
    os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
    os.environ.setdefault('HF_XET_SHARD_CACHE_SIZE_LIMIT', str(1024**3))
    os.environ['HF_HUB_DISABLE_XET'] = '0' if args.use_xet else '1'
    if args.repo_dir:
        root = args.repo_dir.expanduser().resolve()
        get_meta = lambda name: safe_path(root, name)
        fetch = None
        revision = 'local'
    else:
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, constants
        constants.HF_HUB_DISABLE_XET = not args.use_xet
        token = (None if not args.anonymous and args.endpoint.rstrip('/') == 'https://huggingface.co'
                 else False)
        api = HfApi(endpoint=args.endpoint, token=token)
        revision = network_retry(lambda: api.repo_info(args.repo_id, repo_type='dataset', revision=args.revision)).sha
        def get_meta(name):
            return Path(network_retry(lambda: hf_hub_download(args.repo_id, name, repo_type='dataset', revision=revision,
                                       local_dir=cache / 'metadata' / revision, endpoint=args.endpoint, token=token)))
        def fetch(name, force):
            if not args.use_xet:
                url = hf_hub_url(args.repo_id, name, repo_type='dataset', revision=revision, endpoint=args.endpoint) + '?download=true'
                return download_http_ranges(url, safe_path(cache / 'archives', name), checksums[name], force,
                                            chunk_size=args.http_chunk_mb * 1024**2)
            return Path(network_retry(lambda: hf_hub_download(args.repo_id, name, repo_type='dataset', revision=revision,
                                       local_dir=cache / 'archives', endpoint=args.endpoint, token=token, force_download=force)))
    index_name = 'core.files.jsonl.gz' if args.profile == 'core' else 'files.jsonl.gz'
    index = get_meta(f'{args.group}/metadata/{index_name}')
    checksums = {row['remote_path']: row for row in json.loads(get_meta('checksums.json').read_text())['verified_archives']}
    split_ids = None
    if args.split != 'all':
        split_ids = set(json.loads(get_meta(f'Scan3dTac/metadata/split0111/{args.split}_split.json').read_text()))
    plan, summary = make_plan(index, args.group, args.profile, set(args.objects), set(args.categories), split_ids, args.patterns)
    missing = sorted(set(plan) - set(checksums))
    summary.update(revision=revision, published_archives=len(plan) - len(missing),
                   published_compressed_bytes=sum(checksums[name]['archive_bytes'] for name in plan if name in checksums),
                   output=str(output), cache=str(cache))
    summary['logical_GB'] = round(summary['logical_bytes'] / 1e9, 3)
    summary['published_download_GB'] = round(summary['published_compressed_bytes'] / 1e9, 3)
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return summary
    if missing:
        raise RuntimeError(f'The release is still uploading: {len(missing)} required archives are not published yet. '
                           'Choose an available object, use Scan3dTac --profile core, or retry later.')
    from filelock import FileLock
    with FileLock(str(cache / 'download.lock'), timeout=1):
        result = process_plan(plan, checksums, output, cache, fetch,
                              'hardlink' if args.hardlinks else 'copy', args.keep_archives,
                              root if args.repo_dir else None, args.verify_existing, args.group)
        save_metadata(get_meta, output, revision, args, plan)
    print('RESTORE_COMPLETE', json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Stopped: {exc}', file=sys.stderr)
        raise SystemExit(1)
