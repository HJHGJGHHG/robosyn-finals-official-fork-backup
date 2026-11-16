#!/usr/bin/env python3
"""Download the pinned public checkpoints into their policy directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    policy_root = Path(__file__).resolve().parent
    manifest = json.loads((policy_root / 'submission.json').read_text())
    tasks = {entry['task_name']: entry for entry in manifest['tasks']}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', action='append', choices=sorted(tasks),
                        help='Download one task; repeat to select several. Default: all ten.')
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify installed files without network access.')
    args = parser.parse_args()
    failed = []
    for name in args.task or sorted(tasks):
        item = tasks[name]
        destination = policy_root.parent / item['checkpoint_path']
        expected = item['checkpoint_sha256']
        if destination.is_file() and sha256(destination) == expected:
            print(f'{name}: checksum OK', flush=True)
            continue
        if args.verify_only:
            failed.append(name)
            print(f'{name}: missing checkpoint or checksum mismatch', flush=True)
            continue
        from huggingface_hub import hf_hub_download
        revision = item['hf_revision']
        if not revision or len(revision) != 40:
            raise ValueError(f'{name}: expected an immutable checkpoint revision')
        cached = Path(hf_hub_download(repo_id=item['hf_repo_id'],
                                    filename=item['hf_filename'],
                                    revision=revision, token=False))
        if sha256(cached) != expected:
            raise RuntimeError(f'{name}: downloaded checkpoint checksum mismatch')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temp_name = stream.name
                with cached.open('rb') as source:
                    shutil.copyfileobj(source, stream)
            os.replace(temp_name, destination)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        print(f'{name}: downloaded and verified {revision}', flush=True)
    if failed:
        raise SystemExit('Verification failed: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
