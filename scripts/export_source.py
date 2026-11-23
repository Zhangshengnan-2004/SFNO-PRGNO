"""Create a source-only publishing copy, without changing local files or Git history.

Run from any directory:
    python scripts/export_source.py --output ../SFNO-PRGNO-github

Tracked files are also checked with git check-ignore --no-index: adding a
.gitignore alone does not stop Git from tracking already-committed weights.
The output contains current working-tree code, not the old commit's contents.
It intentionally contains no .git directory or inherited binary history.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(*args, data=None, allowed=(0,)):
    result = subprocess.run(
        ['git', '-C', str(ROOT), *args], input=data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode not in allowed:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout


def plan(max_file_bytes):
    names = sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z').split(b'\0')) - {b''})
    ignored = set(git('check-ignore', '--no-index', '--stdin', '-z',
                      data=b'\0'.join(names) + b'\0', allowed=(0, 1)).split(b'\0'))
    included, excluded = [], []
    for raw in names:
        name = raw.decode()
        path = ROOT / name
        if raw in ignored:
            excluded.append({'path': name, 'reason': 'gitignore'})
        elif path.is_symlink():
            excluded.append({'path': name, 'reason': 'local symlink; not copied into portable release'})
        elif not path.is_file():
            excluded.append({'path': name, 'reason': 'deleted from current working tree'})
        else:
            size = path.stat().st_size
            if size > max_file_bytes:
                raise ValueError(f'Unexcluded large file: {name} ({size:,} bytes); update .gitignore first')
            included.append({'path': name, 'bytes': size})
    if not any(item['path'] == '.gitignore' for item in included):
        raise ValueError('The publishing copy must include .gitignore')
    return included, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / (ROOT.name + '-github'))
    parser.add_argument('--max-file-mib', type=float, default=50)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_file_mib <= 0:
        parser.error('--max-file-mib must be positive')
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Choose a new directory outside the original project')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; refusing to overwrite it')
    included, excluded = plan(int(args.max_file_mib * 1024**2))
    summary = {
        'source': str(ROOT), 'output': str(output),
        'files': len(included), 'bytes': sum(item['bytes'] for item in included),
        'excluded_candidates': len(excluded),
        'excluded_by_reason': dict(Counter(item['reason'] for item in excluded)),
        'original_git_history_modified': False, 'local_weights_deleted': False,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    for item in included:
        source = ROOT / item['path']
        destination = output / item['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        item['sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        item['bytes'] = destination.stat().st_size
    summary['bytes'] = sum(item['bytes'] for item in included)
    manifest = dict(summary, included=included, excluded=excluded)
    (output / 'SOURCE_EXPORT.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Ready for a new source-only Git history; no commit or push was performed.')


if __name__ == '__main__':
    main()
