"""Sealed copies of Task source at step entry, separate from published Agents."""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from rsi.common import digest, write_json
from rsi.store import hashes, safe_path, seal


def source_revision(files):
    return digest(files)


def verify_revision(directory, revision, files, base_version, source=None):
    """Verify host-owned evidence without trusting paths supplied by an Agent."""
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{64}', revision):
        raise ValueError('Invalid Agent source revision')
    root = Path(directory)/'source-revisions'
    target = safe_path(root, revision)
    if root.is_symlink() or target.is_symlink():
        raise ValueError('Source revisions must not be symlinks')
    manifest = target/'manifest.json'
    value = json.loads(manifest.read_text())
    expected = dict(format=1, revision=revision, base_version=base_version, files=files)
    if value != expected or source_revision(files) != revision:
        raise ValueError('Agent source revision manifest mismatch')
    if hashes(target/'source') != files:
        raise ValueError('Agent source revision integrity violation')
    for path in [target, *target.rglob('*')]:
        info = path.lstat()
        if path.is_symlink() or info.st_uid != os.geteuid() or info.st_mode & 0o222:
            raise ValueError('Agent source revision must remain host-owned and read-only')
    if source is not None:
        source(manifest)
        for name in files:
            source(safe_path(target/'source', name))
    return target


def save_revision(directory, current, files, base_version):
    """Reuse identical source; never publish a successor or choose inheritance."""
    revision = source_revision(files)
    root = Path(directory)/'source-revisions'
    root.mkdir(exist_ok=True)
    target = root/revision
    if target.exists():
        verify_revision(directory, revision, files, base_version)
        return revision
    staging = Path(tempfile.mkdtemp(prefix='.partial-', dir=root))
    try:
        shutil.copytree(current, staging/'source', ignore=shutil.ignore_patterns('__pycache__'))
        if hashes(staging/'source') != files:
            raise ValueError('Task source changed while preserving its revision')
        write_json(staging/'manifest.json', dict(format=1, revision=revision,
                   base_version=base_version, files=files))
        seal(staging)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return revision
