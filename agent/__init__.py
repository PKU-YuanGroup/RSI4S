"""Initial M0/T0 source bundles, loaded by rsi.runtime in isolated processes."""
"""Initial Agent templates; shared files become independent editable bundles."""
from pathlib import Path
import shutil


def materialize(kind, destination):
    """Copy shared utilities and role-specific code into a self-contained seed."""
    if kind not in ('task', 'meta'):
        raise ValueError('Unknown Agent kind')
    root = Path(__file__).resolve().parent
    destination = Path(destination)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(root / 'shared', destination, ignore=ignore)
    shutil.copytree(root / kind, destination, dirs_exist_ok=True, ignore=ignore)
    return destination
