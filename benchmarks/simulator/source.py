"""Verify the prepared simulators and their separate Python environments."""
import hashlib
import json
import subprocess
from pathlib import Path

RECORD = Path(__file__).with_name('sources.json')
# Packaging tools vary with how an environment was created; they never run inside a simulator.
PACKAGING_TOOLS = frozenset({'pip', 'setuptools', 'wheel'})


def comparable_environment(environment):
    """Python release (not the machine-specific build string) and every non-packaging package version."""
    return dict(python=environment['version'].split()[0],
                packages=sorted([name, version] for name, version in environment['packages']
                                if name.lower() not in PACKAGING_TOOLS))


def verify_source(task, data_root=None, python=None):
    entry = json.loads(RECORD.read_text())[task]
    data_root, python = data_root or entry.get('data_root'), python or entry.get('python')
    if not data_root or not python:
        raise ValueError(f'{task}: set simulator_environments.{task}.data_root and .python in '
                         'configs/simulator.json (tools/setup_data.py writes them)')
    root = Path(data_root).resolve()
    python = str(python)
    for name, expected in entry['files'].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Simulator source/state changed: ' + name)
    actual = subprocess.check_output([python, '-I', '-c',
        'import importlib.metadata as m,json,sys; '
        'print(json.dumps(dict(version=sys.version,packages=sorted('
        '(d.metadata["Name"],d.version) for d in m.distributions()))))'],
        text=True, timeout=30)
    if comparable_environment(json.loads(actual)) != comparable_environment(entry['environment']):
        raise ValueError('Simulator Python environment differs from verified source record')
    return root, python
