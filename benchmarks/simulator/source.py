"""Verify the prepared simulators and the rsi4s environment they run in."""
import hashlib
import json
from pathlib import Path

from rsi.environment import verify as verify_environment

RECORD = Path(__file__).with_name('sources.json')


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
    verify_environment(python)
    return root, python
