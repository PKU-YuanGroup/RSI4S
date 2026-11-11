"""Content-addressed infrastructure freeze, checked before live experiments."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from rsi.common import ROOT, digest, write_json
from rsi.tasks import task_directory, selected_tasks

MANIFEST = ROOT / 'provenance/frozen.json'


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def code_hashes():
    paths = [ROOT / 'eval_service.py', ROOT / '_evaluator_runner.py', ROOT / 'run_rsi.py',
             ROOT / 'configs/tasks_44.txt', ROOT / 'upstream.json',
             ROOT / 'judge_source_papers.jsonl']
    paths += sorted((ROOT / 'rsi').glob('*.py'))
    # The paired study scheduler, terminal scheduling and retention/reporting are
    # part of the experiment, not mutable out-of-band helpers.
    paths += sorted((ROOT / 'maintenance').glob('task48*.py'))
    paths += [ROOT/'maintenance/stress.py', ROOT/'maintenance/naturebench/terminal_queue.py']
    paths += sorted((ROOT / 'rsi/interfaces').glob('*.md'))
    paths += sorted((ROOT / 'judge').glob('*.py'))
    # Include the independently editable Agent packages and shared templates.
    paths += [p for p in sorted((ROOT / 'agent').rglob('*'))
              if p.is_file() and p.suffix in ('.py', '.txt', '.json', '.md') and '__pycache__' not in p.parts]
    paths += [p for p in sorted((ROOT/'benchmarks').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT/'configs/tasks_medrsee.txt']
    paths += [ROOT/'configs/tasks_simulator.txt']
    paths += [ROOT/'configs/tasks_resp.txt']
    paths += [p for p in sorted((ROOT/'tasks/simulator').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in sorted((ROOT/'tasks/medrsee').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def task_files(data_root, task):
    root = task_directory(data_root,task)
    return [root / 'metadata.json', *sorted((root / 'evaluation').rglob('*')),
            root / 'problem/README.md', root / 'problem/data_description.md',
            root / 'problem/execution.md']


def task_hashes(data_root, task):
    root = task_directory(data_root,task)
    return {str(Path(task)/p.relative_to(root)): sha(p) for p in task_files(data_root, task)
            if p.is_file() and '__pycache__' not in p.parts}


def freeze(data_root, evidence=None, manifest=None, tasks=None):
    """Bind code, task contracts and the Python environment; `evidence` must pass for these exact hashes."""
    manifest_path = Path(manifest) if manifest else MANIFEST
    if manifest_path.exists():
        raise RuntimeError('Already frozen; do not overwrite an experiment baseline')
    hashes = code_hashes()
    if evidence is not None:
        report = json.loads(Path(evidence).read_text())
        if report.get('success') is not True or report.get('code_hashes') != hashes:
            raise RuntimeError('Freeze requires successful tests of these exact code hashes')
    tasks = selected_tasks() if tasks is None else list(tasks)
    if not tasks or len(tasks) != len(set(tasks)) or not set(tasks) <= set(selected_tasks()):
        raise ValueError('Freeze requires distinct registered tasks')
    manifest = {'schema': 1, 'code': hashes, 'tasks': {t: task_hashes(data_root, t) for t in tasks},
                'data_root': str(Path(data_root).resolve()),
                'evidence': None if evidence is None else str(Path(evidence).resolve()),
                'evidence_sha256': None if evidence is None else sha(evidence), 'python': sys.executable,
                'python_version': platform.python_version(),
                'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    manifest['identity'] = digest(manifest)
    write_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    return manifest['identity']


def verify_freeze(data_root, task=None, manifest=None):
    value = json.loads((Path(manifest) if manifest else MANIFEST).read_text())
    identity = value.pop('identity')
    if digest(value) != identity:
        raise RuntimeError('Freeze manifest changed')
    if code_hashes() != value['code']:
        raise RuntimeError('Frozen infrastructure changed; experiment refused')
    if str(Path(data_root).resolve()) != value['data_root']:
        raise RuntimeError('Dataset root differs from frozen task packages')
    if sys.executable != value['python']:
        raise RuntimeError('Use the frozen evaluator/science Python interpreter')
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
    if packages != value['packages'] or platform.python_version() != value['python_version']:
        raise RuntimeError('Frozen Python environment package versions changed')
    for name in ([task] if task else value['tasks']):
        if task_hashes(data_root, name) != value['tasks'][name]:
            raise RuntimeError(f'Frozen evaluation/task contract changed: {name}')
    return identity
