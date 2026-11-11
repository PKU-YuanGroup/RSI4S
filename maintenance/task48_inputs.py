"""Prepare or verify each shared ENV_02 input cache before launching any worker."""
import json
from pathlib import Path
import subprocess

from rsi.common import ROOT
from rsi.freeze import sha


def check_inputs(jobs, directory, *, prepare=False):
    from benchmarks.medrsee.source import DATA
    from benchmarks.medrsee.test_inputs import resolve_cache_root
    groups = {}
    for job in jobs:
        if job['task'] != 'ENV_02' or job.get('status', 'queued') != 'queued':
            continue
        config = json.loads(Path(job['config']).read_text())
        data = Path(config.get('medrsee_data_root', str(DATA))).resolve()
        cache = resolve_cache_root(data, config.get('medrsee_test_cache_root')).resolve()
        key = (job['python'], str(data), str(cache))
        groups.setdefault(key, []).append(job)
    if not groups:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    receipts = []
    for group in groups.values():
        job = group[0]
        report = directory / f'ENV_02-{len(receipts):02d}.json'
        command = [job['python'], '-m', 'maintenance.medrsee.prepare_test_inputs',
                   '--config', job['config'], '--report', str(report)]
        if not prepare:
            command.append('--verify-only')
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise RuntimeError('ENV_02 input preparation/check failed; see ' + str(report))
        value = json.loads(report.read_text())
        if value.get('status') != 'ready':
            raise ValueError('ENV_02 input check did not report readiness: ' + str(report))
        receipts.append(dict(jobs=[j['id'] for j in group], report=str(report),
                             report_sha256=sha(report), **value))
    return receipts
