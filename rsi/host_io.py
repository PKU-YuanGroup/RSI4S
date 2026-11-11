"""Deadline-bounded trusted artifact I/O, isolated from the threaded host."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rsi.common import ROOT


class HostIODeadline(TimeoutError):
    def __init__(self, operation, expires):
        super().__init__('Host I/O deadline reached: ' + operation)
        self.operation, self.expires = operation, expires


def run_io(operation, expires, **arguments):
    remaining = expires - time.monotonic()
    if remaining <= 0:
        raise HostIODeadline(operation, expires)
    try:
        result = subprocess.run([sys.executable, '-m', 'rsi.host_io'], cwd=ROOT,
            input=json.dumps(dict(operation=operation, arguments=arguments)),
            capture_output=True, text=True,
            timeout=None if remaining == float('inf') else remaining)
    except subprocess.TimeoutExpired as exc:
        raise HostIODeadline(operation, expires) from exc
    if result.returncode:
        raise RuntimeError('Host I/O failed: ' + result.stderr[-2000:])
    return json.loads(result.stdout.rsplit('HOST_IO=', 1)[1])


def execute(operation, arguments):
    from rsi.store import hashes, seal
    from rsi.sandbox import assert_plain_tree
    if operation == 'agent_source_revision':
        from rsi.agent_sources import save_revision
        return dict(revision=save_revision(**arguments))
    if operation == 'artifact_signature':
        from rsi.store import artifact_signature
        signature, reusable = artifact_signature(arguments['source'])
        return dict(signature=signature, reusable=reusable)
    if operation == 'hashes':
        source = Path(arguments['source'])
        assert_plain_tree(source, ignore=('.tmp', '__pycache__'))
        return hashes(source, ignore=('.tmp', '__pycache__'))
    if operation in ('snapshot', 'copy'):
        source, target = Path(arguments['source']), Path(arguments['target'])
        ignore = tuple(arguments.get('ignore', ('.tmp', '__pycache__')))
        assert_plain_tree(source, ignore=ignore)
        partial = target.with_name(target.name + '.partial')
        if target.exists() or partial.exists():
            raise ValueError('Refuse to overwrite artifact copy: ' + str(target))
        shutil.copytree(source, partial, ignore=shutil.ignore_patterns(*ignore))
        files = hashes(partial)
        if operation == 'snapshot':
            seal(partial)
        os.rename(partial, target)
        return files
    if operation in ('cohort', 'pool_snapshot'):
        from benchmarks.medrsee.submission import validation_cohort
        pool = arguments.get('pool')
        if operation == 'pool_snapshot':
            import sqlite3
            pool = arguments['target']
            with sqlite3.connect(arguments['source']) as source, sqlite3.connect(pool) as dest:
                source.backup(dest)
            Path(pool).chmod(0o444)
        return validation_cohort(pool)
    if operation == 'prepared_inputs':
        from benchmarks.medrsee.test_inputs import PreparedTestInputs
        cache = object.__new__(PreparedTestInputs)
        cache.__dict__.update(arguments['state'])
        for key in ('root', 'index', 'data_root', 'entry'):
            setattr(cache, key, Path(getattr(cache, key)))
        return cache.materialize(arguments['destination'], deadline=arguments['deadline'])
    if operation == 'audit_training':
        from benchmarks.medrsee.training import audit_training
        return audit_training(**arguments)
    if operation == 'inputs':
        from benchmarks.medrsee.runtime import export_inference_inputs
        return export_inference_inputs(**arguments)
    if operation == 'prepare_workspace':
        from rsi.sandbox import prepare_workspace
        prepare_workspace(Path(arguments['workspace']), arguments['uid'])
        return True
    if operation == 'prepare_frozen_workspace':
        from rsi.sandbox import prepare_frozen_workspace
        prepare_frozen_workspace(Path(arguments['workspace']), arguments['uid'])
        return True
    if operation == 'source_snapshot':
        from rsi.science_audit import snapshot
        return snapshot(**arguments)
    raise ValueError('Unknown trusted I/O operation: ' + operation)


if __name__ == '__main__':
    request = json.loads(sys.stdin.read())
    print('HOST_IO=' + json.dumps(execute(**request)))
