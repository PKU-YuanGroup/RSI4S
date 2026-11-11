"""Experiment-only payload removal after verified Test scoring and trajectory audit.

Raw histories remain at their original paths. Scientific source/configuration and
execution logs in disposable workspaces are archived losslessly before removal.
No data inputs, predictions, fitted models or checkpoints enter these archives.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from rsi.freeze import sha
from rsi.common import write_json
from maintenance.task48_archive import durable_json

POLICY = 'protein_trajectory_only_v1'
TASKS = {'LIFE_13', 'LIFE_06', 'LIFE_08'}
CODE = {'.py', '.sh', '.bash', '.r', '.jl', '.m', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.ipynb'}
MODELS = {'.pt', '.pth', '.pkl', '.pickle', '.ckpt', '.safetensors', '.h5', '.hdf5',
          '.joblib', '.onnx', '.pb', '.tflite', '.ubj', '.cbm', '.bst', '.msgpack'}
REPORTS = {'frozen_model.json', 'freeze.json', 'validation_report.json', 'validation_score.json',
           'inference_result.json', 'fit_result.json', 'host_execution.json', 'metrics.json',
           'status.json', 'result.json', 'selection.json', 'requirements.txt', 'environment.yml'}


def walk(root):
    """Enumerate regular files and symlinks once without traversing link targets."""
    for base, dirs, names in os.walk(root, followlinks=False):
        base = Path(base)
        for name in list(dirs):
            path = base/name
            if path.is_symlink():
                dirs.remove(name)
                yield path
        for name in sorted(names):
            yield base/name


def payload(path, run):
    parts = path.relative_to(run).parts
    if parts[0] == 'final':
        return parts != ('final', 'freeze.json')
    if parts[0] == 'terminal_native':
        return True
    if parts[:3] == ('evidence', 'problem', 'data'):
        return True
    if parts[0] == 'sessions' and len(parts) > 2:
        return 'workspace' in parts[2:] or parts[2] == 'attempts'
    return False


def source_record(path, run, declared):
    """Separate retained source/diagnostics from disposable data, including JSON."""
    if path in declared or path.suffix.lower() in MODELS:
        return False
    parts = path.relative_to(run).parts
    if parts[:3] == ('evidence', 'problem', 'data'):
        return False
    if path.suffix.lower() in CODE or path.name == 'Makefile':
        return True
    if path.suffix.lower() in {'.log', '.stdout', '.stderr'}:
        return True
    name = path.name.lower()
    if any(word in name for word in ('prediction', 'embedding', 'checkpoint', 'weights')):
        return False
    if name in REPORTS or name.endswith('.audit.json'):
        return True
    if path.suffix.lower() in {'.md', '.toml', '.ini', '.yaml', '.yml', '.cfg'}:
        return True
    if path.suffix.lower() in {'.json', '.csv', '.jsonl'}:
        return any(word in name for word in ('config', 'metric', 'loss', 'training_log', 'history'))
    return path.suffix.lower() == '.txt' and any(word in name for word in ('note', 'readme', 'summary', 'requirement'))


def audit_trajectory(run, records):
    from maintenance.task48_results import executions
    from maintenance.task48_sources import audit_sources
    from rsi.store import verify_task_source_archive
    required = ('events.jsonl', 'config.json', 'release.json', 'allocated_resources.json',
                'token_report.json', 'final_selection.json', 'final_test.json',
                'terminal_score.json', 'evidence/organizer/ledger.sqlite')
    for name in required:
        path = run/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Required trajectory record missing or linked: '+name)
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    if not records or not any(r.get('kind') == 'task' for r in records):
        raise ValueError('Task session trajectory is missing')
    inventory = executions(records, events)
    if any(row['evidence_issues'] for row in inventory):
        raise ValueError('Incomplete session execution evidence')
    checks = []
    def source(path):
        path = Path(path)
        if not path.resolve().is_relative_to(run) or not path.is_file():
            raise ValueError('Trajectory source is missing or outside the run: '+str(path))
    def check(name, passed, detail=None):
        checks.append(dict(check=name, passed=passed, detail=detail))
    release = json.loads((run/'release.json').read_text())
    audit_sources(run, release, events, inventory, source, check)
    for record in records:
        if record.get('kind') == 'task':
            verify_task_source_archive(run, record)
    if any(row['passed'] is not True for row in checks):
        raise ValueError('Executed Agent source versions are not fully archived')
    calls = json.loads((run/'token_report.json').read_text())
    starts = {e['call'] for e in events if e.get('event') == 'api_start'}
    accounted = {r['call'] for r in calls.get('calls', []) if 'start_time' in r}
    if starts != accounted:
        raise ValueError('API call accounting differs from events')
    errors = {e['call'] for e in events if e.get('event') == 'api_error'}
    for number in starts:
        base = run/'calls'/f'{number:04d}'
        if not base.with_suffix('.request.json').is_file():
            raise ValueError('Raw model request is missing')
        if number not in errors and not any(base.with_suffix(s).is_file() for s in ('.response.json', '.response.txt', '.error.json')):
            raise ValueError('Raw model reply or error is missing')
    # Ledger tables are the exact data/label request and charging record. They
    # contain IDs and request provenance, not the copied scientific data pool.
    ledger = run/'evidence/organizer/ledger.sqlite'
    with sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True) as db:
        counts = dict(queries=db.execute('SELECT count(*) FROM queries').fetchone()[0],
                      charged=db.execute('SELECT coalesce(sum(count),0) FROM queries').fetchone()[0],
                      acquired=db.execute('SELECT count(*) FROM acquired').fetchone()[0])
        if counts['charged'] != counts['acquired']:
            raise ValueError('Label acquisition ledger is inconsistent')
    if (run/'config.json').is_file() and json.loads((run/'config.json').read_text()).get('chain'):
        from rsi.chain import check_handoff_status
        handoff = run/'chain_handoff/manifest.json'
        check_handoff_status(json.loads(handoff.read_text()))
        from rsi.store import hashes
        manifest = json.loads(handoff.read_text())
        for kind in ('task', 'meta'):
            if hashes(handoff.parent/kind) != manifest[kind]['files']:
                raise ValueError('Chain handoff Agent changed')
        if sha(handoff.parent/'research_experience.md') != manifest['experience']['sha256']:
            raise ValueError('Chain handoff experience changed')
    return dict(status='verified', session_count=len(records), api_calls=len(starts),
                source_checks=checks, annotation_counts=counts)


def membership(run):
    """Keep dataset membership provenance without copied features or labels."""
    pool = run/'evidence/problem/data/samples.sqlite'
    if not pool.is_file():
        raise ValueError('Public evidence pool missing before retention')
    target = run/'retention/data-membership.jsonl'
    count = 0
    with sqlite3.connect(pool.as_uri()+'?mode=ro', uri=True) as db, target.open('w') as out:
        columns = [r[1] for r in db.execute('PRAGMA table_info(samples)')]
        selected = [c for c in columns if c in {'sample_id', 'instance', 'partition', 'role', 'group_id', 'acquired'}]
        if 'sample_id' not in selected:
            raise ValueError('Public sample membership schema is missing')
        for row in db.execute('SELECT '+','.join(selected)+' FROM samples ORDER BY sample_id'):
            out.write(json.dumps(dict(zip(selected, row)), sort_keys=True)+'\n')
            count += 1
        out.flush(); os.fsync(out.fileno())
    return dict(path='retention/data-membership.jsonl', sha256=sha(target), samples=count,
                source_sha256=sha(pool), retained_values='IDs, instances, partitions and acquisition status only')


def verify_trajectory(run, meta):
    if meta.get('policy') != POLICY:
        return
    receipt = run/meta['trajectory_manifest']
    if sha(receipt) != meta['trajectory_manifest_sha256']:
        raise ValueError('Trajectory inventory changed')
    for line in receipt.read_text().splitlines():
        row = json.loads(line)
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe trajectory path')
        path = run/relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(run):
            raise ValueError('Retained trajectory file is missing: '+row['path'])
        if row.get('append_only'):
            with path.open('rb') as stream:
                actual = hashlib.sha256(stream.read(row['bytes'])).hexdigest()
            valid = path.stat().st_size >= row['bytes'] and actual == row['sha256']
        else:
            valid = path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        if not valid:
            raise ValueError('Retained trajectory file changed: '+row['path'])
    if sha(run/meta['data_membership']['path']) != meta['data_membership']['sha256']:
        raise ValueError('Dataset membership evidence changed')


def finish(run, result):
    if result.get('policy') != POLICY or result.get('status') != 'completed':
        return result
    verify_trajectory(run, result)
    remaining = [str(p.relative_to(run)) for p in walk(run) if payload(p, run)]
    if remaining:
        raise ValueError('Scientific payload remains after retention: '+str(remaining[:8]))
    # Remove only empty directories after the durable deletion journal commits.
    for base, dirs, names in os.walk(run, topdown=False, followlinks=False):
        path = Path(base)
        if path != run and payload(path/'__empty__', run):
            try:path.rmdir()
            except OSError:pass
    return result


def prepare(run, archives, records, provenance):
    task = json.loads((run/'summary.json').read_text())['task']
    if task not in TASKS:
        raise ValueError('Protein retention policy is limited to the three requested tasks')
    for name in ('sessions', 'final', 'evidence', 'terminal_native'):
        if (run/name).is_symlink():
            raise ValueError('Scientific root is a symlink')
    audit = audit_trajectory(run, records)
    (run/'retention').mkdir(exist_ok=True)
    data = membership(run)
    files = list(walk(run))
    declared = set()
    for path in files:
        if payload(path, run) and path.name == 'frozen_model.json' and not path.is_symlink():
            for name in json.loads(path.read_text()).get('checkpoints', []):
                target = path.parent/name
                if not target.resolve().is_relative_to(run):
                    raise ValueError('Declared checkpoint escaped its run')
                declared.add(target)
    planned, index, protected = [], {}, []
    for path in files:
        relative = str(path.relative_to(run))
        if relative.startswith(('retained-source/', 'retention/')) or path.name in ('retention.json', 'retention-preparation.json', 'retention.lock'):
            continue
        if payload(path, run):
            if path.is_symlink():
                target = os.readlink(path)
                raw = os.fsencode(target)
                info = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                            symlink=True, target=target, model=False)
            else:
                info = dict(path=relative, sha256=archives.hash(path), bytes=path.stat().st_size,
                            model=path.suffix.lower() in MODELS or path in declared)
                if source_record(path, run, declared):
                    index[relative] = archives.add(path)
                    info['archive'] = index[relative]['archive']
            info.update(provenance=provenance.get(relative), run_config='config.json', release='release.json',
                        reason='Protein repeat: archive source/diagnostics, delete every scientific payload after native Test scoring')
            planned.append(info)
        else:
            if path.is_symlink():
                raise ValueError('Unclassified trajectory symlink; preserve for review: '+relative)
            if path.suffix.lower() in MODELS:
                raise ValueError('Unclassified model outside scientific workspaces: '+relative)
            protected.append(dict(path=relative, sha256=archives.hash(path), bytes=path.stat().st_size,
                                  append_only=relative == 'events.jsonl'))
    durable_json(archives.root/'index.json', index)
    durable_json(run/'retained-links.json', dict(links=[]))
    trajectory = run/'retention/trajectory.jsonl'
    with trajectory.open('w') as out:
        for row in protected:out.write(json.dumps(row, sort_keys=True)+'\n')
        out.flush(); os.fsync(out.fileno())
    durable_json(run/'retention/trajectory-audit.json', dict(audit, preserved_files=len(protected), archived_files=len(index)))
    result = dict(schema_version=1, status='deleting', policy=POLICY, task=task, created_at=time.time(),
                  source_index='retained-source/index.json', retained_checkpoints=[], retained_link_targets=[],
                  symbolic_links='retained-links.json', symbolic_links_sha256=sha(run/'retained-links.json'),
                  trajectory_manifest='retention/trajectory.jsonl', trajectory_manifest_sha256=sha(trajectory),
                  trajectory_audit='retention/trajectory-audit.json', data_membership=data,
                  planned=planned, deleted=[], replay_note='All scientific weights and predictions intentionally removed; original source and trajectory retained.')
    from maintenance.task48_retention_journal import install, execute
    verify_trajectory(run, result)
    archives.verify_index(index)
    install(run, result)
    return finish(run, execute(run, archives=archives))
