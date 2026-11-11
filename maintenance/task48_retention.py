"""Conservative post-terminal retention with a hash-bound deletion ledger.

Only classified model/array payloads and archived large worker inputs are removed. Unknown files, all small
records, and failed/interrupted runs remain available for diagnosis. This never
recursively removes a scientific workspace or follows an external symlink.
"""
from benchmarks.task_names import public_name
from rsi.tasks import MEDRSEE_TASKS, SIMULATOR_TASKS
import json
import os
from pathlib import Path
import time

import psutil

from rsi.common import write_json
from rsi.freeze import sha

SHOWCASE = {'LIFE_13','LIFE_08','CHEM_06',
            'CHEM_07','MAT_02','MAT_01','MED_02','RS_01'}
MODEL_SUFFIXES = {'.pt','.pth','.pkl','.pickle','.ckpt','.safetensors','.h5','.hdf5','.joblib'}
ARRAY_SUFFIXES = {'.npy','.npz','.nii'}
SOURCE_SUFFIXES = {'.py','.sh','.json','.jsonl','.yaml','.yml','.toml','.txt','.md','.csv','.log'}


def agent_source_file(path, run):
    parts = path.relative_to(run).parts
    return (len(parts) >= 5 and parts[0] == 'sessions' and parts[2] == 'agent'
            and parts[3] in ('source', 'source-revisions'))


def validate_terminal(run, *, verify_artifact=True):
    run = Path(run).resolve()
    completion = json.loads((run/'completion.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    if completion.get('status') != 'completed':
        result = dict(status='deferred', reason='Failed or incomplete run preserved for diagnosis')
        return result
    pid = completion.get('pid')
    if (pid and pid != os.getpid() and psutil.pid_exists(pid)
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE):
        raise RuntimeError('Refuse retention while the owning worker is alive')
    if pid == os.getpid() and any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                                 for p in psutil.Process().children(recursive=True)):
        raise RuntimeError('Refuse retention while worker descendants are alive')
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    if not any(e.get('event') == 'experiment_end' for e in events):
        raise ValueError('Research and frozen inference finalization have not completed')
    if summary.get('cleanup_errors'):
        raise ValueError('Framework cleanup errors require review before retention')
    task = summary['task']
    if public_name(task) in SIMULATOR_TASKS:
        if not (summary.get('simulator_episode') or {}).get('terminal'):
            raise ValueError('Episode is not terminal')
    elif completion.get('terminal_status') != 'completed':
        raise ValueError('Independent final scoring is not complete')
    if summary.get('benchmark_protocol') == 'naturebench-active':
        score = json.loads((run/'terminal_score.json').read_text())
        if not score.get('test_score_computed'):
            raise ValueError('Native Test scoring has not completed')
        from benchmarks.naturebench_active.scoring.terminal import frozen_result
        # Includes model, predictions and native output hashes, before any deletion.
        if verify_artifact:frozen_result(run)
    return None


def record_failure(run, exc):
    """Keep maintenance errors separate from the immutable scientific receipt."""
    path = Path(run)/'retention.json'
    existing = json.loads(path.read_text()) if path.exists() else {}
    result = dict(existing, status='failed', error=f'{type(exc).__name__}: {exc}', updated_at=time.time())
    write_json(path, result)
    return result


def retain(run):
    from maintenance.task48_retention_journal import locked, execute
    run=Path(run).resolve()
    with locked(run):
        target=run/'retention.json'
        if target.exists():
            existing=json.loads(target.read_text())
            if existing['status']=='completed':
                from maintenance.task48_protein_retention import finish
                return finish(run, existing)
        deferred=validate_terminal(run,verify_artifact=not (run/'retention/plan.json').exists())
        if deferred is not None:
            write_json(target,deferred);return deferred
        if (run/'retention/plan.json').exists():
            from maintenance.task48_protein_retention import finish
            return finish(run, execute(run))
        if target.exists() and existing.get('status')=='deleting':
            raise ValueError('Legacy retention needs explicit migration; do not re-plan deleted files')
        try:
            return prepare_and_retain(run)
        except BaseException as exc:
            progress=run/'retention-preparation.json'
            if progress.exists():
                state=json.loads(progress.read_text())
                state.update(status='failed',error=f'{type(exc).__name__}: {exc}',updated_at=time.time())
                write_json(progress,state)
            record_failure(run, exc)
            raise


def prepare_and_retain(run):
    from maintenance.task48_archive import Archives, archived_input, durable_json
    archives = Archives(run)
    archives.publish('scan_records', run, force=True)
    summary=json.loads((run/'summary.json').read_text());task=summary['task']
    records = [json.loads(p.read_text()) for p in sorted((run/'records').glob('*.json'))]
    provenance = {}
    artifact_info = {}
    for record in records:
        for artifact in record.get('artifacts', []):
            artifact_info[artifact['id']] = dict(run_id=record['run_id'],artifact=artifact['id'],
                task_version=artifact.get('task_version'),meta_version=artifact.get('meta_version'))
            workspace = Path(artifact['workspace']).resolve()
            if not workspace.is_relative_to(run):
                raise ValueError('Artifact escaped its experiment')
            for name, expected in artifact['files'].items():
                path = workspace/name
                if not path.is_file() or archives.hash(path) != expected:
                    raise ValueError('Recorded trajectory artifact changed: '+str(path))
                provenance[str(path.relative_to(run))] = artifact_info[artifact['id']]
        for evaluation in record.get('evaluations',[]):
            fitted=evaluation.get('fitted_artifact') or {}
            if fitted.get('path'):
                fitted_root=Path(fitted['path']).resolve()
                if not fitted_root.is_relative_to(run):
                    raise ValueError('Fitted checkpoint escaped its experiment')
                for name,expected in fitted.get('files',{}).items():
                    path=fitted_root/name
                    if not path.is_file() or archives.hash(path)!=expected:
                        raise ValueError('Fitted checkpoint changed before retention')
                    provenance[str(path.relative_to(run))]=dict(run_id=record['run_id'],
                        artifact=evaluation.get('artifact'),task_version=evaluation.get('task_version'),
                        meta_version=evaluation.get('meta_version'))
    config_path = run/'config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config.get('retention_policy') == 'protein_trajectory_only_v1':
        from maintenance.task48_protein_retention import prepare
        return prepare(run, archives, records, provenance)
    # Known scientific trees; raw API histories, ledgers, configurations and Agent
    # versions never enter this classification or deletion pass.
    roots = [run/'sessions', run/'final']
    files = []
    links = []
    protected_targets = set()
    declared = set()
    for root in roots:
        if root.is_symlink():
            raise ValueError('Scientific root is a symlink; manual review required')
        for path in root.rglob('*'):
            if path.is_symlink():
                target = os.readlink(path)
                # Lexical normalization only: never traverse or hash a link target.
                lexical = Path(os.path.abspath(path.parent/target))
                links.append(dict(path=str(path.relative_to(run)), target=target,
                                  target_within_run=lexical.is_relative_to(run),
                                  action='preserved_without_following'))
                if lexical.is_relative_to(run):
                    protected_targets.add(lexical)
                continue
            if path.is_file():
                files.append(path)
                if path.name == 'frozen_model.json' and not agent_source_file(path, run):
                    declaration = json.loads(path.read_text())
                    for name in declaration.get('checkpoints', []):
                        resolved = (path.parent/name).resolve()
                        if not resolved.is_relative_to(run):
                            raise ValueError('Declared checkpoint escaped the experiment')
                        declared.add(resolved)
    plans, kept, linked_targets = [], [], []
    for path in files:
        relative = str(path.relative_to(run))
        if agent_source_file(path, run):
            continue  # Agent definitions/assets are evidence, not scientific payloads.
        is_model = path.suffix.lower() in MODEL_SUFFIXES or path in declared
        is_array = path.suffix.lower() in ARRAY_SUFFIXES or path.name.endswith('.nii.gz')
        if not is_model and not is_array:
            continue
        attribution=provenance.get(relative)
        if attribution is None and path.is_relative_to(run/'final'):
            attribution=artifact_info.get((summary.get('final_test') or {}).get('artifact'))
        info = dict(path=relative, sha256=archives.hash(path), bytes=path.stat().st_size,
                    provenance=attribution, model=is_model,run_config='config.json',release='release.json')
        if any(path == target or path.is_relative_to(target) for target in protected_targets):
            info['reason'] = 'Preserved direct in-run symbolic-link target; no target deletion'
            linked_targets.append(info)
        elif is_model and task in SHOWCASE:
            info['reason'] = 'Showcase checkpoint; adjacent source/configuration and trajectory retained'
            kept.append(info)
        else:
            info['reason'] = ('Non-showcase model removed by retention policy' if is_model else
                              'Recomputable array/input/prediction after verified terminal scoring')
            plans.append(info)
    # Independent compressed source/configuration copy, verified before deletion.
    archive = archives.root
    source_index = {}
    for path in files:
        if path.suffix.lower() not in SOURCE_SUFFIXES or path in declared:
            continue
        relative = str(path.relative_to(run))
        source_index[relative] = archives.add(path)
        if archived_input(path,run) and not any(path == target or path.is_relative_to(target) for target in protected_targets):
            plans.append(dict(path=relative,sha256=source_index[relative]['sha256'],bytes=path.stat().st_size,
                provenance=provenance.get(relative),model=False,run_config='config.json',release='release.json',
                archive=source_index[relative]['archive'],
                reason='Large worker input retained losslessly in verified archive; original path restorable'))
    durable_json(archive/'index.json',source_index)
    durable_json(run/'retained-links.json',dict(links=links))
    result = dict(schema_version=1,status='deleting',task=task,created_at=time.time(),
                  source_index='retained-source/index.json',retained_checkpoints=kept,deleted=[],
                  symbolic_links='retained-links.json',symbolic_links_sha256=sha(run/'retained-links.json'),
                  retained_link_targets=linked_targets,
                  planned=plans, replay_note='Policy-deleted models cannot be replayed directly; source is not a byte-identical reconstruction guarantee.')
    from maintenance.task48_retention_journal import install, execute
    install(run,result)
    archives.progress['status'] = 'completed'
    archives.publish('ready_for_deletion', run, force=True)
    return execute(run, archives=archives)


def verify_links(run, meta, source, *, deep):
    """Verify link text and separately preserved payloads without following links."""
    if 'symbolic_links' not in meta:
        return  # Older plans predate link evidence.
    if meta['symbolic_links'] != 'retained-links.json':
        raise ValueError('Unexpected symbolic-link receipt path')
    receipt = run/'retained-links.json'
    if sha(receipt) != meta['symbolic_links_sha256']:
        raise ValueError('Symbolic-link receipt changed')
    source(receipt)
    from maintenance.task48_retention_journal import safe_path
    for row in json.loads(receipt.read_text())['links']:
        relative = Path(row['path'])
        # Validate parent components without resolving the final symlink.
        parent = safe_path(run, str(relative.parent))
        link = parent/relative.name
        if not link.is_symlink() or os.readlink(link) != row['target']:
            raise ValueError('Preserved symbolic link changed: '+row['path'])
    for row in meta.get('retained_link_targets', []):
        path = safe_path(run, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or (deep and sha(path) != row['sha256']):
            raise ValueError('Preserved symbolic-link target changed: '+row['path'])
