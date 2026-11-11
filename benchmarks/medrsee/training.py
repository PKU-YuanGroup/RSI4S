"""Audit declared fitting membership and checkpoint ancestry without hidden labels.

This is an auditable declaration, not proof of arbitrary Python's data flow.
"""
import json
from pathlib import Path
import sqlite3

from rsi.store import safe_path


def execution_evidence(current, history):
    """Follow host-recorded workspace inheritance, independent of Agent declarations."""
    parents = {a['id']: record for record in history for a in record.get('artifacts', [])}
    runs, seen = [], set()
    record = current
    while record is not None and record['run_id'] not in seen:
        seen.add(record['run_id'])
        actions = []
        for turn in record.get('trace', []):
            request, response = turn.get('request', {}), turn.get('response', {})
            if request.get('op') not in ('bash', 'python'):
                continue
            actions.append(dict(request=request,
                outcome={key: response[key] for key in ('returncode', 'reason', 'seconds', 'log', 'execution_audit')
                         if key in response}))
        parent = record.get('artifact_parent')
        runs.append(dict(run_id=record['run_id'], artifact_parent=parent, actions=actions))
        record = parents.get(parent)
    return dict(runs=runs, source='Host execution records and workspace inheritance',
                scope='Observed commands and available source files; inheritance is not proof of checkpoint use',
                actual_fitting_membership_verified=False, in_memory_fitting_excluded=False)


def provenance(source, frozen):
    name = frozen.get('training_provenance', 'training_provenance.json')
    path = safe_path(source, name)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Training provenance exceeds 1 MiB; put sample IDs in streamed files')
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Training provenance must be an object')
    names = [name]
    for key in ('labeled_fit_ids_file', 'input_fit_ids_file'):
        value = manifest.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError('Training provenance requires ' + key)
        safe_path(source, value)
        names.append(value)
    parents = manifest.get('parent_artifacts')
    if not isinstance(parents, list) or any(not isinstance(p, str) or not p for p in parents):
        raise ValueError('Training provenance requires parent_artifacts (empty for fresh fitting)')
    return manifest, names


def audit_training(task, source, pool, artifacts=None, source_files=None):
    from .submission import declaration
    from rsi.store import hashes
    known = {a['id']: a for a in (artifacts or [])}
    seen, active, counts = set(), set(), {}
    with sqlite3.connect(f'file:{Path(pool).resolve()}?mode=ro', uri=True) as db:
        db.execute('CREATE TEMP TABLE fit_ids (sample_id TEXT PRIMARY KEY)')
        db.execute('''CREATE TEMP TABLE validation_fit (sample_id TEXT PRIMARY KEY,
            labeled_fit INTEGER NOT NULL DEFAULT 0, input_fit INTEGER NOT NULL DEFAULT 0)''')

        def inspect(workspace, identity, artifact=None):
            if identity in active:
                raise ValueError('Cyclic checkpoint ancestry')
            if identity in seen:
                return
            active.add(identity)
            files = artifact['files'] if artifact else (source_files or hashes(workspace))
            frozen = declaration(task, dict(workspace=str(workspace), files=files))
            manifest, fitting_files = provenance(workspace, frozen)
            # Ancestor weights are not executed here. Verify the exact recorded
            # declaration and ID files needed for the membership audit, avoiding
            # rereading every historical multi-GiB checkpoint at each generation.
            import hashlib
            for name in ['frozen_model.json', *fitting_files]:
                hasher = hashlib.sha256()
                with safe_path(workspace, name).open('rb') as stream:
                    while chunk := stream.read(1024 * 1024):
                        hasher.update(chunk)
                if hasher.hexdigest() != files.get(Path(name).as_posix()):
                    raise ValueError('Fitting provenance changed: ' + identity + '/' + name)
            counts[identity] = {}
            for key, require_label in [('labeled_fit_ids_file', True), ('input_fit_ids_file', False)]:
                db.execute('DELETE FROM fit_ids')
                with safe_path(workspace, manifest[key]).open() as stream:
                    batch = []
                    for line in stream:
                        sample = line.rstrip('\r\n')
                        if not sample:
                            raise ValueError('Blank sample ID in fitting membership')
                        batch.append((sample,))
                        if len(batch) == 4096:
                            db.executemany('INSERT OR IGNORE INTO fit_ids VALUES (?)', batch)
                            batch.clear()
                    db.executemany('INSERT OR IGNORE INTO fit_ids VALUES (?)', batch)
                invalid = db.execute('''SELECT f.sample_id FROM fit_ids f LEFT JOIN samples s USING(sample_id)
                    WHERE s.sample_id IS NULL OR s.role NOT IN ('train', 'val') ''' +
                    (' OR s.acquired != 1' if require_label else '') + ' LIMIT 1').fetchone()
                if invalid:
                    raise ValueError('Fitting membership contains Test, unknown or unauthorized labels: ' + invalid[0])
                counts[identity][key] = db.execute('SELECT count(*) FROM fit_ids').fetchone()[0]
                # Union across all fitting files and ancestors; count only the
                # acquired Val members scored in this evaluation's snapshot.
                db.execute('''INSERT OR IGNORE INTO validation_fit (sample_id)
                    SELECT f.sample_id FROM fit_ids f JOIN samples s USING(sample_id)
                    WHERE s.role='val' AND s.acquired=1''')
                flag = 'labeled_fit' if require_label else 'input_fit'
                db.execute(f'UPDATE validation_fit SET {flag}=1 WHERE sample_id IN (SELECT sample_id FROM fit_ids)')
            for parent in manifest['parent_artifacts']:
                if parent not in known:
                    raise ValueError('Unknown checkpoint ancestor: ' + parent)
                inspect(Path(known[parent]['workspace']), parent, known[parent])
            active.remove(identity)
            seen.add(identity)

        inspect(Path(source), 'current')
        labeled, inputs, total = db.execute('''SELECT coalesce(sum(labeled_fit),0),
            coalesce(sum(input_fit),0),count(*) FROM validation_fit''').fetchone()
    return dict(status='passed', artifacts=counts,
                verification_level='declared_membership_and_ancestry',
                actual_fitting_membership_verified=False,
                in_memory_fitting_excluded=False,
                validation_overlap=dict(labeled_fit_samples=labeled, input_fit_samples=inputs,
                                        any_fit_samples=total),
                scope='Declared fitting membership and ancestry; not proof of arbitrary code data flow')
