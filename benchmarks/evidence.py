"""Shared annotation rules; callers own database layout and label materialization."""
import time


READ_OPS = {'fetch_data', 'annotation_status'}
TASK_OPS = READ_OPS | {'select_validation', 'request_labels'}
META_OPS = READ_OPS | {'query_labels'}
OPS = TASK_OPS | META_OPS


def normalize_ids(values):
    if not isinstance(values, list) or any(not isinstance(s, str) or not s for s in values):
        raise ValueError('sample_ids must be a list of nonempty strings')
    return list(dict.fromkeys(values))


def stage_ids(db, ids, table='requested'):
    """Table names come from adapters, never from an Agent request."""
    db.execute(f'CREATE TEMP TABLE {table} (sample_id TEXT PRIMARY KEY)')
    db.executemany(f'INSERT INTO {table} VALUES (?)', [(s,) for s in ids])


def require_acquired(db, samples='samples', requested='requested'):
    bad = db.execute(f'''SELECT r.sample_id FROM {requested} r LEFT JOIN {samples} s USING(sample_id)
        WHERE s.sample_id IS NULL OR s.acquired!=1 LIMIT 1''').fetchone()
    if bad:
        raise ValueError('Validation requires already acquired labels: ' + bad[0])


def acquire(db, ids, *, actor, reason, budget, samples='samples', acquisitions='acquisitions', check_open=None):
    """Begin one atomic query. The caller reveals labels and commits, or rolls back.

    Both adapters use their existing tables. No raw label is read here, and a
    failed reveal must roll back the receipt and authorization in the same transaction.
    """
    ids = normalize_ids(ids)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Explain the information gap and annotation selection rationale')
    db.execute('BEGIN IMMEDIATE')
    if check_open:
        check_open(db)
    stage_ids(db, ids)
    bad = db.execute(f'''SELECT r.sample_id FROM requested r LEFT JOIN {samples} s USING(sample_id)
        WHERE s.sample_id IS NULL OR s.split!='Q' LIMIT 1''').fetchone()
    if bad:
        raise ValueError('Queries must refer only to original Q members: ' + bad[0])
    new = db.execute(f'SELECT count(*) FROM requested JOIN {samples} USING(sample_id) WHERE acquired=0').fetchone()[0]
    used = db.execute(f'SELECT count(*) FROM {acquisitions}').fetchone()[0]
    if new > budget - used:
        raise ValueError(f'Annotation budget exceeded: new={new}, remaining={budget-used}')
    qid = db.execute('INSERT INTO queries(time,actor,reason,count) VALUES (?,?,?,?)',
                     (time.time(), actor, reason, new)).lastrowid
    db.execute(f'''INSERT INTO {acquisitions} SELECT sample_id,? FROM requested JOIN {samples}
        USING(sample_id) WHERE acquired=0''', (qid,))
    db.execute(f'UPDATE {samples} SET acquired=1 WHERE sample_id IN (SELECT sample_id FROM requested)')
    return dict(query_id=qid, requested=len(ids), newly_annotated=new, already_annotated=len(ids)-new)


def request_payload(*, actor, reason, sample_ids=None, criteria=None, count=None):
    if actor != 'task':
        raise PermissionError('Only Task proposes annotation requests through this method')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Explain the information gap')
    ids = [] if sample_ids is None else normalize_ids(sample_ids)
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError('count must be positive')
    return dict(reason=reason, sample_ids=ids, criteria=criteria, count=count)
