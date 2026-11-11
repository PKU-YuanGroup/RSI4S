"""Host-owned expert simulator and cumulative annotation ledger.

Only whitelisted native input selectors cross the sandbox boundary. Label access
is authorized per sample, never per raw file, patient, or source archive.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from contextlib import closing

import numpy as np

from benchmarks.evidence import acquire, normalize_ids, request_payload, require_acquired, stage_ids
from .source import COUNTS, NativeReader
from .membership import connect, native_row, spatial_groups


class Evidence:
    def __init__(self, task, package, directory, index, data_root, journal=None, cache_root=None):
        self.task, self.index, self.data_root = task, Path(index), Path(data_root)
        self.directory = Path(directory)
        self.problem = self.directory/'problem'
        self.data = self.problem/'data'
        self.data.mkdir(parents=True, exist_ok=False)
        self.journal = journal
        for name in ('README.md','data_description.md'):
            shutil.copyfile(Path(package)/'problem'/name, self.problem/name)
        from rsi.submission import medrsee_execution_instructions
        (self.problem/'execution.md').write_text(
            medrsee_execution_instructions(task, Path(package)/'problem'))
        shutil.copyfile(Path(__file__).with_name('reader.py'), self.problem/'reader.py')
        self.pool = self.data/'pool.sqlite'
        db = connect(self.pool)
        db.executescript('''
          CREATE TABLE samples (ordinal INTEGER PRIMARY KEY, sample_id TEXT UNIQUE NOT NULL,
            group_id TEXT NOT NULL, holdout_group TEXT NOT NULL, split TEXT NOT NULL,
            acquired INTEGER NOT NULL, role TEXT NOT NULL);
          CREATE TABLE queries (query_id INTEGER PRIMARY KEY, time REAL, actor TEXT, reason TEXT,
            count INTEGER NOT NULL);
          CREATE TABLE acquisitions (sample_id TEXT PRIMARY KEY, query_id INTEGER NOT NULL);
          CREATE TABLE validation (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        db.execute('ATTACH DATABASE ? AS native', (str(self.index),))
        db.execute('''INSERT INTO samples SELECT ordinal,sample_id,group_id,group_id,split,
          CASE WHEN split IN ('D0','V') THEN 1 ELSE 0 END,
          CASE WHEN split='V' THEN 'val' ELSE 'train' END FROM native.members WHERE split!='T' ''')
        if task == 'ENV_04':
            mapping = spatial_groups(index, data_root)
            db.executemany('UPDATE samples SET holdout_group=? WHERE group_id=?',
                           [(v,k) for k,v in mapping.items()])
        db.execute('CREATE INDEX pool_selection ON samples(split,acquired,role,ordinal)')
        db.execute('CREATE INDEX pool_group ON samples(holdout_group)')
        db.commit()
        db.close()
        self.reader = NativeReader(self.data_root)
        from .cache import VolumeCache, VOLUME_TASKS
        self.cache = VolumeCache(cache_root, self.reader) if cache_root and task in VOLUME_TASKS else None
        self.fetch_index = 0
        self._permissions()

    def _permissions(self, changed=None):
        for base in [self.directory, self.problem, self.data]:
            base.chmod(0o755)
        paths = self.problem.rglob('*') if changed is None else [changed, *changed.rglob('*')]
        for path in paths:
            path.chmod(0o755 if path.is_dir() else 0o644)

    def status(self):
        with connect(self.pool) as db:
            used = db.execute('SELECT count(*) FROM acquisitions').fetchone()[0]
            counts = [dict(r) for r in db.execute(
                'SELECT split,role,acquired,count(*) AS count FROM samples GROUP BY split,role,acquired')]
            validation = dict(db.execute('SELECT key,value FROM validation'))
        budget = COUNTS[self.task][2]
        return dict(task=self.task, initial_labels=COUNTS[self.task][1], annotation_budget=budget,
                    annotations_used=used, annotations_remaining=budget-used, counts=counts,
                    validation=validation, official_validation=bool(COUNTS[self.task][3]),
                    pool_database=str(self.pool), reader=str(self.problem/'reader.py'),
                    expert='Organizer native labels revealed only for D0, official V, or approved Q samples')

    _ids = staticmethod(normalize_ids)

    def request(self, *, actor, reason, sample_ids=None, criteria=None, count=None, run_id, tool_call=0):
        from rsi.common import write_json
        payload = request_payload(actor=actor, reason=reason, sample_ids=sample_ids, criteria=criteria, count=count)
        with closing(connect(self.pool)) as db:
            stage_ids(db, payload['sample_ids'])
            if db.execute('''SELECT 1 FROM requested r LEFT JOIN samples s USING(sample_id)
                    WHERE s.sample_id IS NULL OR s.split!='Q' LIMIT 1''').fetchone():
                raise ValueError('Requests may refer only to Q members')
        path = self.data/'requests'/f'{run_id}-{tool_call:06d}.json'
        request = dict(op='request_labels', **payload)
        write_json(path, dict(request, run_id=run_id))
        path.chmod(0o644)
        if self.journal:
            self.journal.add('annotation_request', run_id=run_id, path=str(path), request=request)
        return dict(request_path=str(path), status='pending_meta_decision', annotations_charged=0)

    def query(self, sample_ids, actor, reason, deadline=float('inf'), *, run_id=None):
        # Historic host callers pass the Meta run ID as actor; the shared
        # capability dispatcher supplies the role and attribution separately.
        if actor == 'task':
            raise PermissionError('Only Meta can approve annotation acquisition')
        attribution = run_id or actor
        with closing(connect(self.pool)) as db, db:
            receipt = acquire(db, sample_ids, actor=attribution, reason=reason, budget=COUNTS[self.task][2])
        result = dict(receipt, **self.status())
        if self.journal:
            self.journal.add('annotation_query', actor=attribution, reason=reason, **result)
        return result

    def select_validation(self, sample_ids, deadline=float('inf'), *, actor='task'):
        """Replace internal membership with the Agent's exact authorized sample IDs."""
        if actor != 'task':
            raise PermissionError('Only Task declares validation')
        ids = self._ids(sample_ids)
        if time.monotonic() >= deadline:
            raise TimeoutError('Evidence operation deadline expired')
        if COUNTS[self.task][3]:
            if ids:
                raise ValueError('This task has official V; use it without creating another validation split')
            return self.status()
        with closing(connect(self.pool)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            stage_ids(db, ids)
            require_acquired(db)
            db.execute("UPDATE samples SET role='train' WHERE role='val'")
            db.execute("UPDATE samples SET role='val' WHERE sample_id IN (SELECT sample_id FROM requested)")
            db.execute('DELETE FROM validation')
            policy = dict(kind='exact_sample_ids', count=str(len(ids))) if ids else {}
            db.executemany('INSERT INTO validation VALUES (?,?)', policy.items())
            if time.monotonic() >= deadline:
                raise TimeoutError('Evidence operation deadline expired')
        if self.journal:
            self.journal.add('validation_declared', validation=policy)
        return self.status()

    def members(self, split, sample_ids=None, offset=0, limit=None):
        """Public selections in immutable fixed order. T is absent from this DB."""
        where = {'D0':"s.split='D0'", 'Q':"s.split='Q'", 'V':"s.split='V'",
                 'pool':"s.split='Q' AND acquired=0", 'train':"role='train' AND acquired=1",
                 'val':"role='val' AND acquired=1", 'labeled':"acquired=1", 'all':'1'}
        if split not in where:
            raise ValueError('Unknown public split; T is reserved for final frozen inference')
        db = connect(self.pool)
        try:
            db.execute('ATTACH DATABASE ? AS native', (str(self.index),))
            clause = where[split]
            if sample_ids is not None:
                ids = self._ids(sample_ids)
                stage_ids(db, ids)
                if db.execute('''SELECT count(*) FROM requested LEFT JOIN samples USING(sample_id)
                        WHERE samples.sample_id IS NULL''').fetchone()[0]:
                    raise ValueError('Unknown or private sample ID requested')
                clause += ' AND s.sample_id IN (SELECT sample_id FROM requested)'
            sql = f'''SELECT n.*,s.holdout_group,s.acquired,s.role FROM samples s
                JOIN native.members n USING(sample_id) WHERE {clause} ORDER BY s.ordinal'''
            params = []
            if limit is not None:
                sql += ' LIMIT ? OFFSET ?'
                params = [limit,offset]
            for member in db.execute(sql, params):
                yield dict(member)
        finally:
            db.close()

    def fetch(self, split='train', sample_ids=None, offset=0, limit=256, deadline=float('inf'), pages=1):
        if type(limit) is not int or not 1 <= limit <= 4096 or type(offset) is not int or offset < 0:
            raise ValueError('Require offset>=0 and 1<=limit<=4096; paginate large datasets')
        if type(pages) is not int or not 1 <= pages <= 256:
            raise ValueError('Require 1<=pages<=256; pages run without intervening model calls')
        if sample_ids is not None and len(sample_ids) > limit * pages:
            raise ValueError('Explicit sample_ids exceed limit*pages; increase pages or use smaller batches')
        if pages == 1:
            return self._fetch_members(split, list(self.members(split, sample_ids, offset, limit)), deadline)
        batches, count, complete = [], 0, True
        members = self.members(split, sample_ids, offset, limit * pages)
        import itertools
        try:
            for _ in range(pages):
                chunk = list(itertools.islice(members, limit))
                if not chunk:
                    break
                try:
                    result = self._fetch_members(split, chunk, deadline)
                except TimeoutError:
                    complete = False
                    break
                batches.append(result)
                count += result['count']
        finally:
            members.close()
        return dict(batches=batches, count=count, completed_pages=len(batches), complete=complete,
                    next_offset=offset+count, labeled=sum(b['labeled'] for b in batches),
                    order='fixed ordinal', manifest=str(self.data/'batches.json'))

    def _fetch_members(self, split, members, deadline):
        started = time.monotonic()
        self.fetch_index += 1
        destination = self.data/'batches'/f'{self.fetch_index:06d}'
        try:
            result = export_batch(self.task, members, self.reader, destination, labels=True,
                                  deadline=deadline, cache=self.cache)
            (destination/'membership.json').write_text(
                json.dumps({'pool_database': '../../pool.sqlite'})+'\n')
        except BaseException:
            if destination.exists():
                shutil.rmtree(destination)
            raise
        self._permissions(destination)
        manifest = self.data/'batches.json'
        batches = json.loads(manifest.read_text()) if manifest.exists() else []
        result['seconds'] = time.monotonic()-started
        batches.append(result)
        from rsi.common import write_json
        write_json(manifest, batches)
        manifest.chmod(0o644)
        if self.journal:
            self.journal.add('evidence_fetch', split=split, count=len(members), batch=result['path'],
                             seconds=result['seconds'], cache_hits=result.get('cache_hits',0))
        return result

    def close(self):
        self.reader.close()


def export_batch(task, members, reader, destination, labels=False, deadline=float('inf'), cache=None):
    """Lossless native input export. Tabular columns stay named, including strings."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    tables, targets, cache_hits = [], [], 0
    if task == 'ENV_02':
        from .tabular import TableColumns
        column_reader = TableColumns(reader)
    for i, member in enumerate(members):
        if time.monotonic() >= deadline:
            raise TimeoutError('Shared research time exhausted while reading evidence')
        row = native_row(task, member)
        permitted = labels and bool(member.get('acquired',False))
        item = {k: member[k] for k in ('sample_id','group_id','split','ordinal')}
        item.update(holdout_group=member.get('holdout_group',member['group_id']),
                    role=member.get('role','test'), labeled=permitted)
        if task.startswith('ENV_'):
            if task == 'ENV_02':
                tables.append(column_reader.inputs(row))
                if permitted:
                    label = column_reader.load(row['label_path'][0], row['label_selector'][0])
                    target = np.column_stack(list(label.values())).astype(float).ravel()
                else:
                    target = np.full(1, np.nan)
            else:
                sample = reader.read(row, labels=permitted)
                tables.append(sample['inputs'])
                target = sample['labels'][0].to_numpy(float).ravel() if permitted else np.full(24 if task=='ENV_03' else 1,np.nan)
            if task == 'ENV_01' and permitted:
                target = np.log10(target)
            targets.append(target)
        elif cache is not None:
            arrays, annotations, hits = cache.export(row, destination, i, labels=permitted, deadline=deadline)
            item['arrays'] = arrays
            cache_hits += hits
            if permitted and task == 'MED_04':
                item['annotations'] = annotations
        else:
            from .images import decode
            parts = decode(row, reader, labels=permitted)
            values = {'sample_id':row['sample_id']}
            for j, part in enumerate(parts):
                values[f'input_{j}'] = part['x']
                if part['affine'] is not None:
                    values[f'affine_{j}'] = part['affine']
                if permitted:
                    if task == 'MED_04':
                        item['annotations'] = part['y']
                    else:
                        values[f'label_{j}'] = part['y']
            filename = f'{i:06d}.npz'
            np.savez_compressed(destination/filename, **values)
            item['file'] = filename
        manifest.append(item)
    if tables:
        for part in range(len(tables[0])):
            arrays = task == 'ENV_02'
            columns = list(tables[0][part] if arrays else tables[0][part].columns)
            values = {'columns':np.asarray(columns,dtype=str),
                      'offsets':np.cumsum([0]+[len(t[part][columns[0]]) if arrays else len(t[part])
                                               for t in tables],dtype=np.int64)}
            for c, name in enumerate(columns):
                column = np.concatenate([t[part][name] if arrays else t[part][name].to_numpy() for t in tables])
                if column.dtype.kind == 'O':
                    column = column.astype(str)
                values[f'column_{c}'] = column
            np.savez_compressed(destination/f'input_{part}.npz', **values)
        np.save(destination/'labels.npy',np.asarray(targets,dtype=np.float64),allow_pickle=False)
    (destination/'samples.json').write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False)+'\n')
    return dict(path=str(destination), count=len(manifest), order='fixed ordinal', cache_hits=cache_hits,
                labeled=sum(m['labeled'] for m in manifest))
