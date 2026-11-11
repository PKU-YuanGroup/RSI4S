"""Host-only per-run authorization, shared label ledger, and research data view.

Give scientific sandboxes only `problem`; organizer files and original sources
must never be added to their filesystem allowlist.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sqlite3
import time
from contextlib import closing

from benchmarks.evidence import acquire, normalize_ids, request_payload, require_acquired, stage_ids
from benchmarks.naturebench_active.common import stable, digest
from benchmarks.naturebench_active.common import connect, sha256
from rsi.common import write_json


class Evidence:
    def __init__(self, package, directory, *, wall_seconds=14400, validation_policy=None):
        if type(wall_seconds) is not int or wall_seconds < 1:
            raise ValueError('Research wall_seconds must be a positive integer')
        from benchmarks.naturebench_active.validation_design import validate_policy
        validation_policy = validate_policy(validation_policy)
        self.package=Path(package).resolve();self.directory=Path(directory).resolve()
        self.manifest=json.loads((self.package/'metadata.json').read_text())
        from benchmarks.naturebench_active.adapters.multi_properties import check_package
        check_package(self.manifest)
        self.reference=self.package/'organizer/reference.sqlite'
        if sha256(self.reference)!=self.manifest['reference_sha256']:raise ValueError('Organizer reference changed')
        if sha256(self.package/'problem/data/samples.sqlite')!=self.manifest['public_sha256']:raise ValueError('Public template changed')
        self.directory.mkdir(parents=True,exist_ok=False)
        self.problem=self.directory/'problem'
        shutil.copytree(self.package/'problem',self.problem)
        self.pool=self.problem/'data/samples.sqlite';self.pool.chmod(0o644)
        self.private=self.directory/'organizer';self.private.mkdir(mode=0o700)
        self.ledger=self.private/'ledger.sqlite'
        with connect(self.ledger) as db:
            db.executescript('''
              CREATE TABLE state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
              CREATE TABLE queries(id INTEGER PRIMARY KEY,time REAL NOT NULL,actor TEXT NOT NULL,reason TEXT NOT NULL,count INTEGER NOT NULL);
              CREATE TABLE acquired(sample_id TEXT PRIMARY KEY,query_id INTEGER NOT NULL);
              CREATE TABLE requests(id INTEGER PRIMARY KEY,time REAL NOT NULL,payload TEXT NOT NULL);
            ''')
            db.executemany('INSERT INTO state VALUES (?,?)',
                           [('started',str(time.time())),('deadline',str(time.time()+wall_seconds)),('phase','research'),
                            ('validation',('partial_official' if self.manifest.get('internal_validation_instances') else 'official') if self.manifest['official_validation'] else ''),
                            ('validation_policy',stable(validation_policy)),
                            ('task_id',self.manifest['task_id']),('reference_sha256',self.manifest['reference_sha256'])])
        self.ledger.chmod(0o600)

    @classmethod
    def resume(cls,package,directory):
        obj=cls.__new__(cls);obj.package=Path(package).resolve();obj.directory=Path(directory).resolve()
        obj.manifest=json.loads((obj.package/'metadata.json').read_text());obj.reference=obj.package/'organizer/reference.sqlite'
        from benchmarks.naturebench_active.adapters.multi_properties import check_package
        check_package(obj.manifest)
        if sha256(obj.reference)!=obj.manifest['reference_sha256']:raise ValueError('Organizer reference changed')
        obj.problem=obj.directory/'problem';obj.pool=obj.problem/'data/samples.sqlite'
        obj.private=obj.directory/'organizer';obj.ledger=obj.private/'ledger.sqlite'
        if not obj.ledger.is_file():raise ValueError('No existing annotation ledger to resume')
        with connect(obj.ledger) as db:
            state=dict(db.execute('SELECT key,value FROM state'))
        if state['task_id']!=obj.manifest['task_id'] or state['reference_sha256']!=obj.manifest['reference_sha256']:
            raise ValueError('Cannot resume a ledger with another task or data release')
        return obj

    def _connect(self):
        db=sqlite3.connect(self.ledger,uri=True);db.row_factory=sqlite3.Row
        db.execute('ATTACH DATABASE ? AS public',(str(self.pool),))
        # A writable attachment makes private-run commits seek an exclusive
        # lock on this shared reference, even when no reference rows changed.
        db.execute('ATTACH DATABASE ? AS reference',(self.reference.as_uri()+'?mode=ro',))
        return db

    _ids = staticmethod(normalize_ids)

    @staticmethod
    def _research(db):
        state=dict(db.execute('SELECT key,value FROM state'))
        if state['phase']!='research' or time.time()>=float(state['deadline']):
            raise ValueError('Research is closed; no further evidence acquisition or dataset changes')

    def status(self):
        with self._connect() as db:
            used=db.execute('SELECT count(*) FROM acquired').fetchone()[0]
            state=dict(db.execute('SELECT key,value FROM state'))
            counts=[dict(r) for r in db.execute('SELECT instance,split,role,acquired,count(*) AS count FROM public.samples GROUP BY instance,split,role,acquired')]
        return dict(task=self.manifest['task_id'],train_pool=self.manifest['train_pool'],
                    initial_labels=self.manifest['D0'],annotation_budget=self.manifest['annotation_budget'],
                    annotations_used=used,annotations_remaining=self.manifest['annotation_budget']-used,
                    official_validation=bool(self.manifest['official_validation']),validation=state['validation'],
                    validation_policy=json.loads(state.get('validation_policy','{"policy":"agent_selected_v1"}')),
                    validation_design=json.loads(state['validation_design']) if 'validation_design' in state else None,
                    phase=state['phase'],research_seconds_remaining=max(0,float(state['deadline'])-time.time()),counts=counts)

    def request(self,*,actor,reason,sample_ids=None,criteria=None,count=None,run_id=None,tool_call=0):
        payload=request_payload(actor=actor,reason=reason,sample_ids=sample_ids,criteria=criteria,count=count)
        with self._connect() as db:
            self._research(db)
            for sid in payload['sample_ids']:
                if db.execute("SELECT 1 FROM public.samples WHERE sample_id=? AND split='Q'",(sid,)).fetchone() is None:
                    raise ValueError('Requests may refer only to Q members')
            query=db.execute('INSERT INTO requests(time,payload) VALUES (?,?)',(time.time(),stable(payload))).lastrowid
        return dict(request_id=query,status='pending_meta_decision',annotations_charged=0)

    def query(self,sample_ids,*,actor,reason,deadline=float('inf'),run_id=None):
        if actor!='meta':raise PermissionError('Only Meta can approve annotation acquisition')
        with closing(self._connect()) as db, db:
            receipt=acquire(db,sample_ids,actor=actor,reason=reason,budget=self.manifest['annotation_budget'],
                            samples='public.samples',acquisitions='acquired',check_open=self._research)
            db.execute('''UPDATE public.samples SET label=(SELECT label FROM reference.samples r WHERE r.sample_id=public.samples.sample_id)
                        WHERE sample_id IN (SELECT sample_id FROM requested)''')
        return dict(receipt,**self.status())

    def select_validation(self,sample_ids,*,actor,deadline=float('inf')):
        """Replace internal membership without choosing or diagnosing the Agent's split."""
        if actor!='task':raise PermissionError('Only Task declares validation')
        ids=self._ids(sample_ids)
        if time.monotonic()>=deadline:raise TimeoutError('Evidence operation deadline expired')
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE');self._research(db)
            stage_ids(db,ids,'validation_requested')
            require_acquired(db,'public.samples','validation_requested')
            requested=db.execute('''SELECT s.instance FROM validation_requested r
                                    JOIN public.samples s USING(sample_id)''').fetchall()
            allowed=self.manifest.get('internal_validation_instances')
            if allowed and any(r['instance'] not in allowed for r in requested):
                raise ValueError('This instance already has native validation')
            official=db.execute('''SELECT 1 FROM validation_requested r JOIN public.samples s USING(sample_id)
                WHERE s.instance IN (SELECT DISTINCT instance FROM public.samples WHERE split='V') LIMIT 1''').fetchone()
            if official:raise ValueError('Internal validation would change an instance with native validation')
            db.execute("""UPDATE public.samples SET role='train' WHERE role='val'
                AND instance NOT IN (SELECT DISTINCT instance FROM public.samples WHERE split='V')""")
            db.execute("UPDATE public.samples SET role='val' WHERE sample_id IN (SELECT sample_id FROM validation_requested)")
            counts=dict(db.execute('SELECT role,count(*) FROM public.samples WHERE acquired=1 GROUP BY role'))
            state='agent_sample_holdout' if ids else ''
            if not ids and self.manifest['official_validation']:
                state='partial_official' if self.manifest.get('internal_validation_instances') else 'official'
            db.execute("UPDATE state SET value=? WHERE key='validation'",(state,))
            if time.monotonic()>=deadline:raise TimeoutError('Evidence operation deadline expired')
        return dict(declared=True,counts=counts,selection_mode='replace_internal_validation')

    def fetch(self,split='train',sample_ids=None,offset=0,limit=256,*,deadline=float('inf'),pages=1):
        if type(pages) is not int or pages!=1:
            raise ValueError('Active NatureBench fetch returns one page; use offset/limit or public samples.sqlite')
        from .preparation import check, file_hash, deadline_connection
        check(deadline)
        filters={'train':"role='train' AND acquired=1",'val':"role='val' AND acquired=1",'pool':"acquired=0",
                 'all':'1','labeled':'acquired=1','D0':"split='D0'",'Q':"split='Q'",'V':"split='V'"}
        if split not in filters:raise ValueError('Test is unavailable during research')
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=4096:raise ValueError('Invalid pagination')
        with self._connect() as db, deadline_connection(db,deadline):
            self._research(db)
            clause=filters[split];params=[]
            if sample_ids is not None:
                ids=self._ids(sample_ids)
                stage_ids(db,ids)
                if db.execute('SELECT 1 FROM requested LEFT JOIN public.samples USING(sample_id) WHERE public.samples.sample_id IS NULL LIMIT 1').fetchone():
                    raise ValueError('Unknown or private sample ID')
                clause+=' AND sample_id IN (SELECT sample_id FROM requested)'
            records=[dict(r) for r in db.execute(f'SELECT * FROM public.samples WHERE {clause} ORDER BY instance,ordinal LIMIT ? OFFSET ?',(*params,limit,offset))]
            for r in records:
                check(deadline)
                r['inputs']=json.loads(r['inputs']);r['label']=None if r['label'] is None else json.loads(r['label'])
                native=db.execute('SELECT attachments FROM reference.samples WHERE sample_id=?',(r['sample_id'],)).fetchone()
                files=[]
                if self.manifest['task_id']=='LIFE_16':
                    from benchmarks.naturebench_active.adapters.noncoding import checked_feature
                    feature=checked_feature(self.problem/'data/files',r)
                    files.append(str(feature.relative_to(self.problem)))
                from benchmarks.naturebench_active.adapters.pending import FEATURE_TASKS
                if self.manifest['task_id'] in FEATURE_TASKS:
                    from benchmarks.naturebench_active.adapters.pending import checked_feature
                    feature=checked_feature(self.problem/'data/files',r)
                    files.append(str(feature.relative_to(self.problem)))
                for attachment in json.loads(native[0]):
                    source_root=Path((self.package/'organizer/source_root.txt').read_text().strip())
                    source=source_root/attachment['source']
                    expected=db.execute('SELECT sha256 FROM reference.sources WHERE path=?',(attachment['source'],)).fetchone()
                    if expected is None or file_hash(source,deadline)!=expected[0]:raise ValueError('Native feature integrity check failed')
                    out=self.problem/'data/files'/digest(r['sample_id'])/attachment['name']
                    from benchmarks.naturebench_active.adapters.attachments import materialize
                    materialize(source,out,expected[0],deadline=deadline)
                    files.append(str(out.relative_to(self.problem)))
                if files:r['files']=files
        check(deadline)
        return dict(records=records,next_offset=offset+len(records))

    def close_research(self,artifact):
        """Host-only terminal handoff, never an Agent query capability."""
        from rsi.sandbox import assert_plain_tree
        artifact=Path(artifact).resolve();assert_plain_tree(artifact)
        files={str(p.relative_to(artifact)):sha256(p) for p in sorted(artifact.rglob('*')) if p.is_file()}
        if not files:raise ValueError('Freeze a nonempty final artifact before Test')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if dict(db.execute('SELECT key,value FROM state'))['phase'] not in ('research','selection'):raise ValueError('Research already closed')
            db.execute("UPDATE state SET value='closed' WHERE key='phase'")
            db.execute('INSERT INTO state VALUES (?,?)',('artifact',stable(dict(path=str(artifact),files=files))))
        return dict(phase='closed',artifact_sha256=digest(files))

    def stop_research(self):
        """Close all evidence mutations before Meta's read-only selection."""
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if dict(db.execute('SELECT key,value FROM state'))['phase']=='research':
                db.execute("UPDATE state SET value='selection' WHERE key='phase'")

    def extend_deadline(self,seconds):
        """Host-only: move the research deadline with the host clock (returned infrastructure time)."""
        if not seconds>0:raise ValueError('Deadline extension must be positive')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            state=dict(db.execute('SELECT key,value FROM state'))
            if state['phase']=='research':
                db.execute("UPDATE state SET value=? WHERE key='deadline'",(str(float(state['deadline'])+seconds),))

    def export_test_inputs(self,destination):
        """Host-only one-shot export for frozen final inference; never exposes labels."""
        destination=Path(destination).resolve()
        if destination==self.problem or self.problem in destination.parents:raise ValueError('Final Test must stay outside the research view')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE');state=dict(db.execute('SELECT key,value FROM state'))
            resuming=state['phase']=='test_exporting' and state.get('test_destination')==str(destination)
            if state['phase']!='closed' and not resuming:raise ValueError('Test requires a closed research run and may be released once')
            if destination.exists() and not resuming:raise FileExistsError(destination)
            artifact=json.loads(state['artifact']);root=Path(artifact['path'])
            from rsi.sandbox import assert_plain_tree
            assert_plain_tree(root)
            current={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
            if current!=artifact['files']:raise ValueError('Final artifact changed after freeze')
            db.execute("UPDATE state SET value='test_exporting' WHERE key='phase'")
            db.execute("INSERT OR REPLACE INTO state VALUES ('test_destination',?)",(str(destination),))
        destination.mkdir(parents=True,exist_ok=resuming)
        out=connect(destination/'inputs.sqlite')
        try:
            out.executescript('CREATE TABLE IF NOT EXISTS samples(sample_id TEXT PRIMARY KEY,instance TEXT,ordinal INTEGER,inputs TEXT); CREATE TABLE IF NOT EXISTS assets(asset_id TEXT PRIMARY KEY,payload TEXT); DELETE FROM samples; DELETE FROM assets;')
            with self._connect() as db:
                out.executemany('INSERT INTO samples VALUES (?,?,?,?)',db.execute("SELECT sample_id,instance,ordinal,inputs FROM reference.samples WHERE split='T' ORDER BY instance,ordinal"))
                out.executemany('INSERT INTO assets VALUES (?,?)',db.execute('SELECT asset_id,payload FROM reference.assets'))
                # Feature files must be copied individually; raw source directories
                # containing unqueried labels never cross the inference boundary.
                source_root=Path((self.package/'organizer/source_root.txt').read_text().strip())
                from benchmarks.naturebench_active.inputs import export_identifiers
                export_identifiers(self,db,destination,source_root)
                if self.manifest['task_id']=='LIFE_16':
                    from benchmarks.naturebench_active.adapters.noncoding import export_features
                    export_features(self,db,destination)
                from benchmarks.naturebench_active.adapters.pending import FEATURE_TASKS, TASKS as PENDING_TASKS
                if self.manifest['task_id'] in FEATURE_TASKS:
                    from benchmarks.naturebench_active.adapters.pending import export_features
                    export_features(self,db,destination)
                if self.manifest['task_id'] in PENDING_TASKS:
                    from benchmarks.naturebench_active.adapters.pending import copy_resources
                    copy_resources(self.package,destination,self.manifest)
                for row in db.execute("SELECT sample_id,attachments FROM reference.samples WHERE split='T'"):
                    for a in json.loads(row['attachments']):
                        expected=db.execute('SELECT sha256 FROM reference.sources WHERE path=?',(a['source'],)).fetchone()[0]
                        source=source_root/a['source']
                        if sha256(source)!=expected:raise ValueError('Test input source changed')
                        path=destination/'files'/digest(row['sample_id'])/a['name']
                        from benchmarks.naturebench_active.adapters.attachments import materialize
                        # Frozen maps can be large; stored gzip blocks avoid
                        # spending inference time recompressing numeric inputs.
                        materialize(source,path,expected,map_compresslevel=0)
            out.commit()
        finally:
            out.close()
        (destination/'inputs.sqlite').chmod(0o444)
        with self._connect() as db:
            db.execute("UPDATE state SET value='test_released' WHERE key='phase'")
        return dict(inputs=str(destination/'inputs.sqlite'),count=self.manifest['T'],feedback_to_research=False)
