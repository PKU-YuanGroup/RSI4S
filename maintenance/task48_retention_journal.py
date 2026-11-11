"""Linear retention bookkeeping: immutable plan and durable batch intents/commits."""
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import time
from contextlib import contextmanager

from rsi.common import write_json
from rsi.freeze import sha

BATCH_SIZE = 256


@contextmanager
def locked(run):
    with (Path(run)/'retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def safe_path(run, relative, *, protein=False, final_symlink=False):
    path = Path(relative)
    allowed = ('sessions','final','evidence','terminal_native') if protein else ('sessions','final')
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in allowed:
        raise ValueError('Unsafe retention path')
    target = run/path
    # Reject symlinks, including an ancestor that redirects inside the run.
    current = run
    for part in path.parts:
        current = current/part
        if current.is_symlink() and not (final_symlink and current == target):
            raise ValueError('Symlink in retention path')
    checked = target.parent if final_symlink else target
    if not checked.resolve().is_relative_to(run):
        raise ValueError('Retention path escaped run')
    return target


def append(stream, value):
    stream.write(json.dumps(value,sort_keys=True)+'\n')
    stream.flush()
    os.fsync(stream.fileno())


def read_journal(path, total):
    """Only a partial final line can be truncated; complete corrupt rows fail."""
    committed = 0; deleted_bytes = 0; pending = None; end = 0
    with path.open('r+b') as stream:
        while True:
            line = stream.readline()
            if not line:break
            if not line.endswith(b'\n'):
                stream.truncate(end);stream.flush();os.fsync(stream.fileno());break
            row = json.loads(line)
            count = row.get('count')
            if type(count) is not int or count <= 0 or row.get('start') != committed or committed+count > total:
                raise ValueError('Invalid retention journal sequence')
            if row.get('event') == 'intent' and pending is None:
                pending = row
            elif row.get('event') == 'commit' and pending is not None and count == pending['count']:
                if type(row.get('bytes')) is not int or row['bytes'] < 0:
                    raise ValueError('Invalid committed byte count')
                committed += count;deleted_bytes += row['bytes'];pending = None
            else:
                raise ValueError('Invalid retention journal transition')
            end = stream.tell()
    return committed, deleted_bytes, pending


def install(run, result, *, legacy=None, legacy_prefix=0):
    """Called under the run lock, before the first v2 deletion."""
    directory = run/'retention';directory.mkdir(exist_ok=True)
    meta_path = directory/'plan.json'
    if meta_path.exists():
        raise ValueError('A retention plan already exists; resume it')
    plans = result['planned']
    temp = directory/'plan.jsonl.tmp'
    h=hashlib.sha256();total_bytes=0
    with temp.open('wb') as stream:
        for info in plans:
            line=(json.dumps(info,sort_keys=True)+'\n').encode()
            stream.write(line);h.update(line);total_bytes+=info['bytes']
        stream.flush();os.fsync(stream.fileno())
    temp.replace(directory/'plan.jsonl')
    metadata={k:v for k,v in result.items() if k not in ('planned','deleted','status')}
    metadata.update(schema_version=2,plan='retention/plan.jsonl',plan_sha256=h.hexdigest(),
                    planned_count=len(plans),planned_bytes=total_bytes,journal='retention/deletions.jsonl',
                    source_index_sha256=sha(run/result['source_index']))
    metadata['closure_sha256']={name:sha(run/name) for name in ('completion.json','summary.json','final_test.json','terminal_score.json') if (run/name).is_file()}
    if legacy is not None:metadata['legacy']=legacy
    # The initial prefix is carried over from the verified v1 manifest, never re-deleted.
    with (directory/'deletions.jsonl').open('w') as journal:
        if legacy_prefix:
            append(journal,dict(event='intent',start=0,count=legacy_prefix))
            append(journal,dict(event='commit',start=0,count=legacy_prefix,
                               bytes=sum(x['bytes'] for x in plans[:legacy_prefix]),origin='legacy_verified'))
        else:
            journal.flush();os.fsync(journal.fileno())
    from maintenance.task48_archive import durable_json
    durable_json(meta_path,metadata)
    progress=dict(metadata,status='deleting',deleted_count=legacy_prefix,
                  deleted_bytes=sum(x['bytes'] for x in plans[:legacy_prefix]),updated_at=time.time())
    write_json(run/'retention.json',progress)
    return progress


def execute(run, *, archives=None):
    """Resume a verified plan without rescanning or rewriting the entire tree."""
    run=Path(run).resolve();meta_path=run/'retention/plan.json'
    meta=json.loads(meta_path.read_text())
    from maintenance.task48_protein_retention import POLICY, verify_trajectory
    protein = meta.get('policy') == POLICY
    verify_trajectory(run, meta)
    for name,digest in meta['closure_sha256'].items():
        if sha(run/name)!=digest:raise ValueError('Terminal closure changed during retention')
    plan=run/meta['plan'];journal_path=run/meta['journal']
    if meta['plan']!='retention/plan.jsonl' or meta['journal']!='retention/deletions.jsonl':
        raise ValueError('Unexpected retention journal paths')
    if sha(plan)!=meta['plan_sha256'] or sha(run/meta['source_index'])!=meta['source_index_sha256']:
        raise ValueError('Retention plan or source index changed')
    from maintenance.task48_archive import Archives
    archives = archives or Archives(run)
    archives.verify_index(json.loads((run/meta['source_index']).read_text()))
    committed,deleted_bytes,pending=read_journal(journal_path,meta['planned_count'])
    progress=dict(meta,status='deleting',deleted_count=committed,deleted_bytes=deleted_bytes,updated_at=time.time())
    with plan.open() as records,journal_path.open('a') as journal:
        # Verify byte accounting against the immutable plan, including the legacy prefix.
        actual_bytes=0
        for _ in range(committed):
            line=next(records,None)
            if line is None:raise ValueError('Truncated retention plan')
            actual_bytes+=json.loads(line)['bytes']
        if actual_bytes!=deleted_bytes:raise ValueError('Retention byte accounting changed')
        while committed<meta['planned_count']:
            count=pending['count'] if pending else min(BATCH_SIZE,meta['planned_count']-committed)
            batch=[json.loads(line) for line in itertools.islice(records,count)]
            if len(batch)!=count:raise ValueError('Truncated retention plan')
            targets=[];missing=0
            for info in batch:
                path=safe_path(run,info['path'],protein=protein,final_symlink=protein and info.get('symlink',False))
                if not path.exists() and not path.is_symlink():
                    if pending is None:raise ValueError('Unexplained missing retention payload: '+info['path'])
                    missing+=1;targets.append(None);continue
                if protein and info.get('symlink'):
                    valid = path.is_symlink() and os.readlink(path) == info['target'] and hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest() == info['sha256']
                else:
                    valid = path.is_file() and path.stat().st_size==info['bytes'] and sha(path)==info['sha256']
                if not valid:
                    raise ValueError('Payload changed after retention planning: '+info['path'])
                targets.append(path)
            if pending is None:
                append(journal,dict(event='intent',start=committed,count=count))
            # Durable intent covers a crash between unlink and the commit below.
            for path in targets:
                if path is not None:path.unlink()
            # Make removals durable before their commit receipt.
            for parent in {(run/info['path']).parent for info in batch}:
                fd=os.open(parent,os.O_RDONLY | os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            size=sum(info['bytes'] for info in batch)
            append(journal,dict(event='commit',start=committed,count=count,bytes=size,recovered_missing=missing))
            committed+=count;deleted_bytes+=size;pending=None
            progress.update(deleted_count=committed,deleted_bytes=deleted_bytes,updated_at=time.time())
            write_json(run/'retention.json',progress)
        if next(records,None) is not None:raise ValueError('Extra retention plan rows')
    if deleted_bytes!=meta['planned_bytes']:raise ValueError('Incomplete retention byte accounting')
    progress.update(status='completed',finished_at=time.time())
    write_json(run/'retention.json',progress)
    return progress


def migrate_legacy(run, snapshot, *, allow_unrecorded_next=False):
    """Explicit migration of a stopped v1 deletion loop; no scientific rerun."""
    from maintenance.task48_retention import validate_terminal
    run=Path(run).resolve();snapshot=Path(snapshot).resolve()
    with locked(run):
        if validate_terminal(run) is not None:raise ValueError('Incomplete run cannot migrate retention')
        legacy=json.loads(snapshot.read_text())
        if legacy.get('schema_version')!=1 or legacy.get('status')!='deleting':
            raise ValueError('Require a v1 in-progress retention snapshot')
        if sha(run/'retention.json')!=sha(snapshot):raise ValueError('Live legacy manifest changed')
        plans=legacy['planned'];done=legacy['deleted'];count=len(done)
        if done!=plans[:count]:raise ValueError('Legacy deleted list is not the plan prefix')
        for info in done:
            if safe_path(run,info['path']).exists():raise ValueError('Legacy deleted payload reappeared')
        extra=0
        if count<len(plans) and not safe_path(run,plans[count]['path']).exists():
            if not allow_unrecorded_next:raise ValueError('Legacy interrupted unlink needs explicit reconciliation')
            extra=1
        # The old loop writes its manifest before advancing, so at most one extra unlink is possible.
        if count+extra<len(plans) and not safe_path(run,plans[count+extra]['path']).is_file():
            raise ValueError('More than one unrecorded legacy deletion')
        for info in legacy['retained_checkpoints']:
            if sha(safe_path(run,info['path']))!=info['sha256']:raise ValueError('Retained checkpoint changed')
        dest=run/'retention';dest.mkdir(exist_ok=True)
        saved=dest/'legacy-v1.json'
        with snapshot.open('rb') as src,saved.open('xb') as out:shutil.copyfileobj(src,out)
        result=install(run,legacy,legacy=dict(manifest='retention/legacy-v1.json',sha256=sha(saved),
                      recorded_deleted=count,reconciled_unrecorded_next=extra),legacy_prefix=count+extra)
        return result


def deleted_records(run):
    """Stream detailed receipts from v1 or v2 without a duplicated giant summary."""
    run=Path(run);summary=json.loads((run/'retention.json').read_text())
    if summary.get('schema_version')!=2:
        yield from summary.get('deleted',[]);return
    committed,_,_=read_journal(run/summary['journal'],summary['planned_count'])
    with (run/summary['plan']).open() as stream:
        for line in itertools.islice(stream,committed):yield json.loads(line)
