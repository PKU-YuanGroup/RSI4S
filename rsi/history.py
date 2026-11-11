"""Read-only, on-demand access to completed research, without prompt injection."""
from __future__ import annotations

import codecs
import json
import os
from pathlib import Path
import stat

from rsi.common import write_json
from rsi.agent_sources import verify_revision
from rsi.history_view import PAGE_BYTES, json_bytes, row_page
from rsi.store import safe_path, seal, verify_task_source_archive

OPS = {'list_history_files', 'read_history_file'}


def publish(root, session, record):
    """Link explicit research-log files only; never expose an experiment root.

    Called after the record is complete and all sandbox descendants are stopped.
    Links avoid copying long transcripts. No interpretation, ranking or model
    summarization is performed here. API headers/credentials and scorer logs
    are deliberately outside this allowlist.
    """
    root, session = Path(root), Path(session)
    run_id = record['run_id']
    target = safe_path(root/'history', run_id)
    if target.exists():
        raise ValueError('Completed history cannot be replaced')
    staging = target.with_name(target.name + '.partial')
    staging.mkdir(parents=True, exist_ok=False)
    entries = []

    def link(source, name, kind):
        source = Path(source)
        if not source.exists():
            return
        if not stat.S_ISREG(source.lstat().st_mode) or source.is_symlink():
            raise ValueError('History source must be a regular host log: ' + str(source))
        if source.stat().st_uid != os.geteuid():
            raise ValueError('History source must be owned by the host: ' + str(source))
        destination = safe_path(staging, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        destination.chmod(0o444)
        entries.append(dict(path=name, kind=kind, bytes=source.stat().st_size))

    link(root/'records'/f'{run_id}.json', 'record.json', 'record')
    link(session/'context.json', 'context.json', 'context')
    archive = record.get('agent_source_archive')
    if archive:
        archive = verify_task_source_archive(root, record)
        source = Path(archive['path'])
        for name in archive['files']:
            path = safe_path(source, name)
            entries.append(dict(path='agent/source/' + Path(name).as_posix(), kind='agent_source',
                                bytes=path.stat().st_size, source_path=str(path)))
    for manifest in sorted((session/'agent/source-revisions').glob('*/manifest.json')):
        revision = manifest.parent.name
        if revision.startswith('.partial-'):
            continue  # An interrupted copy is not a published source revision.
        value = json.loads(manifest.read_text())
        verified = verify_revision(session/'agent', revision, value['files'], record['task_version'])
        for path in [manifest, *(safe_path(verified/'source', name) for name in value['files'])]:
            entries.append(dict(path=str(path.relative_to(session)), kind='agent_source_revision',
                                bytes=path.stat().st_size, source_path=str(path)))
    for folder, pattern, kind in [
        ('tool-results', '*.json', 'tool_result'), ('tool-results', '*.stdout', 'tool_stdout'),
        ('prompt-audit', '*.json', 'model_input'), ('protocol-errors', '*.json', 'protocol_error'),
        ('agent/processes', 'exec-*.log', 'agent_log'),
        ('science', 'exec-*.log', 'science_log'),
        ('science', 'exec-*.audit.json', 'scientific_execution_audit'),
        ('science/source-audit', '*.json', 'scientific_source_manifest'),
        ('science/source-audit/sources', '*.txt', 'scientific_source'),
        ('development_commands', 'exec-*.log', 'development_log')]:
        for source in sorted((session/folder).glob(pattern)):
            link(source, str(source.relative_to(session)), kind)
    first = record.get('budget_before', {}).get('calls', 0) + 1
    last = record.get('budget_after', {}).get('calls', 0)
    for call in range(first, last + 1):
        for suffix in ('request.json', 'response.json', 'response.txt', 'error.json'):
            name = f'{call:04d}.{suffix}'
            link(root/'calls'/name, 'api/'+name, 'api_'+suffix.split('.')[0])
    # These are label-free scientific inference logs, not private scorer logs.
    for item in record.get('artifacts', []):
        replay = Path(item['workspace']).parent/'replay'
        if not replay.resolve().is_relative_to(session.resolve()):
            continue
        prefix = 'validation/'+item['id']
        link(replay.parent/'host_execution.json', prefix+'/host_execution.json', 'scientific_execution_evidence')
        link(replay/'inference_result.json', prefix+'/inference_result.json', 'validation_inference')
        for source in sorted((replay/'inference_logs').glob('exec-*.log')):
            link(source, prefix+'/'+source.name, 'validation_inference')
        # Active NatureBench replays use separate fit/predict workers. Include
        # every attempt, including failed fits, rather than only the tool's tail.
        # Keep private execution configs and terminal Test logs outside history.
        for phase in ('fit_logs','prediction_logs'):
            for source in sorted((replay/phase).glob('attempt-*/exec-*.log')):
                link(source, prefix+'/'+str(source.relative_to(replay)), 'scientific_worker_log')
    manifest = dict(run_id=run_id, files=sorted(entries, key=lambda e:e['path']),
                    read_only=True, completed_research_only=True,
                    content_in_prompt=False, offset_unit='UTF-8 bytes')
    write_json(staging/'manifest.json', manifest)
    seal(staging)
    os.rename(staging, target)
    return dict(path=str(target), files=len(entries), bytes=sum(e['bytes'] for e in entries))


def read_page(path, offset=0, limit=4096):
    """Seek directly to a UTF-8 byte offset; read at most one bounded page."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 4 <= limit <= 50000:
        raise ValueError('Use offset>=0 and limit=4..50000 UTF-8 bytes; default limit=4096')
    size = path.stat().st_size
    with path.open('rb') as stream:
        stream.seek(offset)
        chunk = stream.read(limit)
    if chunk and chunk[0] & 0xc0 == 0x80:
        raise ValueError('Offset splits a UTF-8 character; use the returned next_offset')
    # Console logs may contain malformed UTF-8; use the same replacement policy
    # as their original tool result. Byte ranges remain explicit and recoverable.
    def decode(length):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        text = decoder.decode(chunk[:length], final=offset+length >= size)
        return text, length-len(decoder.getstate()[0])
    text, consumed = decode(len(chunk))
    if json_bytes(text) > PAGE_BYTES:
        low, high = 0, len(chunk)
        while low < high:
            middle = (low + high + 1) // 2
            if json_bytes(decode(middle)[0]) <= PAGE_BYTES:
                low = middle
            else:
                high = middle - 1
        text, consumed = decode(low)
    return dict(text=text, offset=offset, next_offset=offset+consumed if offset+consumed < size else None,
                offset_unit='UTF-8 bytes', bytes_read=consumed, total_bytes=size, page_byte_limit=PAGE_BYTES)


def resolve_history_file(root, run_id, entry):
    """Resolve sealed virtual source references using their verified session record."""
    base = safe_path(Path(root)/'history', run_id)
    name = entry['path']
    if entry['kind'] == 'agent_source':
        record = json.loads(safe_path(Path(root)/'records', run_id+'.json').read_text())
        archive = verify_task_source_archive(root, record)
        path = safe_path(Path(archive['path']), str(Path(name).relative_to('agent/source')))
    elif entry['kind'] == 'agent_source_revision':
        relative = Path(name).relative_to('agent/source-revisions')
        revision = relative.parts[0]
        record = json.loads(safe_path(Path(root)/'records', run_id+'.json').read_text())
        directory = safe_path(Path(root)/'sessions', run_id)/'agent'
        value = json.loads((safe_path(directory/'source-revisions', revision)/'manifest.json').read_text())
        verified = verify_revision(directory, revision, value['files'], record['task_version'])
        path = safe_path(verified, str(Path(*relative.parts[1:])))
    else:
        path = safe_path(base, name)
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('History file must remain a regular file')
    return path


def query(root, request):
    base = safe_path(Path(root)/'history', request['run_id'])
    if not (Path(root)/'records'/f"{request['run_id']}.json").is_file():
        raise ValueError('Only completed research sessions are visible')
    manifest = json.loads((base/'manifest.json').read_text())
    entries = manifest['files']
    if request['op'] == 'list_history_files':
        offset, limit = request.get('offset', 0), request.get('limit', 20)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('History file listing requires offset>=0 and limit=1..100')
        shown, end = row_page(iter(entries[offset:offset+limit]), offset, limit, PAGE_BYTES - 2048)
        return dict(run_id=request['run_id'], total_files=len(entries), files=shown,
                    offset=offset, next_offset=end if end < len(entries) else None,
                    page_byte_limit=PAGE_BYTES,
                    detail='read_history_file(run_id, path, offset, limit); content is never loaded automatically')
    name = request['path']
    entry = next((e for e in entries if e['path'] == name), None)
    if entry is None:
        raise ValueError('Not a published research history file')
    path = resolve_history_file(root, request['run_id'], entry)
    return dict(run_id=request['run_id'], path=name,
                **read_page(path, request.get('offset', 0), request.get('limit', 4096)))
