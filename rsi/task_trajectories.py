"""Lossless, deduplicated review input from published completed Task histories."""
import hashlib
import json
from pathlib import Path
import stat
import time

from rsi.store import safe_path


def history_review_purpose(context, request, messages):
    """Limit structured review settings to an intact, host-supplied BIO-07 segment."""
    if 'history_review_segment' not in request:
        return None
    index = request['history_review_segment']
    review = context.get('complete_task_trajectories') or {}
    chunks = review.get('chunks', [])
    if (context.get('kind') != 'meta' or context.get('task_id') != 'LIFE_03'
            or review.get('protocol') != 'full_task_trajectories_v1'
            or type(index) is not int or not 0 <= index < len(chunks)):
        raise ValueError('Invalid complete-history review request')
    segment = '<complete_task_history_segment>\n'+chunks[index]+'\n</complete_task_history_segment>'
    if not any(message['role'] == 'user' and segment in message['content'] for message in messages):
        raise ValueError('Complete-history review must include the entire original segment')
    return 'complete_task_history'


def complete_history_context(context, store, *, deadline=float('inf')):
    if context.get('kind') != 'meta' or context.get('task_id') != 'LIFE_03':
        return context
    return {**context, 'complete_task_trajectories':
            task_trajectories(store.root, store.records(), deadline=deadline)}


def task_trajectories(root, records, *, deadline=float('inf'), chunk_chars=90000):
    if type(chunk_chars) is not int or not 1024 <= chunk_chars <= 90000:
        raise ValueError('Use 1024..90000 characters per review chunk')
    root = Path(root)
    seen, sections, files, task_ids = {}, [], [], []

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError('Complete Task history preparation exceeded the stage deadline')

    def encoded(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def emit(source, content):
        check_time()
        text = content if isinstance(content, str) else encoded(content)
        identity = hashlib.sha256(text.encode()).hexdigest()
        if identity in seen:
            return seen[identity]
        else:
            seen[identity] = len(seen)
            sections.append(encoded(dict(section=seen[identity], source=source, sha256=identity, characters=len(text)))+'\n')
            sections.append(text+'\n')
        return seen[identity]

    def message(source, value):
        if not isinstance(value, dict) or not isinstance(value.get('content'), str):
            return emit(source, value)
        fields = {key: ({'section': emit(source+'/'+key, item)} if isinstance(item, str) else item)
                  for key, item in value.items()}
        return emit(source+'/metadata', fields)

    for record in sorted(records, key=lambda item: item['run_id']):
        if record['kind'] != 'task':
            continue
        run_id = record['run_id']
        base = safe_path(root/'history', run_id)
        manifest = json.loads((base/'manifest.json').read_text())
        if not manifest.get('read_only') or not manifest.get('completed_research_only'):
            raise ValueError('Require published completed research history')
        task_ids.append(run_id)
        emit(run_id+'/manifest.json', manifest)
        for info in manifest['files']:
            check_time()
            from rsi.history import resolve_history_file
            path = resolve_history_file(root, run_id, info)
            status = path.lstat()
            if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o222 or status.st_size != info['bytes']:
                raise ValueError('Historical file changed: '+str(path))
            raw = path.read_bytes()
            text = raw.decode('utf-8', errors='replace')
            name = run_id+'/'+info['path']
            files.append(dict(source=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            value = json.loads(text) if path.suffix == '.json' else None
            if info['kind'] == 'api_response' and path.suffix == '.txt':
                try:
                    value = json.loads(text)
                except ValueError:
                    pass  # A raw failed/streamed response still enters in full.
            if info['kind'] in ('api_request', 'model_input') and isinstance(value, dict):
                key = 'messages' if info['kind'] == 'api_request' else 'original_messages'
                if isinstance(value.get(key), list):
                    refs = [message(name+'/'+key+'/'+str(i), m) for i, m in enumerate(value[key])]
                    files[-1]['section'] = emit(name, {**value, key: {'message_sections': refs}})
                    continue
            if info['kind'] == 'api_response' and isinstance(value, dict) and isinstance(value.get('choices'), list):
                for i, choice in enumerate(value['choices']):
                    if isinstance(choice, dict) and 'message' in choice:
                        identity = message(name+'/choices/'+str(i)+'/message', choice['message'])
                        choice['message'] = {'message_section': identity}
                files[-1]['section'] = emit(name, value)
            else:
                files[-1]['section'] = emit(name, value if value is not None else text)
    emit('complete_file_coverage', files)
    transcript = ''.join(sections)
    chunks = [transcript[start:start+chunk_chars] for start in range(0, len(transcript), chunk_chars)]
    return dict(protocol='full_task_trajectories_v1', task_ids=task_ids, files=files,
                transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
                characters=len(transcript), chunks=chunks,
                rule='All published Task history content; only identical content is referenced once. '
                     'No generated summaries or omitted tails. Invalid log UTF-8 uses replacement decoding.')
