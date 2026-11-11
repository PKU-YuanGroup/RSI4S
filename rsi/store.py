"""Append-only history and content-addressed, independently bound Agent programs."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from rsi.common import digest, encoded, write_json
from rsi.history_view import PAGE_BYTES, evaluation_view, manifest_view, object_view, row_page, text_page, text_prefix
from rsi.sandbox import assert_plain_tree


def hashes(root, ignore=()):
    root = Path(root)
    assert_plain_tree(root, ignore=ignore)
    result = {}
    for path in sorted(root.rglob('*')):
        if any(part in ignore for part in path.relative_to(root).parts):
            continue
        if path.is_file() and '__pycache__' not in path.parts:
            # Scientific artifacts include multi-GiB arrays and checkpoints.
            # Integrity checks must not allocate one whole file in the host.
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
            result[str(path.relative_to(root))] = hasher.hexdigest()
    return result


def seal(root):
    for p in [Path(root), *Path(root).rglob('*')]:
        p.chmod(0o555 if p.is_dir() else 0o444)


def artifact_signature(root, deadline=None):
    """Metadata guard for host-owned sealed files; never trust writable trees."""
    root = Path(root)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Artifact verification deadline reached')
    assert_plain_tree(root)
    signature = []
    reusable = True
    # Linux may stamp multiple rapid changes with the same coarse clock tick.
    # Never reuse a digest while a ctime is fresh, including after a wall-clock
    # rollback. A full content hash remains mandatory during this guard window.
    stable_before = time.time_ns() - 2_000_000_000
    for path in [root, *sorted(root.rglob('*'))]:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Artifact verification deadline reached')
        stat = path.lstat()
        signature.append((str(path.relative_to(root)), stat.st_dev, stat.st_ino,
                          stat.st_mode, stat.st_uid, stat.st_gid, stat.st_nlink,
                          stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        reusable &= (stat.st_uid == os.geteuid() and not stat.st_mode & 0o222
                     and stat.st_ctime_ns < stable_before)
    return tuple(signature), reusable


def safe_path(root, name):
    root = Path(root).resolve()
    path = root / name
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Expected a relative path without traversal')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escaped its allowed root')
    return path


def read_text(path, offset=0, limit=16000, deadline=None):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50000:
        raise ValueError('Invalid pagination; limit must be 1..50000')
    path = Path(path)
    # Preserve the character-offset contract without allocating a whole log or
    # checkpoint. Exact total_chars is known only after reaching EOF.
    with path.open(encoding='utf-8') as stream:
        skipped = 0
        while skipped < offset:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('History read deadline reached')
            part = stream.read(min(65536, offset-skipped))
            if not part:
                return dict(text='', offset=offset, total_chars=skipped, next_offset=None,
                            total_bytes=path.stat().st_size)
            skipped += len(part)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('History read deadline reached')
        text = stream.read(limit+1)
    if '\0' in text:
        raise ValueError('Text reads do not accept binary artifacts')
    shown = text_prefix(text[:limit])
    more = len(text) > len(shown)
    return dict(text=shown, offset=offset, total_chars=None if more else offset+len(text),
                next_offset=offset+len(shown) if more else None, total_bytes=path.stat().st_size,
                offset_unit='characters', page_byte_limit=PAGE_BYTES)


def verify_task_source_archive(root, record):
    """Resolve only a recorded, sealed source tree from a completed Task."""
    archive = record.get('agent_source_archive')
    if record.get('kind') != 'task' or not archive or not archive.get('read_only'):
        raise ValueError('Completed Task has no sealed Agent source archive')
    run_id = record['run_id']
    source = safe_path(Path(root) / 'sessions', run_id) / 'agent' / 'source'
    if (archive.get('run_id') != run_id or archive.get('base_version') != record['task_version']
            or Path(archive['path']).absolute() != source.absolute()):
        raise ValueError('Task source archive does not match its completed record')
    actual = hashes(source)
    for path in [source, *source.rglob('*')]:
        info = path.lstat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o222:
            raise RuntimeError('Completed Task source must remain host-owned and read-only')
    if actual != archive['files']:
        raise RuntimeError('Task source archive integrity violation')
    return archive


class Store:
    def __init__(self, root, journal, dependency):
        self.root, self.journal, self.dependency = Path(root), journal, dependency
        for name in ('versions', 'candidates', 'records', 'history'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.candidates = {}
        self._artifact_hash_cache = {}

    def version(self, identity):
        path = safe_path(self.root / 'versions', identity)
        info = json.loads((path / 'manifest.json').read_text())
        body = {k: info[k] for k in ('kind', 'files', 'dependency')}
        if identity != info['kind'][0].upper() + '-' + digest(body)[:20] or hashes(path / 'source') != info['files']:
            raise RuntimeError('Published Agent integrity violation')
        if info['dependency'] != self.dependency:
            raise RuntimeError('Agent dependency binding differs')
        return path, info

    def publish_seed(self, kind, **metadata):
        """Publish one assembled seed; successors remain independent source trees."""
        from tempfile import TemporaryDirectory
        from agent import materialize
        with TemporaryDirectory(prefix='seed-', dir=self.root) as directory:
            source = materialize(kind, Path(directory) / 'source')
            return self.publish(kind, source, **metadata)

    def publish(self, kind, source, base=None, sources=(), author='initial', rationale='', checks=None,
                source_run=None):
        if kind not in ('task', 'meta'):
            raise ValueError('Unknown Agent kind')
        if source_run is not None:
            if kind != 'task':
                raise ValueError('Only Task may use a completed Task source archive')
            self.task_source(source_run)
        files = hashes(source)
        body = {'kind': kind, 'files': files, 'dependency': self.dependency}
        identity = kind[0].upper() + '-' + digest(body)[:20]
        target = self.root / 'versions' / identity
        if target.exists():
            self.version(identity)
            self.journal.add('version_reused', version=identity, base_version=base, source_versions=list(sources),
                             source_run=source_run, author=author)
            return identity
        target.mkdir()
        shutil.copytree(source, target / 'source', ignore=shutil.ignore_patterns('__pycache__'))
        patch = []
        old = self.version(base)[0] / 'source' if base else None
        paths = sorted(set(files) | (set(hashes(old)) if old else set()))
        for name in paths:
            before = (old / name).read_text().splitlines(True) if old and (old / name).exists() else []
            after = (Path(source) / name).read_text().splitlines(True) if (Path(source) / name).exists() else []
            patch.extend(difflib.unified_diff(before, after, fromfile='a/' + name, tofile='b/' + name))
        (target / 'change.patch').write_text(''.join(patch))
        write_json(target / 'manifest.json', {**body, 'id': identity, 'base_version': base,
                   'source_versions': list(sources), 'author': author, 'rationale': rationale,
                   'source_run': source_run,
                   'checks': checks, 'patch_sha256': hashlib.sha256(''.join(patch).encode()).hexdigest()})
        seal(target)
        self.journal.add('version_published', version=identity, agent_kind=kind, base_version=base,
                         source_versions=list(sources), source_run=source_run, author=author, files=files)
        return identity

    def create(self, kind, base_version, source_versions=(), rationale='', source_run=None):
        path, info = self.version(base_version)
        if info['kind'] != kind:
            raise ValueError('Base kind mismatch')
        if source_run is not None and kind != 'task':
            raise ValueError('Only Task may record a completed Task source archive')
        for identity in source_versions:
            if self.version(identity)[1]['kind'] != kind:
                raise ValueError('Combined source kind mismatch')
        candidate = f'C-{len(self.candidates):04d}'
        destination = self.root / 'candidates' / candidate
        if destination.exists():
            raise RuntimeError('Candidate collision: existing experiments may not be restarted')
        shutil.copytree(path / 'source', destination)
        for p in [destination, *destination.rglob('*')]:
            p.chmod(0o755 if p.is_dir() else 0o644)
        self.candidates[candidate] = {'kind': kind, 'base_version': base_version,
                                     'source_versions': list(source_versions), 'rationale': rationale,
                                     'source_run': source_run}
        self.journal.add('candidate_created', candidate=candidate, metadata=self.candidates[candidate])
        return candidate

    def candidate(self, identity):
        if identity not in self.candidates:
            raise ValueError('Unknown candidate')
        return self.root / 'candidates' / identity

    def edit(self, candidate, path, content):
        if not isinstance(content, str) or len(content.encode()) > 150000:
            raise ValueError('Source text too large')
        target = safe_path(self.candidate(candidate), path)
        if target.suffix not in ('.py', '.txt', '.json', '.md'):
            raise ValueError('Only source/text/config files allowed')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self.journal.add('candidate_edited', candidate=candidate, path=path,
                         sha256=hashlib.sha256(content.encode()).hexdigest())

    def syntax(self, candidate):
        from rsi.context_policy import load_policy
        root = self.candidate(candidate)
        assert_plain_tree(root)
        load_policy(root)
        for required in ('entry.py', 'tools.py', 'prompt.txt'):
            if not (root / required).is_file():
                raise ValueError('Missing entry interface file: ' + required)
        for p in root.rglob('*.py'):
            ast.parse(p.read_text(), filename=str(p.relative_to(root)))
        return hashes(root)

    def save_record(self, record):
        target = self.root / 'records' / (record['run_id'] + '.json')
        if target.exists():
            raise RuntimeError('Cannot overwrite completed history')
        write_json(target, record)
        target.chmod(0o444)
        self.journal.add('history_appended', run_id=record['run_id'], sha256=hashlib.sha256(target.read_bytes()).hexdigest())

    def records(self):
        return [json.loads(p.read_text()) for p in sorted((self.root / 'records').glob('*.json'))]

    def record(self, run_id):
        return json.loads(safe_path(self.root / 'records', run_id + '.json').read_text())

    def archive_task_source(self, run_id, base_version):
        """Freeze the completed Task's own checkout in place; do not publish a successor."""
        source = safe_path(self.root / 'sessions', run_id) / 'agent' / 'source'
        files = hashes(source)
        # The sandbox UID must not retain ownership of a completed archive.
        for path in [source, *source.rglob('*')]:
            os.chown(path, os.geteuid(), os.getegid(), follow_symlinks=False)
        seal(source)
        archive = dict(run_id=run_id, path=str(source), base_version=base_version,
                       files=files, read_only=True)
        self.journal.add('agent_source_archived', **archive)
        return archive

    def task_source(self, run_id):
        return verify_task_source_archive(self.root, self.record(run_id))

    def artifact(self, identity, deadline=None):
        for record in self.records():
            for artifact in record['artifacts']:
                if artifact['id'] == identity:
                    workspace = artifact['workspace']
                    def signature():
                        if deadline is None:
                            return artifact_signature(workspace)
                        from rsi.host_io import run_io
                        value = run_io('artifact_signature', deadline, source=workspace)
                        return tuple(tuple(row) for row in value['signature']), value['reusable']
                    before, reusable = signature()
                    cached = self._artifact_hash_cache.get(workspace)
                    if reusable and cached is not None and cached[0] == before:
                        actual = cached[1]
                    else:
                        if deadline is None:
                            actual = hashes(workspace)
                        else:
                            from rsi.host_io import run_io
                            actual = run_io('hashes', deadline, source=workspace)
                        after, still_sealed = signature()
                        if before != after:
                            raise RuntimeError('Artifact changed during integrity verification')
                        if actual != artifact['files']:
                            self._artifact_hash_cache.pop(workspace, None)
                            raise RuntimeError('Artifact integrity violation')
                        if still_sealed:
                            self._artifact_hash_cache[workspace] = (after, actual)
                        else:
                            self._artifact_hash_cache.pop(workspace, None)
                    if actual != artifact['files']:
                        raise RuntimeError('Artifact integrity violation')
                    return artifact
        raise ValueError('Artifact is not in completed visible history')

    def query(self, request, reader, deadline=None):
        op = request['op']
        self.journal.add('history_read', reader=reader, request=request)
        from rsi.history import OPS, query as history_query
        if op in OPS:
            return history_query(self.root, request)
        if op == 'list_history':
            records = self.records()
            offset, limit = request.get('offset', 0), request.get('limit', 20)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
                raise ValueError('Invalid history pagination')
            versions = sorted((self.root / 'versions').iterdir())
            version_offset = request.get('version_offset', 0)
            if type(version_offset) is not int or version_offset < 0:
                raise ValueError('Invalid version pagination')
            runs, end = row_page(({**self.record_index(r), 'research_history_available':
                                  (self.root/'history'/r['run_id']/'manifest.json').is_file()}
                                 for r in records[offset:offset + limit]), offset, limit, 19000)
            version_rows, version_end = row_page(
                (object_view({k: info[k] for k in ('id', 'kind', 'base_version', 'author')}, 2048)[0]
                 for info in (self.version(p.name)[1] for p in versions[version_offset:version_offset + limit])),
                version_offset, limit, 3500)
            return {'total_runs': len(records), 'offset': offset,
                    'next_offset': end if end < len(records) else None, 'runs': runs,
                    'total_versions': len(versions), 'version_offset': version_offset,
                    'next_version_offset': version_end if version_end < len(versions) else None,
                    'versions': version_rows, 'page_byte_limit': PAGE_BYTES,
                    'details': 'Compact index, not a ranking. Follow next offsets; read_record(run_id, section=evaluations or artifacts) pages all entries. Omitted fields remain in exact read_record JSON. Meta can list_history_files/read_history_file for completed transcripts.'}
        if op == 'read_record':
            section = request.get('section')
            if section is not None:
                if section not in ('evaluations', 'artifacts'):
                    raise ValueError('Record section must be evaluations or artifacts')
                record = self.record(request['run_id'])
                offset, limit = request.get('offset', 0), request.get('limit', 20)
                if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
                    raise ValueError('Invalid record entry pagination')
                source = record.get(section, [])
                view = evaluation_view if section == 'evaluations' else self.artifact_index
                entries, end = row_page((dict(view(e), record_entry_index=i)
                                        for i, e in enumerate(source[offset:], offset)), offset, limit,
                                       PAGE_BYTES - 1024)
                return dict(run_id=request['run_id'], section=section, entries=entries,
                            offset=offset, total_entries=len(source), next_offset=end if end < len(source) else None,
                            page_byte_limit=PAGE_BYTES,
                            details='Compact views in original order. Omit section to read exact record JSON.')
            return read_text(safe_path(self.root / 'records', request['run_id'] + '.json'),
                             request.get('offset', 0), request.get('limit', 4096), deadline)
        if op == 'read_version':
            if request.get('section') not in (None, 'manifest'):
                raise ValueError('Version section must be manifest')
            path, info = self.version(request['version'])
            if request.get('section') == 'manifest':
                return read_text(path/'manifest.json', request.get('offset', 0), request.get('limit', 4096), deadline)
            if not request.get('path'):
                return manifest_view(info, request.get('offset', 0), request.get('limit', 20))
            return read_text(safe_path(path / 'source', request['path']), request.get('offset', 0), request.get('limit', 16000), deadline)
        if op == 'read_artifact':
            if request.get('section') not in (None, 'manifest'):
                raise ValueError('Artifact section must be manifest')
            info = self.artifact(request['artifact'], deadline=deadline)
            if request.get('section') == 'manifest':
                return text_page(encoded(info), request.get('offset', 0), request.get('limit', 4096))
            if not request.get('path'):
                return manifest_view(info, request.get('offset', 0), request.get('limit', 20))
            return read_text(safe_path(info['workspace'], request['path']), request.get('offset', 0), request.get('limit', 16000), deadline)
        raise ValueError('Unknown history operation')

    @staticmethod
    def artifact_index(artifact):
        info, clipped = object_view({k: artifact.get(k) for k in ('id', 'workspace', 'task_version', 'meta_version')}, 1536)
        return dict(info, details_omitted=True) if clipped else info

    @staticmethod
    def record_index(record):
        """Bounded prefixes with explicit continuation; never choose candidates for Meta."""
        base, clipped = object_view({k: record.get(k) for k in (
            'run_id', 'task_version', 'meta_version', 'status', 'error')}, 1536)
        archive = record.get('agent_source_archive')
        if archive:
            base['agent_source_archive'] = {k: archive[k] for k in ('run_id', 'path', 'base_version', 'read_only')}
        if record.get('agent_source_archive_error'):
            base['agent_source_archive_error'] = text_prefix(record['agent_source_archive_error'], 512)
        artifacts = record.get('artifacts', [])
        evaluations = record.get('evaluations', [])
        a_page, a_end = row_page((Store.artifact_index(a) for a in artifacts), 0, 20, 2048)
        e_page, e_end = row_page((evaluation_view(e) for e in evaluations), 0, 20, 6500)
        summary = str(record.get('summary', ''))
        shown = text_prefix(summary[:1000], 1536)
        return {**base, 'metadata_clipped': clipped,
                'artifacts': a_page, 'total_artifacts': len(artifacts),
                'next_artifact_offset': a_end if a_end < len(artifacts) else None,
                'evaluations': e_page, 'total_evaluations': len(evaluations),
                'next_evaluation_offset': e_end if e_end < len(evaluations) else None,
                'summary': shown, 'summary_clipped': shown != summary}
