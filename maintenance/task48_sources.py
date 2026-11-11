"""Audit published Agents and Task revisions against their separate evidence."""
from pathlib import Path

from rsi.agent_sources import verify_revision
from rsi.common import digest
from rsi.store import hashes, safe_path, verify_task_source_archive
from maintenance.task48_results import read


def audit_sources(run, release, events, inventory, source, check):
    versions, revisions = {}, {}
    for row in inventory:
        session = safe_path(run/'sessions', row['session_id'])/'agent'
        task = row['kind'] == 'task'
        mutable = task and any(events[i].get('source') == str(session/'source')
                               for i in row['execution_events'])
        archive = None
        if mutable:
            path = safe_path(run/'records', row['session_id']+'.json')
            record = read(path, {})
            source(path)
            valid, detail = None, 'Completed Task source archive is missing'
            if record.get('agent_source_archive_error'):
                valid, detail = False, record['agent_source_archive_error']
            elif record.get('agent_source_archive'):
                try:
                    if (record.get('run_id') != row['session_id']
                            or record.get('task_version') != row['configured_version']):
                        raise ValueError('Task source archive belongs to a different session or starting version')
                    archive = verify_task_source_archive(run, record)
                    valid, detail = True, None
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    valid, detail = False, str(exc)
            check('task_source_archive:'+row['session_id'], valid, detail)
            if archive:
                for name in archive['files']:
                    source(safe_path(Path(archive['path']), name))
        for index in row['execution_events']:
            event = events[index]
            version = event['version']
            if version not in versions:
                paths = []
                try:
                    path = safe_path(run/'versions', version)
                    manifest = read(path/'manifest.json', {})
                    paths.append(path/'manifest.json')
                    actual = hashes(path/'source') if (path/'source').is_dir() else {}
                    body = {k:manifest.get(k) for k in ('kind', 'files', 'dependency')}
                    valid = (bool(manifest and actual) and actual == manifest.get('files')
                             and manifest.get('dependency') == release.get('identity')
                             and version == str(manifest.get('kind', ''))[:1].upper()+'-'+digest(body)[:20]
                             and manifest.get('kind') == row['kind'])
                    paths.extend(safe_path(path/'source', name) for name in actual)
                except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                    valid, actual = False, {}
                versions[version] = valid, actual
                for path in paths:
                    source(path)
            base_valid, actual = versions[version]
            valid, detail = base_valid and actual == event['files'], None
            revision = event.get('source_revision')
            if revision is not None:
                if not task or event.get('source') != str(session/'source'):
                    valid, detail = False, 'Source revision is not bound to this Task working copy'
                else:
                    key = row['session_id'], version, revision
                    if key not in revisions:
                        paths = []
                        try:
                            verify_revision(session, revision, event['files'], version, paths.append)
                            revisions[key] = True, None
                        except FileNotFoundError as exc:
                            revisions[key] = None, str(exc)
                        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                            revisions[key] = False, str(exc)
                        for path in paths:
                            source(path)
                    valid, detail = revisions[key]
                    if not base_valid:
                        valid, detail = False, 'Published starting version failed integrity verification'
            elif mutable and base_valid and actual != event['files']:
                # Older self-modifying runs saved only their final source. Never
                # invent a missing intermediate copy or call a legal edit invalid.
                if archive and archive['files'] == event['files']:
                    valid = True
                else:
                    valid, detail = None, 'Intermediate Task source was not archived'
            check('executed_source:'+row['session_id']+':'+str(index), valid, detail)
