"""Read native result receipts and executed sessions without judging methods."""
from benchmarks.task_names import public_name
from rsi.tasks import MEDRSEE_TASKS, SIMULATOR_TASKS
import json
import math
from collections import Counter

from rsi.agent_sources import source_revision


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def events_at(path):
    if not path.is_file():
        return None
    # Corrupt or truncated evidence must not silently become zero activity.
    return [json.loads(line) for line in path.read_text().splitlines()]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def annotation_counts(run):
    """Query counters only; no label values are read."""
    import sqlite3
    for name, table in (('evidence/organizer/ledger.sqlite', 'acquired'),
                        ('evidence/problem/data/pool.sqlite', 'acquisitions')):
        path = run/name
        if path.is_file():
            db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
            try:
                queries, charged = db.execute('SELECT count(*), coalesce(sum(count),0) FROM queries').fetchone()
                acquired = db.execute('SELECT count(*) FROM '+table).fetchone()[0]
            finally:
                db.close()
            return dict(queries=queries, charged=charged, acquired=acquired)
    return None


def escape(value):
    return str(value).replace('~', '~0').replace('/', '~1')


def leaves(value, names=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*names, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, (*names, str(index)))
    elif value is None or type(value) in (int, float):
        yield '/'.join(names), value if finite(value) else None, '/' + '/'.join(map(escape, names))


def executions(records, events):
    rows = {}
    for record in records:
        key = record['run_id']
        if key in rows:
            raise ValueError('Duplicate session record: ' + key)
        rows[key] = dict(session_id=key, kind=record['kind'], phase=record.get('phase', 'research'),
                         configured_version=record.get('executed_version'), status=record['status'],
                         started=False if events is not None else None,
                         executed=False if events is not None else None, executed_versions=[],
                         executed_source_revisions=[], execution_events=[], evidence_issues=[])
    active = None
    for index, event in enumerate(events or []):
        kind = event.get('event')
        if kind == 'session_start':
            active = event['run_id']
            row = rows.setdefault(active, dict(session_id=active, kind=event['agent_kind'],
                phase=event.get('phase', 'research'), configured_version=event.get('version'),
                status='record_missing', started=False, executed=False, executed_versions=[],
                executed_source_revisions=[], execution_events=[], evidence_issues=['Session record missing']))
            row['started'] = True
            if event.get('version') != row['configured_version']:
                row['evidence_issues'].append('Session start version differs from record')
        elif kind == 'session_end':
            active = None
        elif kind == 'agent_source_executed':
            key = event.get('run_id', active)
            if key not in rows:
                continue  # Initial/candidate checks are not research sessions.
            row = rows[key]
            if event.get('version') != row['configured_version']:
                if event.get('run_id'):
                    row['evidence_issues'].append('Executed source differs from configured version')
                continue
            if not event.get('files') or not any(event.get('coverage', {}).values()) or (event.get('process') or {}).get('returncode') != 0:
                row['evidence_issues'].append('Execution event lacks source/coverage/success evidence')
                continue
            if not row['started']:
                row['evidence_issues'].append('Execution without session_start')
                continue
            revision = source_revision(event['files'])
            if event.get('source_revision') is not None and event['source_revision'] != revision:
                row['evidence_issues'].append('Source revision differs from executed source hashes')
                continue
            row['executed'] = True
            row['execution_events'].append(index)
            if event['version'] not in row['executed_versions']:
                row['executed_versions'].append(event['version'])
            if revision not in row['executed_source_revisions']:
                row['executed_source_revisions'].append(revision)
    return list(rows.values())


def session_counts(records, events=None):
    inventory = executions(records, events)
    result = {}
    # Closing Meta sessions are counted apart from research evolution.
    closing = {'meta_selection': 'final_selection', 'meta_distillation': 'chain_distillation'}
    for kind in ('task', 'meta', 'meta_selection', 'meta_distillation'):
        rows = [r for r in inventory if (r['phase'] == closing[kind] if kind in closing
                else r['kind'] == kind and r['phase'] not in closing.values())]
        # Legacy *_sessions is now the evidenced execution count, not attempts.
        result[kind+'_attempts'] = len(rows)
        result[kind+'_sessions_started'] = sum(r['started'] is True for r in rows) if events is not None else None
        result[kind+'_sessions'] = sum(r['executed'] is True for r in rows) if events is not None else None
        result[kind+'_sessions_completed'] = sum(r['status'] == 'finished' and r['executed'] is True for r in rows) if events is not None else None
        result[kind+'_session_statuses'] = dict(Counter(r['status'] for r in rows))
        result[kind+'_configured_versions'] = len({r['configured_version'] for r in rows if r['configured_version']})
        result[kind+'_published_versions'] = len({v for r in rows for v in r['executed_versions']}) if events is not None else None
        result[kind+'_versions'] = len({v for r in rows for v in r['executed_source_revisions']}) if events is not None else None
    result['execution_evidence_issues'] = sum(len(r['evidence_issues']) for r in inventory)
    return result


def scientific_completion_status(completion):
    """Interpret old retention-only failure receipts without rewriting raw evidence."""
    if (completion.get('status') == 'failed' and completion.get('retention_error')
            and not completion.get('error')
            and completion.get('research_status') in ('completed', 'budget_exhausted')
            and completion.get('terminal_status') in ('completed', 'not_applicable')):
        return 'completed'
    return completion.get('status')


def terminal(run, task, summary, completion):
    """Return metrics plus receipt validity; preserve even incomplete scores."""
    rows, reasons = [], []
    if scientific_completion_status(completion) != 'completed':
        reasons.append('Scientific completion is not completed')
    receipt = dict(split='test', coverage=None, samples=None, test_samples=None,
                   membership=None, protocol=None, trajectory=None)

    def emit(instance, values, path, prefix):
        for name, value, pointer in leaves(values):
            rows.append(dict(instance=instance, metric=name, value=value, path=path,
                             pointer=prefix+pointer))

    if public_name(task) in SIMULATOR_TASKS:
        # One episode per Task stage; the result is the best completed valid episode.
        from benchmarks.simulator.runtime import EPISODE_PROTOCOL, SELECTION_RULE, best_episode, primary_value
        receipt.update(split='episode', protocol=EPISODE_PROTOCOL)
        path = run/'summary.json'
        episode = summary.get('simulator_episode') or {}
        episodes = summary.get('simulator_episodes') or []
        saved = read(run/'simulator/result.json', {})
        actions = events_at(run/'simulator/actions.jsonl') or []
        valid = (episode.get('protocol') == EPISODE_PROTOCOL and episode.get('selection') == SELECTION_RULE
                 and episode.get('independent_test') is False and primary_value(task, episode) is not None)
        if not valid:
            reasons.append('Missing selected terminal episode or finite primary metric')
        best = best_episode(task, episodes)
        if best is None or best.get('episode_index') != episode.get('episode_index'):
            reasons.append('Reported episode is not the best completed valid episode')
        for key in ('protocol', 'terminal', 'metrics', 'usage', 'trajectory_sha256', 'independent_test',
                    'episode_index', 'selection'):
            if key not in saved or saved[key] != episode.get(key):
                reasons.append('Simulator terminal receipt differs: '+key)
        selected_actions = [a for a in actions if a.get('episode_index') == episode.get('episode_index')]
        if not selected_actions or selected_actions[-1].get('trajectory_hash') != episode.get('trajectory_sha256'):
            reasons.append('Missing or mismatched action trajectory')
        completed = [e for e in episodes if e.get('terminal') is True]
        receipt.update(coverage=valid, trajectory=episode.get('trajectory_sha256'),
                       selection=SELECTION_RULE, selected_episode=episode.get('episode_index'),
                       last_completed_episode=completed[-1].get('episode_index') if completed else None,
                       completed_episodes=len(completed),
                       valid_episodes=sum(primary_value(task, e) is not None for e in episodes),
                       episodes=[dict(episode_index=e.get('episode_index'), terminal=e.get('terminal'),
                                      primary=primary_value(task, e)) for e in episodes])
        emit(task, episode.get('metrics') or {}, path, '/simulator_episode/metrics')
    elif public_name(task) in MEDRSEE_TASKS:
        path = run/'final_test.json'
        final = read(path, {})
        receipt.update(samples=final.get('samples'), test_samples=final.get('test_samples'),
                       membership=final.get('membership'))
        if final.get('status') != 'completed' or final.get('task') != task or final.get('split') != 'T':
            reasons.append('Final receipt is not a completed Test result for this task')
        if not final.get('metric') or not finite(final.get('score')):
            reasons.append('No finite named primary metric')
        count, expected = final.get('samples'), final.get('test_samples')
        receipt['coverage'] = type(count) is int and count > 0 and count == expected
        if not receipt['coverage']:
            reasons.append('Test sample coverage is missing or incomplete')
        membership = final.get('membership') or {}
        digest = membership.get('sample_ids_sha256', '')
        if (membership.get('count') != count or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            reasons.append('Test membership identity is not recorded')
        for stage in ('inference', 'scoring'):
            if (final.get('executions', {}).get(stage) or {}).get('returncode') != 0:
                reasons.append('Missing or failed '+stage+' execution')
        ignored = {'task', 'split', 'samples', 'status', 'artifact', 'method', 'test_samples',
                   'elapsed_seconds', 'executions', 'selection', 'metric', 'membership', 'error', 'phase'}
        values = {k:v for k,v in final.items() if k not in ignored}
        if 'score' in values and final.get('metric'):
            value = values.pop('score')
            rows.append(dict(instance=task, metric=final['metric'], value=value if finite(value) else None,
                             path=path, pointer='/score'))
        emit(task, values, path, '')
    else:
        path = run/'terminal_score.json'
        final = read(path, {})
        receipt['coverage'] = final.get('coverage_complete')
        if final.get('test_score_computed') is not True or completion.get('terminal_status') != 'completed':
            reasons.append('Native Test scoring receipt is incomplete')
        if final.get('feedback_to_research') is not False:
            reasons.append('Terminal feedback isolation receipt is missing')
        if receipt['coverage'] is False:
            reasons.append('Native Test coverage is partial')
        values = final.get('metrics', {})
        if 'instances' in values:
            for instance, value in values['instances'].items():
                emit(instance, value.get('metrics', {}), path, '/metrics/instances/'+escape(instance)+'/metrics')
        else:
            for instance, value in values.items():
                if instance != 'native_macro_metrics':
                    emit(instance, value, path, '/metrics/'+escape(instance))
        if 'native_macro_metrics' in values:
            emit('gtex_v8', values['native_macro_metrics'], path, '/metrics/native_macro_metrics')
    status = 'ok' if not reasons else ('partial_coverage' if receipt['coverage'] is False else 'invalid_or_incomplete')
    for row in rows:
        row.update(split=receipt['split'], status=status if row['value'] is not None else 'missing',
                   coverage=receipt['coverage'], evidence_issues=reasons)
    receipt.update(status=status, evidence_issues=reasons, source=path)
    return rows, receipt
