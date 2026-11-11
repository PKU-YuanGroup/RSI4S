"""Evidence audit for a new study; never restarts, repairs or cleans a run."""
from benchmarks.task_names import public_name
from rsi.tasks import MEDRSEE_TASKS, SIMULATOR_TASKS
from collections import Counter
from pathlib import Path
import json

from rsi.chain import EXPERIENCE, HANDOFF, check_handoff_status, sha256_text
from rsi.common import ROOT
from rsi.freeze import sha
from rsi.store import hashes, safe_path
from maintenance.task48_results import read, annotation_counts, scientific_completion_status
from maintenance.task48_sources import audit_sources
from maintenance.task48 import CONCURRENT_POLICY, SCHEDULING_POLICY, COEVOLVE_FIRST_POLICY


def audit_run(job, study, summary, completion, events, inventory, receipt, source, *, deep=False):
    run = Path(job['directory'])
    checks = []
    def check(name, passed, detail=None):
        checks.append(dict(check=name, status='missing' if passed is None else 'passed' if passed else 'failed', detail=detail))
    if not run.exists():
        return dict(run_id=job['id'], status='not_started', checks=[], scientific_validity='not_assessed')
    config = read(run/'config.json', {})
    release = read(run/'release.json', {})
    allocated = read(run/'allocated_resources.json', {})
    for name in ('config.json', 'release.json', 'allocated_resources.json', 'workflow.json',
                 'simulator/result.json', 'simulator/actions.jsonl', 'final/freeze.json', 'final_selection.json'):
        if (run/name).is_file():
            source(run/name)
    planned_path = Path(job['config']) if job.get('config') else None
    planned = read(planned_path, {}) if planned_path else {}
    if planned_path and planned_path.is_file():
        source(planned_path)
    expected_config = study.get('job_config_sha256', {}).get(job['id'])
    check('prepared_config_identity', sha(planned_path) == expected_config if expected_config and planned else None)
    # load_config adds defaults and removes credentials; compare supplied policy values.
    ignored = {'credentials', 'tasks', 'judge', 'stage_seconds', 't00_seconds', 'meta_seconds', 'task_seconds'}
    expected_settings = dict(planned)
    if expected_settings.get('freeze_manifest'):
        # load_config resolves repository-relative freeze paths before saving config.json.
        expected_settings['freeze_manifest'] = str((ROOT/expected_settings['freeze_manifest']).resolve())
    differences = [key for key,value in expected_settings.items() if key not in ignored and config.get(key) != value]
    check('executed_config_matches_plan', not differences if config and planned else None, differences)
    expected_code = study.get('code_revision', {}).get('code_hashes')
    check('released_infrastructure', release.get('code') == expected_code if expected_code and release else None)
    check('summary_release_identity', summary.get('release') == release.get('identity') if summary.get('release') and release else None)
    check('allocated_resources', allocated.get('gpu') == job.get('gpu') and
          allocated.get('cpu_affinity') == job.get('cpu_affinity') if allocated else None)
    check('session_execution_evidence', not any(row['evidence_issues'] for row in inventory) if events is not None else None)
    audit_sources(run, release, events, inventory, source, check)
    if job['mode'] == 'fixed':
        check('fixed_meta_identity', all(v == summary.get('m0') for row in inventory
              if row['kind'] == 'meta' for v in row['executed_versions']) if summary.get('m0') else None)
    check('native_result_receipt', receipt['status'] == 'ok', receipt['evidence_issues'])
    kinds = [event.get('event') for event in events or []]
    if public_name(job['task']) not in SIMULATOR_TASKS:
        check('research_closed_before_test', kinds.index('exploration_end') < kinds.index('final_test_start')
              if 'exploration_end' in kinds and 'final_test_start' in kinds else None)
        after_test = (events or [])[kinds.index('final_test_start')+1:] if 'final_test_start' in kinds else []
        late = [event.get('run_id') for event in after_test
                if event.get('event') in ('session_start', 'api_start', 'annotation_query')]
        check('no_agent_calls_after_test', not late if 'final_test_start' in kinds else None, late)
        annotation = summary.get('annotation_status')
        used = (annotation or {}).get('annotations_used')
        budget = (annotation or {}).get('annotation_budget')
        check('annotation_budget', 0 <= used <= budget if type(used) is int and type(budget) is int else None,
              annotation)
        ledger = annotation_counts(run)
        check('annotation_ledger_matches_closure', ledger['acquired'] == ledger['charged'] == used
              if ledger and used is not None else None, ledger)
        frozen = read(run/'final/freeze.json', {})
        final = read(run/'final_test.json', {})
        selected = frozen.get('artifact', final.get('artifact'))
        check('selected_artifact_identity', selected == final.get('artifact') if selected else None)
        if public_name(job['task']) in MEDRSEE_TASKS:
            if deep:
                try:
                    verify_test_membership(run, final, source)
                except (OSError, ValueError, KeyError) as exc:
                    check('test_input_membership', False, str(exc))
                else:
                    check('test_input_membership', True)
            else:
                check('test_input_membership', None, 'Full input-ID check is performed by task48 audit')
    calls = read(run/'token_report.json', {})
    for row in calls.get('calls', []):
        request = run/'calls'/f"{row['call']:04d}.request.json"
        response = run/'calls'/f"{row['call']:04d}.response.json"
        valid = request.is_file() and (not row.get('usage_known') or response.is_file())
        check('raw_api_interaction:'+str(row['call']), valid)
        for path in (request, response):
            if path.is_file():
                source(path)
    starts = {e['call'] for e in events or [] if e.get('event') == 'api_start'}
    recorded = {r['call'] for r in calls.get('calls', []) if 'start_time' in r}
    check('api_call_accounting', starts == recorded if calls and events is not None else None)
    if calls:
        rows = calls.get('calls', [])
        totals = calls.get('total', {})
        check('api_usage_totals', totals.get('calls') == len(rows) and
              totals.get('unknown_usage_calls') == sum(not r.get('usage_known') for r in rows) and
              totals.get('prompt_tokens') == sum(r.get('prompt_tokens') or 0 for r in rows) and
              totals.get('completion_tokens') == sum(r.get('completion_tokens') or 0 for r in rows))
        check('api_billing', None, calls.get('api_cost_reason', 'No verified billing receipt; unknown, not zero'))
    retention = read(run/'retention.json', {})
    source(run/'retention.json')
    if scientific_completion_status(completion) != 'completed':
        check('failed_run_preserved', retention.get('status') == 'deferred' and not (run/'retention/plan.json').exists()
              if retention else None)
    elif retention.get('status') != 'completed':
        check('retention_complete', False if retention else None)
    else:
        try:
            verify_retention(run, source, deep=deep)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            check('retention_integrity', False, str(exc))
        else:
            check('retention_integrity', True, 'Full archive contents checked' if deep else 'Receipts checked; full archive check pending')
        if not deep:
            check('archive_content_integrity', None, 'Run task48 audit for full decompression and hash verification')
    counts = Counter(c['status'] for c in checks)
    return dict(run_id=job['id'], status='failed' if counts['failed'] else 'incomplete' if counts['missing'] else 'passed',
                checks=checks, counts=dict(counts), scientific_validity='not_assessed',
                note='Checks recorded execution and access boundaries; not a scientific validity verdict or proof against in-memory fitting')


def verify_retention(run, source, *, deep):
    """Read-only journal parsing: never call the truncating recovery reader."""
    from maintenance.task48_retention import SHOWCASE
    from maintenance.task48_archive import Archives
    meta = read(run/'retention/plan.json', {})
    if meta.get('schema_version') != 2:
        raise ValueError('Missing v2 retention plan')
    from maintenance.task48_retention import verify_links
    verify_links(run, meta, source, deep=deep)
    for name, expected in meta['closure_sha256'].items():
        if sha(run/name) != expected:
            raise ValueError('Terminal closure changed: '+name)
        source(run/name)
    for key in ('plan', 'source_index'):
        if sha(run/meta[key]) != meta[key+'_sha256']:
            raise ValueError('Retention hash mismatch: '+key)
        source(run/meta[key])
    plan = [json.loads(line) for line in (run/meta['plan']).read_text().splitlines()]
    if len(plan) != meta['planned_count']:
        raise ValueError('Retention plan length differs')
    committed, pending = 0, None
    for line in (run/meta['journal']).read_text().splitlines():
        row = json.loads(line)
        if row['start'] != committed or type(row['count']) is not int or row['count'] <= 0:
            raise ValueError('Invalid deletion journal sequence')
        if row['event'] == 'intent' and pending is None:
            pending = row['count']
        elif row['event'] == 'commit' and row['count'] == pending:
            if row['bytes'] != sum(r['bytes'] for r in plan[committed:committed+pending]):
                raise ValueError('Deletion bytes differ from plan')
            committed += pending; pending = None
        else:
            raise ValueError('Invalid deletion journal transition')
    if pending is not None or committed != len(plan):
        raise ValueError('Deletion journal incomplete')
    source(run/meta['journal']); source(run/'retention/plan.json')
    for row in plan:
        path = safe_path(run, row['path'])
        if path.exists():
            # Explicit restored copies are valid, but still must match the old identity.
            if sha(path) != row['sha256']:
                raise ValueError('Restored payload changed: '+row['path'])
        if row.get('model') and meta['task'] in SHOWCASE:
            raise ValueError('Showcase model was classified for deletion')
    for row in meta.get('retained_checkpoints', []):
        path = safe_path(run, row['path'])
        if not path.is_file() or (deep and sha(path) != row['sha256']):
            raise ValueError('Retained checkpoint missing or changed: '+row['path'])
    if deep:
        index = read(run/meta['source_index'])
        Archives(run, progress=False).verify_index(index)
        for value in index.values():
            source(run/value['archive'])
        deleted = {row['path']:row['sha256'] for row in plan}
        for record in (run/'records').glob('*.json'):
            for artifact in read(record).get('artifacts', []):
                workspace = Path(artifact['workspace']).resolve()
                if not workspace.is_relative_to(run):
                    raise ValueError('Artifact workspace escaped its run')
                for name, expected in artifact['files'].items():
                    path = safe_path(workspace, name)
                    relative = path.relative_to(run).as_posix()
                    observed = sha(path) if path.is_file() else deleted.get(relative, index.get(relative, {}).get('sha256'))
                    if observed != expected:
                        raise ValueError('Recorded artifact changed or lost: '+relative)


def verify_test_membership(run, final, source):
    """Compare scored IDs to exported input metadata without opening label arrays."""
    import hashlib
    path = run/'final/problem/data/batches.json'
    batches = read(path)
    if not batches:
        raise ValueError('Final input batch manifest missing or empty')
    source(path)
    identity = hashlib.sha256(); count = 0
    for batch in batches:
        directory = Path(batch['path']).resolve()
        if not directory.is_relative_to(run/'final/problem/data'):
            raise ValueError('Final input metadata escaped its run')
        path = directory/'samples.json'
        samples = read(path)
        if not isinstance(samples, list) or len(samples) != batch['count']:
            raise ValueError('Final input batch count differs')
        source(path)
        for sample in samples:
            if sample.get('split') != 'T' or sample.get('labeled') is not False:
                raise ValueError('Final inputs contain non-Test or labeled metadata')
            identity.update((json.dumps(sample['sample_id'], ensure_ascii=True)+'\n').encode())
            count += 1
    membership = final.get('membership', {})
    if count != membership.get('count') or identity.hexdigest() != membership.get('sample_ids_sha256'):
        raise ValueError('Exported Test membership differs from the scored membership')


def audit_pairs(jobs):
    groups = {}
    for job in jobs:
        groups.setdefault(job['task'], {})[job['mode']] = job
    results = []
    for task, pair in groups.items():
        status, detail = 'missing', 'Both modes and their planned configs are required'
        differences, resource_issues = [], []
        policies = {job.get('scheduling_policy', SCHEDULING_POLICY) for job in pair.values()}
        if set(pair) == {'fixed', 'coevolve'}:
            configs = [read(Path(pair[m]['config']), {}) if pair[m].get('config') else {} for m in ('fixed', 'coevolve')]
            if all(configs):
                values = [{k:v for k,v in cfg.items() if k not in ('mode', 'credentials')} for cfg in configs]
                detail = 'Compare full paired configuration including data, seeds, splits, budget and resources'
                if policies == {CONCURRENT_POLICY}:
                    masks = []
                    for mode, cfg in zip(('fixed', 'coevolve'), values):
                        resources = dict(cfg.get('sandbox_resources', {}))
                        mask = resources.pop('cpu_affinity', None)
                        valid = (isinstance(mask, list) and len(mask) == 4 and
                                 all(type(cpu) is int and cpu >= 0 for cpu in mask) and len(set(mask)) == 4)
                        if not valid or mask != pair[mode].get('cpu_affinity'):
                            resource_issues.append(mode + ': CPU affinity must match its four-core plan')
                        if resources.get('threads') != 4 or cfg.get('gpus') != [pair[mode].get('gpu')]:
                            resource_issues.append(mode + ': GPU or thread allocation differs from concurrent policy')
                        masks.append(set(mask) if valid else set())
                        cfg['sandbox_resources'] = resources
                    if masks[0] & masks[1]:
                        resource_issues.append('Concurrent paired CPU affinities overlap')
                    detail += '; concurrent modes require disjoint planned four-core CPU sets with equal quotas'
                elif policies not in ({SCHEDULING_POLICY}, {COEVOLVE_FIRST_POLICY}):
                    resource_issues.append('Inconsistent or unknown scheduling policy')
                differences = sorted(key for key in values[0].keys() | values[1].keys()
                                     if key not in values[0] or key not in values[1] or values[0][key] != values[1][key])
                status = 'passed' if not differences and not resource_issues else 'failed'
        initial, contracts = None, None
        if set(pair) == {'fixed', 'coevolve'}:
            summaries = [read(Path(pair[m]['directory'])/'summary.json', {}) for m in ('fixed', 'coevolve')]
            releases = [read(Path(pair[m]['directory'])/'release.json', {}) for m in ('fixed', 'coevolve')]
            if all(s.get('m0') and s.get('t0') for s in summaries):
                initial = all(summaries[0][k] == summaries[1][k] for k in ('m0','t0'))
            if all(r.get('task_contract') for r in releases):
                contracts = releases[0]['task_contract'] == releases[1]['task_contract']
        if initial is False or contracts is False:
            status = 'failed'
        elif status == 'passed' and (initial is None or contracts is None):
            status = 'missing'
        results.append(dict(task_id=task, status=status, detail=detail,
                            same_initial_agents=initial, same_data_contract=contracts,
                            scheduling_policies=sorted(policies), configuration_differences=differences,
                            resource_issues=resource_issues))
    return results


def audit_chains(jobs):
    """Each chain step must start from the exact sealed handoff of its predecessor."""
    by_id = {job['id']: job for job in jobs}
    results = []
    for job in jobs:
        chain = job.get('chain')
        if not chain:
            continue
        run = Path(job['directory'])
        record = read(run/'chain.json', {})
        handoff = record.get('handoff') or {}
        row = dict(run_id=job['id'], task_id=job['task'], mode=job['mode'], chain=chain['name'],
                   position=chain['position'], after=chain['after'], job_status=job['status'],
                   inherited_manifest_sha256=(record.get('inherited') or {}).get('manifest_sha256'),
                   distillation_status=(record.get('distillation') or {}).get('status'),
                   handoff_status=handoff.get('status'), handoff_error=record.get('handoff_error'),
                   experience_characters=handoff.get('experience_characters'), issues=[])
        if not record:
            row['status'] = 'not_started' if job['status'] in ('queued', 'blocked', 'cancelled') else 'missing'
            results.append(row)
            continue
        if row['handoff_error']:
            row['issues'].append('Chain handoff failed: ' + row['handoff_error'])
        summary = read(run/'summary.json', {})
        if chain['position'] > 1:
            source = Path(by_id[chain['after']]['directory']) if chain['after'] else Path(record.get('inherit') or '')
            path = source/HANDOFF/'manifest.json'
            if not path.is_file():
                row['issues'].append('Predecessor handoff is missing')
            else:
                if row['inherited_manifest_sha256'] != sha(path):
                    row['issues'].append('Inherited handoff differs from the predecessor seal')
                predecessor = read(path)
                try:
                    check_handoff_status(predecessor)
                except ValueError as exc:
                    row['issues'].append('Predecessor handoff: ' + str(exc))
                for kind, key in (('task', 't0'), ('meta', 'm0')):
                    version = summary.get(key)
                    files = read(run/'versions'/version/'manifest.json', {}).get('files') if version else None
                    if files is not None and files != predecessor[kind]['files']:
                        row['issues'].append(f'Initial {kind} Agent differs from the inherited handoff')
        elif record.get('inherited') is not None:
            row['issues'].append('The first chain step must start from the initial Agents')
        sealed = run/HANDOFF
        if (sealed/'manifest.json').is_file():
            manifest = read(sealed/'manifest.json')
            try:
                check_handoff_status(manifest)
            except ValueError as exc:
                row['issues'].append(str(exc))
            if handoff.get('manifest_sha256') != sha(sealed/'manifest.json'):
                row['issues'].append('Recorded handoff differs from the sealed manifest')
            for kind in ('task', 'meta'):
                if hashes(sealed/kind) != manifest[kind]['files']:
                    row['issues'].append(f'Sealed {kind} Agent changed')
            if sha256_text((sealed/EXPERIENCE).read_text()) != manifest['experience']['sha256']:
                row['issues'].append('Sealed research experience changed')
        elif job['status'] in ('finished', 'failed'):
            row['issues'].append('Completed chain step has no valid handoff')
        row['status'] = ('failed' if row['issues'] else
                         'in_progress' if not (sealed/'manifest.json').is_file() else 'passed')
        results.append(row)
    return results
