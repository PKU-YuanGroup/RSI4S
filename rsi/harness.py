"""Run published Agents and editable Task copies with a cumulative task budget."""
from __future__ import annotations

import os
import math
import platform
import queue
import signal
import subprocess
import sys
import time
from pathlib import Path

from rsi.api import APIClient, Budget, BudgetExceeded
from rsi.chain import (DISTILLATION_RUN, agent_context, chain_settings, inherited_experience, lineage,
                       load_handoff, seal_handoff)
from rsi.common import ROOT, Journal, digest, git_commit, write_json, record_path
from rsi.closing import Finalizer
from benchmarks.runtime import create_evaluation
from rsi.freeze import code_hashes, task_hashes, verify_freeze
from rsi.resources import GPUPool
from rsi.providers import API_REQUEST_TIMEOUT_SECONDS
from rsi.reporting import token_report
from rsi.schedule import (research_budget, session_context, evolution_round_limit, meta_selection_seconds,
                          retire_stage_limits, final_inference_seconds)
from rsi.runtime import Session
from rsi.store import Store
from rsi.tasks import task_directory, MEDRSEE_TASKS, SIMULATOR_TASKS

AGENTS = ROOT / 'agent'


def infrastructure():
    return code_hashes()


def task_context(data_dir):
    problem = Path(data_dir) / 'problem'
    return {name: (problem / name).read_text()[:40000]
            for name in ('README.md', 'data_description.md') if (problem / name).exists()}


def release(config, task_id):
    if config.get('naturebench_active'):
        from benchmarks.naturebench_active.release import release as active_release
        return active_release(config,task_id)
    baseline = verify_freeze(Path(config['data_root']), task_id,config.get('freeze_manifest'))
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
    body = {'code': infrastructure(), 'python': sys.executable, 'python_version': platform.python_version(),
            'packages': packages, 'task_contract': task_hashes(Path(config['data_root']), task_id),
            'model': config['model'], 'baseline': baseline,
            'git_commit': git_commit()}
    return {**body, 'identity': digest(body)}


def run_task(config, task_id, directory, client_factory=None, verify=True):
    config = {**config, 'task': task_id, 'rounds': evolution_round_limit(config)}
    retire_stage_limits(config)
    # Legacy configs cannot re-enable the removed post-hoc model reviewer.
    config.pop('judge', None)
    config.setdefault('api_max_retries', 3)
    config.setdefault('api_request_timeout_seconds', API_REQUEST_TIMEOUT_SECONDS)
    config.setdefault('meta_recovery_limit', 3)
    if type(config['meta_recovery_limit']) is not int or config['meta_recovery_limit'] < 1:
        raise ValueError('meta_recovery_limit must be a positive integer')
    if config.get('finalize_seconds', 0) != 0:
        raise ValueError('finalize_seconds was removed; final inference follows research closure')
    config.pop('finalize_seconds', None)
    config.pop('max_tool_calls', None)
    queue_limit=config.get('max_gpu_queue_seconds')
    if queue_limit is not None and (type(queue_limit) not in (int,float) or not math.isfinite(queue_limit) or queue_limit<=0):
        raise ValueError('max_gpu_queue_seconds must be positive and finite, or null for an unbounded resource wait')
    if task_id in MEDRSEE_TASKS or config.get('naturebench_active'):
        config['meta_selection_seconds'] = meta_selection_seconds(config)
    if config.get('naturebench_active'):
        config['final_inference_seconds'] = final_inference_seconds(config)
    chain = chain_settings(config)

    # Offline fixture clients may use temporary scratch directories.
    directory = (record_path(directory, area='runs') if verify or client_factory is None
                 else Path(directory).resolve())
    directory.mkdir(parents=True, exist_ok=False)
    journal = Journal(directory / 'events.jsonl')
    journal.add('workflow_start', task=task_id, mode=config['mode'])
    if config['mode'] not in ('fixed', 'coevolve'):
        raise ValueError('Explicit fixed/coevolve mode required')
    from rsi.providers import validate_model
    validate_model(config['model'])
    if os.path.realpath(sys.executable) != os.path.realpath(config['python']):
        raise ValueError('Launch using configured science Python')
    frozen = release(config, task_id) if verify else {'identity': 'offline-fixture', 'code': infrastructure()}
    write_json(directory / 'release.json', frozen)
    write_json(directory / 'config.json', {k: v for k, v in config.items() if k != 'credentials'})
    if config.get('naturebench_active'):
        from benchmarks.naturebench_active.timing import phase_plan
        write_json(directory/'workflow_plan.json',dict(
            research_wall_seconds=config['budget']['wall_seconds'],gpu_queue_limit_seconds=queue_limit,
            environment_preflight_limit_seconds=180,final_selection_limit_seconds=meta_selection_seconds(config),
            frozen_inference_limit_seconds=final_inference_seconds(config),
            scientific_review='disabled',
            evaluation_phases=phase_plan(config.get('evaluation_phases')),
            native_scoring='Separate terminal job outside research; no outer wall-clock limit by default',
            native_scoring_limit_seconds=config.get('terminal_scoring_seconds'),
            estimated_complete_at=None,note='Limits are not completion estimates. The configured research budget excludes these terminal phases.'))
    budget = Budget(config['budget'], journal)
    store = Store(directory, journal, frozen['identity'])
    pool = GPUPool(config['gpus'], ROOT / 'runs/gpu_pool.json', skip_busy_mb=1024, skip_busy_util=20,
                   slots_per_gpu=config.get('gpu_slots_per_gpu', 2),
                   cpu_cores_per_slot=config.get('cpu_cores_per_slot', 4),
                   cpu_affinity=config.get('sandbox_resources', {}).get('cpu_affinity'),
                   lease_info={'task': task_id, 'mode': config['mode'], 'directory': str(directory)})
    evaluation = None
    gpu = None
    started = time.monotonic()
    status, error, verdict = 'completed', None, None
    research_started = None
    termination_requested = False
    runs = []
    infrastructure_failures = []
    consecutive_meta_failures = 0
    m0 = t0 = meta = task = session = context = None
    inherited = chain_summary = None
    prior = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        nonlocal termination_requested
        termination_requested = True
        if evaluation:
            evaluation.wall_deadline = time.monotonic()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('Terminated; preserve budget and records, do not restart this directory')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if chain:
            chain_summary = dict(name=chain['name'], position=chain['position'], length=len(chain['tasks']),
                                 tasks=chain['tasks'], inherit=chain['inherit'], lineage=None, inherited=None,
                                 distillation=None, handoff=None, handoff_error=None)
            # Verify the predecessor before waiting for resources or starting research.
            inherited = load_handoff(chain, config['mode'])
            chain_summary.update(lineage=lineage(inherited), inherited=None if inherited is None else dict(
                directory=inherited['directory'], manifest_sha256=inherited['manifest_sha256'],
                status=inherited['manifest']['status'], task_version=inherited['manifest']['task']['version'],
                meta_version=inherited['manifest']['meta']['version'],
                experience_sha256=inherited['manifest']['experience']['sha256']))
            write_json(directory / 'chain.json', chain_summary)
            journal.add('chain_step_start', **{k: chain_summary[k] for k in ('name', 'position', 'length', 'inherited')})
        journal.add('resource_queue_start')
        while gpu is None:
            queue_remaining=None if queue_limit is None else queue_limit-(time.monotonic()-started)
            if queue_remaining is not None and queue_remaining<=0:
                raise TimeoutError('GPU resource queue limit exhausted before research started')
            try:
                interval=config.get('gpu_wait_seconds',600)
                gpu = pool.get(timeout=interval if queue_remaining is None else min(interval,queue_remaining))
            except queue.Empty:
                journal.add('resource_queue_wait', waited_seconds=time.monotonic()-started,
                            research_clock_started=False)
        config['sandbox_resources'] = {**config.get('sandbox_resources', {}),
                                       'cpu_affinity': pool.assigned_cpus(gpu)}
        config['sandbox_resources'].setdefault('threads', len(pool.assigned_cpus(gpu)))
        write_json(directory/'allocated_resources.json', dict(gpu=gpu, **config['sandbox_resources']))
        journal.add('gpu_acquired', gpu=gpu, pid=os.getpid(),
                    slots_per_gpu=config.get('gpu_slots_per_gpu', 2),
                    sandbox_resources=config.get('sandbox_resources', {}))
        if config.get('naturebench_active') and verify:
            from benchmarks.naturebench_active.preflight import check_environment
            readiness=check_environment(task_id,config['naturebench_source_root'],directory/'preflight',
                python=config['python'],gpu=gpu,resources=config['sandbox_resources'])
            journal.add('environment_preflight_passed',seconds=readiness['preflight_seconds'],research_clock_started=False)
        data_dir = task_directory(config['data_root'], task_id)
        evaluation = create_evaluation(config, task_id, data_dir, directory, gpu, journal)
        evaluation.start()
        research_started = time.monotonic()
        journal.add('research_start', wall_seconds=config['budget'].get('wall_seconds'),
                    solve_seconds=config['budget']['solve_seconds'])
        client = client_factory(budget, journal) if client_factory else APIClient(config['credentials'], config['model'], budget,
                     directory / 'calls', journal, config.get('max_output_tokens', 'model_max'),
                     request_timeout=config['api_request_timeout_seconds'],
                     max_retries=config['api_max_retries'],
                     stop_after_action=config.get('api_stop_after_action', False))
        session = Session(config, store, client, evaluation, gpu, journal, directory / 'sessions',
                          research_started=research_started, inherited_experience=inherited_experience(inherited))
        if inherited:
            origin = (f"chain {chain['name']} step {chain['position'] - 1} "
                      f"({inherited['manifest']['task_id']}, handoff {inherited['manifest']['status']})")
            m0 = store.publish('meta', Path(inherited['directory']) / 'meta', author='chain_handoff',
                               rationale='Inherited Meta from ' + origin)
            t0 = store.publish('task', Path(inherited['directory']) / 'task', author='chain_handoff',
                               rationale='Inherited Task from ' + origin)
        else:
            m0 = store.publish_seed('meta', rationale='Shared minimal generic M0')
            t0 = store.publish_seed('task', rationale='Shared minimal generic T0; no task-specific strategy')
        meta, task = m0, t0
        context = {'task_id': task_id, 'task': task_context(data_dir),
                   'problem_path': str(evaluation.evidence.problem if hasattr(evaluation,'evidence') else data_dir/'problem'),
                   'research_budget': research_budget(config),
                   'interfaces': 'Agent entry.step(state, context, response) returns {state,request}. A request is one capability operation; llm returns {text}. Other operations return actual tool results. The host enforces permissions and budgets. Newly created tools and flows must use these capabilities. Scientific python runs separately; save files for persistence.'}
        if hasattr(evaluation,'evidence'):
            context['annotation'] = evaluation.annotation_status()
        if chain:
            context['chain'] = agent_context(chain, inherited)
        for identity in (m0, t0):
            info = store.version(identity)[1]
            candidate = store.create(info['kind'], identity)
            session.check(candidate, context)
        artifact_parent = None
        for index in range(config['rounds']):
            if evaluation.remaining() < 1:
                raise BudgetExceeded('Insufficient research time to start another Task')
            if index and hasattr(evaluation, 'prepare_next_task'):
                evaluation.prepare_next_task()
            if infrastructure() != frozen['code']:
                raise RuntimeError('Frozen infrastructure changed mid-experiment')
            if config['mode'] == 'fixed' and meta != m0:
                raise RuntimeError('Fixed Meta invariant violated')
            store.version(m0)
            task_run = session.run('task', task, meta, f'T{index:02d}', context,
                                    config.get('task_calls'), artifact_parent,
                                    session_info=session_context(config, 'task', index))
            runs.append(task_run)
            failure = task_run.get('failure') or {}
            if failure.get('category') == 'infrastructure':
                infrastructure_failures.append(dict(run_id=task_run['run_id'], error=task_run['error'], **failure))
                if not failure['retryable']:
                    raise RuntimeError('Nonrecoverable Task API failure: ' + task_run['error'])
            if task_run['status'] == 'budget_exhausted':
                raise BudgetExceeded(task_run['error'] or 'Shared Task budget exhausted')
            if getattr(evaluation, 'episode_done', False):
                # A terminal episode ends this Task stage only; Meta follows and the
                # next Task receives a fresh copy of the same fixed scenario.
                journal.add('simulator_episode_end', task=task_id,
                            episode_index=evaluation.episode_index,
                            protocol=evaluation.episode_protocol,
                            reason='Terminal episode; the next Task starts the same fixed scenario anew')
            data_handoff = (task_run.get('handoff') or {}).get('reason') == 'annotation_request'
            if index + 1 == config['rounds'] and not data_handoff:
                break
            if evaluation.remaining() < 1:
                raise BudgetExceeded('Insufficient research time before Meta handoff')
            allow_meta = config['mode'] == 'coevolve' and index + 2 < config['rounds']
            meta_context = {**context, 'current_task': task, 'current_meta': meta, 'initial_task': t0,
                            'mode': config['mode'], 'need_meta_successor': False,
                            'meta_evolution_allowed': allow_meta,
                            'next_task_call_allocation': config.get('task_calls')}
            if data_handoff:
                meta_context['task_handoff'] = task_run['handoff']
            meta_run = session.run('meta', meta, meta, f'M{index:02d}', meta_context,
                                  config.get('meta_calls'), allow_meta=allow_meta,
                                  session_info=session_context(config, 'meta', index,
                                                               data_request_handoff=data_handoff))
            if meta_run['status'] == 'budget_exhausted':
                raise BudgetExceeded(meta_run['error'] or 'Shared Meta budget exhausted')
            submitted = meta_run['submitted']
            failure = meta_run.get('failure') or {}
            api_outage = meta_run['status'] == 'api_error' and failure.get('retryable') is True
            # A crashed Meta Agent step publishes nothing, like an API outage:
            # the existing Task continues, bounded by the same recovery limit.
            agent_crash = meta_run['status'] == 'failed' and failure.get('category') == 'agent_process'
            recoverable = api_outage or agent_crash
            if failure.get('category') == 'infrastructure':
                infrastructure_failures.append(dict(run_id=meta_run['run_id'], error=meta_run['error'], **failure))
            consecutive_meta_failures = consecutive_meta_failures + 1 if recoverable and 'task' not in submitted else 0
            if consecutive_meta_failures >= config['meta_recovery_limit']:
                if api_outage:
                    raise RuntimeError(f'Infrastructure failure: {consecutive_meta_failures} consecutive Meta API failures')
                raise RuntimeError(f'{consecutive_meta_failures} consecutive Meta failures without a Task successor; '
                                   'last: ' + str(meta_run['error']))
            if 'task' not in submitted:
                if meta_run['status'] != 'stage_deadline' and not recoverable:
                    raise RuntimeError('Meta did not submit the required Task successor: ' + str(meta_run['error'] or meta_run['status']))
                # No valid update is invented after a deadline or API outage.
                # Continue the existing Agent and ledger; record
                # this as retention, never as a new Agent version.
                if config.get('reuse_artifacts') and task_run['artifacts']:
                    artifact_parent = task_run['artifacts'][-1]['id']
                journal.add('task_retained', version=task, after_run=meta_run['run_id'],
                            reason=('Recoverable Meta API failure' if api_outage else
                                    'Meta Agent process failed without a successor' if agent_crash else
                                    'Meta stage deadline without a valid successor'),
                            retention_kind=('api_failure' if api_outage else
                                            'agent_process_failure' if agent_crash else 'stage_deadline'),
                            consecutive_meta_failures=consecutive_meta_failures, artifact_parent=artifact_parent)
            else:
                task, artifact_parent = submitted['task']['version'], submitted['task']['artifact_parent']
                if submitted['task'].get('decision') == 'retain':
                    journal.add('task_retained', version=task, after_run=meta_run['run_id'],
                                reason=submitted['task'].get('reason', ''),
                                retention_kind='agent_decision', artifact_parent=artifact_parent)
            if allow_meta:
                parent = meta
                meta = submitted.get('meta', {}).get('version', parent)
                if meta != parent:
                    journal.add('meta_handoff', parent=parent, successor=meta, after_run=meta_run['run_id'],
                                next_meta_session=f'M{index+1:02d}')
                else:
                    journal.add('meta_retained', version=meta, after_run=meta_run['run_id'],
                                reason=meta_run.get('summary'),
                                retention_kind=('agent_decision' if meta_run['status'] == 'finished'
                                                else meta_run['status']),
                                next_meta_session=f'M{index+1:02d}')
    except BudgetExceeded as exc:
        status, error = 'budget_exhausted', str(exc)
    except (Exception, KeyboardInterrupt) as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        if isinstance(exc, KeyboardInterrupt):
            termination_requested = True
    finally:
        closing = Finalizer(directory, journal)
        research_status = status
        released = False
        summary = dict(status=status, error=error, task=task_id, mode=config['mode'],
                       research_status=research_status, budget=budget.snapshot(),
                       completed_task_runs=len(runs), infrastructure_failures=infrastructure_failures)
        try:
            research_wall_seconds = time.monotonic() - research_started if research_started is not None else None
            effective_solve_seconds = closing.call('solve_time', lambda: evaluation.state.get_effective_elapsed()) if evaluation else None
            # The official pause total also carries returned infrastructure time; report them apart.
            infrastructure_research_seconds = getattr(evaluation, 'infrastructure_excluded_seconds', 0.0) if evaluation else None
            evaluation_paused_seconds = closing.call(
                'paused_time', lambda: evaluation.state.total_paused - infrastructure_research_seconds) if evaluation else None
            official_result = closing.call('evaluation_results', evaluation.results) if evaluation else None
            research_api_budget = budget.snapshot()
            annotation_status = (closing.call('annotation_status', evaluation.annotation_status)
                                 if evaluation and hasattr(evaluation, 'annotation_status') else None)
            closing.event('exploration_end', status=status, research_wall_seconds=research_wall_seconds,
                        effective_solve_seconds=effective_solve_seconds, budget=research_api_budget,
                        infrastructure_excluded_seconds=infrastructure_research_seconds,
                        annotation_status=annotation_status)
            # Close scientific scoring before the read-only Meta review. Its clock
            # is separate; no research deadline, ledger or artifact is reset.
            if session:
                session.close_research()
            closed = evaluation is None or closing.call('evaluation_close', evaluation.close, False) is not False
            selection_run = None
            selection_wall_seconds = 0
            selection_budget_before = budget.snapshot()
            if (session and context and runs and hasattr(evaluation, 'evidence')
                    and closed and not termination_requested and not closing.interrupted):
                selection_started = time.monotonic()
                closing.event('meta_final_review_start', outside_research_budget=True,
                            seconds=config['meta_selection_seconds'], read_only=True)
                try:
                    closing_context = {**context, 'current_task': task, 'current_meta': meta,
                        'initial_task': t0, 'mode': config['mode'], 'final_selection_only': True,
                        'meta_evolution_allowed': False, 'need_meta_successor': False}
                    selection_run = session.run('meta', meta, meta, 'M-final', closing_context,
                                                config.get('meta_calls'), allow_meta=False)
                except (Exception, KeyboardInterrupt) as exc:
                    closing.event('meta_final_review_failed', error=f'{type(exc).__name__}: {exc}')
                    if isinstance(exc, KeyboardInterrupt):
                        termination_requested = True
                        status, error = 'failed', 'Interrupted during final Meta selection'
                selection_wall_seconds = time.monotonic() - selection_started
                closing.event('meta_final_review_end', wall_seconds=selection_wall_seconds,
                            status=selection_run['status'] if selection_run else 'failed',
                            selection=getattr(evaluation, 'meta_selection', None))
            selection_budget_after = budget.snapshot()
            if (chain and session and context and runs and not termination_requested
                    and not closing.interrupted):
                # Distill before frozen Test inference, so Test never informs the next task.
                distill_run, distill_started, distill_before = None, time.monotonic(), budget.snapshot()
                distill_error = None
                closing.event('chain_distillation_start', outside_research_budget=True,
                              seconds=chain['distillation_seconds'])
                try:
                    distill_context = {**{k: v for k, v in context.items() if k != 'annotation'},
                        'current_task': task, 'current_meta': meta, 'initial_task': t0, 'mode': config['mode'],
                        'chain_distillation': True, 'meta_evolution_allowed': config['mode'] == 'coevolve',
                        'need_meta_successor': False, 'final_selection': getattr(evaluation, 'meta_selection', None)}
                    distill_run = session.run('meta', meta, meta, DISTILLATION_RUN, distill_context,
                                              config.get('meta_calls'), allow_meta=config['mode'] == 'coevolve')
                except (Exception, KeyboardInterrupt) as exc:
                    distill_error = f'{type(exc).__name__}: {exc}'
                    closing.event('chain_distillation_failed', error=distill_error)
                    if isinstance(exc, KeyboardInterrupt):
                        termination_requested = True
                        status, error = 'failed', 'Interrupted during chain distillation'
                distill_after = budget.snapshot()
                submitted = (distill_run or {}).get('submitted') or {}
                distilled = bool(distill_run and distill_run['status'] == 'finished'
                                 and 'task' in submitted and 'experience' in submitted)
                deadline_handoff = bool(distill_run and distill_run['status'] == 'stage_deadline'
                    and distill_run.get('deadline_handoff') is True and 'task' in submitted and 'experience' in submitted)
                distillation = dict(run_id=DISTILLATION_RUN, distilled=distilled, deadline_handoff=deadline_handoff,
                    status=distill_run['status'] if distill_run else 'failed',
                    error=distill_run['error'] if distill_run else distill_error or 'Distillation session did not start',
                    summary=distill_run['summary'] if distill_run else None,
                    wall_seconds=time.monotonic() - distill_started,
                    api_usage={k: distill_after[k] - distill_before[k] for k in (
                        'calls', 'charged_tokens', 'reported_prompt_tokens', 'reported_completion_tokens',
                        'unknown_usage_calls')})
                closing.event('chain_distillation_end', **{k: distillation[k] for k in (
                    'status', 'distilled', 'deadline_handoff', 'wall_seconds', 'api_usage')})
                chain_summary['distillation'] = distillation
                if (distilled or deadline_handoff) and not termination_requested:
                    try:
                        chain_summary['handoff'] = seal_handoff(
                            directory, store, chain, mode=config['mode'], task_id=task_id,
                            release=frozen['identity'], task_version=submitted['task']['version'],
                            meta_version=submitted.get('meta', {}).get('version', meta),
                            experience=submitted['experience']['text'],
                            distillation={k: v for k, v in distillation.items() if k != 'api_usage'},
                            inherited=inherited)
                        closing.event('chain_handoff_sealed', **chain_summary['handoff'])
                    except Exception as exc:
                        chain_summary['handoff_error'] = f'{type(exc).__name__}: {exc}'
                elif not distilled:
                    chain_summary['handoff_error'] = ('Chain distillation did not complete: '
                        + distillation['status'] + (': ' + distillation['error'] if distillation['error'] else ''))
            if chain_summary is not None:
                if chain_summary['handoff'] is None:
                    chain_summary['handoff_error'] = (chain_summary['handoff_error'] or error
                                                      or 'Chain ended without a valid handoff')
                    status = 'failed'
                    if not error:
                        error = chain_summary['handoff_error']
                    elif chain_summary['handoff_error'] not in error:
                        error += '; ' + chain_summary['handoff_error']
                    closing.event('chain_handoff_failed', error=chain_summary['handoff_error'])
                closing.call('chain_record', lambda: write_json(directory / 'chain.json', chain_summary))
            final_test = None
            if evaluation:
                if (hasattr(evaluation,'finalize') and runs and closed and not termination_requested
                        and not closing.interrupted and (chain_summary is None or chain_summary['handoff'] is not None)):
                    closing.event('final_test_start', feedback_to_agents=False)
                    final_test = closing.call('final_test',
                        lambda: evaluation.finalize(runs,config['python'],final_inference_seconds(config)),
                        dict(status='failed', error='Final test raised an exception; see cleanup_errors'))
                    closing.event('final_test_end', **final_test)
            released = closing.call('release_resources', pool.release_all, False) is not False
            verdict = {'status': 'disabled', 'is_valid': None,
                       'reason': 'Independent scientific validity review is not performed.'}
            summary = {'status': status, 'error': error, 'mode': config['mode'], 'task': task_id,
                       'infrastructure_failures': infrastructure_failures, 'research_status': research_status,
                       'm0': m0, 't0': t0, 'final_meta': meta, 'final_task': task, 'completed_task_runs': len(runs),
                       'requested_task_runs': config['rounds'], 'budget': budget.snapshot(), 'judge': verdict,
                       'valid_score_observed': any(e['valid_primary_score'] for r in runs for e in r['evaluations']),
                       'official_result': official_result, 'gpu': gpu,
                       'final_test': final_test,
                       'research_api_budget': research_api_budget,
                       'annotation_status': annotation_status,
                       'meta_selection_wall_seconds': selection_wall_seconds,
                       'meta_selection': {
                           'status': selection_run['status'] if selection_run else 'not_completed',
                           'error': selection_run['error'] if selection_run else None,
                           'decision': getattr(evaluation, 'meta_selection', None),
                           'api_usage': {k: selection_budget_after[k] - selection_budget_before[k]
                                         for k in ('calls', 'charged_tokens', 'reported_prompt_tokens',
                                                   'reported_completion_tokens', 'unknown_usage_calls')}},
                       'wall_seconds': time.monotonic() - started,
                       'research_wall_seconds': research_wall_seconds, 'judge_wall_seconds': 0,
                       'result_policy': 'native_metrics_without_judge_v1',
                       'effective_solve_seconds': effective_solve_seconds,
                       'evaluation_paused_seconds': evaluation_paused_seconds,
                       'infrastructure_excluded_seconds': infrastructure_research_seconds,
                       'infrastructure_time': evaluation.infrastructure_time()
                                              if hasattr(evaluation, 'infrastructure_time') else None,
                       'release': frozen['identity'], 'restart_policy': 'Refuse existing directory. A retry must continue the same live process and cumulative ledger; no implicit reset.'}
            if config.get('naturebench_active'):summary['benchmark_protocol']='naturebench-active'
            if chain_summary is not None:
                summary['chain'] = chain_summary
            if evaluation and hasattr(evaluation, 'simulator_status'):
                # Report the best completed valid episode; all episodes stay in the ledger.
                episode = closing.call('simulator_episode', evaluation.reported_episode)
                if episode is not None:
                    summary['simulator_episode'] = {k: episode[k] for k in (
                        'task', 'terminal', 'metrics', 'usage', 'protocol', 'independent_test',
                        'episode_index', 'trajectory_sha256', 'selection', 'selected')}
                    summary['simulator_episodes'] = evaluation.episodes()
                    summary['simulator_completed_episodes'] = episode['completed_episodes']
                    summary['simulator_valid_episodes'] = episode['valid_episodes']

        except (Exception, KeyboardInterrupt) as exc:
            closing.failed('finalization', exc)
        finally:
            try:
                if not released:
                    closing.call('release_resources', pool.release_all)
                closing.call('token_report', lambda: token_report(directory))
                closing.event('experiment_end', **closing.update(summary))
                closing.save(summary)
            finally:
                signal.signal(signal.SIGTERM, prior)
    return summary
