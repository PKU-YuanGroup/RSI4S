"""Run paired modes in the selected order, in waves of at most sixteen task_48 workers.

Research is never restarted implicitly. A wave barrier includes frozen terminal
scoring and retention; modes share the same hardware at different times.
Task chains instead run their steps in order on one slot per chain and mode.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmarks.task_names import public_name
from rsi.tasks import MEDRSEE_TASKS, SIMULATOR_TASKS
from rsi.chain import HANDOFF, SCHEDULING_POLICY as CHAIN_POLICY, check_handoff_status
from rsi.common import ROOT, record_path, write_json
from rsi.monitoring import Monitor, append_json
from rsi.resources import GPUPool
from rsi.schedule import terminal_scoring_seconds
from maintenance.stress import cpu_slots

MODE_ORDER = ('fixed', 'coevolve')
SCHEDULING_POLICY = 'fixed_then_coevolve_v1'
CONCURRENT_POLICY = 'concurrent_paired_modes_v1'
COEVOLVE_FIRST_POLICY = 'coevolve_then_fixed_v1'


def scheduling_policy(mode_order=MODE_ORDER, concurrent_modes=False):
    order = tuple(mode_order)
    if order not in (MODE_ORDER, tuple(reversed(MODE_ORDER))):
        raise ValueError('Require each mode exactly once: fixed and coevolve')
    return CONCURRENT_POLICY if concurrent_modes else (
        SCHEDULING_POLICY if order == MODE_ORDER else COEVOLVE_FIRST_POLICY)


def policy_mode_order(policy):
    if policy == COEVOLVE_FIRST_POLICY:
        return tuple(reversed(MODE_ORDER))
    if policy in (SCHEDULING_POLICY, CONCURRENT_POLICY, CHAIN_POLICY):
        return MODE_ORDER
    raise ValueError('Unknown scheduling policy: ' + str(policy))


def task_configs(index_path=ROOT / 'configs/task_48.json'):
    index = json.loads(Path(index_path).read_text())
    tasks = (ROOT / index['task_manifest']).read_text().split()
    groups = {}
    for group in index['groups']:
        for task in group['tasks']:
            if task in groups:
                raise ValueError('Duplicate task in protocol groups: ' + task)
            groups[task] = group['config']
    if len(tasks) != 48 or len(set(tasks)) != 48 or set(tasks) != set(groups):
        raise ValueError('Require exactly the 48 manifest tasks, each in one group')
    return [(task, groups[task]) for task in tasks]


def plan(directory, pairs=None, slots=None, *, concurrent_modes=False, mode_order=MODE_ORDER, include=None):
    """Pure planning: no credentials, API calls, files, or scientific imports.

    `include` (mode/task pairs) keeps only those jobs of a sequential study, packed into full
    waves per mode in the usual task order: the continuation of a stopped study."""
    if include is not None:
        return plan_included(directory, pairs, slots, mode_order=mode_order, include=include)
    mode_order = tuple(mode_order)
    policy = scheduling_policy(mode_order, concurrent_modes)
    if concurrent_modes and mode_order != MODE_ORDER:
        raise ValueError("Mode order applies only to sequential studies")
    directory = Path(directory)
    slots = cpu_slots(16, physical_cores_per_slot=4, threads_per_core=1) if slots is None else slots
    if len(slots) != 16 or any(len(c) != 4 for c in slots):
        raise ValueError('Require sixteen four-logical-CPU slots')
    if len({c for slot in slots for c in slot}) != 64:
        raise ValueError('CPU slots must be disjoint')
    jobs = []
    pairs = list(pairs if pairs is not None else task_configs())
    waves_per_mode = (len(pairs) + 15) // 16
    order = ([(phase, mode, index, pair) for index, pair in enumerate(pairs)
              for phase, mode in enumerate(mode_order)] if concurrent_modes else
             [(phase, mode, index, pair) for phase, mode in enumerate(mode_order)
              for index, pair in enumerate(pairs)])
    for phase, mode, index, (task, source) in order:
        local = index % 16
        slot = local % 8 * 2 + local // 8
        if concurrent_modes:
            slot = index % 8 * 2 + phase
        jobs.append(planned_job(directory, mode, task, source, slot, slots,
                                index // 8 if concurrent_modes else phase * waves_per_mode + index // 16,
                                policy))
    return jobs


def planned_job(directory, mode, task, source, slot, slots, wave, policy):
    config = json.loads((ROOT / source).read_text())
    cfg = {**config, 'mode': mode, 'tasks': [task], 'gpus': [slot // 2],
           'gpu_slots_per_gpu': 2,
           'sandbox_resources': {**config.get('sandbox_resources', {}),
                                 'threads': 4, 'cpu_affinity': slots[slot]}}
    return dict(id=f'{mode}/{task}', task=task, mode=mode, wave=wave,
                scheduling_policy=policy,
                slot=slot, gpu=slot // 2,
                cpu_affinity=slots[slot], python=config['python'],
                config=str(directory / 'job-configs' / f'{mode}-{task}.json'),
                directory=str(directory / mode / task),
                source_config=source, settings=cfg, status='queued')


def plan_included(directory, pairs, slots, *, mode_order, include):
    mode_order = tuple(mode_order)
    policy = scheduling_policy(mode_order, False)
    directory = Path(directory)
    slots = cpu_slots(16, physical_cores_per_slot=4, threads_per_core=1) if slots is None else slots
    if len(slots) != 16 or len({c for slot in slots for c in slot}) != 64 or any(len(c) != 4 for c in slots):
        raise ValueError('Require sixteen disjoint four-logical-CPU slots')
    pairs = list(pairs if pairs is not None else task_configs())
    include = {tuple(job.split('/', 1)) if isinstance(job, str) else tuple(job) for job in include}
    known = {(mode, task) for mode in MODE_ORDER for task, _ in pairs}
    if not include or not include <= known:
        raise ValueError('Include distinct mode/task jobs of this study')
    jobs, wave = [], 0
    for mode in mode_order:
        selected = [(task, source) for task, source in pairs if (mode, task) in include]
        for position, (task, source) in enumerate(selected):
            local = position % 16
            jobs.append(planned_job(directory, mode, task, source, local % 8 * 2 + local // 8, slots,
                                    wave + position // 16, policy))
        wave += (len(selected) + 15) // 16
    return jobs


def plan_selected(directory, selected, slots=None, *, slot_offset=0):
    """Explicit mode/task reruns: one concurrent wave, one planned slot each.

    Offsets keep CPU sets disjoint from studies that run at the same time.
    """
    directory = Path(directory)
    slots = cpu_slots(16, physical_cores_per_slot=4, threads_per_core=1) if slots is None else slots
    if len(slots) != 16 or len({c for slot in slots for c in slot}) != 64:
        raise ValueError('Require sixteen disjoint four-logical-CPU slots')
    sources = dict(task_configs())
    selected = [tuple(job.split('/', 1)) if isinstance(job, str) else tuple(job) for job in selected]
    if (not selected or len(set(selected)) != len(selected) or
            any(len(job) != 2 or job[0] not in MODE_ORDER or job[1] not in sources for job in selected)):
        raise ValueError('Select distinct mode/task jobs from task_48')
    if type(slot_offset) is not int or slot_offset < 0 or slot_offset + len(selected) > 16:
        raise ValueError('Selected jobs must fit the sixteen planned slots')
    return [planned_job(directory, mode, task, sources[task], slot_offset + i, slots, 0, CONCURRENT_POLICY)
            for i, (mode, task) in enumerate(selected)]


def plan_chains(directory, chains, modes=('coevolve',), slots=None, *, slot_offset=0):
    """One planned slot per chain and mode; each step inherits its predecessor's run.

    A chain is dict(name, tasks, start=1, inherit=None). A later start continues a
    chain from an existing sealed handoff given as that predecessor's run directory.
    """
    directory = Path(directory)
    slots = cpu_slots(16, physical_cores_per_slot=4, threads_per_core=1) if slots is None else slots
    if len(slots) != 16 or len({c for slot in slots for c in slot}) != 64:
        raise ValueError('Require sixteen disjoint four-logical-CPU slots')
    sources = dict(task_configs())
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODE_ORDER):
        raise ValueError('Chain modes must be distinct values of fixed/coevolve')
    names = [chain['name'] for chain in chains]
    tasks = [task for chain in chains for task in chain['tasks']]
    if not chains or len(set(names)) != len(names) or len(set(tasks)) != len(tasks):
        raise ValueError('Chains need distinct names and each task appears in one chain only')
    if any(task not in sources for task in tasks):
        raise ValueError('Chain tasks must belong to task_48')
    if type(slot_offset) is not int or slot_offset < 0 or slot_offset + len(chains) * len(modes) > 16:
        raise ValueError('Chains must fit the sixteen planned slots')
    jobs = []
    for index, (mode, chain) in enumerate((mode, chain) for mode in modes for chain in chains):
        start, inherit = chain.get('start', 1), chain.get('inherit')
        if type(start) is not int or not 1 <= start <= len(chain['tasks']) or (inherit is None) != (start == 1):
            raise ValueError('A chain starts at its first task, or later from an inherited handoff')
        previous = None
        for position in range(start, len(chain['tasks']) + 1):
            task = chain['tasks'][position - 1]
            job = planned_job(directory, mode, task, sources[task], slot_offset + index, slots,
                              position, CHAIN_POLICY)
            job['chain'] = dict(name=chain['name'], position=position, length=len(chain['tasks']),
                                after=previous['id'] if previous else None)
            settings = dict(name=chain['name'], tasks=list(chain['tasks']), position=position)
            if position > 1:
                settings['inherit'] = previous['directory'] if previous else str(record_path(inherit, area='runs'))
            job['settings']['chain'] = settings
            jobs.append(job)
            previous = job
    return jobs


def block_chains(jobs):
    """A chain step requires its predecessor's finished or deadline-transferred handoff."""
    by_id = {job['id']: job for job in jobs}
    changed = True
    while changed:
        changed = False
        for job in jobs:
            after = (job.get('chain') or {}).get('after')
            if job['status'] != 'queued' or after is None:
                continue
            previous = by_id[after]
            reason = None
            if previous['status'] in ('blocked', 'cancelled'):
                reason = f"Chain predecessor {after} is {previous['status']}"
            elif previous['status'] in ('finished', 'failed'):
                path = Path(previous['directory']) / HANDOFF / 'manifest.json'
                if not path.is_file():
                    reason = f'Chain predecessor {after} ended without a sealed handoff'
                else:
                    try:
                        check_handoff_status(json.loads(path.read_text()))
                    except (OSError, ValueError) as exc:
                        reason = f'Chain predecessor {after} has no valid handoff: {exc}'
            if reason:
                job.update(status='blocked', ended_at=time.time(), result=dict(status='blocked', error=reason))
                changed = True


def eligible_wave(jobs):
    """A terminal failure counts as finished, but never as a successful score."""
    pending = [j for j in jobs if j['status'] in ('queued', 'running')]
    if not pending:
        return []
    policies = {j.get('scheduling_policy', SCHEDULING_POLICY) for j in jobs}
    if len(policies) != 1 or not policies <= {SCHEDULING_POLICY, COEVOLVE_FIRST_POLICY, CONCURRENT_POLICY,
                                              CHAIN_POLICY}:
        raise ValueError('Inconsistent or unknown scheduling policy')
    if policies == {CHAIN_POLICY}:
        # Blocked steps are resolved by block_chains before this pure selection.
        status = {j['id']: j['status'] for j in jobs}
        busy = {j['slot'] for j in pending if j['status'] == 'running'}
        return [j for j in pending if j['status'] == 'queued' and j['slot'] not in busy
                and (j['chain']['after'] is None or status[j['chain']['after']] in ('finished', 'failed'))]
    if policies == {CONCURRENT_POLICY}:
        wave = min(j['wave'] for j in pending)
        return [j for j in pending if j['wave'] == wave and j['status'] == 'queued']
    mode_order = policy_mode_order(next(iter(policies)))
    mode = next(mode for mode in mode_order if any(j['mode'] == mode for j in pending))
    if any(j['status'] == 'running' and j['mode'] != mode for j in pending):
        return []
    wave = min(j['wave'] for j in pending if j['mode'] == mode)
    return [j for j in pending if j['mode'] == mode and j['wave'] == wave and j['status'] == 'queued']


def launch(directory):
    directory = record_path(directory, area='runs')
    # Preparation must have already written the scientific plan and verified freezes.
    study = json.loads((directory / 'study.json').read_text())
    if study.get('preflight_status') != 'ready':
        raise ValueError('Preparation, tested retention and all freezes must be ready')
    jobs = json.loads((directory / 'jobs.json').read_text())
    import fcntl
    with (directory / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'supervisor.json').exists():
            raise FileExistsError('No implicit restart: inspect the original supervisor and workers')
        from maintenance.task48_inputs import check_inputs
        check_inputs(jobs, directory / 'launch-input-checks' / str(time.time_ns()))
        return supervise(directory, jobs)


def resume_maintenance(directory, receipt_path):
    """Explicit, audited handoff after closed workers finish maintenance elsewhere.

    No running science can be adopted, restarted or silently marked successful.
    The original supervisor snapshot and interruption records remain in diagnostics.
    """
    import fcntl
    import psutil
    from rsi.freeze import sha, code_hashes
    directory=record_path(directory,area='runs')
    receipt_path=record_path(receipt_path,area='runs')
    receipt=json.loads(receipt_path.read_text())
    with (directory/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path=directory/'supervisor.json'
        if sha(state_path)!=receipt['prior_supervisor_sha256']:
            raise ValueError('Supervisor changed since maintenance handoff')
        state=json.loads(state_path.read_text())
        def alive(pid,start):
            return (GPUPool.process_start(pid)==start and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE)
        if alive(state['pid'],state['process_start']):
            raise ValueError('Original supervisor is still alive')
        if receipt['code_hashes']!=code_hashes():raise ValueError('Maintenance code changed')
        if set(receipt['job_configs'])!={j['config'] for j in state['jobs']}:
            raise ValueError('Incomplete maintenance job configuration set')
        for name,digest in receipt['job_configs'].items():
            if sha(name)!=digest:raise ValueError('Job config changed after maintenance')
        workers=receipt['completed_workers']
        if set(workers)!={j['id'] for j in state['jobs'] if j['status']=='running'}:
            raise ValueError('Maintenance worker set differs from running jobs')
        for job in state['jobs']:
            if job['status']!='running':continue
            if alive(job['pid'],job['process_start']):raise ValueError('Original worker is still alive')
            run=Path(job['directory']);proof=workers[job['id']]
            if any(sha(run/name)!=digest for name,digest in proof['files'].items()):
                raise ValueError('Maintenance completion evidence changed')
            if not {'completion.json','retention.json'}<=set(proof['files']):
                raise ValueError('Incomplete maintenance handoff evidence')
            completion=json.loads((run/'completion.json').read_text())
            retention=json.loads((run/'retention.json').read_text())
            if (completion['status']!='completed' or completion['terminal_status']!='completed'
                    or retention['status']!='completed'):
                raise ValueError('Maintenance recovery has not completed')
            job.update(status='finished',ended_at=time.time(),returncode=None,result=completion,
                       maintenance_handoff=str(receipt_path))
        return supervise(directory,state['jobs'])


def supervise(directory, jobs):
    policy = jobs[0].get('scheduling_policy', SCHEDULING_POLICY) if jobs else SCHEDULING_POLICY
    state = dict(pid=os.getpid(), process_start=GPUPool.process_start(os.getpid()),
                 started_at=time.time(), status='running', concurrency=16,
                 scheduling_policy=policy, mode_order=list(policy_mode_order(policy)),
                 slots_per_gpu=2, jobs=jobs)
    monitor = Monitor(directory, directory / 'report/summary.md')
    processes, logs = {}, {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    write_json(directory / 'supervisor.json', state)
    try:
        while True:
            for job in jobs:
                process = processes.get(job['id'])
                if process is None or process.poll() is None:
                    continue
                outcome = Path(job['directory']) / 'completion.json'
                try:
                    result = json.loads(outcome.read_text())
                except (OSError, ValueError):
                    result = dict(status='failed', error='Worker exited without completion.json')
                job.update(status='finished' if process.returncode == 0 else 'failed',
                           ended_at=time.time(), returncode=process.returncode, result=result)
                logs.pop(job['id']).close()
                processes.pop(job['id'])
            if stopping:
                for job in jobs:
                    if job['status'] == 'queued':
                        job['status'] = 'cancelled'
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                state['status'] = 'interrupted'
            else:
                if policy == CHAIN_POLICY:
                    block_chains(jobs)
                for job in eligible_wave(jobs):
                    log_path = directory / 'worker-logs' / (job['id'].replace('/', '-') + '.log')
                    log_path.parent.mkdir(exist_ok=True)
                    log = log_path.open('xb', buffering=0)
                    process = subprocess.Popen(
                        [job['python'], '-m', 'maintenance.task48', 'worker',
                         '--config', job['config'], '--task', job['task'], '--out', job['directory']],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True)
                    job.update(status='running', pid=process.pid,
                               process_start=GPUPool.process_start(process.pid), started_at=time.time())
                    processes[job['id']], logs[job['id']] = process, log
            try:
                monitor.sample(jobs)
            except Exception as exc:
                append_json(directory / 'monitor-errors.jsonl', dict(time=time.time(), error=repr(exc)))
            state.update(heartbeat=time.time(), monitor_samples=monitor.samples, peaks=monitor.peaks)
            if not processes and not eligible_wave(jobs):
                state['status'] = 'interrupted' if stopping else 'finished'
            write_json(directory / 'supervisor.json', state)
            if state['status'] in ('finished', 'interrupted') and not processes:
                break
            time.sleep(5)
    except BaseException as exc:
        state.update(status='supervisor_failed', error=repr(exc), heartbeat=time.time())
        write_json(directory / 'supervisor.json', state)
        # Leave live workers' ledgers intact. Do not start another supervisor.
        raise
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
        for stream in logs.values():
            stream.close()
    from maintenance.task48_report import report
    report(directory)
    return state


def worker(config_path, task, directory):
    from run_rsi import load_config
    from rsi.harness import run_task
    config = load_config(config_path)
    # Apply the affinity to host/scorer as well as scientific sandbox descendants.
    os.sched_setaffinity(0, config['sandbox_resources']['cpu_affinity'])
    directory = record_path(directory, area='runs')
    if directory.exists():
        raise FileExistsError('Refuse existing worker directory: ' + str(directory))
    started = time.time()
    begin = time.monotonic()
    try:
        result = run_task(config, task, directory)
    except FileExistsError:
        # A competing launch must not turn another worker's directory into ours.
        raise
    except (Exception, KeyboardInterrupt) as exc:
        # Initialization may fail before the harness can write a summary.
        result = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
        directory.mkdir(parents=True, exist_ok=True)
    from rsi.common import Journal
    journal = Journal(directory/'events.jsonl')
    completion = dict(status='completed', research_status=result['status'], pid=os.getpid(),
                      started_at=started, terminal_status='not_applicable', research_error=result.get('error'))
    try:
        if config.get('chain'):
            chain = result.get('chain') or {}
            try:
                check_handoff_status({**(chain.get('handoff') or {}), 'distillation': chain.get('distillation')})
            except ValueError:
                completion['terminal_status'] = 'blocked'
                raise ValueError(result.get('error') or 'Chain stopped without a valid handoff')
        if config.get('naturebench_active'):
            completion['terminal_status'] = 'failed'
            if (result.get('final_test') or {}).get('status') != 'completed':
                raise ValueError('Frozen inference did not complete; terminal scoring unavailable')
            score_started = time.time()
            journal.add('terminal_scoring_start')
            if task == 'LIFE_01':
                from benchmarks.naturebench_active.scoring.terminal_resp import terminal_score
                terminal_score(directory, Path(config['data_root']) / task, config['naturebench_source_root'])
            else:
                from maintenance.naturebench.terminal_queue import run_job
                terminal = run_job(dict(id=task, run=str(directory), score_root=str(directory),
                    resources=dict(device='previous', cpu_cores=4, slots_per_gpu=2),
                    scoring=dict(seconds=terminal_scoring_seconds(config))),
                    config['naturebench_source_root'], sys.executable)
                if terminal['status'] != 'completed':
                    raise RuntimeError(terminal.get('error', 'Native scoring incomplete'))
            completion.update(terminal_status='completed', terminal_seconds=time.time()-score_started)
            journal.add('terminal_scoring_end', status='completed', seconds=completion['terminal_seconds'])
        elif public_name(task) in MEDRSEE_TASKS:
            completion['terminal_status'] = (result.get('final_test') or {}).get('status', 'missing')
            if completion['terminal_status'] != 'completed':
                raise ValueError('Final scoring did not complete')
        elif public_name(task) in SIMULATOR_TASKS:
            terminal=(result.get('simulator_episode') or {}).get('terminal') is True
            completion['terminal_status']='completed' if terminal else 'incomplete'
            if not terminal:
                raise ValueError('Native episode did not reach a terminal state; metrics unavailable')
        if result['status'] not in ('completed', 'budget_exhausted'):
            raise ValueError(result.get('error') or result['status'])
    except Exception as exc:
        completion.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if config.get('naturebench_active') and 'score_started' in locals():
            completion['terminal_seconds'] = time.time()-score_started
            journal.add('terminal_scoring_end', status='failed', seconds=completion['terminal_seconds'])
    completion['ended_at'] = time.time()
    write_json(directory / 'completion.json', completion)
    # Retention is separate and must refuse any unverified/incomplete final state.
    from maintenance.task48_retention import retain, record_failure
    retention_started = time.monotonic()
    journal.add('retention_start')
    try:
        retention = retain(directory)
    except Exception as exc:
        retention = record_failure(directory, exc)
    journal.add('retention_end', status=retention['status'], seconds=time.monotonic()-retention_started)
    # Separate receipt: completion.json is already hash-bound by retention.
    workflow_status = 'failed' if retention['status'] == 'failed' else completion['status']
    workflow = dict(status=workflow_status, scientific_status=completion['status'],
                    maintenance_error=retention.get('error'), started_at=started, ended_at=time.time(),
                    wall_seconds=time.monotonic()-begin, retention_status=retention['status'])
    journal.add('workflow_end', **workflow)
    write_json(directory/'workflow.json', workflow)
    return dict(completion, workflow_status=workflow_status, retention_status=retention['status'],
                maintenance_error=retention.get('error'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['launch', 'worker', 'report', 'audit', 'resume-maintenance'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--task')
    parser.add_argument('--config')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    if args.command=='resume-maintenance':
        if not args.receipt:parser.error('--receipt is required for maintenance resumption')
        resume_maintenance(args.out,args.receipt)
        return 0
    if args.command == 'worker':
        result = worker(args.config, args.task, args.out)
        return 0 if result.get('workflow_status', result['status']) == 'completed' else 1
    if args.command in ('report', 'audit'):
        from maintenance.task48_report import report
        report(record_path(args.out, area='runs'), deep_audit=args.command == 'audit')
    else:
        launch(args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
