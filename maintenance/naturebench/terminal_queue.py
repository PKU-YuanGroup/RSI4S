"""Run independent baseline/final terminal jobs with resumable durable records.

The JSON plan supplies jobs, each with id, run and score_root. Resource settings
belong in resources (device=previous/auto/cpu/integer, cpu_cores, memory_mib...);
scoring settings belong in scoring. All inputs must already be frozen. No Test
score or scheduling state is returned to a research agent.
"""
import argparse
from concurrent.futures import as_completed
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from benchmarks.naturebench_active.scoring.resources import terminal_resources
from benchmarks.naturebench_active.scoring.terminal import frozen_result, terminal_score
from rsi.common import digest, write_json
from rsi.processes import task_executor
from rsi.resources import GPUPool


def run_job(job, source_root, python):
    directory = Path(job['score_root']).resolve(); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = directory/'attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        state = dict(job=job, job_sha256=digest(job), status='checking_frozen_inference',
                     pid=os.getpid(), process_start=GPUPool.process_start(os.getpid()),
                     started_at=time.time(), attempt=str(attempt))
        def publish():
            write_json(attempt/'status.json', state)
            write_json(directory/'job_status.json', state)
        def interrupted(signum, frame):
            raise InterruptedError('Terminal job interrupted')
        prior = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, interrupted)
        publish()
        try:
            frozen=frozen_result(Path(job['run']).resolve())
            if job.get('frozen_identity') is not None and job['frozen_identity']!=frozen:
                raise ValueError('Frozen result changed since plan preparation')
            options=dict(job.get('scoring',{}))
            limit=options.get('seconds')
            if limit is not None and (type(limit) not in (int,float) or not math.isfinite(limit) or limit<=0):
                raise ValueError('scoring.seconds must be positive and finite, or null for no time limit')
            options['seconds']=limit
            state['phase_budgets']=dict(native_scoring_seconds=limit,
                resource_queue_seconds=job.get('resources',{}).get('queue_seconds',600))
            if frozen['task']=='s41592-023-02148-8':
                from benchmarks.naturebench_active.scoring.capacity import verify_profile
                if not job.get('capacity_profile'):
                    raise ValueError('Profile representative RNA CM/sequence costs before queueing; provide capacity_profile')
                state['capacity']=verify_profile(job['capacity_profile'],frozen,
                    Path(source_root)/frozen['task']/'evaluation/evaluator.py',job.get('resources',{}),options)
                state['phase_budgets']['rna_family_seconds']=options.get('rna_family_seconds',3600)
            resource_run = Path(job.get('resource_run', job['run'])).resolve()
            if resource_run != Path(job['run']).resolve():
                # A T00 reconstruction may reuse its own original task's resource
                # record, but never borrow a different task's previous GPU.
                if frozen_result(resource_run)['task'] != frozen_result(Path(job['run']))['task']:
                    raise ValueError('Resource provenance belongs to a different task')
            state['status'] = 'waiting_for_resources'; publish()
            with terminal_resources(resource_run, **job.get('resources', {})) as resources:
                state.update(status='scoring', resources=resources, scoring_started_at=time.time()); publish()
                if set(options) & {'cuda_visible_devices', 'cpu_threads', 'score_root', 'python'}:
                    raise ValueError('Device, CPU and result paths are controlled by the terminal scheduler')
                report = terminal_score(job['run'], source_root, python=python, score_root=directory,
                    cuda_visible_devices=resources['cuda_visible_devices'], cpu_threads=resources['cpu_threads'], **options)
                for key, expected in job.get('required_coverage', {}).items():
                    if report.get('coverage', {}).get(key)!=expected:
                        raise ValueError('Terminal coverage requirement not met: '+key)
                state.update(status='completed', report=str(directory/'terminal_score.json'),
                             coverage_complete=report.get('coverage_complete'),
                             finished_at=time.time(), test_score_computed=report['test_score_computed'])
        except BaseException as exc:
            state.update(status='failed', finished_at=time.time(), error=f'{type(exc).__name__}: {exc}')
        finally:
            publish()
            signal.signal(signal.SIGTERM, prior)
        return state


def run_queue(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get('execution_ready') is False:
        raise ValueError('Terminal plan requires its recorded CPU quota deployment before execution; regenerate the plan with a delegated cgroup')
    jobs = plan['jobs']
    if len({j['id'] for j in jobs}) != len(jobs) or len({str(Path(j['score_root']).resolve()) for j in jobs}) != len(jobs):
        raise ValueError('Terminal job IDs and score directories must be unique')
    python = plan.get('python', sys.executable)
    if os.path.realpath(python) != os.path.realpath(sys.executable):
        raise ValueError('Launch the queue with its configured science Python')
    state_path = plan_path.with_suffix('.state.json')
    with plan_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='running', plan_sha256=digest(plan), pid=os.getpid(),
                     process_start=GPUPool.process_start(os.getpid()), started_at=time.time(), jobs={})
        write_json(state_path, state)
        try:
            with task_executor(plan.get('max_jobs', len(jobs) or 1)) as executor:
                pending = {executor.submit(run_job, job, plan['source_root'], python): job['id'] for job in jobs}
                for future in as_completed(pending):
                    state['jobs'][pending[future]] = future.result()
                    state['updated_at'] = time.time()
                    write_json(state_path, state)
            state['status'] = 'completed' if all(j['status']=='completed' for j in state['jobs'].values()) else 'incomplete'
            if state['status']=='completed' and any(j.get('coverage_complete') is False for j in state['jobs'].values()):
                state['status']='completed_with_partial_coverage'
        except BaseException as exc:
            state.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            state['finished_at'] = time.time()
            write_json(state_path, state)
    return state


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args=parser.parse_args()
    result=run_queue(args.plan)
    print(json.dumps(result))
    sys.exit(0 if result['status']=='completed' else 1)
