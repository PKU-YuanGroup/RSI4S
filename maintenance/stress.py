"""Durable paired GPU runs with independent resource/API probes.

Launch: python -m maintenance.stress launch --config configs/parallel16.json --out NEW_DIRECTORY
The supervisor never retries or overwrites a research run. Each run retains its own four-hour clock.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from rsi.api import APIClient, Budget
from rsi.common import ROOT, Journal, parse_json, write_json, record_path
from rsi.freeze import verify_freeze
from rsi.harness import run_task
from rsi.providers import API_REQUEST_TIMEOUT_SECONDS
from rsi.monitoring import Monitor, append_json


def cpu_slots(concurrency=16, physical_cores_per_slot=2, threads_per_core=None):
    """Disjoint physical cores per slot, split evenly between CPU sockets.

    Legacy stress runs include all SMT siblings. task_48 explicitly requests
    four distinct physical cores with one hardware thread per core.
    """
    if type(physical_cores_per_slot) is not int or physical_cores_per_slot < 1:
        raise ValueError('Require a positive physical core count')
    if threads_per_core is not None and (type(threads_per_core) is not int or threads_per_core < 1):
        raise ValueError('Require a positive hardware thread count')
    packages = defaultdict(lambda: defaultdict(list))
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        package = int((root / 'physical_package_id').read_text())
        core = int((root / 'core_id').read_text())
        packages[package][core].append(cpu)
    if type(concurrency) is not int or concurrency < 2 or concurrency % 2:
        raise ValueError('Paired concurrency must be a positive even integer')
    per_mode = concurrency // 2
    required = physical_cores_per_slot * per_mode
    if len(packages) < 2 or any(len(cores) < required for cores in packages.values()):
        raise RuntimeError(f'{concurrency}-slot profile requires two CPU sockets with at least {required} allowed physical cores each')
    result = []
    for package in sorted(packages)[:2]:
        cores = [cpus for _, cpus in sorted(packages[package].items())]
        for i in range(per_mode):
            selected = cores[physical_cores_per_slot*i:physical_cores_per_slot*(i+1)]
            if threads_per_core is not None and any(len(c) < threads_per_core for c in selected):
                raise RuntimeError('Insufficient allowed hardware threads on a physical core')
            result.append(sorted(cpu for siblings in selected for cpu in siblings[:threads_per_core]))
    return result


def make_jobs(config, directory, tasks):
    gpus = config.get('gpus', list(range(8)))
    shares = config.get('gpu_slots_per_gpu', 2)
    if len(gpus) < 2 or len(gpus) % 2:
        raise ValueError('Paired launcher requires an even number of GPUs')
    concurrency = len(gpus) * shares
    slots = cpu_slots(concurrency)
    per_mode, gpus_per_mode = concurrency // 2, len(gpus) // 2
    jobs = []
    for index, task in enumerate(tasks):
        for mode, offset in [('fixed', 0), ('coevolve', per_mode)]:
            slot = (index % gpus_per_mode) * shares + (index // gpus_per_mode) % shares + offset
            gpu = gpus[slot // shares]
            limits = {'threads': 2, 'max_processes': 128,
                      **config.get('sandbox_resources', {}), 'cpu_affinity': slots[slot]}
            job_config = {**config, 'mode': mode, 'gpus': [gpu], 'gpu_slots_per_gpu': shares,
                          'sandbox_resources': limits}
            config_path = directory / 'job-configs' / f'{mode}-{task}.json'
            write_json(config_path, job_config)
            jobs.append({'id': f'{mode}/{task}', 'task': task, 'mode': mode,
                         'slot': slot, 'gpu': gpu, 'cpu_affinity': slots[slot],
                         'directory': str(directory / mode / task), 'config': str(config_path),
                         'status': 'queued'})
    return jobs


def compute_probe(gpu, cpus):
    """Fixed workload; repeats and sizes are identical before/during the run."""
    os.sched_setaffinity(0, cpus)
    os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      CUDA_VISIBLE_DEVICES=str(gpu))
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    rng = np.random.default_rng(71)
    a = rng.standard_normal((1536, 1536), dtype=np.float32)
    b = rng.standard_normal((1536, 1536), dtype=np.float32)
    a @ b
    cpu = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(4):
            a @ b
        cpu.append((time.perf_counter() - started) / 4)
    torch.manual_seed(71)
    x = torch.randn((4096, 4096), device='cuda')
    y = torch.randn((4096, 4096), device='cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    for _ in range(3):
        x @ y
    torch.cuda.synchronize()
    values = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(16):
            x @ y
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) / 16)
    return [{'name': f'cpu-gpu{gpu}-slot0', 'seconds': statistics.median(cpu), 'status': 'ok'},
            {'name': f'gpu{gpu}-matmul', 'seconds': statistics.median(values), 'status': 'ok'}]


def api_probe(config, directory):
    journal = Journal(directory / 'events.jsonl')
    budget = Budget({'api_calls': None, 'api_tokens': None}, journal)
    client = APIClient(config['credentials'], config['model'], budget, directory / 'calls', journal,
                       config.get('max_output_tokens', 'model_max'),
                       request_timeout=config.get('api_request_timeout_seconds', API_REQUEST_TIMEOUT_SECONDS), max_retries=0)
    results = []
    # Input is fixed across rounds. These calls are telemetry, never task evidence.
    for size in (2000, 32000):
        padding = ('A neutral telemetry record. No scientific task is included.\n' * (size//59+1))[:size]
        messages = [{'role': 'system', 'content': 'This is a latency probe. Return exactly this JSON object: {"ok":true}.'},
                    {'role': 'user', 'content': padding}]
        started = time.monotonic()
        try:
            text = client.complete(messages, 'probe', 'fixed_latency_probe_v1', client.request_timeout)
            parsed = parse_json(text)
            status = 'ok' if parsed == {'ok': True} else 'unexpected_response'
        except Exception as exc:
            status = type(exc).__name__ + ': ' + str(exc)[:150]
        results.append({'name': f'api-{size}-chars', 'seconds': time.monotonic()-started, 'status': status})
    return results


def require_baseline(baseline):
    names = [f'{kind}{gpu}{suffix}' for gpu in range(8)
             for kind, suffix in [('gpu', '-matmul'), ('cpu-gpu', '-slot0')]]
    names += ['api-2000-chars', 'api-32000-chars']
    missing = [name for name in names if not baseline.get(name)]
    if missing:
        raise RuntimeError('No successful baseline; research has not started: ' + ', '.join(missing))


def probe_round(config_path, directory, label):
    config = json.loads(Path(config_path).read_text())
    probe_dir = Path(directory) / 'probe-runs' / label
    probe_dir.mkdir(parents=True, exist_ok=False)
    shares = config.get('gpu_slots_per_gpu', 2)
    gpus = config['gpus']
    slots = cpu_slots(len(gpus) * shares)
    rows = []
    for index, gpu in enumerate(gpus):
        target = probe_dir / f'gpu{gpu}.json'
        command = [sys.executable, '-m', 'maintenance.stress', 'compute-probe', '--gpu', str(gpu),
                   '--cpus', ','.join(map(str, slots[index*shares])), '--output', str(target)]
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True, timeout=90)
            if result.returncode:
                raise RuntimeError(result.stderr[-300:])
            rows.extend(json.loads(target.read_text()))
        except Exception as exc:
            rows.append({'name': f'gpu{gpu}-matmul', 'seconds': None, 'status': str(exc)[:300]})
    rows.extend(api_probe(config, probe_dir / 'api'))
    result = {'label': label, 'time': time.time(), 'rows': rows}
    write_json(probe_dir / 'result.json', result)
    return result


def launch(config_path, directory, progress):
    from run_rsi import load_config, preflight
    directory = record_path(directory, area='runs')
    progress = record_path(progress, area='runs')
    if not progress.is_relative_to(directory):
        raise ValueError('Progress reports must be inside their experiment directory')
    config = load_config(config_path)
    if config['budget']['solve_seconds'] != 14400 or config['budget'].get('wall_seconds') != 14400:
        raise ValueError('This stress profile requires exactly 14400-second exploration budgets')
    if config.get('gpu_slots_per_gpu') not in (2, 3) or config['gpus'] != list(range(8)):
        raise ValueError('This paired profile requires all eight GPUs and two or three slots per GPU')
    tasks = config.get('tasks') or (ROOT / 'configs/tasks_44.txt').read_text().split()
    if config.get('naturebench_active'):
        from benchmarks.naturebench_active.release import release
        freeze = dict(protocol='naturebench-active', tasks={t: release(config, t)['identity'] for t in tasks})
    else:
        freeze = verify_freeze(config['data_root'], manifest=config.get('freeze_manifest'))
    readiness_root = config['naturebench_source_root'] if config.get('naturebench_active') else config['data_root']
    readiness = preflight(readiness_root, tasks)
    readiness['public_data_root'] = config['data_root']
    blockers = [r for r in readiness['tasks'] if r['metadata_issues'] or r['missing_evaluator_imports'] or not r['data_directory']]
    if blockers:
        raise RuntimeError('Task readiness blockers: ' + json.dumps(blockers))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'worker-logs').mkdir()
    write_json(directory / 'readiness.json', readiness)
    state = {'pid': os.getpid(), 'process_start': __import__('rsi.resources', fromlist=['GPUPool']).GPUPool.process_start(os.getpid()),
             'started_at': time.time(), 'status': 'calibrating', 'freeze': freeze,
             'directory': str(directory), 'concurrency': len(config['gpus']) * config['gpu_slots_per_gpu'],
             'slots_per_gpu': config['gpu_slots_per_gpu'],
             'exploration_seconds_per_run': 14400, 'jobs': make_jobs(config, directory, tasks)}
    monitor = Monitor(directory, progress)
    processes, logs = {}, {}
    stop = False
    def interrupted(signum, frame):
        nonlocal stop
        stop = True
    old = signal.signal(signal.SIGTERM, interrupted)
    old_int = signal.signal(signal.SIGINT, interrupted)
    probe = None
    next_probe = 0
    probe_number = 0
    baseline = defaultdict(list)
    latest_probes = []
    last_report = 0
    try:
        while True:
            now = time.time()
            for job in state['jobs']:
                process = processes.get(job['id'])
                if process is None or process.poll() is None:
                    continue
                result_path = Path(job['directory']) / 'summary.json'
                try:
                    result = json.loads(result_path.read_text())
                except (OSError, ValueError):
                    result = {'status': 'failed', 'error': f'Worker exited {process.returncode} without a complete summary'}
                job.update(status=result['status'], ended_at=now, returncode=process.returncode, result=result)
                logs.pop(job['id']).close()
                processes.pop(job['id'])
            if probe and probe.poll() is not None:
                path = directory / 'probe-runs' / probe_label / 'result.json'
                try:
                    record = json.loads(path.read_text())
                    if probe_label.startswith('baseline'):
                        for row in record['rows']:
                            if row['status'] == 'ok' and row['seconds'] is not None:
                                baseline[row['name']].append(row['seconds'])
                    latest_probes = [{**row, 'baseline_seconds': statistics.median(baseline[row['name']])
                                      if baseline[row['name']] else None} for row in record['rows']]
                    append_json(directory / 'probes.jsonl', {**record, 'rows': latest_probes})
                    write_json(directory / 'probe-baseline.json', dict(baseline))
                except (OSError, ValueError) as exc:
                    append_json(directory / 'probes.jsonl', {'time': now, 'label': probe_label,
                                                          'error': str(exc), 'returncode': probe.returncode})
                probe_log.close(); probe = None
                if probe_number >= 2 and state['status'] == 'calibrating':
                    require_baseline(baseline)
                    state['status'] = 'running'
                next_probe = now + (0 if state['status'] == 'calibrating' else 900)
            if stop:
                state['status'] = 'interrupted'
                for process in processes.values():
                    process.terminate()
                if probe and probe.poll() is None:
                    os.killpg(probe.pid, signal.SIGTERM)
                for job in state['jobs']:
                    if job['status'] == 'queued':
                        job['status'] = 'cancelled'
                # Give workers time to clean confined descendants and persist summaries.
                for process in processes.values():
                    try:
                        process.wait(timeout=25)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                stop = False
            if state['status'] == 'running':
                occupied = {j['slot'] for j in state['jobs'] if j['status'] == 'running'}
                for job in state['jobs']:
                    if job['status'] != 'queued' or job['slot'] in occupied:
                        continue
                    log = (directory / 'worker-logs' / (job['id'].replace('/', '-') + '.log')).open('ab', buffering=0)
                    process = subprocess.Popen([sys.executable, '-m', 'maintenance.stress', 'worker', '--config', job['config'],
                                                '--task', job['task'], '--out', job['directory']],
                                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True)
                    job.update(status='running', pid=process.pid, started_at=now)
                    processes[job['id']], logs[job['id']] = process, log
                    occupied.add(job['slot'])
            pending = any(j['status'] in ('queued', 'running') for j in state['jobs'])
            if not pending and not processes:
                state['status'] = 'finished' if state['status'] != 'interrupted' else 'interrupted'
            if now >= next_probe and probe is None and state['status'] in ('running', 'calibrating'):
                probe_label = f'baseline-{probe_number}' if state['status'] == 'calibrating' else f'load-{probe_number:03d}'
                probe_number += 1
                probe_log = (directory / 'worker-logs' / f'probe-{probe_label}.log').open('ab', buffering=0)
                probe = subprocess.Popen([sys.executable, '-m', 'maintenance.stress', 'probe-round', '--config', str(Path(config_path).resolve()),
                                          '--out', str(directory), '--label', probe_label], cwd=ROOT,
                                         stdout=probe_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
            try:
                sample = monitor.sample(state['jobs'])
            except Exception as exc:
                append_json(directory / 'monitor-errors.jsonl', {'time': time.time(), 'error': repr(exc)})
                state['monitor_error'] = repr(exc)
                write_json(directory / 'supervisor.json', state)
                time.sleep(5)
                continue
            state.update(heartbeat=time.time(), monitor_samples=monitor.samples, peaks=monitor.peaks)
            write_json(directory / 'supervisor.json', state)
            if now - last_report >= 30 or state['status'] in ('finished', 'interrupted'):
                monitor.report(state, sample, latest_probes)
                last_report = now
            if state['status'] in ('finished', 'interrupted') and not processes:
                if probe and probe.poll() is None:
                    os.killpg(probe.pid, signal.SIGTERM)
                    probe.wait(timeout=20)
                    probe_log.close()
                write_json(directory / 'finished.json', {'time': time.time(), 'status': state['status'],
                                                       'jobs': len(state['jobs']), 'peaks': monitor.peaks})
                break
            time.sleep(5)
    except BaseException as exc:
        state.update(status='supervisor_failed', error=repr(exc), heartbeat=time.time())
        write_json(directory / 'supervisor.json', state)
        if 'sample' in locals():
            monitor.report(state, sample, latest_probes)
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if probe and probe.poll() is None:
            os.killpg(probe.pid, signal.SIGTERM)
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        signal.signal(signal.SIGINT, old_int)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('launch', 'worker', 'probe-round'):
        p = sub.add_parser(command)
        p.add_argument('--config', required=True)
        p.add_argument('--out', required=True, type=record_path)
        if command == 'launch':
            p.add_argument('--progress', help='Progress document; defaults to OUT/progress.md')
        elif command == 'worker':
            p.add_argument('--task', required=True)
        else:
            p.add_argument('--label', required=True)
    p = sub.add_parser('compute-probe')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--output', required=True, type=record_path)
    args = parser.parse_args()
    if args.command == 'launch':
        launch(args.config, args.out, args.progress or str(Path(args.out) / 'progress.md'))
    elif args.command == 'worker':
        from run_rsi import load_config
        result = run_task(load_config(args.config), args.task, args.out)
        print(json.dumps({'status': result['status'], 'judge': result['judge']}), flush=True)
    elif args.command == 'probe-round':
        probe_round(args.config, args.out, args.label)
    else:
        write_json(args.output, compute_probe(args.gpu, list(map(int, args.cpus.split(',')))))


if __name__ == '__main__':
    main()
