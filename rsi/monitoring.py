"""Incremental, secret-free telemetry and a continuously rewritten supervision report."""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import psutil

from rsi.common import encoded, write_json
from rsi.workflow import phase_after


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))]


def append_json(path, value):
    with Path(path).open('a') as out:
        out.write(encoded(value) + '\n')


def proc_sample():
    rows = {}
    for process in psutil.process_iter(['pid', 'ppid', 'uids', 'cpu_times', 'memory_info', 'num_threads', 'status']):
        try:
            info = process.info
            if info['status'] == psutil.STATUS_ZOMBIE:
                continue
            cpu = info['cpu_times']
            row = {'pid': info['pid'], 'ppid': info['ppid'], 'uid': info['uids'].real,
                   'cpu_seconds': cpu.user + cpu.system, 'rss_bytes': info['memory_info'].rss,
                   'threads': info['num_threads']}
            try:
                with open(f'/proc/{info["pid"]}/schedstat') as f:
                    fields = f.read().split()
                    row['runqueue_wait_seconds'] = int(fields[1]) / 1e9
            except (OSError, ValueError, IndexError):
                pass
            rows[info['pid']] = row
        except (psutil.Error, AttributeError, TypeError):
            continue
    return rows


def job_processes(processes, pid):
    """Include host descendants and reparented workers with the job's sandbox UID."""
    if not pid:
        return []
    sandbox_uids = (200000 + pid, 1200000 + pid, 3000000 + pid)
    owned = {pid} | {p['pid'] for p in processes.values() if p['uid'] in sandbox_uids}
    children = {}
    for process in processes.values():
        children.setdefault(process['ppid'], []).append(process['pid'])
    pending = list(owned)
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in owned:
                owned.add(child)
                pending.append(child)
    return [processes[p] for p in sorted(owned) if p in processes]


def read_pairs(path):
    try:
        return {k: int(v) for k, v in (line.split() for line in Path(path).read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def pressure(name):
    try:
        return {line.split()[0]: {k: float(v) for k, v in (item.split('=') for item in line.split()[1:])}
                for line in Path('/sys/fs/cgroup', name + '.pressure').read_text().splitlines()}
    except (OSError, ValueError):
        return {}


def nvidia_sample():
    fields = ['index', 'uuid', 'utilization.gpu', 'utilization.memory', 'memory.used', 'memory.total',
              'power.draw', 'temperature.gpu', 'clocks.sm', 'clocks.mem']
    def query(option, columns):
        result = subprocess.run(['nvidia-smi', '--' + option + '=' + ','.join(columns),
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[:500])
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    gpus, processes, errors = [], [], []
    try:
        for row in query('query-gpu', fields):
            value = dict(zip(fields, row))
            for key in fields:
                if key == 'uuid':
                    continue
                try:
                    value[key] = float(value[key])
                except (ValueError, KeyError):
                    value[key] = None
            gpus.append(value)
    except Exception as exc:
        errors.append(str(exc))
    try:
        for row in query('query-compute-apps', ['gpu_uuid', 'pid', 'used_gpu_memory']):
            processes.append({'gpu_uuid': row[0], 'pid': int(row[1]),
                              'memory_mib': float(row[2]) if row[2].replace('.', '', 1).isdigit() else None})
    except Exception as exc:
        errors.append(str(exc))
    return {'gpus': gpus, 'processes': processes, 'errors': errors}


class RunEvents:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.offset = 0
        self.pending = {}
        self.latencies = deque(maxlen=2000)
        self.recent = deque(maxlen=2000)
        self.errors = Counter()
        self.calls = self.successes = self.retries = self.evaluations = self.task_rounds = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.stage = 'queued'
        self.workflow_phase = 'unattributed'
        self.last_event = self.research_started = None
        self.best = None
        self.tool_seconds = deque(maxlen=200)
        self.resource_failures = Counter()
        self.cuda_oom_output_events = 0

    def read(self):
        path = self.directory / 'events.jsonl'
        if not path.exists():
            return
        with path.open() as stream:
            stream.seek(self.offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith('\n'):
                    self.offset = position
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    self.offset = stream.tell()
                    continue
                self.consume(event)
                self.offset = stream.tell()

    def consume(self, event):
        kind, now = event['event'], event['time']
        self.workflow_phase = phase_after(event, self.workflow_phase)
        self.last_event = now
        if kind == 'gpu_acquired':
            self.research_started = now
        elif kind == 'session_start':
            self.stage = event['run_id']
        elif kind == 'session_end' and event['run_id'].startswith('T'):
            self.task_rounds += 1
        elif kind == 'final_test_start':
            self.stage = 'final_test'
        elif kind == 'final_test_end':
            self.stage = 'final_test_' + event['status']
        elif kind == 'validity_judge_start':
            self.stage = 'judge'
        elif kind == 'exploration_end':
            self.stage = 'exploration_end'
        elif kind == 'api_start':
            self.calls += 1
            self.pending[event['call']] = now
        elif kind in ('api_end', 'api_error'):
            started = self.pending.pop(event['call'], now)
            seconds = event.get('seconds', now - started)
            self.recent.append((now, seconds, kind == 'api_error'))
            if kind == 'api_end':
                self.successes += 1
                self.latencies.append(seconds)
            else:
                message = event.get('message', '').lower()
                category = ('rate_limit' if event.get('http_status') == 429 else
                            'timeout' if 'timeout' in message or 'deadline' in message else
                            'empty_response' if 'nonempty' in message else
                            'http_' + str(event.get('http_status')) if event.get('http_status') else
                            event.get('error_type', 'unknown'))
                self.errors[category] += 1
        elif kind == 'api_retry':
            self.retries += 1
        elif kind == 'api_settled' and event.get('usage'):
            self.prompt_tokens += event['usage']['prompt_tokens']
            self.completion_tokens += event['usage']['completion_tokens']
        elif kind == 'evaluation':
            self.evaluations += 1
            value = event.get('aggregate_improvement')
            if isinstance(value, (int, float)) and (self.best is None or value > self.best):
                self.best = value
        elif kind in ('agent_action', 'agent_source_failed'):
            response = event.get('response', event.get('process', {}))
            if isinstance(response, dict):
                if isinstance(response.get('seconds'), (int, float)):
                    self.tool_seconds.append(response['seconds'])
                if response.get('reason') in ('memory_limit', 'timeout', 'output_limit'):
                    self.resource_failures[response['reason']] += 1
                # A shell may mask a failed Python command with a successful
                # tail/echo. Report explicit CUDA exceptions independently of
                # its exit code. Replayed logs can repeat these reports, so
                # this is an output-event count, not unique OOM failures.
                stdout = response.get('stdout', '')
                if isinstance(stdout, str) and any(marker in stdout for marker in (
                        'OutOfMemoryError: CUDA out of memory', 'CUDA error: out of memory')):
                    self.cuda_oom_output_events += 1

    def summary(self, now):
        recent = [row for row in self.recent if now - row[0] <= 300]
        return {'stage': self.stage, 'workflow_phase': self.workflow_phase,
                'last_event': self.last_event, 'task_rounds': self.task_rounds,
                'evaluations': self.evaluations, 'best_g': self.best, 'api_calls': self.calls,
                'api_successes': self.successes, 'api_errors': dict(self.errors), 'api_retries': self.retries,
                'api_inflight': len(self.pending),
                # A request can start after the sample timestamp but before
                # its event is read. Its observed age is then zero.
                'oldest_request_seconds': max((max(0, now - t) for t in self.pending.values()), default=0),
                'api_p50_seconds': percentile(self.latencies, .5), 'api_p95_seconds': percentile(self.latencies, .95),
                'recent_api_calls': len(recent), 'recent_api_errors': sum(row[2] for row in recent),
                'recent_api_p95_seconds': percentile([r[1] for r in recent if not r[2]], .95),
                'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens,
                'resource_failures': dict(self.resource_failures),
                'cuda_oom_output_events': self.cuda_oom_output_events,
                'recent_tool_p95_seconds': percentile(self.tool_seconds, .95)}


class Monitor:
    def __init__(self, directory, progress_path):
        self.directory, self.progress_path = Path(directory), Path(progress_path)
        self.events = {}
        self.previous_processes = {}
        self.previous_time = time.monotonic()
        self.previous_cgroup = read_pairs('/sys/fs/cgroup/cpu.stat')
        self.previous_cgroup_time = time.monotonic()
        self.peaks = {'gpu_memory_mib': {}, 'gpu_utilization': {}, 'job_rss_bytes': 0,
                      'cpu_quota_percent': 0, 'host_memory_current_bytes': 0,
                      'api_inflight': 0, 'running_jobs': 0}
        self.samples = 0
        psutil.cpu_percent(percpu=True)

    def sample(self, jobs):
        now, monotonic = time.time(), time.monotonic()
        elapsed = max(.001, monotonic - self.previous_time)
        processes = proc_sample()
        for pid, row in processes.items():
            old = self.previous_processes.get(pid, {})
            row['cpu_percent'] = max(0, row['cpu_seconds'] - old.get('cpu_seconds', row['cpu_seconds'])) / elapsed * 100
            row['runqueue_wait_delta_seconds'] = max(0, row.get('runqueue_wait_seconds', 0) - old.get('runqueue_wait_seconds', row.get('runqueue_wait_seconds', 0)))
        gpu = nvidia_sample()
        gpu_by_pid = {row['pid']: row for row in gpu['processes']}
        rows = []
        for job in jobs:
            key = job['id']
            event = self.events.setdefault(key, RunEvents(job['directory']))
            event.read()
            row = {k: job.get(k) for k in ('id', 'task', 'mode', 'gpu', 'slot', 'pid', 'status', 'started_at', 'ended_at')}
            row.update(event.summary(now))
            pid = job.get('pid')
            owned = job_processes(processes, pid)
            row.update(cpu_percent=sum(p['cpu_percent'] for p in owned),
                       rss_bytes=sum(p['rss_bytes'] for p in owned), process_count=len(owned),
                       threads=sum(p['threads'] for p in owned),
                       runqueue_wait_seconds=sum(p['runqueue_wait_delta_seconds'] for p in owned),
                       gpu_memory_mib=(None if gpu['errors'] or any(
                           gpu_by_pid[p['pid']]['memory_mib'] is None for p in owned if p['pid'] in gpu_by_pid)
                           else sum(gpu_by_pid.get(p['pid'], {}).get('memory_mib') or 0 for p in owned)))
            if job.get('result'):
                result = job['result']
                row['judge'] = result.get('judge')
                row['error'] = result.get('error')
                row['official_result'] = result.get('official_result')
            rows.append(row)
        cgroup = read_pairs('/sys/fs/cgroup/cpu.stat')
        cgroup_time = time.monotonic()
        cgroup_elapsed = max(.001, cgroup_time - self.previous_cgroup_time)
        quota_text = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        quota = int(quota_text[0]) / int(quota_text[1]) if quota_text[0] != 'max' else psutil.cpu_count()
        quota_used = max(0, cgroup.get('usage_usec', 0) - self.previous_cgroup.get('usage_usec', 0)) / 1e6 / cgroup_elapsed
        memory = read_pairs('/sys/fs/cgroup/memory.stat')
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        per_cpu = psutil.cpu_percent(percpu=True)
        disk = psutil.disk_io_counters()
        sample = {'time': now, 'interval_seconds': elapsed, **gpu, 'jobs': rows,
                  'host': {'cpu_quota_cores': quota, 'cpu_cores_used': quota_used,
                           'cpu_quota_percent': quota_used / quota * 100,
                           'cpu_per_logical_percent': per_cpu, 'load_average': os.getloadavg(),
                           'cpu_throttled_usec_delta': max(0, cgroup.get('throttled_usec', 0) - self.previous_cgroup.get('throttled_usec', 0)),
                           'cpu_throttled_periods_delta': max(0, cgroup.get('nr_throttled', 0) - self.previous_cgroup.get('nr_throttled', 0)),
                           'memory_current_bytes': current, 'memory_anon_bytes': memory.get('anon'),
                           'memory_file_cache_bytes': memory.get('file'),
                           'memory_events': read_pairs('/sys/fs/cgroup/memory.events'),
                           'pressure': {k: pressure(k) for k in ('cpu', 'memory', 'io')},
                           'disk_free_bytes': shutil.disk_usage(self.directory).free,
                           'disk_counters': disk._asdict() if disk else {}},
                  'api_inflight': sum(r['api_inflight'] for r in rows),
                  'api_errors': sum(sum(r['api_errors'].values()) for r in rows),
                  'running_jobs': sum(j['status'] == 'running' for j in jobs)}
        for g in gpu['gpus']:
            key = str(int(g['index']))
            for name, field in [('gpu_memory_mib', 'memory.used'), ('gpu_utilization', 'utilization.gpu')]:
                self.peaks[name][key] = max(self.peaks[name].get(key, 0), g[field] or 0)
        for key, value in [('job_rss_bytes', max((r['rss_bytes'] for r in rows), default=0)),
                           ('cpu_quota_percent', sample['host']['cpu_quota_percent']),
                           ('host_memory_current_bytes', current), ('api_inflight', sample['api_inflight']),
                           ('running_jobs', sample['running_jobs'])]:
            self.peaks[key] = max(self.peaks[key], value)
        self.samples += 1
        sample['peaks'] = self.peaks.copy()
        append_json(self.directory / 'metrics.jsonl', sample)
        write_json(self.directory / 'metrics-latest.json', sample)
        self.previous_processes, self.previous_time, self.previous_cgroup = processes, monotonic, cgroup
        self.previous_cgroup_time = cgroup_time
        return sample

    def report(self, state, sample, probes=None):
        def number(value, digits=1):
            return '—' if value is None else f'{value:.{digits}f}'
        now = datetime.fromtimestamp(sample['time'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        counts = Counter(j['status'] for j in state['jobs'])
        concurrency = state.get('concurrency', 16)
        shares = state.get('slots_per_gpu', 2)
        tasks = len({j['task'] for j in state['jobs']})
        lines = [f'# RSI4S {concurrency} 并发压力测试进度', '',
                 f'更新时间：{now}。监督进程 PID：{os.getpid()}；状态：{state["status"]}。', '',
                 f'{tasks} 个任务，共 {len(state["jobs"])} 个评估。最多 {concurrency} 并发，每 GPU {shares} 个；每评估探索 {state.get("exploration_seconds_per_run", 14400):,} 秒，judge 在探索后单独执行。', '',
                 f'状态计数：{dict(counts)}；已采样 {self.samples} 次；实际同时运行峰值 {self.peaks["running_jobs"]}。', '',
                 'GPU 0–3 运行 fixed，4–7 运行 coevolve；对应任务使用匹配的 CPU/内存/共享 GPU 配置。完成后继续从该槽位队列启动下一任务。', '',
                 '## 当前资源及累计峰值', '',
                 '| GPU | 利用率 % | 显存 GiB | 利用率峰值 % | 显存峰值 GiB |',
                 '|---|---:|---:|---:|---:|']
        if state.get('error'):
            lines[2:2] = ['监督进程异常：' + str(state['error']), '']
        for gpu in sample['gpus']:
            key = str(int(gpu['index']))
            lines.append(f'| {key} | {number(gpu["utilization.gpu"])} | {number((gpu["memory.used"] or 0)/1024)} | {number(self.peaks["gpu_utilization"][key])} | {number(self.peaks["gpu_memory_mib"][key]/1024)} |')
        host = sample['host']
        lines += ['', f'CPU：当前使用约 {number(host["cpu_cores_used"])} 个逻辑核的计算时间 / {number(host["cpu_quota_cores"])} 配额；使用率峰值 {number(self.peaks["cpu_quota_percent"])}%。本采样间隔限流 {host["cpu_throttled_periods_delta"]} 次。', '',
                  f'内存：匿名页 {number((host["memory_anon_bytes"] or 0)/1024**3)} GiB；文件缓存 {number((host["memory_file_cache_bytes"] or 0)/1024**3)} GiB（可回收部分不能当作程序常驻内存）；磁盘剩余 {number(host["disk_free_bytes"]/1024**3)} GiB。', '',
                  f'API：当前在途 {sample["api_inflight"]}；累计请求错误 {sample["api_errors"]}（包含已重试恢复的请求）。延迟包含服务端排队和完整生成，非首 token 延迟。', '',
                  '## 固定探针与变慢证据', '']
        if probes:
            lines.append('以下倍率来自相同输入/计算探针相对于启动前基线；不将不同科研程序的执行时间直接当作性能退化。')
            lines.append('')
            lines.append('| 探针 | 当前耗时 s | 基线耗时 s | 倍率 | 状态 |')
            lines.append('|---|---:|---:|---:|---|')
            for row in probes:
                base = row.get('baseline_seconds')
                value = row.get('seconds')
                ratio = value / base if base and value is not None else None
                lines.append(f'| {row["name"]} | {number(value,3)} | {number(base,3)} | {number(ratio,2)} | {row.get("status","ok")} |')
        else:
            lines.append('基线/负载探针尚未产生完整数据，不据此断言已经变慢。')
        warnings = []
        if any(row.get('cuda_oom_output_events', 0) for row in sample['jobs']):
            warnings.append('科研命令输出出现 CUDA 显存不足异常。此计数可能包含重放日志，应结合执行记录确认实际失败次数。')
        if host['cpu_quota_percent'] >= 85:
            warnings.append('CPU 使用接近容器配额；结合 runqueue/PSI 和固定 CPU 探针判断争抢。')
        if host['cpu_throttled_periods_delta']:
            warnings.append('本间隔出现 CPU 配额限流。')
        if any((g['memory.used'] or 0) / max(g['memory.total'] or 1, 1) > .9 for g in sample['gpus']):
            warnings.append('至少一张 GPU 显存超过 90%。')
        if host['disk_free_bytes'] < 200 * 1024**3:
            warnings.append('磁盘余量低于 200 GiB。')
        if probes and any(r.get('baseline_seconds') and r.get('seconds', 0) > 2*r['baseline_seconds'] for r in probes if r.get('seconds') is not None):
            warnings.append('至少一项固定探针耗时超过基线 2 倍；单次异常需结合后续样本确认。')
        if sample['errors']:
            warnings += ['监控读取错误：' + e for e in sample['errors']]
        lines += ['', '当前观察：' + ('；'.join(warnings) if warnings else '本次采样未触发资源阈值告警；不代表不存在短时峰值。'), '',
                  '## 各评估进度', '',
                  '| 模式 / task | GPU | 状态 / 当前阶段 | 完成 Task 轮数 | 评分次数 | 最佳 g | API 次数 / 错误 | API p50 / p95 s | CPU % / RSS GiB |',
                  '|---|---:|---|---:|---:|---:|---|---|---|']
        for row in sample['jobs']:
            lines.append(f'| {row["mode"]} / {row["task"]} | {row["gpu"]} | {row["status"]} / {row["stage"]} | {row["task_rounds"]} | {row["evaluations"]} | {number(row["best_g"],4)} | {row["api_calls"]} / {sum(row["api_errors"].values())} | {number(row["api_p50_seconds"])} / {number(row["api_p95_seconds"])} | {number(row["cpu_percent"])} / {number(row["rss_bytes"]/1024**3)} |')
        failures = [row for row in sample['jobs'] if row.get('error') and row['status'] == 'failed']
        lines += ['', '## 失败与审核', '']
        if not failures:
            lines.append('暂未记录终态运行故障。四小时正常耗尽以 budget_exhausted 保留；原生指标不以 Judge 判定为条件。')
        for row in failures:
            lines.append(f'- {row["id"]}：{str(row["error"])[:500]}')
        judgments = Counter(
            'disabled' if (job['result'].get('judge') or {}).get('status') == 'disabled'
            else 'historical:' + str((job['result'].get('judge') or {}).get('is_valid'))
            for job in state['jobs'] if job.get('result'))
        lines += ['', f'审查记录状态：{dict(judgments)}。disabled 表示未进行独立科学有效性审查；历史结论仅保留溯源，不作为成绩门槛。', '',
                  '## 隔离与解释边界', '',
                  '- 每个评估使用独立 UID、工作目录、源码版本、历史记录、API 账本；训练输入只读，隐藏标签只供独立评分进程读取。',
                  '- CPU 亲和性由宿主指定并禁止子进程扩大；同一评估的子进程共享该 CPU 集合。线程变量只是默认值，不能代替亲和性。',
                  '- cgroup 挂载只读：内存采用每 0.5 秒采样的进程树 RSS 看门狗；共享页可能重复计数，短时超量可能发生，不宣称硬内存配额。',
                  '- 每卡 3 个任务共享算力和显存，没有 MIG/MPS 硬分区；CUDA 地址空间、文件与进程权限相互隔离，GPU 故障和带宽竞争仍可能影响同卡任务。',
                  '- GPU/CPU 采样可能漏掉瞬时尖峰；高利用率本身不等于争抢，需结合固定探针、排队时间、OOM/超时和压力指标。',
                  '- 不自动重启失败评估、不重置四小时预算、不改写历史结果；持续消费队列中的其余任务。', '',
                  f'原始记录：`{self.directory}`。`metrics.jsonl` 为资源/进度采样；`probes.jsonl` 为固定基准；`supervisor.json` 为队列与进程；每个评估的 `calls/` 和 `events.jsonl` 保存请求、错误及用量。', '']
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.progress_path.with_suffix('.tmp')
        temp.write_text('\n'.join(lines))
        temp.replace(self.progress_path)
