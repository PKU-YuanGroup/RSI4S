"""Terminal evaluators share research GPU/CPU leases, including the prior GPU."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import subprocess

from benchmarks.naturebench_active.scoring.rna_terminal import available_memory_mib
from rsi.common import ROOT
from rsi.resources import GPUPool


class EvaluationPool(GPUPool):
    """CPU-only terminal work participates in the same resource registry."""

    def __init__(self, gpu_ids, pool_path, *, memory_mib=0, memory_budget_mib=None, **kwargs):
        if any(type(gpu) is not int or gpu < 0 for gpu in gpu_ids):
            raise ValueError('GPU candidates must be nonnegative integers')
        if type(memory_mib) is not int or memory_mib < 0:
            raise ValueError('memory_mib must be a nonnegative integer')
        self.memory_mib = memory_mib
        self.memory_budget_mib = memory_budget_mib or int(available_memory_mib()*.8)
        if self.memory_budget_mib < memory_mib:
            raise ValueError('One evaluation exceeds the entire memory reservation budget')
        Path(pool_path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(gpu_ids, Path(pool_path), **kwargs)

    def _probe_external_busy(self, gid):
        if self._skip_busy_mb <= 0:
            return False
        # A broken/unknown GPU probe must not be treated as a free device.
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits', '-i', str(gid)],
                capture_output=True, text=True, timeout=3, check=True)
            memory, utilization = map(int, result.stdout.strip().split(','))
        except (OSError, ValueError, subprocess.SubprocessError):
            return True
        return memory > self._skip_busy_mb or utilization > self._skip_busy_util

    def _try_acquire(self, state):
        entries = list(state['gpus'].values())+[state.setdefault('cpu', {'holders': []})]
        holders = [holder for entry in entries for holder in self._clean(entry)]
        if self.memory_mib and (sum(h.get('memory_mib', 0) for h in holders)+self.memory_mib > self.memory_budget_mib or
                                self.memory_mib > int(available_memory_mib()*.9)):
            return None
        if self._registered:
            gpu = super()._try_acquire(state)
            if gpu is not None:
                state['gpus'][str(gpu)]['holders'][-1]['memory_mib'] = self.memory_mib
            return gpu
        used = {cpu for h in holders for cpu in h.get('cpu_affinity', [])}
        available = sorted(os.sched_getaffinity(0)-used)
        cpus = self.requested_cpus or available[:self.cpu_cores]
        if not set(cpus) <= set(available) or len(cpus) < self.cpu_cores:
            return None
        state['cpu']['holders'].append(dict(pid=os.getpid(), process_start=self.process_start(os.getpid()),
            acquired_at=time.time(), cpu_affinity=cpus, memory_mib=self.memory_mib, **self.lease_info))
        return -1

    def assigned_cpus(self, gid):
        if gid != -1:
            return super().assigned_cpus(gid)
        return self._with_lock(lambda state: next(h['cpu_affinity'] for h in self._clean(state['cpu'])
            if h['pid'] == os.getpid() and h['process_start'] == self.process_start(os.getpid())))

    def put(self, gid):
        if gid != -1:
            return super().put(gid)
        def release(state):
            holders = self._clean(state['cpu'])
            for holder in holders:
                if holder['pid'] == os.getpid() and holder['process_start'] == self.process_start(os.getpid()):
                    holders.remove(holder)
                    break
        self._with_lock(release)
        if gid in self._held:
            self._held.remove(gid)


@contextmanager
def terminal_resources(run, *, device='previous', gpu_ids=None, cpu_cores=None, cpu_affinity=None,
                       slots_per_gpu=None, queue_seconds=600, memory_mib=0, memory_budget_mib=None,
                       pool_path=None, skip_busy_mb=1024, skip_busy_util=20,
                       distinct_physical_cores=False, cpu_quota_root=None, cpu_quota_cores=None):
    """Reacquire a released device; an old allocation is not an active lease."""
    from .cpu_quota import cpu_quota, check_physical
    if cpu_quota_cores is not None and (cpu_quota_root is None or cpu_quota_cores!=cpu_cores):
        raise ValueError('Explicit terminal CPU quota and affinity allocation must agree; provide a delegated cgroup')
    run = Path(run).resolve()
    config_path, allocation_path = run/'config.json', run/'allocated_resources.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    allocated = json.loads(allocation_path.read_text()) if allocation_path.exists() else {}
    prior = allocated.get('gpu')
    if device == 'cpu':
        candidates = []
    elif device == 'previous':
        if type(prior) is not int:
            raise ValueError('No previous task GPU is recorded; select auto, cpu, or an explicit GPU')
        candidates = [prior]
    elif device == 'auto':
        candidates = gpu_ids or config.get('gpus') or ([prior] if prior is not None else [])
        if not candidates:
            raise ValueError('Automatic GPU scoring requires configured GPU candidates')
    elif type(device) is int and device >= 0:
        candidates = [device]
    else:
        raise ValueError('device must be previous, auto, cpu, or a nonnegative GPU integer')
    if cpu_cores is None:
        cpu_cores = len(allocated.get('cpu_affinity', [])) or config.get('cpu_cores_per_slot', 4)
    pool = EvaluationPool(candidates, pool_path or ROOT/'runs/gpu_pool.json',
        skip_busy_mb=skip_busy_mb, skip_busy_util=skip_busy_util,
        slots_per_gpu=slots_per_gpu or config.get('gpu_slots_per_gpu', 2), preferred_gpu=prior,
        cpu_cores_per_slot=cpu_cores, cpu_affinity=cpu_affinity,
        memory_mib=memory_mib, memory_budget_mib=memory_budget_mib,
        lease_info=dict(task='terminal-evaluation', directory=str(run)))
    before = os.sched_getaffinity(0)
    started = time.monotonic()
    try:
        gpu = pool.get(timeout=queue_seconds)
        cpus = pool.assigned_cpus(gpu)
        if distinct_physical_cores:check_physical(cpus)
        os.sched_setaffinity(0, cpus)
        with cpu_quota(cpu_quota_root,cpu_quota_cores) as quota:
            yield dict(gpu=None if gpu == -1 else gpu, cuda_visible_devices='' if gpu == -1 else str(gpu),
                   cpu_affinity=cpus, cpu_threads=len(cpus), previous_task_gpu=prior,
                   requested_device=device, queue_seconds=time.monotonic()-started,
                   memory_mib=memory_mib, pool_path=str(pool._pool_path), cpu_quota=quota)
    finally:
        os.sched_setaffinity(0, before)
        pool.release_all()
