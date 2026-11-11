"""Cross-process GPU slots; exclusive by default, explicitly shared when configured."""
import os
import time
from contextlib import contextmanager
from pathlib import Path

from rsi.official_gpu import _FileGpuPool


class GPUPool(_FileGpuPool):
    def __init__(self, gpu_ids, pool_path, skip_busy_mb=0, skip_busy_util=100,
                 slots_per_gpu=1, lease_info=None, cpu_cores_per_slot=None, cpu_affinity=None,
                 preferred_gpu=None):
        if type(slots_per_gpu) is not int or not 1 <= slots_per_gpu <= 8:
            raise ValueError('gpu_slots_per_gpu must be an integer in 1..8')
        if cpu_cores_per_slot is not None and (type(cpu_cores_per_slot) is not int or cpu_cores_per_slot < 1):
            raise ValueError('cpu_cores_per_slot must be a positive integer')
        self.cpu_cores = cpu_cores_per_slot
        self.requested_cpus = sorted(cpu_affinity) if cpu_affinity else None
        if self.requested_cpus:
            if not set(self.requested_cpus) <= os.sched_getaffinity(0):
                raise ValueError('Requested CPUs are outside the allowed affinity')
            self.cpu_cores = len(self.requested_cpus)
        if self.cpu_cores and self.cpu_cores > len(os.sched_getaffinity(0)):
            raise ValueError('Not enough allowed CPUs for one experimental lease')
        self.capacity = slots_per_gpu
        self.preferred_gpu = preferred_gpu
        self.lease_info = lease_info or {}
        super().__init__(gpu_ids, pool_path, skip_busy_mb, skip_busy_util)

    @staticmethod
    def process_start(pid):
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return fields[19] if fields[0] != 'Z' else None
        except (OSError, IndexError):
            return None

    def _clean(self, entry):
        holders = entry.setdefault('holders', [])
        holders[:] = [h for h in holders if self._is_pid_alive(h['pid'])
                      and self.process_start(h['pid']) == h.get('process_start')]
        legacy = entry.get('holder_pid')
        if legacy and not self._is_pid_alive(legacy):
            entry['holder_pid'] = None
        return holders

    def _merge_ids(self, state, gpu_ids):
        super()._merge_ids(state, gpu_ids)
        for gid in gpu_ids:
            entry = state['gpus'][str(gid)]
            holders = self._clean(entry)
            if holders and entry.get('capacity', 1) != self.capacity:
                raise RuntimeError('Cannot change GPU slot capacity while leases are active')
            entry['capacity'] = self.capacity

    def _try_acquire(self, state):
        assigned = []
        if self.cpu_cores or self.requested_cpus:
            entries = list(state['gpus'].values())+[state.setdefault('cpu', {'holders': []})]
            used_cpus = {cpu for entry in entries for h in self._clean(entry)
                         for cpu in h.get('cpu_affinity', [])}
            available = sorted(os.sched_getaffinity(0) - used_cpus)
            assigned = self.requested_cpus or available[:self.cpu_cores]
            if not set(assigned) <= set(available) or len(assigned) < (self.cpu_cores or 1):
                return None
        candidates = []
        for gid in self._registered:
            entry = state['gpus'][str(gid)]
            holders = self._clean(entry)
            if holders and entry.get('capacity') != self.capacity:
                raise RuntimeError('GPU slot capacity differs from active leases')
            if entry.get('holder_pid') or len(holders) >= self.capacity:
                continue
            # Usage from admitted peers is expected. External-busy detection
            # applies before the first lease, never to a legitimate shared peer.
            if not holders and self._probe_external_busy(gid):
                continue
            candidates.append((gid != self.preferred_gpu if self.preferred_gpu is not None else False,
                               len(holders), gid))
        if not candidates:
            return None
        _, _, gid = min(candidates)
        entry = state['gpus'][str(gid)]
        entry['capacity'] = self.capacity
        used = {h['slot'] for h in entry['holders']}
        slot = next(i for i in range(self.capacity) if i not in used)
        entry['holders'].append({'pid': os.getpid(), 'process_start': self.process_start(os.getpid()),
                                 'slot': slot, 'acquired_at': time.time(), **self.lease_info, 'cpu_affinity': assigned})
        return gid

    def assigned_cpus(self, gid):
        def read(state):
            return next(h.get('cpu_affinity', []) for h in self._clean(state['gpus'][str(gid)])
                        if h['pid'] == os.getpid() and h['process_start'] == self.process_start(os.getpid()))
        return self._with_lock(read)

    def put(self, gid):
        def release(state):
            entry = state['gpus'].get(str(gid), {})
            holders = self._clean(entry)
            for holder in holders:
                if holder['pid'] == os.getpid() and holder['process_start'] == self.process_start(os.getpid()):
                    holders.remove(holder)
                    break
        self._with_lock(release)
        if gid in self._held:
            self._held.remove(gid)


@contextmanager
def scientific_gpu(config, *, lease_info=None):
    """Diagnostics use the same shared leases and CPU allocation as formal runs."""
    from rsi.common import ROOT
    pool = GPUPool(config['gpus'], ROOT / 'runs/gpu_pool.json', skip_busy_mb=1024, skip_busy_util=20,
                   slots_per_gpu=config.get('gpu_slots_per_gpu', 2), lease_info=lease_info,
                   cpu_cores_per_slot=config.get('cpu_cores_per_slot', 4),
                   cpu_affinity=config.get('sandbox_resources', {}).get('cpu_affinity'))
    try:
        gpu = pool.get(timeout=config.get('gpu_wait_seconds', 600))
        limits = {**config.get('sandbox_resources', {}), 'cpu_affinity': pool.assigned_cpus(gpu)}
        yield gpu, limits
    finally:
        pool.release_all()
