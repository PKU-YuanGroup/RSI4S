"""Verbatim exclusive GPU pools extracted from NatureBench solve.py. See provenance."""
import json, logging, os, queue, subprocess, time
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

class _InMemoryGpuPool:
    """Original single-process queue-based pool. `get(block=True)` blocks
    until a GPU is freed; `get(block=False)` raises queue.Empty."""

    def __init__(self, gpu_ids: List[int]) -> None:
        self._q: "queue.Queue[int]" = queue.Queue()
        for gid in gpu_ids:
            self._q.put(gid)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> int:
        return self._q.get(block=block, timeout=timeout)

    def put(self, gid: int) -> None:
        self._q.put(gid)

class _FileGpuPool:
    """Cross-process GPU pool. Coordinates multiple solve.py instances via a
    JSON file under fcntl.flock. Acquire blocks until a GPU is free; release
    marks it free. Stale holders (dead pid) are auto-reclaimed.

    State file schema:
        {"gpus": {"<id>": {"holder_pid": int|null, "task": str|null,
                           "acquired_at": float|null}}}
    """

    POLL_INTERVAL = 1.0   # seconds between retries when blocked
    _PROBE_CACHE_TTL = 5.0  # seconds — reuse nvidia-smi result within this window

    def __init__(self, gpu_ids: List[int], pool_path: Path,
                 skip_busy_mb: int = 0, skip_busy_util: int = 100) -> None:
        import fcntl as _fcntl
        self._fcntl = _fcntl
        self._pool_path = pool_path
        self._lock_path = pool_path.with_suffix(pool_path.suffix + ".lock")
        self._registered: List[int] = list(gpu_ids)
        self._held: List[int] = []  # GPUs this process currently holds
        # External-busy detection thresholds. skip_busy_mb<=0 disables.
        self._skip_busy_mb = int(skip_busy_mb)
        self._skip_busy_util = int(skip_busy_util)
        self._probe_cache: Dict[int, tuple] = {}  # gid -> (mem_mb, util_pct, ts)
        # initialize / merge own gpu_ids into the file pool
        self._with_lock(lambda state: self._merge_ids(state, gpu_ids))

    def _probe_external_busy(self, gid: int) -> bool:
        """Return True if GPU `gid` is currently used by an *external* process
        (memory or utilization above thresholds). Cached for _PROBE_CACHE_TTL
        to limit nvidia-smi calls. Falls back to "not busy" on probe error.
        """
        if self._skip_busy_mb <= 0:
            return False
        now = time.time()
        cached = self._probe_cache.get(gid)
        if cached and (now - cached[2]) < self._PROBE_CACHE_TTL:
            mem_mb, util_pct, _ = cached
        else:
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits",
                     "-i", str(gid)],
                    capture_output=True, text=True, timeout=3,
                )
                if out.returncode != 0 or not out.stdout.strip():
                    return False
                parts = [p.strip() for p in out.stdout.strip().split(",")]
                mem_mb = int(parts[0])
                util_pct = int(parts[1])
                self._probe_cache[gid] = (mem_mb, util_pct, now)
            except Exception:
                return False
        return mem_mb > self._skip_busy_mb or util_pct > self._skip_busy_util

    def _open_lock(self):
        # open or create the lock file
        return open(self._lock_path, "a+")

    def _read_state(self) -> Dict[str, Any]:
        if not self._pool_path.exists():
            return {"gpus": {}}
        try:
            return json.loads(self._pool_path.read_text(encoding="utf-8"))
        except Exception:
            return {"gpus": {}}

    def _write_state(self, state: Dict[str, Any]) -> None:
        tmp = self._pool_path.with_suffix(self._pool_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self._pool_path)

    def _with_lock(self, fn):
        # exclusive lock around read-modify-write of pool file
        f = self._open_lock()
        try:
            self._fcntl.flock(f.fileno(), self._fcntl.LOCK_EX)
            state = self._read_state()
            ret = fn(state)
            self._write_state(state)
            return ret
        finally:
            try:
                self._fcntl.flock(f.fileno(), self._fcntl.LOCK_UN)
            finally:
                f.close()

    @staticmethod
    def _is_pid_alive(pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _merge_ids(self, state: Dict[str, Any], gpu_ids: List[int]) -> None:
        gpus = state.setdefault("gpus", {})
        for gid in gpu_ids:
            key = str(gid)
            if key not in gpus:
                gpus[key] = {"holder_pid": None, "task": None, "acquired_at": None}

    def _try_acquire(self, state: Dict[str, Any]) -> Optional[int]:
        gpus = state.setdefault("gpus", {})
        # prefer GPUs we initially registered (own list) before others
        my_pid = os.getpid()
        for gid in self._registered + sorted(int(k) for k in gpus.keys() if int(k) not in self._registered):
            key = str(gid)
            entry = gpus.get(key)
            if entry is None:
                continue
            holder = entry.get("holder_pid")
            if holder is None or not self._is_pid_alive(holder):
                # Pool says this GPU is free, but check if an external process
                # (not in our pool) is using it. Skip in that case so we don't
                # contend with someone else's training.
                if self._probe_external_busy(gid):
                    continue
                # free or stale → take it
                entry["holder_pid"] = my_pid
                entry["acquired_at"] = time.time()
                return gid
        return None

    def get(self, block: bool = True, timeout: Optional[float] = None) -> int:
        deadline = (time.time() + timeout) if (timeout is not None) else None
        first = True
        while True:
            gid_holder = []
            def _take(state, _h=gid_holder):
                gid = self._try_acquire(state)
                _h.append(gid)
            self._with_lock(_take)
            gid = gid_holder[0]
            if gid is not None:
                self._held.append(gid)
                return gid
            if not block:
                raise queue.Empty()
            if deadline is not None and time.time() >= deadline:
                raise queue.Empty()
            if first:
                logger.info("GPU pool: all GPUs busy, waiting for one to free up...")
                first = False
            time.sleep(self.POLL_INTERVAL)

    def put(self, gid: int) -> None:
        my_pid = os.getpid()
        def _release(state):
            entry = state.get("gpus", {}).get(str(gid))
            if entry and entry.get("holder_pid") == my_pid:
                entry["holder_pid"] = None
                entry["acquired_at"] = None
        self._with_lock(_release)
        try:
            self._held.remove(gid)
        except ValueError:
            pass

    def release_all(self) -> None:
        """Best-effort cleanup on shutdown: release everything still held."""
        for gid in list(self._held):
            try:
                self.put(gid)
            except Exception:
                pass
