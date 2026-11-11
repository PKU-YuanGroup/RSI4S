"""Bounded host transport to one persistent simulator per experiment."""
import json
import os
import selectors
import signal
import subprocess
import threading
import time
from pathlib import Path

from benchmarks.simulator.source import verify_source


class EpisodeClient:
    def __init__(self, task, directory, data_root=None, python=None):
        root, python = verify_source(task, data_root, python)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory/'simulator.log').open('ab')
        self.lock = threading.RLock()
        self.buffer = b''
        self.failed = False
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '',
               'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
               'OMP_NUM_THREADS': '1', 'NUMBA_DISABLE_JIT': '1', 'MPLBACKEND': 'Agg'}
        self.proc = subprocess.Popen([python, '-I', str(Path(__file__).with_name('worker.py')),
            task, str(root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env=env, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            self.current = self._receive(60)['result']
        except BaseException:
            self.close()
            raise

    def _receive(self, seconds):
        deadline = time.monotonic() + seconds
        while b'\n' not in self.buffer:
            if not self.selector.select(max(0, deadline-time.monotonic())):
                raise TimeoutError('Simulator deadline exceeded; episode cannot restart')
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('Simulator exited; see simulator.log')
            self.buffer += chunk
            if len(self.buffer) > 8 * 1024 * 1024:
                raise ValueError('Simulator response exceeds 8 MiB')
        line, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(line)

    def call(self, request, seconds):
        with self.lock:
            if self.failed or self.proc.poll() is not None:
                raise RuntimeError('Simulator episode unavailable; automatic reset is forbidden')
            if seconds <= 0:
                raise TimeoutError('No simulator action time remains')
            encoded = json.dumps(request, allow_nan=False).encode() + b'\n'
            if len(encoded) > 4096:
                raise ValueError('Simulator action arguments exceed 4096 bytes')
            try:
                self.proc.stdin.write(encoded)
                self.proc.stdin.flush()
                reply = self._receive(seconds)
            except BaseException:
                self.failed = True
                self.close()
                raise
            self.current = reply['result']
            return reply

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout, self.log):
            stream.close()
        self.selector.close()
