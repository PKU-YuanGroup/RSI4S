"""Owned native subprocesses, with timeout and interruption cleanup."""
from contextlib import contextmanager
import math
import os
import signal
import subprocess
import threading
import time


class NativeProcesses:
    """Keep process groups alive only for the lifetime of their scoring owner."""

    def __init__(self):
        self._lock = threading.Lock()
        self._children = set()
        self._stopped = False

    @property
    def stopped(self):
        return self._stopped

    @staticmethod
    def _signal(child, signum):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    def stop(self):
        with self._lock:
            self._stopped = True
            children = list(self._children)
        for child in children:
            self._signal(child, signal.SIGTERM)
        # A native executable can leave descendants after its leader exits.
        # Kill the owned group as well as reaping the direct child.
        for child in children:
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            self._signal(child, signal.SIGKILL)
            child.wait()

    def run(self, command, *, timeout, **kwargs):
        if timeout is not None and (type(timeout) not in (int, float)
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Native subprocess timeout must be positive and finite, or null')
        with self._lock:
            if self._stopped:
                raise InterruptedError('Native scoring was interrupted')
            child = subprocess.Popen(command, start_new_session=True, **kwargs)
            self._children.add(child)
        started = time.monotonic()
        try:
            returncode = child.wait(timeout=timeout)
            if self._stopped:
                raise InterruptedError('Native scoring was interrupted')
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            return dict(returncode=returncode, seconds=time.monotonic()-started,
                        time_limit_seconds=timeout, pid=child.pid)
        except BaseException:
            # Give a cooperating native wrapper time to cancel grandchildren
            # which own separate process groups (e.g. parallel Infernal jobs).
            self._signal(child, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            raise
        finally:
            self._signal(child, signal.SIGKILL)
            child.wait()
            with self._lock:
                self._children.discard(child)

    @contextmanager
    def managed(self):
        """Forward terminal cancellation before an outer owner escalates to KILL."""
        previous = None
        if threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGTERM)

            def interrupted(signum, frame):
                raise InterruptedError('Native scoring received SIGTERM')

            signal.signal(signal.SIGTERM, interrupted)
        try:
            yield self
        finally:
            try:
                if previous is not None:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                self.stop()
            finally:
                if previous is not None:
                    signal.signal(signal.SIGTERM, previous)
