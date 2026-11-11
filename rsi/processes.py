"""Reap the task worker pool when its CLI receives an interruption."""
import concurrent.futures
import multiprocessing
import signal
import time
from contextlib import contextmanager


@contextmanager
def task_executor(max_workers, stop_grace=20):
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Task runner terminated')

    signal.signal(signal.SIGTERM, interrupted)
    executor = None
    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=multiprocessing.get_context('spawn'))
        yield executor
    except BaseException:
        # Repeated termination signals must not interrupt child reaping.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if executor is not None:
            # Python 3.11 has no public terminate_workers API. Capture the owned
            # Process handles before shutdown clears the executor's registry.
            workers = list((executor._processes or {}).values())
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            deadline = time.monotonic() + stop_grace
            for worker in workers:
                worker.join(max(0, deadline - time.monotonic()))
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
            for worker in workers:
                worker.join()
        raise
    finally:
        try:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            signal.signal(signal.SIGTERM, previous)
