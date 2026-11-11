"""Observe native docking attempts and the exact subset used by its aggregate.

The original top-K selection, per-docking settings and averaging remain in the
upstream evaluator. The aggregate deadline is disabled by default and recorded.
Repeated identical SMILES occupy distinct
trial slots, as in that evaluator; stochastic docking trials are not deduplicated.
Meeko keeps macrocycles closed and rigid because this QuickVina version cannot
apply the CG/G ring-closure constraints used by flexible macrocycle docking.
"""
import builtins
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import types
import uuid

from benchmarks.naturebench_active.common import sha256
from benchmarks.naturebench_active.scoring.process import NativeProcesses
from rsi.common import Journal, write_json


LIGAND_PREPARATION = dict(policy='quickvina2_rigid_macrocycles_v1', rigid_macrocycles=True)


def quickvina_preparation():
    """Keep ring connectivity and the native RDKit conformer for legacy QVina.

    Do not relabel CG atoms or remove G atoms from an already opened ring:
    that would discard the ring-closure constraints without restoring the ring.
    """
    from meeko import MoleculePreparation
    return MoleculePreparation(rigid_macrocycles=True)


def run_evaluation(native, *, progress_path, cpu_budget, workers=None, docking_seconds=None):
    if type(cpu_budget) is not int or not 1 <= cpu_budget <= len(os.sched_getaffinity(0)):
        raise ValueError('Pocket CPU budget must fit the allocated affinity')
    if workers is not None and (type(workers) is not int or not 1 <= workers <= cpu_budget):
        raise ValueError('Pocket workers must fit the allocated CPU budget')
    progress_path = Path(progress_path)
    directory = progress_path.parent/'docking_attempts'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    journal = Journal(directory/'events.jsonl')
    lock, local = threading.Lock(), threading.local()
    identity = dict(evaluator_sha256=sha256(Path(native.__file__)),
                    adapter_sha256=sha256(Path(__file__)),
                    qvina_sha256=sha256(Path(native.QVINA_BIN)),
                    ligand_preparation=dict(LIGAND_PREPARATION),
                    predictions_sha256={instance: sha256(Path(native.OUTPUT_DIR)/instance/native.OUTPUT_FILE)
                                        for instance in native.INSTANCES})
    from .pocket_budget import with_budget
    with_budget(native.run_evaluation, docking_seconds)  # Validate before any trial.
    identity['budget_protocol'] = dict(
        policy='pocket_unlimited_docking_v1' if docking_seconds is None else 'pocket_extended_docking_v1',
        native_seconds=3000, docking_seconds=docking_seconds,
        single_docking_seconds=120, cpu_per_docking=1, exhaustiveness=16)
    state = dict(status='running', started_at=time.time(), identity=identity, cpu_budget=cpu_budget,
                 attempts_directory=str(directory), attempts={}, workers=[])

    last_publish = 0.

    def publish(force=False):
        # Caller holds the lock, so callbacks cannot race the atomic rename.
        # Per-attempt receipts and the append-only journal remain synchronous.
        # Rewriting all 910 attempts at every event made submission quadratic.
        nonlocal last_publish
        now = time.monotonic()
        if not force and now-last_publish < 1.:
            return
        write_json(progress_path, state)
        last_publish = now

    def update(index, event, **fields):
        with lock:
            state['attempts'][index].update(fields)
            journal.add(event, attempt=index, **fields)
            write_json(directory/index/'attempt.json', state['attempts'][index])
            publish()

    processes = NativeProcesses()

    def recorded_subprocess(command, *, capture_output, text, timeout):
        # Only the native compute_vina_score function sees this import adapter.
        # Meeko/RDKit and the rest of the process keep their original imports.
        if not capture_output or not text:
            raise ValueError('Unexpected upstream docking subprocess contract')
        index = local.attempt
        work = directory/index
        ligand = Path(command[command.index('--ligand')+1])
        receptor = Path(command[command.index('--receptor')+1])
        shutil.copyfile(ligand, work/'ligand.pdbqt')
        invocation = dict(command=list(command), ligand_sha256=sha256(ligand),
                          receptor_sha256=sha256(receptor), timeout_seconds=timeout,
                          started_at=time.time(), status='running')
        update(index, 'native_started', native_process=invocation)
        try:
            with (work/'stdout.log').open('w') as stdout, (work/'stderr.log').open('w') as stderr:
                try:
                    execution = processes.run(command, timeout=timeout, stdout=stdout, stderr=stderr)
                    returncode = execution['returncode']
                except subprocess.CalledProcessError as exc:
                    returncode = exc.returncode
                    execution = dict(returncode=returncode)
            invocation.update(status='completed', finished_at=time.time(), execution=execution,
                              stdout_sha256=sha256(work/'stdout.log'), stderr_sha256=sha256(work/'stderr.log'))
            output_path = Path(command[command.index('--out')+1])
            if output_path.is_file():
                shutil.copyfile(output_path, work/'docked.pdbqt')
                invocation['docked_sha256'] = sha256(work/'docked.pdbqt')
            update(index, 'native_completed', native_process=invocation)
            return subprocess.CompletedProcess(command, returncode,
                (work/'stdout.log').read_text(errors='replace'), (work/'stderr.log').read_text(errors='replace'))
        except BaseException as exc:
            invocation.update(status='failed', finished_at=time.time(), error=f'{type(exc).__name__}: {exc}')
            update(index, 'native_failed', native_process=invocation)
            raise

    original_executor, original_compute = native.ThreadPoolExecutor, native.compute_vina_score
    proxy = types.SimpleNamespace(run=recorded_subprocess, TimeoutExpired=subprocess.TimeoutExpired)

    def native_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'subprocess' and level == 0:
            return proxy
        if name == 'meeko' and level == 0:
            module = builtins.__import__(name, globals, locals, fromlist, level)
            # Scope the preparation policy to this evaluator function. Other
            # callers retain Meeko's defaults, including concurrent callers.
            return types.SimpleNamespace(**{**vars(module), 'MoleculePreparation': quickvina_preparation})
        return builtins.__import__(name, globals, locals, fromlist, level)

    globals_copy = {**original_compute.__globals__, '__builtins__': {**vars(builtins), '__import__': native_import}}
    observed_compute = types.FunctionType(original_compute.__code__, globals_copy,
                                         original_compute.__name__, original_compute.__defaults__, original_compute.__closure__)

    class ObservedExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers, **kwargs):
            # The upstream 16-worker ceiling is a scheduling default. Each
            # unchanged qvina invocation still uses exactly --cpu 1.
            allocated = workers if workers is not None else cpu_budget
            with lock:
                state['workers'].append(dict(requested=max_workers, allocated=allocated))
                publish(force=True)
            super().__init__(max_workers=allocated, **kwargs)

        def submit(self, function, *args, **kwargs):
            pocket_index, smiles, pocket_id, center = args[0]
            with lock:
                index = f'{len(state["attempts"]):05d}'
                (directory/index).mkdir()
                state['attempts'][index] = dict(pocket_index=pocket_index, pocket_id=pocket_id,
                    smiles=smiles, pocket_center=center, status='submitted', submitted_at=time.time(),
                    collected=False)
                journal.add('submitted', attempt=index, **state['attempts'][index])
                write_json(directory/index/'attempt.json', state['attempts'][index])
                publish()

            def execute():
                local.attempt = index
                update(index, 'started', status='running', started_at=time.time())
                try:
                    result = function(*args, **kwargs)
                    update(index, 'completed', status='completed', finished_at=time.time(), vina=result[1])
                    return result
                except BaseException as exc:
                    update(index, 'failed', status='failed', finished_at=time.time(), error=f'{type(exc).__name__}: {exc}')
                    raise

            future = super().submit(execute)
            original_result = future.result

            def collected_result(*args, **kwargs):
                # Only the native acceptance loop calls this. A task completing
                # after the budget cutoff must not become an included score.
                try:
                    return original_result(*args, **kwargs)
                finally:
                    update(index, 'collected', collected=True, collected_at=time.time())
            future.result = collected_result
            def on_done(future):
                if future.cancelled():
                    update(index, 'cancelled', status='cancelled', finished_at=time.time())
            future.add_done_callback(on_done)
            return future

        def __exit__(self, exception_type, value, traceback):
            if exception_type is not None:
                processes.stop()
                self.shutdown(wait=True, cancel_futures=True)
                return False
            return super().__exit__(exception_type, value, traceback)

    with lock:
        publish(force=True)
    native.ThreadPoolExecutor = ObservedExecutor
    native.compute_vina_score = observed_compute
    try:
        with processes.managed():
            runner = with_budget(native.run_evaluation, docking_seconds)
            metrics = runner()
            if processes.stopped:
                raise InterruptedError('Pocket scoring interrupted; attempt evidence is preserved')
        attempts = list(state['attempts'].values())
        successful = [a for a in attempts if isinstance(a.get('vina'), (int, float)) and math.isfinite(a['vina'])]
        included = [a for a in successful if a['collected']]
        best = {}
        for attempt in included:
            pocket = attempt['pocket_id']
            best[pocket] = min(best.get(pocket, math.inf), attempt['vina'])
        coverage = dict(expected_pockets=native.NUM_POCKETS, submitted=len(attempts),
            started=sum('started_at' in a for a in attempts),
            completed=sum(a['status']=='completed' for a in attempts),
            cancelled=sum(a['status']=='cancelled' for a in attempts),
            failed=sum(a['status']=='failed' for a in attempts),
            collected=sum(a['collected'] for a in attempts),
            successful_completed=len(successful), successful_collected=len(included),
            successful_pockets=len(best), vina_denominator=len(best),
            all_submitted_collected=all(a['collected'] for a in attempts),
            all_pockets_scored=len(best)==native.NUM_POCKETS,
            all_docking_attempts_succeeded=len(included)==len(attempts),
            per_pocket={f'pocket_{i:03d}': dict(
                submitted=sum(a['pocket_index']==i for a in attempts),
                collected=sum(a['pocket_index']==i and a['collected'] for a in attempts),
                successful=sum(a['pocket_index']==i for a in included),
                best_vina=best.get(f'pocket_{i:03d}')) for i in range(native.NUM_POCKETS)})
        coverage['complete'] = coverage['all_submitted_collected'] and coverage['all_pockets_scored']
        expected_mean = float(native.np.mean([best[f'pocket_{i:03d}'] for i in range(native.NUM_POCKETS)
                                             if f'pocket_{i:03d}' in best])) if best else None
        actual_mean = metrics.get('crossdock', {}).get('Vina Score')
        if actual_mean != expected_mean:
            raise ValueError('Native Vina aggregate disagrees with collected-trial evidence')
        metrics['crossdock']['coverage'] = coverage
        state.update(status='completed', finished_at=time.time(), coverage=coverage, metrics=metrics)
        with lock:
            publish(force=True)
        return metrics
    except BaseException as exc:
        with lock:
            state.update(status='failed', finished_at=time.time(), error=f'{type(exc).__name__}: {exc}')
            publish(force=True)
        raise
    finally:
        native.ThreadPoolExecutor, native.compute_vina_score = original_executor, original_compute
