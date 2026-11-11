"""Host-only native Test scoring after irreversible research closure.

No scoring result is published to Task/Meta history, query tools, validation
feedback or model selection.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import shutil
import time
import uuid

from benchmarks.naturebench_active.common import connect, sha256
from rsi.common import ROOT, write_json, record_path
from benchmarks.naturebench_active.scoring.process import NativeProcesses
from benchmarks.naturebench_active.adapters.multi_properties import TASK as PROPERTY_TASK, REACTION_INSTANCES

NATIVE_METRICS = {
    'MAT_05': {'sunset_1': ('MSE_IID_x1e3', 'MSE_OOD_x1e3')},
    'LIFE_12': {'phopt': ('sample_weighted_rmse', 'sample_weighted_r')},
    'LIFE_15': {
        instance: ('AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1-Score')
        for instance in ('human', 'mouse', 'cell_kyse70')
    },
}
from benchmarks.naturebench_active.adapters.plm_interact import TASK as PLM_TASK, INSTANCES as PLM_INSTANCES
NATIVE_METRICS[PLM_TASK] = {
    instance: ('AUPR', 'AUROC') if instance == 'mutation_effect' else ('AUPR',)
    for instance in PLM_INSTANCES
}
from benchmarks.naturebench_active.adapters.pending import (
    YEAST, POCKET, RNA, OPTIMIZATION, ECD, COVID, RNA_GENERATION, MOLECULE_GENERATION,
)
NATIVE_METRICS.update({
    YEAST: {'yeast_promoter': ('Pearson Score','Spearman Score')},
    POCKET: {'crossdock': ('Vina Score','QED','SA','Diversity')},
    RNA: {instance: ('f1_score','precision','recall') for instance in ('nrc','archiveii600','ts0')},
    OPTIMIZATION: {'drd2_qed': ('SR@0.4','SR@0.5','SR@0.6'), 'admet': ('SRA_overall',)},
    ECD: {'cmcds': ('Symbol-Acc','Number-RMSE','Position-RMSE')},
    COVID: {instance: tuple(f'{metric}_t{horizon}' for metric in ('Accuracy','MSE','WMSE','BrierScore','RPS')
                           for horizon in (1,3)) for instance in ('v1','v3')},
    RNA_GENERATION: {instance: ('mean_bit_score','n_families_evaluated') for instance in ('rna_families_628','rna_families_18')},
    MOLECULE_GENERATION: {instance: ('Success Rate','Scaffold Diversity') for instance in ('tanimoto_mol_a','tpsa','qed')},
})


def required_metrics(task, source_root, release):
    """Use explicit adapters or frozen native metadata for other output adapters.

    Run the unchanged native evaluator and preserve its complete metric object;
    the primary contract is a minimum completeness check, not a metric filter.
    """
    if task in NATIVE_METRICS:
        return NATIVE_METRICS[task]
    from benchmarks.naturebench_active.outputs import SUPPORTED
    if task not in SUPPORTED or task == 'LIFE_01':
        raise NotImplementedError('Native terminal scoring adapter pending: ' + task)
    metadata = Path(source_root) / task / 'metadata.json'
    if release['task_contract'].get(str(metadata.resolve())) != sha256(metadata):
        raise ValueError('Native metric metadata differs from the experiment release')
    from eval_service import _get_per_instance_primaries
    table, issues = _get_per_instance_primaries(metadata, task)
    if issues or not table:
        raise ValueError('Invalid native primary metric contract: ' + str(issues))
    return {instance: (info['metric'],) for instance, info in table.items()}


def frozen_result(run):
    """Validate terminal state and all scored artifacts before any truth access."""
    summary = json.loads((run/'final_test.json').read_text())
    if summary.get('status') != 'completed':
        raise ValueError('Require completed frozen final inference')
    with connect(run/'evidence/organizer/ledger.sqlite') as db:
        state = dict(db.execute('SELECT key,value FROM state'))
    if state.get('phase') != 'test_released':
        raise ValueError('Research must be irreversibly closed before Test scoring')
    final = run/'final'
    hashes = {p.name: sha256(p) for p in (final/'artifact').iterdir()}
    if hashes != summary['artifact_sha256'] or hashes != json.loads(state['artifact'])['files']:
        raise ValueError('Frozen selected model changed')
    if sha256(final/'final_predictions.jsonl') != summary['predictions_sha256']:
        raise ValueError('Frozen final predictions changed')
    outputs = summary.get('native_output', {}).get('files', {})
    if not outputs:
        raise ValueError('Require frozen native output manifest')
    actual_files={str((Path(directory)/name).relative_to(final/'final_output'))
                  for directory,_,names in os.walk(final/'final_output') for name in names}
    if actual_files!=set(outputs):
        raise ValueError('Frozen native output membership changed')
    for relative, expected in outputs.items():
        path = (final/'final_output'/relative).resolve()
        if not path.is_relative_to((final/'final_output').resolve()) or sha256(path) != expected:
            raise ValueError('Frozen native final output changed')
    return summary


def terminal_score(run, source_root, *, python=sys.executable, seconds=None,
                   cuda_visible_devices=None, cpu_threads=None, score_root=None,
                   rna_workers=4, rna_threads=6, rna_memory_mib=None,
                   rna_family_seconds=3600, rna_cache_root=None, rna_chunk_size=64,
                   pocket_workers=None, pocket_docking_seconds=None):
    """Score with the caller's GPU visibility unless explicitly overridden.

    The organizer must allocate devices before calling. An empty explicit
    visibility selects CPU execution; None preserves the scheduler environment.
    Native scoring has no outer wall-clock limit unless explicitly configured.
    """
    if cuda_visible_devices is not None and not isinstance(cuda_visible_devices, str):
        raise ValueError('cuda_visible_devices must be a string or None')
    explicit_threads = cpu_threads is not None
    if cpu_threads is None:
        cpu_threads = len(os.sched_getaffinity(0))
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    if seconds is not None and (type(seconds) not in (int, float)
                                or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('Terminal scoring timeout must be positive and finite, or null')
    if pocket_docking_seconds is not None:
        if (type(pocket_docking_seconds) is not int or pocket_docking_seconds < 3000
                or (seconds is not None and seconds <= pocket_docking_seconds)):
            raise ValueError('Outer terminal budget must exceed the revised pocket docking budget (>= 3000)')
    run, source_root = Path(run).resolve(), Path(source_root).resolve()
    score_root = Path(score_root).resolve() if score_root is not None else run
    score_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OUTPUT_DIR=str(run/'final/final_output'),
                       PATH=str(Path(python).absolute().parent)+os.pathsep+os.environ.get('PATH',''),
                       OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads),
                       OPENBLAS_NUM_THREADS=str(cpu_threads))
    if cuda_visible_devices is not None:
        environment['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    resources = dict(cuda_visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
                     cpu_threads=cpu_threads, cpu_affinity=sorted(os.sched_getaffinity(0)),
                     python=str(python))
    with (score_root/'terminal_score.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = frozen_result(run)
        task = summary['task']
        evaluation_protocol = None
        if task == PROPERTY_TASK:
            from benchmarks.naturebench_active.scoring.reaction_terminal import validate_submission
            evaluation_protocol = validate_submission(run, summary)
        if task == 'LIFE_16':
            from benchmarks.naturebench_active.scoring.terminal_noncoding import terminal_score as noncoding_score
            return noncoding_score(run, source_root)
        source = source_root/task/'evaluation/evaluator.py'
        source_hash = sha256(source)
        release = json.loads((run/'release.json').read_text())
        if release['task_contract'].get(str(source.resolve())) != source_hash:
            raise ValueError('Native evaluator differs from the experiment release')
        required = required_metrics(task, source_root, release)
        native_binary={}
        if task == RNA_GENERATION:
            for name in ('cmalign', 'cmfetch'):
                executable = shutil.which(name, path=environment['PATH'])
                if executable is None:
                    raise RuntimeError('Missing native RNA executable: '+name)
                native_binary[str(Path(executable).resolve())] = sha256(Path(executable))
        if task==POCKET:
            binary=source.parent/'qvina02'
            native_binary[str(binary)]=sha256(binary)
            if release['task_contract'].get(str(binary.resolve()))!=native_binary[str(binary)]:
                raise ValueError('QuickVina binary differs from the experiment release')
        output = score_root/'terminal_score.json'
        adapter_paths = [Path(__file__), Path(__file__).with_name('native_runner.py'),
                         Path(__file__).with_name('process.py')]
        if task == PROPERTY_TASK:
            adapter_paths.extend([Path(__file__).with_name('reaction_terminal.py'),
                                  Path(__file__).parents[1]/'adapters/multi_properties.py'])
        if task == RNA_GENERATION:
            adapter_paths.append(Path(__file__).with_name('rna_terminal.py'))
            resources['rna'] = dict(workers=rna_workers, threads_per_family=rna_threads,
                                    memory_mib=rna_memory_mib, family_seconds=rna_family_seconds,
                                    chunk_size=rna_chunk_size)
        if task == POCKET:
            adapter_paths.extend([Path(__file__).with_name('pocket_terminal.py'),
                                  Path(__file__).with_name('pocket_budget.py')])
            resources['pocket'] = dict(workers=pocket_workers if pocket_workers is not None else cpu_threads,
                                      docking_seconds=pocket_docking_seconds)
            evaluation_protocol = dict(
                policy='pocket_unlimited_docking_v1' if pocket_docking_seconds is None else 'pocket_extended_docking_v1',
                native_docking_seconds=3000, docking_seconds=pocket_docking_seconds, outer_seconds=seconds,
                cpu_per_docking=1, exhaustiveness=16, single_docking_seconds=120)
        adapters = {path.name: sha256(path) for path in adapter_paths}
        if output.exists():
            saved = json.loads(output.read_text())
            if (saved['artifact_sha256'] != summary['artifact_sha256'] or
                    saved['predictions_sha256'] != summary['predictions_sha256'] or
                    saved['native_output_sha256'] != summary['native_output']['files'] or
                    saved['evaluator_sha256'] != source_hash or
                    saved.get('native_binary_sha256',{}) != native_binary or
                    saved.get('evaluation_protocol') != evaluation_protocol):
                raise ValueError('Terminal score belongs to another frozen result')
            if saved.get('adapter_sha256') != adapters:
                raise ValueError('Cached terminal score uses another adapter version; use a separate score_root')
            if task == RNA_GENERATION and saved.get('execution_resources', {}).get('rna') != resources['rna']:
                raise ValueError('Cached terminal score uses different RNA resources; use a separate score_root')
            if task == POCKET and saved.get('execution_resources', {}).get('pocket') != resources['pocket']:
                raise ValueError('Cached terminal score uses different pocket resources; use a separate score_root')
            if cuda_visible_devices is not None or explicit_threads:
                cached_resources = saved.get('execution_resources', {})
                if any(cached_resources.get(key) != resources[key]
                       for key in ('cuda_visible_devices', 'cpu_threads')):
                    raise ValueError('Cached terminal score uses different or unknown resources; '
                                     'use a separate score run for a device comparison')
            if any(not Path(path).is_file() or sha256(Path(path)) != expected
                   for path, expected in saved.get('test_truth_sha256', {}).items()):
                raise ValueError('Native Test truth changed since the cached score')
            return saved
        # Native truth is first touched here, after every terminal-state check.
        truth = {str(p): sha256(p) for p in sorted((source.parent/'ground_truth').rglob('*')) if p.is_file()
                 and not (task == PROPERTY_TASK and p.parent.name in REACTION_INSTANCES
                          and p.name != 'ground_truth_split_0.csv')}
        if not truth:
            raise ValueError('Native Test truth is missing')
        directory = score_root/'terminal_native'; directory.mkdir(exist_ok=True)
        if task == PROPERTY_TASK:
            environment['RSI_NATIVE_OPTIONS'] = json.dumps(dict(task=task))
        if task == RNA_GENERATION:
            environment['RSI_NATIVE_OPTIONS'] = json.dumps(dict(task=task, rna=dict(
                cache_root=str(Path(rna_cache_root).resolve() if rna_cache_root else directory/'rna_cache'),
                progress_path=str(directory/'rna_progress.json'), cpu_budget=cpu_threads,
                threads=rna_threads, workers=rna_workers, memory_mib=rna_memory_mib,
                family_seconds=rna_family_seconds, chunk_size=rna_chunk_size)))
        execution_source=source
        if task==POCKET:
            environment['RSI_NATIVE_OPTIONS'] = json.dumps(dict(task=task, pocket=dict(
                progress_path=str(directory/'pocket_progress.json'), cpu_budget=cpu_threads,
                workers=pocket_workers, docking_seconds=pocket_docking_seconds)))
            # The published QuickVina file has no executable bit. Execute an
            # identical private copy beside an unchanged evaluator copy; the
            # original source package and scoring implementation stay intact.
            staged=directory/'evaluation';staged.mkdir(exist_ok=True)
            execution_source=staged/'evaluator.py'
            shutil.copyfile(source,execution_source)
            shutil.copyfile(binary,staged/'qvina02');(staged/'qvina02').chmod(0o755)
            truth_link=staged/'ground_truth'
            if not truth_link.exists():truth_link.symlink_to(source.parent/'ground_truth',target_is_directory=True)
            if truth_link.resolve()!=(source.parent/'ground_truth').resolve():
                raise ValueError('Staged receptor path changed')
        started = time.time()
        if (directory/'evaluator.log').exists():
            attempts = directory/'previous_attempts'; attempts.mkdir(exist_ok=True)
            attempt = uuid.uuid4().hex
            shutil.copyfile(directory/'evaluator.log', attempts/(attempt+'.log'))
            if (directory/'status.json').exists():
                shutil.copyfile(directory/'status.json', attempts/(attempt+'.json'))
        status = dict(status='running', run=str(run), task=task, started_at=started,
                      time_limit_seconds=seconds,
                      execution_resources=resources, adapter_sha256=adapters)
        write_json(directory/'status.json', status)
        try:
            with NativeProcesses().managed() as processes, (directory/'evaluator.log').open('w') as log:
                execution = processes.run(
                    [str(python), str(Path(__file__).with_name('native_runner.py')), str(execution_source)],
                    cwd=directory, env=environment, stdout=log, stderr=log, timeout=seconds)
        except BaseException as exc:
            write_json(directory/'status.json', dict(status, status='failed', finished_at=time.time(),
                                                     error=f'{type(exc).__name__}: {exc}'))
            raise
        try:
            marker = '===EVAL_RESULT_JSON===\n'
            text = (directory/'evaluator.log').read_text()
            if marker not in text:
                raise ValueError('Native evaluator returned no score')
            metrics = json.loads(text.rsplit(marker, 1)[1])
            for instance, names in required.items():
                for name in names:
                    value = metrics.get(instance, {}).get(name)
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError('Native terminal score is missing or nonfinite: '+instance+'/'+name)
            if task==RNA_GENERATION:
                for instance in NATIVE_METRICS[task]:
                    expected=sum(name.startswith(instance+'/') and name.endswith(('.fa','.fasta'))
                                 for name in summary['native_output']['files'])
                    if metrics[instance]['n_families_evaluated']!=expected:
                        raise ValueError('Native RNA scoring omitted a frozen target family')
            if (frozen_result(run) != summary or sha256(source) != source_hash or
                    (task == PROPERTY_TASK and validate_submission(run, summary) != evaluation_protocol)):
                raise ValueError('Frozen inference or evaluator changed during scoring')
            if any(sha256(Path(path)) != expected for path, expected in truth.items()):
                raise ValueError('Native Test truth changed during scoring')
            if any(sha256(Path(path)) != expected for path,expected in native_binary.items()):
                raise ValueError('Native scoring binary changed during scoring')
            if any(sha256(path) != adapters[path.name] for path in adapter_paths):
                raise ValueError('Native scoring adapter changed during scoring')
        except BaseException as exc:
            write_json(directory/'status.json', dict(status, status='failed', finished_at=time.time(),
                                                     error=f'{type(exc).__name__}: {exc}'))
            raise
        report = dict(task=task, phase='terminal_only', metrics=metrics,
                      time_limit_seconds=seconds,
                      selected_artifact=summary['artifact'], artifact_sha256=summary['artifact_sha256'],
                      predictions_sha256=summary['predictions_sha256'],
                      native_output_sha256=summary['native_output']['files'], evaluator_sha256=source_hash,
                      native_binary_sha256=native_binary,
                      adapter_sha256=adapters, execution=execution,
                      execution_resources=resources,
                      test_truth_sha256=truth, test_truth_read=True,
                      test_score_computed=True, feedback_to_research=False)
        if evaluation_protocol is not None:
            report['evaluation_protocol'] = evaluation_protocol
        if task == POCKET:
            report['coverage_complete'] = metrics['crossdock']['coverage']['complete']
            report['coverage'] = metrics['crossdock']['coverage']
            report['docking_progress_sha256'] = sha256(directory/'pocket_progress.json')
            report['ligand_preparation'] = json.loads(
                (directory/'pocket_progress.json').read_text())['identity']['ligand_preparation']
        elif task == RNA_GENERATION:
            report['coverage_complete'] = all(row['complete'] for row in metrics.values())
        write_json(output, report)
        write_json(directory/'status.json', dict(status, status='completed', finished_at=time.time()))
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=record_path)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--seconds',type=float,default=None,
                        help='Optional native scoring timeout; omitted means no outer wall-clock limit')
    parser.add_argument('--cuda-visible-devices', default=None,
                        help='Allocated CUDA devices; inherit scheduler visibility by default, or pass an empty string for CPU')
    parser.add_argument('--cpu-threads', type=int, default=None,
                        help='Native CPU budget; defaults to the allocated process affinity')
    parser.add_argument('--score-root', help='Separate host-only result directory; preserves earlier scores', type=record_path)
    parser.add_argument('--rna-workers', type=int, default=4)
    parser.add_argument('--rna-threads', type=int, default=6)
    parser.add_argument('--rna-memory-mib', type=int)
    parser.add_argument('--rna-family-seconds', type=float, default=3600)
    parser.add_argument('--rna-cache-root', help='Organizer-owned shared family checkpoints')
    parser.add_argument('--rna-chunk-size', type=int, default=64,
                        help='Maximum sequences per RNA scheduling unit; 0 disables partitioning')
    parser.add_argument('--pocket-docking-seconds', type=int,
                        help='Aggregate docking limit; omitted means unlimited; any outer limit must exceed it')
    parser.add_argument('--pocket-workers', type=int,
                        help='Concurrent native docking trials; defaults to the allocated CPU budget')
    args = parser.parse_args()
    print(json.dumps(terminal_score(args.run, args.source_root, python=args.python,seconds=args.seconds,
                                   cuda_visible_devices=args.cuda_visible_devices, cpu_threads=args.cpu_threads,
                                   score_root=args.score_root, rna_workers=args.rna_workers,
                                   rna_threads=args.rna_threads, rna_memory_mib=args.rna_memory_mib,
                                   rna_family_seconds=args.rna_family_seconds, rna_cache_root=args.rna_cache_root,
                                   rna_chunk_size=args.rna_chunk_size, pocket_workers=args.pocket_workers, pocket_docking_seconds=args.pocket_docking_seconds)))
