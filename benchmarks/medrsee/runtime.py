"""Active-evidence lifecycle and terminal, frozen test evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

from rsi.common import ROOT, write_json, digest, record_path
from rsi.host_io import run_io
from benchmarks.runtime import EvidenceEvaluation
from rsi.sandbox import Sandbox
from .evidence import Evidence, export_batch
from .membership import build_index, fingerprint, connect
from .source import DATA, NativeReader, COUNTS
from . import submission
from .submission import declaration


class MedRSEEEvaluation(EvidenceEvaluation):
    selection_module = submission

    def __init__(self, task, data_dir, run_dir, seconds, gpu, wall_seconds=None, resource_limits=None,
                 raw_data=DATA, journal=None, cache_root=None, test_cache_root=None, index_root=None):
        index_root = record_path(index_root, area='runs') if index_root else ROOT/'runs/medrsee-indices'
        cache = index_root/fingerprint()/f'{task}.sqlite'
        started, existed = time.monotonic(), cache.exists()
        if journal:
            journal.add('membership_index_start', task=task, path=str(cache), existed=existed)
        self.index = build_index(task,cache)
        if journal:
            journal.add('membership_index_ready', task=task, path=str(cache), existed_at_start=existed,
                        elapsed_seconds=time.monotonic()-started)
        self.test_inputs = None
        if task == 'ENV_02':
            from .test_inputs import PreparedTestInputs, resolve_cache_root
            test_cache_root = resolve_cache_root(raw_data, test_cache_root)
            self.test_inputs = PreparedTestInputs(test_cache_root, task, self.index, raw_data)
            prepared = self.test_inputs.verify()
            # This is before the research clock/API start. Do not pass the cache
            # path or its manifest to either Agent's evidence/context.
            if journal:
                journal.add('test_inputs_prepared', task=task, key=self.test_inputs.key,
                            samples=prepared['identity']['count'], labels_included=False,
                            exposed_to_agents=False)
        self.evidence = Evidence(task,data_dir,Path(run_dir)/'evidence',self.index,raw_data,journal,cache_root)
        self.cache_root = str(cache_root) if cache_root else None
        self.python = __import__('sys').executable
        self.raw_data = Path(raw_data)
        super().__init__(task,data_dir,run_dir,seconds,gpu,wall_seconds,resource_limits)

    def scorer_access(self, task_dir, output_dir):
        reference = Path(output_dir).parents[1]/'validation_reference.sqlite'
        return ([ROOT/'benchmarks',self.index,reference,self.raw_data],
                dict(MEDRSEE_TASK=self.task, MEDRSEE_INDEX=str(self.index),
                     MEDRSEE_POOL=str(reference), MEDRSEE_DATA=str(self.raw_data)))

    def record_execution(self, current, history):
        from .training import execution_evidence
        self.host_execution = execution_evidence(current, history)

    def validation_cohort(self, *, deadline=None):
        return run_io('cohort', deadline if deadline is not None else time.monotonic()+self.remaining(),
                      pool=str(self.evidence.pool))

    def evaluate(self, round_dir, timeout=None):
        """Score only outputs reproduced by the artifact's saved inference entry."""
        start = time.monotonic()
        seconds = min(self.remaining(), timeout) if timeout is not None else self.remaining()
        deadline = start + seconds
        round_dir = Path(round_dir)
        snapshot = round_dir/'validation_reference.sqlite'
        cohort = run_io('pool_snapshot', deadline, source=str(self.evidence.pool), target=str(snapshot))
        source = round_dir/'workspace'
        files = run_io('hashes', deadline, source=str(source))
        frozen = declaration(self.task, dict(workspace=str(source), files=files))
        if not cohort['samples']:
            raise ValueError('Declare a nonempty authorized validation set before evaluation')
        audit = run_io('audit_training', deadline, task=self.task, source=str(source), pool=str(snapshot),
                       artifacts=getattr(self, 'historical_artifacts', []), source_files=files)
        execution = getattr(self, 'host_execution', None)
        if execution is not None:
            report = round_dir/'host_execution.json'
            write_json(report, dict(artifact_files_sha256=digest(files), **execution))
            report.chmod(0o444)
            import hashlib
            audit['host_execution'] = dict(path=str(report), sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                runs=len(execution['runs']), commands=sum(len(r['actions']) for r in execution['runs']))
        replay = round_dir/'replay'
        replay.mkdir()
        workspace, problem = replay/'workspace', replay/'problem'
        shutil.copyfile(snapshot, replay/'validation_reference.sqlite')
        before = run_io('copy', deadline, source=str(source), target=str(workspace),
                        ignore=['output', '.tmp', '__pycache__'])
        run_io('inputs', deadline, task=self.task, index=str(self.index), pool=str(snapshot),
               raw_data=str(self.raw_data), problem=str(problem), cache_root=self.cache_root,
               deadline=deadline)
        split = 'V' if self.task in ('RS_02', 'RS_01', 'MED_07') else 'validation'
        output = workspace/'output/validation'
        output.mkdir(parents=True)
        science = Sandbox(self.python, problem, workspace, replay/'inference_logs', self.gpu,
            env_extra=dict(MEDRSEE_PHASE='validation', MEDRSEE_SPLIT=split, MEDRSEE_TASK=self.task,
                           OUTPUT_DIR=str(output)), limits=self.resource_limits, frozen_workspace=True)
        entry = frozen.get('inference_entry', 'predict.py')
        outcome = science.run(f'import runpy;runpy.run_path({entry!r},run_name="__main__")',
                              deadline-time.monotonic())
        write_json(replay/'inference_result.json', outcome)
        if outcome['returncode']:
            raise RuntimeError('Saved validation inference failed: ' + outcome['stdout'][-1500:])
        after = run_io('hashes', deadline, source=str(workspace))
        verify_frozen(before, after, frozen, source)
        shutil.copyfile(source/'frozen_model.json', output/'frozen_model.json')
        workspace.chmod(0o755)  # The independent scorer UID must traverse this parent.
        result = super().evaluate(replay, deadline-time.monotonic())
        self.validation_seconds = [*getattr(self, 'validation_seconds', []), time.monotonic()-start]
        result.update(validation_cohort=cohort, training_audit=audit,
                      host_validation_seconds=time.monotonic()-start,
                      inference_verification=dict(status='completed', source_files_sha256=digest(files),
                                                  log=outcome['log'], seconds=outcome['seconds'],
                                                  filesystem_policy='read_only_model_and_program',
                                                  in_memory_fitting_excluded=False))
        return result

    def close(self):
        self.closed = True
        super().close()
        self.evidence.close()

    def finalize(self, runs, python, seconds=3600):
        """No agent/model call follows this point. Only saved inference code runs."""
        output = self.directory/'final_test.json'
        with output.open('x') as stream:
            json.dump({'status':'started','task':self.task},stream)
        start = time.monotonic()
        submission = {}
        phase = 'select_submission'
        executions = {}
        try:
            artifact, selection = self.select_submission(runs, getattr(self, 'meta_selection', None))
            write_json(self.directory/'final_selection.json', selection)
            submission['selection'] = selection
            if artifact is None:
                raise ValueError(selection['error'])
            submission['artifact'] = artifact['id']
            source = Path(artifact['workspace'])
            if run_io('hashes', start+seconds, source=str(source)) != artifact['files']:
                raise ValueError('Saved research artifact changed before final freeze')
            frozen = declaration(self.task, artifact)
            submission['method'] = frozen['method']
            entry = frozen.get('inference_entry','predict.py')
            checkpoints = frozen.get('checkpoints')
            base = self.directory/'final'
            base.mkdir()
            write_json(base/'freeze.json',dict(artifact=artifact['id'],files=artifact['files'],
                declaration=frozen,selection=selection,annotation_status=self.evidence.status(),frozen_at=time.time()))
            # The final view adds only T inputs. Previously acquired evidence is
            # already present in the immutable artifact if the method needs it.
            problem = base/'problem'
            phase = 'export_test_inputs'
            if getattr(self, 'test_inputs', None) is not None:
                problem.mkdir(parents=True)
                shutil.copyfile(Path(__file__).with_name('reader.py'), problem/'reader.py')
                phase = 'materialize_prepared_test_inputs'
                cache_state = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.test_inputs).items()}
                batches = run_io('prepared_inputs', start+seconds, state=cache_state,
                                 destination=str(problem/'data'), deadline=start+seconds)
                write_json(problem/'data/batches.json', batches)
                executions['input_preparation'] = dict(mode='prepared_cache', key=self.test_inputs.key,
                    seconds=time.monotonic()-start, batches=len(batches))
            else:
                run_io('inputs', start+seconds, task=self.task, index=str(self.index), pool=None,
                       raw_data=str(self.raw_data), problem=str(problem), cache_root=self.cache_root,
                       deadline=start+seconds)
            phase = 'prepare_inference'
            workspace = base/'workspace'
            before = run_io('copy', start+seconds, source=str(source), target=str(workspace),
                            ignore=['output', '.tmp', '__pycache__'])
            for path in [base,problem,*problem.rglob('*')]:
                path.chmod(0o755 if path.is_dir() else 0o444)
            # Same depth as verified validation (output/validation): the entry may
            # clear and recreate OUTPUT_DIR inside the writable output root.
            run = workspace/'output'/'test'
            run.mkdir(parents=True)
            science = Sandbox(python,problem,workspace,base/'inference_logs',self.gpu,
                env_extra={'MEDRSEE_PHASE':'final','MEDRSEE_SPLIT':'T','MEDRSEE_TASK':self.task,
                           'OUTPUT_DIR':str(run)},
                limits=self.resource_limits, frozen_workspace=True)
            phase = 'inference'
            outcome = science.run(f'import runpy;runpy.run_path({entry!r},run_name="__main__")',seconds-(time.monotonic()-start))
            write_json(base/'inference_result.json',outcome)
            executions['inference'] = {k:outcome.get(k) for k in
                ('returncode','reason','seconds','log','peak_sampled_rss_bytes','resource_samples')}
            executions['inference'].update(filesystem_policy='read_only_model_and_program',
                                            in_memory_fitting_excluded=False)
            if outcome['returncode']:
                raise RuntimeError(f"Frozen inference failed: returncode={outcome['returncode']}, "
                    f"reason={outcome.get('reason')}: "+outcome['stdout'][-1500:])
            phase = 'verify_frozen_files'
            after = run_io('hashes', start+seconds, source=str(workspace))
            verify_frozen(before, after, frozen, source)
            shutil.copyfile(source/'frozen_model.json',run/'frozen_model.json')
            workspace.chmod(0o755)  # Dedicated scorer UID must traverse its parents.
            run.parent.chmod(0o755)
            # Scoring runs in the trusted isolated process, never in agent Python.
            scratch = base/'scoring_scratch'
            scorer = Sandbox(python,problem,scratch,base/'scoring_logs',self.gpu,
                uid=1200000+__import__('os').getpid(),
                read_extra=[ROOT/'benchmarks',self.index,self.raw_data,run],limits=self.resource_limits)
            for path in [run,*run.rglob('*')]:
                path.chmod(0o755 if path.is_dir() else 0o444)
            code = (f'import sys,json;sys.path.insert(0,{str(ROOT/"benchmarks")!r});'
                    'from medrsee.scoring import score;'
                    f'r=score({self.task!r},{str(run)!r},{str(self.index)!r},{str(self.raw_data)!r},scratch="luna");'
                    'print("MEDRSEE_FINAL="+json.dumps(r))')
            phase = 'scoring'
            outcome = scorer.run(code,seconds-(time.monotonic()-start))
            write_json(base/'scoring_result.json',outcome)
            executions['scoring'] = {k:outcome.get(k) for k in
                ('returncode','reason','seconds','log','peak_sampled_rss_bytes','resource_samples')}
            if outcome['returncode']:
                raise RuntimeError(f"Final scorer failed: returncode={outcome['returncode']}, "
                    f"reason={outcome.get('reason')}: "+outcome['stdout'][-1500:])
            phase = 'parse_score'
            text = Path(outcome['log']).read_text()
            result = json.loads(text.rsplit('MEDRSEE_FINAL=',1)[1].strip())
            result.update(status='completed',artifact=artifact['id'],method=frozen['method'],
                          test_samples=COUNTS[self.task][4],elapsed_seconds=time.monotonic()-start,
                          executions=executions,selection=selection)
            write_json(base/'metrics.json',result)
        except (Exception,KeyboardInterrupt) as exc:
            result = dict(status='failed',task=self.task,error=f'{type(exc).__name__}: {exc}',
                          elapsed_seconds=time.monotonic()-start,phase=phase,executions=executions,**submission)
        write_json(output,result)
        return result


def verify_frozen(before, after, frozen, source):
    from .training import provenance
    _, fitting_files = provenance(source, frozen)
    protected = {frozen.get('inference_entry', 'predict.py'), *frozen['checkpoints'], 'frozen_model.json'}
    protected.update(p for p in before if p.endswith('.py'))
    protected.update(fitting_files)
    for path in {Path(p).as_posix() for p in protected}:
        if before.get(path) != after.get(path):
            raise ValueError('Frozen inference changed a declared checkpoint or entry: ' + path)


def export_inference_inputs(task, index, pool, raw_data, problem, cache_root=None, deadline=float('inf')):
    """Trusted worker exports native inputs with no labels for Val or terminal Test."""
    from .scoring import selected_members
    from .cache import VolumeCache, VOLUME_TASKS
    problem = Path(problem)
    (problem/'data').mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name('reader.py'), problem/'reader.py')
    batches, chunk = [], []
    with NativeReader(raw_data) as reader:
        cache = VolumeCache(cache_root, reader) if cache_root and task in VOLUME_TASKS else None
        for member in selected_members(index, pool):
            chunk.append(dict(member))
            if len(chunk) == (512 if task.startswith('ENV_') else 1):
                batches.append(export_batch(task, chunk, reader, problem/'data'/f'{len(batches):06d}',
                                             labels=False, deadline=deadline, cache=cache))
                chunk = []
        if chunk:
            batches.append(export_batch(task, chunk, reader, problem/'data'/f'{len(batches):06d}',
                                         labels=False, deadline=deadline, cache=cache))
    write_json(problem/'data/batches.json', batches)
    for path in [problem, *problem.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o444)
    return dict(batches=len(batches))
