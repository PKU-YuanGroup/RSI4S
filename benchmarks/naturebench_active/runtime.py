"""RSI Task/Meta adapter: acquired-label fitting, validation, terminal inference.

The legacy NatureBench Test-scoring service is never started by this adapter.
"""
import json
from pathlib import Path
import shutil
import sys
import time
import copy
import ast

import eval_service
from rsi.common import digest, write_json
from rsi.schedule import final_inference_seconds as inference_limit
from benchmarks.runtime import EvidenceEvaluation
from benchmarks.naturebench_active.execution import development_rows, attach_feature_paths, scientific_worker
from benchmarks.naturebench_active.preparation import jsonl, check, file_hash, digest as input_digest
from benchmarks.naturebench_active.evidence import Evidence
from benchmarks.naturebench_active.common import connect, sha256
from benchmarks.naturebench_active.scoring.metrics import SUPPORTED, score
from benchmarks.naturebench_active import submission
from benchmarks.naturebench_active.adapters.pending import LOSS_TASKS, YEAST


class ActiveEvaluation(EvidenceEvaluation):
    active_naturebench=True
    annotation_protocol='naturebench-active'
    selection_module=submission

    def __init__(self,task,data_dir,run_dir,seconds,gpu,wall_seconds=None,resource_limits=None,*,source_root,journal=None,validation_policy=None,evaluation_phases=None,final_inference_seconds=None):
        if type(seconds) is not int or seconds < 1 or type(wall_seconds) is not int or seconds != wall_seconds:
            raise ValueError('Active NatureBench requires one positive shared research budget')
        if task not in SUPPORTED:raise NotImplementedError('Validation adapter pending: '+task)
        self.task=task;self.directory=Path(run_dir);self.package=Path(data_dir);self.source_root=Path(source_root)
        self.python=sys.executable;self.gpu=gpu;self.resource_limits=resource_limits;self.journal=journal
        self.final_inference_seconds=inference_limit(dict(naturebench_active=True,
                                                         final_inference_seconds=final_inference_seconds))
        from benchmarks.naturebench_active.timing import phase_plan
        self.evaluation_phases=phase_plan(evaluation_phases)
        self.evidence=Evidence(self.package,self.directory/'evidence',wall_seconds=wall_seconds,validation_policy=validation_policy)
        table,issues=eval_service._get_per_instance_primaries(self.source_root/task/'metadata.json',task_name=task)
        if issues:raise ValueError('Native primary metric metadata: '+str(issues))
        with connect(self.evidence.pool) as db:instances=[r[0] for r in db.execute('SELECT DISTINCT instance FROM samples')]
        aliases={'shared_300k':'3bpa_300k','shared_maps':'benchmark'}
        primary={instance:table['gtex_v8' if task=='LIFE_16' else aliases.get(instance,instance)] for instance in instances}
        if task=='MAT_05':
            # Development uses the native MSE formula on an Agent-held-out D0
            # cohort. Its metric is not the independent IID Test evaluation.
            primary={instance:{**info,'native_metric':info['metric'],'metric':'MSE_x1000'}
                     if info['metric']=='MSE_IID_x1e3' else info
                     for instance,info in primary.items()}
        if task in LOSS_TASKS or task==YEAST:
            # A native Test reference is not a development loss/Pearson-r
            # baseline. Self-reported loss has no cross-method ranking scale.
            primary={instance:{**info,'native_metric':info['metric'],
                'metric':'model_loss' if task in LOSS_TASKS else 'Pearson r',
                'higher_is_better':task==YEAST,'sota':1.0,
                'reference_scope':'unranked_self_reported_loss' if task in LOSS_TASKS else 'development_unit_scale'}
                for instance,info in primary.items()}
        # Agents and reports never compare with published values: a unit scale
        # keeps the host's validity checks without exposing any paper score.
        primary={instance:{**info,'sota':1.0,'reference_scope':info.get('reference_scope','development_unit_scale')}
                 for instance,info in primary.items()}
        self.state=eval_service.TaskState(task,self.package,timeout=seconds,primary_table=primary)
        self.wall_seconds=wall_seconds;self.wall_deadline=None;self.closed=False;self.feedback=[]
        self.scoring_seconds={};self.validation_seconds=[]
        self.source_hash=sha256(self.source_root/task/'evaluation/evaluator.py')

    def start(self):
        self.state.start_time=time.time();self.wall_deadline=time.monotonic()+self.wall_seconds

    def close(self):
        self.closed=True
        self.evidence.stop_research()

    def extend_research_deadline(self,seconds):
        # The annotation ledger closes acquisition at its own stored deadline.
        self.evidence.extend_deadline(seconds)

    def results(self):
        return dict(protocol='naturebench-active',validation_only=True,submissions=self.feedback,
                    best_attempt=self.state.best_attempt,test_scored=False)

    def validation_cohort(self, *, deadline=None):
        with connect(self.evidence.pool) as db:
            ids=[r[0] for r in db.execute("SELECT sample_id FROM samples WHERE acquired=1 AND role='val' ORDER BY instance,ordinal")]
        return dict(samples=len(ids),sha256=digest(ids))

    def scientific_signature(self, source, *, deadline=None, records=None):
        """The actual self-contained worker inputs, including labels and files.

        Only model.py enters the fitting workspace. Notes and unused helpers
        remain in the handoff snapshot; they cannot alter this adapter's fit.
        Other benchmark adapters must define their own dependency contract.
        """
        source = Path(source)
        if not (source/'model.py').is_file():
            return None
        check(deadline)
        records = records or {role: development_rows(self.evidence, role, deadline=deadline) for role in ('train', 'val')}
        dependencies = {}
        def checked_hash(path):
            return file_hash(path, deadline)
        for name, expected in self.evidence.manifest.get('reference_files', {}).items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid public resource path')
            value = checked_hash(self.package/'problem/reference'/relative)
            if value != expected:
                raise ValueError('Public scientific resource changed: '+name)
            dependencies['reference/'+name] = value
        for role, rows in records.items():
            for start in range(0, len(rows), 128):
                check(deadline)
                ids = [row['sample_id'] for row in rows[start:start+128]]
                for row in self.evidence.fetch(role, ids, limit=128, deadline=deadline if deadline is not None else float('inf'))['records']:
                    for relative in row.get('files', []):
                        dependencies[relative] = checked_hash(self.evidence.problem/relative)
        scorer = Path(__file__).parent/'scoring'
        host_code = [Path(__file__), Path(__file__).with_name('execution.py'),
                     Path(__file__).with_name('generation.py'), Path(__file__).with_name('timing.py'),
                     Path(__file__).with_name('preparation.py'),
                     *sorted(scorer.glob('*.py'))]
        return dict(protocol='self_contained_model_and_acquired_inputs_v1',
            model_sha256=checked_hash(source/'model.py'), records_sha256=input_digest(records, deadline),
            files_sha256=input_digest(dependencies, deadline), manifest_sha256=input_digest(self.evidence.manifest, deadline),
            native_evaluator_sha256=checked_hash(self.source_root/self.task/'evaluation/evaluator.py'),
            host_code_sha256=digest({str(p): checked_hash(p) for p in host_code}),
            python=self.python, gpu=self.gpu, resource_limits=copy.deepcopy(self.resource_limits),
            evaluation_phases=self.evaluation_phases)

    def reuse_evaluation(self, directory, previous, *, deadline):
        """Reuse an automatic handoff's proved computation, preserving new notes."""
        if self.closed or previous.get('error') or not previous.get('valid_primary_score'):
            return None
        source = Path(directory)/'workspace'
        signature = self.scientific_signature(source, deadline=deadline)
        if signature is None or signature != previous.get('scientific_signature'):
            return None
        fitted = previous.get('fitted_artifact', {})
        artifact = Path(fitted.get('path', ''))
        if not artifact.is_dir() or {p.name: file_hash(p,deadline) for p in artifact.iterdir()} != fitted.get('files'):
            return None
        try:
            report = json.loads((artifact.parent/'validation_report.json').read_text())
            if (report['artifact_sha256'] != fitted['files'] or report.get('scientific_signature') != signature or
                    {key: value['metrics'] for key, value in report['metrics']['instances'].items()} != previous['raw_scores'] or
                    file_hash(artifact.parent/'predictions.jsonl',deadline) != report['predictions_sha256']):
                return None
        except (OSError, ValueError, KeyError):
            return None
        result = copy.deepcopy(previous)
        result.update(attempt=len(self.feedback), reused=True,
            scientific_reuse=dict(reason='automatic_handoff_only', source_artifact=previous.get('artifact'),
                                  source_snapshot=previous.get('snapshot'), signature=signature))
        files = {str(p.relative_to(source)): file_hash(p,deadline) for p in source.rglob('*') if p.is_file()}
        result['inference_verification']['source_files_sha256'] = digest(files)
        for phase in result.get('phase_timing', {}).get('phases', {}).values():
            phase.update(reused=True, original_seconds=phase.get('original_seconds', phase.get('seconds')), seconds=0)
        for field in ('training_audit', 'inference_verification', 'generation_validation'):
            if field not in result:
                continue
            execution = result[field]['execution']
            execution.update(reused=True, original_seconds=execution.get('original_seconds', execution.get('seconds')),
                             seconds=0)
        self.feedback.append(result)
        return result

    def submission_contract(self):
        text=Path(__file__).resolve().parents[2]/'rsi/interfaces/naturebench-active.md'
        from benchmarks.naturebench_active.scoring.structure import OUTPUT_CONTRACTS
        from benchmarks.naturebench_active.scoring.scientific import OUTPUT_CONTRACTS as other
        contracts={**OUTPUT_CONTRACTS,**other}
        from benchmarks.naturebench_active.scoring.pending import OUTPUT_CONTRACTS as pending_contracts
        contracts.update(pending_contracts)
        from benchmarks.naturebench_active.adapters.resp import TASK as RESP_TASK, OUTPUT_CONTRACT as RESP_CONTRACT
        contracts[RESP_TASK]=RESP_CONTRACT
        from benchmarks.naturebench_active.adapters.noncoding import TASK as NONCODING, OUTPUT_CONTRACT as NONCODING_CONTRACT
        contracts[NONCODING]=NONCODING_CONTRACT
        if self.task=='LIFE_07':
            from benchmarks.naturebench_active.scoring.cryo import OUTPUT_CONTRACT
            contracts[self.task]=OUTPUT_CONTRACT
        if self.task=='CHEM_07':
            from benchmarks.naturebench_active.scoring.linker import OUTPUT_CONTRACT
            contracts[self.task]=OUTPUT_CONTRACT
        timing=('has no wall-clock time limit' if self.final_inference_seconds is None else
                f'has a {self.final_inference_seconds}-second limit')
        instructions=(text.read_text().replace('{{TASK_ID}}',self.task)
                      .replace('{{FINAL_TIMING}}',timing))
        if self.task in contracts:instructions+='\n\n'+contracts[self.task]
        return dict(protocol='naturebench-active',task_id=self.task,instructions=instructions,
                    evaluation_phases=self.evaluation_phases)

    def evaluate(self,round_dir,timeout=None):
        if self.closed:raise ValueError('Research is closed')
        started=time.monotonic();deadline=started+min(self.remaining(),timeout if timeout is not None else self.remaining())
        directory=Path(round_dir);source=directory/'workspace'
        replay=directory/'replay';replay.mkdir()
        from benchmarks.naturebench_active.timing import PhaseBudget
        from benchmarks.naturebench_active.execution import enrich_public_inputs
        from benchmarks.naturebench_active.preparation import copy_file
        timing=PhaseBudget(deadline,replay,self.evaluation_phases)
        def source_files(seconds):
            if file_hash(self.source_root/self.task/'evaluation/evaluator.py',deadline)!=self.source_hash:
                raise ValueError('Native metric source changed')
            from rsi.sandbox import assert_plain_tree
            assert_plain_tree(source)
            return {str(p.relative_to(source)):file_hash(p,deadline) for p in source.rglob('*') if p.is_file()}
        files=timing.run('source_verification',source_files)
        if 'model.py' not in files:raise ValueError('Save self-contained model.py with fit(task,records) and predict(model,records)')
        train=timing.run('load_train',lambda _: development_rows(self.evidence,'train',deadline=deadline,enrich=False))
        val=timing.run('load_validation',lambda _: development_rows(self.evidence,'val',deadline=deadline,enrich=False))
        if not val:raise ValueError('No acquired validation records to score')
        for role,rows in [('train',train),('validation',val)]:
            timing.run('expand_'+role,lambda _,rows=rows: enrich_public_inputs(rows,self.evidence.pool,self.task,deadline=deadline))
        signature=timing.run('scientific_signature',lambda _: self.scientific_signature(source,deadline=deadline,records={'train':train,'val':val}))
        train_input=replay/'train_input';val_input=replay/'validation_input'
        timing.run('attach_train',lambda _: attach_feature_paths(self.evidence,train,'train',train_input,deadline=deadline))
        timing.run('attach_validation',lambda _: attach_feature_paths(self.evidence,val,'val',val_input,deadline=deadline))
        timing.run('export_train',lambda _: jsonl(train_input/'records.jsonl',train,deadline))
        loss_validation=self.task in LOSS_TASKS
        if loss_validation:
            def export_early_stopping(_):
                early_stopping=[]
                for row in val:
                    check(deadline)
                    early_stopping.append(copy.deepcopy(row))
                attach_feature_paths(self.evidence,early_stopping,'val',train_input,deadline=deadline)
                jsonl(train_input/'validation_records.jsonl',early_stopping,deadline)
            timing.run('export_early_stopping',export_early_stopping)
        timing.run('export_validation',lambda _: jsonl(val_input/'records.jsonl',
            val if loss_validation else ({k:v for k,v in r.items() if k!='label'} for r in val),deadline))
        fit=replay/'fit_workspace';fit.mkdir()
        timing.run('prepare_fit',lambda _: copy_file(source/'model.py',fit/'model.py',deadline))
        fitting=timing.run('fit',lambda seconds: scientific_worker(self.python,self.task,'fit',train_input,fit,
            replay/'fit_logs',seconds,self.resource_limits,gpu=self.gpu))
        artifact=replay/'artifact';artifact.mkdir()
        for name in ('model.py','model.pkl'):shutil.copyfile(fit/name,artifact/name)
        hashes={p.name:sha256(p) for p in artifact.iterdir()}
        workspace=replay/'validation_workspace';workspace.mkdir()
        for name in ('model.py','model.pkl'):shutil.copyfile(artifact/name,workspace/name)
        inference=timing.run('predict',lambda seconds: scientific_worker(self.python,self.task,
            'loss' if loss_validation else 'predict',val_input,workspace,replay/'prediction_logs',
            seconds,self.resource_limits,gpu=self.gpu))
        with (workspace/'output/predictions.jsonl').open() as stream:
            predictions=[json.loads(line) for line in stream]
        if time.monotonic()>=deadline:raise TimeoutError('Validation deadline exhausted')
        from benchmarks.naturebench_active.adapters.pending import OPTIMIZATION
        if self.task==OPTIMIZATION:
            from benchmarks.naturebench_active.scoring.optimization import isolated_score
            metrics=timing.run('score',lambda seconds: isolated_score(val,predictions,
                self.source_root/self.task/'evaluation/evaluator.py',replay,python=self.python,seconds=seconds,gpu=self.gpu))
        else:
            metrics=timing.run('score',lambda seconds: score(self.task,val,predictions,self.source_root,training_records=train))
        shutil.copyfile(workspace/'output/predictions.jsonl',replay/'predictions.jsonl')
        if hashes!={p.name:sha256(p) for p in artifact.iterdir()}:raise ValueError('Fitted artifact changed during prediction')
        raw={key:value['metrics'] for key,value in metrics['instances'].items()}
        primary={instance:self.state.primary_table[instance] for instance in raw}
        coverage=dict(evaluated_instances=sorted(raw),
            unevaluated_instances=sorted(set(self.state.primary_table)-set(raw)),
            complete=set(raw)==set(self.state.primary_table),
            records_per_instance={key:value['count'] for key,value in metrics['instances'].items()})
        write_json(replay/'validation_report.json',dict(task=self.task,artifact_sha256=hashes,metrics=metrics,
            predictions_sha256=sha256(replay/'predictions.jsonl'),scientific_signature=signature,
            validation_coverage=coverage,phase_timing=timing.report,
            formal_rsi_task_meta_integrated=True,test_truth_read=False))
        if loss_validation:
            # A new formula/normalizer can lower an Agent-defined loss without
            # improving generation. Keep the observed losses, but do not turn
            # them into a global scalar ranking or an automatic best attempt.
            improvements,aggregate={},None
        else:
            improvements,aggregate=eval_service._compute_improvements(raw,primary)
        result=dict(attempt=len(self.feedback),raw_scores=raw,per_instance_improvement=improvements,aggregate_improvement=aggregate,
            aggregate_scope=('Self-reported validation loss is diagnostic; no cross-method scalar ranking' if loss_validation else
                             'Development unit scale; different from native Test metric and reference' if self.task==YEAST
                             else 'Development validation on a fixed unit scale, not relative to any published value; not a Test improvement'),
            validation_cohort=self.validation_cohort(),validation_coverage=coverage,
            benchmark_protocol='naturebench-active',test_truth_read=False,
            training_audit=dict(status='passed',fitting_records=len(train),validation_records=len(val),validation_overlap=0,
                                authorized_training_sha256=digest([r['sample_id'] for r in train]),execution=fitting),
            inference_verification=dict(status='completed',source_files_sha256=digest(files),execution=inference,
                                        filesystem_policy=('labeled validation for model loss and read-only fitted model' if loss_validation
                                                           else 'only unlabeled validation input and read-only fitted model')),
            fitted_artifact=dict(path=str(artifact),files=hashes))
        result['scientific_signature'] = signature
        result['phase_timing'] = timing.report
        result['aggregate_scope'] += '; covers only evaluated instances in validation_coverage'
        if loss_validation:
            result['selection_metric'] = 'self_reported_loss_diagnostic'
            result['metric_comparison'] = dict(automatic_scalar_ranking=False,
                implementation_sha256=digest(ast.dump(ast.parse((source/'model.py').read_text()), include_attributes=False)),
                definition_verified_by_host=False,
                guidance='Compare the stated loss formula, units, normalization and its relation to predict; '
                         'different implementations or validation cohorts are not numerically interchangeable. '
                         'Choose any further generation checks and use scientific reasoning for explicit artifact selection.')
        self.feedback.append(result);self.validation_seconds.append(time.monotonic()-started)
        # A failed instance carries a fixed penalty that can exceed real scores on
        # unbounded metrics, so only fully measured attempts can become the best.
        from rsi.evaluation import valid_score
        if aggregate is not None and valid_score(result,primary) and (self.state.best_aggregate_improvement is None or aggregate>self.state.best_aggregate_improvement):
            self.state.best_aggregate_improvement=aggregate;self.state.best_attempt=result['attempt']
        return result

    def finalize(self,runs,python,seconds=None):
        try:
            artifact,selection=self.select_submission(runs,getattr(self,'meta_selection',None))
            write_json(self.directory/'final_selection.json',selection)
            if artifact is None:raise ValueError(selection['error'])
            feedback=next(e for r in runs for e in r['evaluations'] if e.get('artifact')==artifact['id'] and e.get('fitted_artifact'))
            selected=Path(feedback['fitted_artifact']['path'])
            directory=self.directory/'final';directory.mkdir()
            shutil.copytree(selected,directory/'artifact')
            shutil.copyfile(selected.parent/'validation_report.json',directory/'validation_report.json')
            config=directory/'runtime_config.json'
            write_json(config,dict(python=python,data_root=str(self.source_root),sandbox_resources=self.resource_limits,gpu=self.gpu))
            from benchmarks.naturebench_active.inference import run
            result=run(self.package,directory,config,evidence_directory=self.evidence.directory,seconds=seconds)
            result.update(status='completed',artifact=artifact['id'],selection=selection,formal_rsi_task_meta_integrated=True)
        except Exception as exc:
            result=dict(status='failed',task=self.task,error=f'{type(exc).__name__}: {exc}',test_score_computed=False)
        write_json(self.directory/'final_test.json',result)
        return result
