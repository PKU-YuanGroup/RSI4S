"""Exercise one-way frozen inference after a completed validation diagnostic.

Only unlabeled Test inputs are released; no Test scoring or research feedback.
Shared by the formal RSI runtime and offline diagnostic CLI.
"""
import fcntl
import json
from pathlib import Path
import shutil
import time

from rsi.common import write_json
from rsi.schedule import final_inference_seconds
from rsi.resources import scientific_gpu
from benchmarks.naturebench_active.execution import scientific_worker
from benchmarks.naturebench_active.common import jsonl
from benchmarks.naturebench_active.evidence import Evidence
from benchmarks.naturebench_active.common import sha256
from benchmarks.naturebench_active.scoring.metrics import aligned_predictions
from benchmarks.naturebench_active.outputs import SUPPORTED, write_outputs


def run(package, directory, config_path, *, evidence_directory=None, seconds=None):
    seconds=final_inference_seconds(dict(naturebench_active=True,final_inference_seconds=seconds))
    config = json.loads(Path(config_path).read_text())
    if 'gpu' in config:
        return _run(package, directory, config, evidence_directory=evidence_directory, seconds=seconds)
    with scientific_gpu(config, lease_info={'phase': 'frozen_inference_diagnostic',
                                           'directory': str(Path(directory).resolve())}) as (gpu, limits):
        return _run(package, directory, {**config, 'gpu': gpu, 'sandbox_resources': limits},
                    evidence_directory=evidence_directory, seconds=seconds)


def _run(package, directory, config, *, evidence_directory=None, seconds=None):
    deadline=None if seconds is None else time.monotonic()+seconds
    directory = Path(directory).resolve()
    with (directory/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads((directory/'validation_report.json').read_text())
        if report['task'] not in SUPPORTED:
            raise NotImplementedError('Native final output adapter pending; research remains open')
        artifact = directory/'artifact'
        hashes = {p.name:sha256(p) for p in artifact.iterdir()}
        if hashes != report['artifact_sha256']:
            raise ValueError('Selected validated artifact changed')
        final_report = directory/'final_inference_report.json'
        if final_report.exists():
            saved = json.loads(final_report.read_text())
            if sha256(directory/'final_predictions.jsonl') != saved['predictions_sha256']:
                raise ValueError('Frozen predictions changed')
            for relative, expected in saved.get('native_output',{}).get('files',{}).items():
                if sha256(directory/'final_output'/relative) != expected:
                    raise ValueError('Native final output changed')
            return saved
        evidence = Evidence.resume(package, evidence_directory or directory/'evidence')
        phase = evidence.status()['phase']
        if phase in ('research','selection'):
            write_json(directory/'freeze.json', evidence.close_research(artifact))
        final_input = directory/'final_input'
        if evidence.status()['phase'] in ('closed','test_exporting'):
            write_json(directory/'test_release.json', evidence.export_test_inputs(final_input))
        if evidence.status()['phase'] != 'test_released':
            raise ValueError('Invalid final inference phase')
        from benchmarks.naturebench_active.inputs import prepare_records
        records,input_normalization=prepare_records(report['task'],final_input)
        if len(records) != evidence.manifest['T']:
            raise ValueError('Incomplete Test input release')
        jsonl(final_input/'records.jsonl', records)
        workspace = directory/'final_workspace'; workspace.mkdir(exist_ok=True)
        for name in ('model.py','model.pkl'):
            shutil.copyfile(artifact/name,workspace/name)
        checkpoint = directory/'final_predict_complete.json'
        if checkpoint.exists():
            result = json.loads(checkpoint.read_text())
            if sha256(directory/'final_predictions.jsonl') != result['predictions_sha256']:
                raise ValueError('Frozen prediction checkpoint changed')
        else:
            result = scientific_worker(config['python'], report['task'], 'predict', final_input, workspace,
                                 directory/'final_logs', None if deadline is None else deadline-time.monotonic(),
                                 config.get('sandbox_resources'),gpu=config['gpu'])
            shutil.copyfile(workspace/'output/predictions.jsonl',directory/'final_predictions.jsonl')
            result['predictions_sha256'] = sha256(directory/'final_predictions.jsonl')
            write_json(checkpoint,result)
        with (directory/'final_predictions.jsonl').open() as stream:
            predictions = [json.loads(x) for x in stream]
        values = aligned_predictions(records, predictions)
        identifiers=final_input/'native_identifiers.json'
        native_context=json.loads(identifiers.read_text()) if identifiers.exists() else {}
        native_output = write_outputs(report['task'],records,predictions,directory/'final_output',config['data_root'],native_context=native_context)
        if hashes != {p.name:sha256(p) for p in artifact.iterdir()}:
            raise ValueError('Frozen model changed during inference')
        result.update(status='frozen_inference_completed', task=report['task'], predictions=len(predictions),
                       final_inference_limit_seconds=seconds,
                      artifact_sha256=hashes, predictions_sha256=sha256(directory/'final_predictions.jsonl'),
                      native_output=native_output,
                      input_normalization=input_normalization,
                      test_truth_read=False, test_score_computed=False, feedback_to_research=False,
                      formal_rsi_task_meta_integrated=False)
        write_json(final_report,result)
        return result
