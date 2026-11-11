"""Run the actual generator on bounded unlabeled development conditions."""
import copy
import json
from pathlib import Path
import shutil
import time

from benchmarks.naturebench_active.common import jsonl, sha256
from benchmarks.naturebench_active.execution import attach_feature_paths, scientific_worker
from benchmarks.naturebench_active.scoring.pending import validate_prediction
from rsi.common import write_json


def probe_records(records, limit=6):
    """Cover instances/family conditions without generating for every Val row."""
    selected, seen = [], set()
    for row in records:
        condition = (row['instance'], row['inputs'].get('family', row['sample_id']))
        if condition in seen:
            continue
        seen.add(condition)
        selected.append(copy.deepcopy({key: value for key, value in row.items() if key != 'label'}))
    # Round-robin by instance so a long first corpus cannot hide later tasks.
    buckets = {}
    for row in selected:
        buckets.setdefault(row['instance'], []).append(row)
    chosen = []
    while len(chosen) < limit and any(buckets.values()):
        for rows in buckets.values():
            if rows and len(chosen) < limit:
                chosen.append(rows.pop(0))
    return chosen


def generation_probe(evaluation, records, artifact, directory, deadline):
    selected = probe_records(records)
    if not selected:
        raise ValueError('Generation validation needs acquired development conditions')
    directory = Path(directory)
    problem, workspace = directory/'inputs', directory/'workspace'
    workspace.mkdir(parents=True)
    attach_feature_paths(evaluation.evidence, selected, 'val', problem)
    jsonl(problem/'records.jsonl', selected)
    for name in ('model.py', 'model.pkl'):
        shutil.copyfile(artifact/name, workspace/name)
    execution = scientific_worker(evaluation.python, evaluation.task, 'generation_probe', problem, workspace,
        directory/'logs', deadline-time.monotonic(), evaluation.resource_limits, gpu=evaluation.gpu)
    predictions_path = workspace/'output/predictions.jsonl'
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    if [p['sample_id'] for p in predictions] != [r['sample_id'] for r in selected]:
        raise ValueError('Generation probe prediction membership mismatch')
    details, warnings = [], []
    for row, prediction in zip(selected, predictions):
        value = validate_prediction(evaluation.task, row, prediction['prediction'])
        candidates = next(iter(value.values()))
        unique = len(set(candidates))
        details.append(dict(sample_id=row['sample_id'], instance=row['instance'],
                            candidates=len(candidates), unique_candidates=unique))
        if unique == 1:
            warnings.append('All generated candidates are identical for '+row['sample_id'])
    diagnostics = json.loads((workspace/'output/generation_diagnostics.json').read_text())
    if diagnostics['observed_model_exceptions']:
        warnings.append('Generator handled exceptions; inspect the recorded source locations and messages')
    report = dict(status='passed_with_warnings' if warnings else 'passed', execution=execution,
        records=details, warnings=warnings, **diagnostics, unlabeled_inputs=True,
        test_inputs_read=False, test_truth_read=False, predictions_sha256=sha256(predictions_path),
        scope='Bounded development schema and execution check; not proof of scientific validity or full performance')
    write_json(directory/'report.json', report)
    return report
