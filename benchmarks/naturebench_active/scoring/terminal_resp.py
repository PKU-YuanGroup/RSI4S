"""One organizer-only RESP Test score after model selection and frozen inference."""
import argparse
import json
from pathlib import Path

from benchmarks.task_names import original_id
from benchmarks.naturebench_active.adapters.resp import TASK, labels, tensor
from benchmarks.naturebench_active.common import connect, sha256
from benchmarks.naturebench_active.evidence import Evidence
from benchmarks.naturebench_active.scoring.metrics import score
from rsi.common import write_json, record_path


def terminal_score(run, package, source_root):
    run, package, source_root = map(lambda p: Path(p).resolve(), (run, package, source_root))
    summary = json.loads((run/'final_test.json').read_text())
    if summary.get('status') != 'completed' or summary.get('task') != TASK:
        raise ValueError('Require completed RESP frozen inference')
    evidence = Evidence.resume(package, run/'evidence')
    with connect(evidence.ledger) as db:
        state = dict(db.execute('SELECT key,value FROM state'))
    if state['phase'] != 'test_released':
        raise ValueError('Research must be irreversibly closed before Test scoring')
    final = run/'final'
    hashes = {p.name: sha256(p) for p in (final/'artifact').iterdir()}
    if hashes != summary['artifact_sha256'] or hashes != json.loads(state['artifact'])['files']:
        raise ValueError('Frozen selected model changed')
    prediction_hash = sha256(final/'final_predictions.jsonl')
    if prediction_hash != summary['predictions_sha256']:
        raise ValueError('Frozen predictions changed')
    for relative, expected in summary['native_output']['files'].items():
        if sha256(final/'final_output'/relative) != expected:
            raise ValueError('Frozen CSV output changed')
    with connect(evidence.reference) as db:
        sources = dict(db.execute('SELECT path,sha256 FROM sources'))
        expected_rows = [dict(r) for r in db.execute(
            "SELECT sample_id,instance,ordinal,inputs FROM samples WHERE split='T' ORDER BY instance,ordinal")]
    # Hash-locked packages prepared before the rename record original task IDs.
    prefix = original_id(TASK) if original_id(TASK)+'/evaluation/evaluator.py' in sources else TASK
    truth_relative = prefix+'/evaluation/ground_truth/testy.pt'
    metric_relative = prefix+'/evaluation/evaluator.py'
    for relative in (truth_relative, metric_relative):
        if sha256(source_root/relative) != sources[relative]:
            raise ValueError('RESP scoring source changed: '+relative)
    with connect(final/'final_input/inputs.sqlite') as db:
        released = [dict(r) for r in db.execute('SELECT sample_id,instance,ordinal,inputs FROM samples ORDER BY instance,ordinal')]
    if released != expected_rows or len(released) != summary['predictions']:
        raise ValueError('Final Test inputs differ from the frozen source membership')
    output = run/'terminal_score.json'
    if output.exists():
        saved = json.loads(output.read_text())
        if saved['predictions_sha256'] != prediction_hash or saved['artifact_sha256'] != hashes or saved['test_truth_sha256'] != sources[truth_relative]:
            raise ValueError('Terminal score belongs to another frozen result')
        return saved
    # Test truth is deserialized for the first time only after all guards above.
    category, weight = labels(tensor(source_root/truth_relative), len(released))
    for row in released:
        index = row.pop('ordinal')
        row['inputs'] = json.loads(row['inputs'])
        row['label'] = {'class': int(category[index]), 'confidence_weight': float(weight[index])}
    predictions = [json.loads(line) for line in (final/'final_predictions.jsonl').read_text().splitlines()]
    metrics = score(TASK, released, predictions, source_root)
    metrics['test_truth_read'] = True
    result = dict(task=TASK, phase='terminal_only', metrics=metrics, records=len(released),
        selected_artifact=summary['artifact'], artifact_sha256=hashes,
        predictions_sha256=prediction_hash, test_truth_sha256=sources[truth_relative],
        test_truth_read=True, test_score_computed=True, feedback_to_research=False)
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=record_path)
    parser.add_argument('--config', default='configs/resp.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    print(json.dumps(terminal_score(args.run, Path(config['data_root'])/TASK,
                                   config['naturebench_source_root'])))
