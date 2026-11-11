"""One terminal GTEx score after research is closed and inference is frozen.

This host-only command is not registered as a Task/Meta tool. Its report is not
added to research history, validation feedback, model selection or query tools.
"""
import argparse
import json
from pathlib import Path

from benchmarks.naturebench_active.adapters.noncoding import TASK, BINS, read_native
from benchmarks.naturebench_active.common import connect, sha256
from benchmarks.naturebench_active.scoring.metrics import score
from rsi.common import write_json, record_path


def terminal_score(run, source_root):
    run, source_root = Path(run).resolve(), Path(source_root).resolve()
    summary = json.loads((run/'final_test.json').read_text())
    if summary.get('status') != 'completed' or summary.get('task') != TASK:
        raise ValueError('Require completed noncoding frozen final inference')
    final = run/'final'
    with connect(run/'evidence/organizer/ledger.sqlite') as db:
        state = dict(db.execute('SELECT key,value FROM state'))
    if state['phase'] != 'test_released':
        raise ValueError('Research must be irreversibly closed before Test scoring')
    hashes = {p.name:sha256(p) for p in (final/'artifact').iterdir()}
    if hashes != summary['artifact_sha256'] or hashes != json.loads(state['artifact'])['files']:
        raise ValueError('Frozen selected model changed')
    prediction_hash = sha256(final/'final_predictions.jsonl')
    if prediction_hash != summary['predictions_sha256']:
        raise ValueError('Frozen final predictions changed')
    output = run/'terminal_score.json'
    if output.exists():
        saved = json.loads(output.read_text())
        if saved['predictions_sha256'] != prediction_hash or saved['artifact_sha256'] != hashes:
            raise ValueError('Terminal score belongs to another frozen result')
        return saved
    with connect(final/'final_input/inputs.sqlite') as db:
        records = [dict(r) for r in db.execute('SELECT sample_id,instance,inputs FROM samples ORDER BY instance,ordinal')]
    keys = ['phenotype_id','variant_id','bulk']
    sources = {}
    for bin_name in BINS:
        # This is the first and only independent Test-truth access in the new
        # task pipeline. All research and selection have already ended above.
        path = source_root/TASK/'evaluation/ground_truth/gtex_v8'/bin_name/f'y_ref_{bin_name}.pkl'
        truth = read_native(path)
        if truth.duplicated(keys).any():raise ValueError('Duplicate Test identity')
        by_id = {tuple(row[k] for k in keys):dict(label=int(row['label']),slope=float(row['slope']))
                 for row in truth.to_dict('records')}
        cohort = [r for r in records if r['instance']=='gtex_v8/'+bin_name]
        input_ids = [tuple(json.loads(r['inputs'])[k] for k in keys) for r in cohort]
        if len(input_ids)!=len(set(input_ids)) or set(input_ids)!=set(by_id):
            raise ValueError('Final inputs and native Test truth do not align')
        for row,key in zip(cohort,input_ids):row['label']=by_id[key]
        sources[str(path)] = sha256(path)
    for row in records:row['inputs']=json.loads(row['inputs'])
    predictions=[json.loads(line) for line in (final/'final_predictions.jsonl').read_text().splitlines()]
    metrics=score(TASK,records,predictions,source_root)
    result=dict(task=TASK,phase='terminal_only',metrics=metrics,records=len(records),
                selected_artifact=summary['artifact'],artifact_sha256=hashes,
                predictions_sha256=prediction_hash,test_truth_sha256=sources,
                test_truth_read=True,test_score_computed=True,feedback_to_research=False)
    write_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True, type=record_path);parser.add_argument('--source-root',required=True)
    args=parser.parse_args()
    print(json.dumps(terminal_score(args.run,args.source_root)))
