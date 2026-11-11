"""Development metrics and input-only output contracts for the reviewed batch."""
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from benchmarks.naturebench_active.adapters.pending import (
    TASKS, YEAST, POCKET, RNA, OPTIMIZATION, ECD, COVID, RNA_GENERATION,
    MOLECULE_GENERATION, LOSS_TASKS, RNA_CLASSES,
)

COVID_CLASSES = ('SUBSTANTIAL DECREASING', 'MODERATE DECREASING', 'STABLE',
                 'MODERATE INCREASING', 'SUBSTANTIAL INCREASING')

LOSS_CONTRACT = '''Development uses your model's own validation loss, lower is better.
Define fit(task, training_records, validation_records). Only training_records may
contribute training gradients or fitted target statistics; validation_records are
available for selecting epochs/hyperparameters and early stopping. Define
validation_loss(model, labeled_records), returning one finite scalar loss per
record. The host runs it with the frozen fitted artifact. State your loss formula
and normalization in your research report; different loss definitions are not
scientifically interchangeable. The host reports these losses as diagnostics and
does not compute a cross-method aggregate improvement or automatically rank them.
Explain how the loss relates to the actual generator when selecting an artifact.
Define predict(model, unlabeled_records) for final generation. Loss evaluation
does not call predict or require a generation probe. You choose whether and when
to test generation during development. Final generation must satisfy the output
schema. Each predict value is exactly the documented dictionary with no other
keys: the host attaches sample_id alignment itself, so a value that also contains
sample_id or any extra field fails final validation. Loss validation receives
labels; final predict never does.
Use the native generation contract below for predict, not loss values.
'''

OUTPUT_CONTRACTS = {
    YEAST: 'Return {"expression": finite_float} per DNA sequence. Development uses Pearson r '
           '(constant predictions score 0); Test retains the native grouped/paired scoring. '
           'Optional plasmid context is in inputs.resource_paths["plasmid.json"].',
    POCKET: LOSS_CONTRACT + 'predict returns {"smiles": list_of_exactly_100_SMILES} per pocket. '
            'Read only inputs.feature_paths (one numeric NPZ) for pocket features. All '
            'ligand supervision resides in label. Train on acquired ligands; no Vina validation.',
    RNA: 'For nrc return {"class_label": one_native_class_name}. For archiveii600/ts0 return '
         '{"partners": list_of_integers}, length equal to the input sequence. Use 0 for unpaired '
         'and 1-based reciprocal partner indices for pairs. Native training targets can also '
         'contain -1 (unknown/unpaired); do not treat it as a valid paired position. Test output '
         'identifiers are supplied in inputs.sequence_id. Do not learn from identifiers.',
    OPTIMIZATION: 'Return {"smiles": list_of_SMILES}, exactly 20 per drd2_qed record and 5 per '
                  'admet record. Respect inputs.task_name and re_group. Fit only acquired target '
                  'molecules. Public pretrained property oracles are in inputs.resource_paths. '
                  'Native DRD2/QED validation success uses the official joint-task validation '
                  'rows; other official validation conditions remain visible but do not define '
                  'the joint-task primary. ADMET uses every official validation condition.',
    ECD: 'Return exactly {"peak_num": n, "peak_position": [int,...], "peak_height": [0_or_1,...]}. '
         'n is a nonnegative integer; each output list has length n (no -1 padding); positions '
         'are in [0,19]. Training target vectors use -1 padding. Read only inputs.feature_paths '
         '(one numeric NPZ) for graph and descriptor arrays. Preserve native is_enantiomer.',
    COVID: 'Return {"t1": [p0,p1,p2,p3,p4], "t3": [p0,p1,p2,p3,p4]}; each is a finite '
           'nonnegative probability vector summing to 1. Class order: substantial decrease, '
           'moderate decrease, stable, moderate increase, substantial increase. Labels t1/t3 '
           'are integers in this same order. Handle missing inputs (null), list time series '
           'and v1/v3 column differences. Preserve official validation membership.',
    RNA_GENERATION: LOSS_CONTRACT + 'predict returns {"sequences": list_of_exactly_1000_RNA_strings} '
                    'per family condition, nonempty strings with A,C,G,U only. Shared 18/628 '
                    'families have one acquired training corpus; use inputs.family and '
                    'target_instances. Public Rfam.cm and fixed families.json are accessible '
                    'via inputs.resource_paths. Final families are fixed before training.',
    MOLECULE_GENERATION: LOSS_CONTRACT + 'predict returns {"smiles": list_of_exactly_500_SMILES} '
                         'per target condition. The three corpora stay separate. Acquired label '
                         'contains smiles; unqueried corpus strings are hidden. The shared '
                         'native prior is at inputs.resource_paths["prior_model.pt"]. RDKit '
                         'property reward computations are allowed during development.',
}


def finite(value):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric scalar')
    return float(value)


def sequence_list(value, key, count, *, rna=False):
    if not isinstance(value, dict) or set(value) != {key}:
        raise ValueError('Generation schema requires only ' + key)
    items = value[key]
    if not isinstance(items, list) or len(items) != count:
        raise ValueError(f'Expected exactly {count} generated candidates')
    if any(not isinstance(item, str) or not item or any(c.isspace() for c in item) for item in items):
        raise ValueError('Generated candidates must be nonempty single-line strings')
    if rna and any(set(item) - set('ACGU') for item in items):
        raise ValueError('Generated RNA must contain only ACGU')
    return items


def validate_prediction(task, row, value):
    if not isinstance(value, dict):
        raise ValueError('Prediction must be a dictionary')
    if task == YEAST:
        if set(value) != {'expression'}:
            raise ValueError('Expected expression only')
        finite(value['expression'])
    elif task == RNA:
        if row['instance'] == 'nrc':
            if set(value) != {'class_label'} or value['class_label'] not in RNA_CLASSES:
                raise ValueError('Unknown RNA class prediction')
        else:
            if set(value) != {'partners'} or not isinstance(value['partners'], list):
                raise ValueError('RNA structure needs a partners vector')
            partners = value['partners']
            n = len(row['inputs']['sequence'])
            if len(partners) != n or any(type(p) is not int or not 0 <= p <= n for p in partners):
                raise ValueError('Invalid RNA partner indices')
            if any(p and (p == i or partners[p-1] != i) for i,p in enumerate(partners,1)):
                raise ValueError('RNA pairs must be reciprocal without self-pairs')
    elif task == ECD:
        if set(value) != {'peak_num', 'peak_position', 'peak_height'}:
            raise ValueError('ECD target schema mismatch')
        n = value['peak_num']
        if type(n) is not int or n < 0:
            raise ValueError('ECD peak count must be a nonnegative integer')
        for key, allowed in [('peak_position', range(20)), ('peak_height', (0,1))]:
            a = value[key]
            if not isinstance(a, list) or len(a) != n or any(type(x) is not int or x not in allowed for x in a):
                raise ValueError('Invalid ECD ' + key)
    elif task == COVID:
        if set(value) != {'t1', 't3'}:
            raise ValueError('Both hospitalization horizons are required')
        for values in value.values():
            if not isinstance(values, list) or len(values) != 5:
                raise ValueError('Expected five class probabilities')
            if any(not 0 <= finite(x) <= 1 for x in values) or not math.isclose(sum(values),1,abs_tol=1e-6):
                raise ValueError('Invalid class probability distribution')
    elif task == RNA_GENERATION:
        sequence_list(value, 'sequences', 1000, rna=True)
    elif task in (POCKET, OPTIMIZATION, MOLECULE_GENERATION):
        count = 100 if task == POCKET else (500 if task == MOLECULE_GENERATION else (20 if row['instance']=='drd2_qed' else 5))
        sequence_list(value, 'smiles', count)
    else:
        raise ValueError('Unreviewed prediction task')
    return value


def covid_distributions(values):
    # Preserve the native descending dictionary insertion order for tied argmax.
    return [{COVID_CLASSES[i]: v[i] for i in range(4,-1,-1)} for v in values]


def pairs(partners):
    return {(min(i,p),max(i,p)) for i,p in enumerate(partners,1) if p > 0}


def score(task, records, values, source):
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    instances = {}
    if task == OPTIMIZATION:
        from benchmarks.naturebench_active.scoring.optimization import score as optimization_score
        return optimization_score(records, values, source)
    if task == RNA:
        native = native_functions(source, ['calculate_nrc_metrics', 'calculate_structure_metrics'])
    elif task == COVID:
        native = native_functions(source, ['calculate_metrics', 'calc_accuracy', 'calc_mse',
            'calc_wmse', 'calc_brier_score', 'calc_rps'], ['CLASSES', 'CLASS_VALUES'])
    elif task == ECD:
        native = native_functions(source, ['calculate_metrics'])
    metric_instances = ({i for r in records for i in r['inputs']['target_instances']}
                        if task==RNA_GENERATION else {r['instance'] for r in records})
    for instance in sorted(metric_instances):
        selected = [(r,v) for r,v in zip(records,values)
                    if (instance in r['inputs']['target_instances'] if task==RNA_GENERATION else r['instance']==instance)]
        if task in LOSS_TASKS:
            if any(not isinstance(v,dict) or set(v)!={'loss'} for _,v in selected):
                raise ValueError('Development requires model loss, not generated Test metrics')
            losses = [finite(v['loss']) for _,v in selected]
            metrics = {'model_loss': float(np.mean(losses))}
        else:
            for row,value in selected:
                validate_prediction(task,row,value)
            if task == YEAST:
                y = [r['label']['expression'] for r,_ in selected]
                p = [v['expression'] for _,v in selected]
                if len(y)<2 or np.ptp(y)==0:
                    raise ValueError('Pearson validation needs at least two nonconstant targets')
                metrics = {'Pearson r': 0.0 if np.ptp(p)==0 else float(pearsonr(y,p).statistic)}
            elif task == RNA:
                if instance == 'nrc':
                    metrics = native['calculate_nrc_metrics'](
                        {r['sample_id']:v['class_label'] for r,v in selected},
                        {r['sample_id']:r['label']['class_label'] for r,_ in selected})
                else:
                    metrics = native['calculate_structure_metrics'](
                        {r['sample_id']:pairs(v['partners']) for r,v in selected},
                        {r['sample_id']:pairs(r['label']['partners']) for r,_ in selected})
            elif task == ECD:
                metrics = native['calculate_metrics']([v for _,v in selected], [r['label'] for r,_ in selected])
            elif task == COVID:
                truth = pd.DataFrame([{k:COVID_CLASSES[r['label'][k]] for k in ('t1','t3')} for r,_ in selected])
                metrics = native['calculate_metrics'](
                    covid_distributions([v['t1'] for _,v in selected]),
                    covid_distributions([v['t3'] for _,v in selected]), truth)
            else:
                raise ValueError('Unsupported development metric')
        if any(not math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError('Nonfinite development metric')
        instances[instance] = {'count': len(selected), 'metrics': metrics}
    return {'instances': instances, 'development_metric': 'model_loss' if task in LOSS_TASKS else 'native_or_reviewed',
            'test_truth_read': False, 'test_score_computed': False}


def write_instance(task, instance, selected, out):
    from benchmarks.naturebench_active.outputs import csv_file
    for row, value in selected:
        validate_prediction(task,row,value)
    if task == YEAST:
        np.save(out/'predictions.npy', np.array([v['expression'] for _,v in selected],dtype=float))
    elif task == POCKET:
        (out/'generated_molecules.json').write_text(json.dumps({r['inputs']['pocket_id']:v['smiles'] for r,v in selected}))
    elif task == RNA:
        if instance == 'nrc':
            csv_file(out/'predictions.csv',['sequence_id','predicted_label'],
                     [{'sequence_id':r['inputs']['sequence_id'],'predicted_label':v['class_label']} for r,v in selected])
        else:
            for row, value in selected:
                name = row['inputs']['sequence_id']
                if not re.fullmatch(r'[A-Za-z0-9_.+-]+',name) or name in ('.','..'):
                    raise ValueError('Unsafe native RNA identifier')
                (out/(name+'.bpseq')).write_text(''.join(
                    f'{i} {base} {p}\n' for i,(base,p) in enumerate(zip(row['inputs']['sequence'],value['partners']),1)))
    elif task == OPTIMIZATION:
        columns = ['src_smi', *(['task_name','re_group'] if instance=='admet' else []), 'opt_smi', 'opt_idx']
        rows = []
        for row,value in selected:
            for i,smiles in enumerate(value['smiles'],1):
                rows.append({**{k:row['inputs'][k] for k in columns if k not in ('opt_smi','opt_idx')},
                             'opt_smi':smiles, 'opt_idx':i})
        csv_file(out/'output.csv',columns,rows)
    elif task == ECD:
        (out/'predictions.json').write_text(json.dumps([v for _,v in selected]))
    elif task == COVID:
        (out/'output.json').write_text(json.dumps({k:covid_distributions([v[k] for _,v in selected]) for k in ('t1','t3')}))
    elif task == RNA_GENERATION:
        for row,value in selected:
            family = row['inputs']['family']
            if not re.fullmatch(r'RF\d{5}',family):
                raise ValueError('Invalid RNA family identifier')
            (out/(family+'.fa')).write_text(''.join(f'>generated_{i:04d}\n{s}\n' for i,s in enumerate(value['sequences'])))
    elif task == MOLECULE_GENERATION:
        if len(selected) != 1:
            raise ValueError('Targeted generation has one fixed condition per instance')
        (out/'generated.smi').write_text('\n'.join(selected[0][1]['smiles'])+'\n')
    return 'reviewed native output schema, complete released input membership, no Test truth access'
