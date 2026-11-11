"""Host-side validation adapters. Never open an independent Test truth file."""
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics as skmetrics
from benchmarks.naturebench_active.scoring.structure import TASKS as STRUCTURE_TASKS
from benchmarks.naturebench_active.scoring.scientific import TASKS as SCIENTIFIC_TASKS


SUPPORTED = {'LIFE_12', 'LIFE_02', 'LIFE_14',
             'LIFE_15', 'MAT_04', 'LIFE_10',
             'CHEM_02', 'CHEM_03'}
SUPPORTED |= STRUCTURE_TASKS
SUPPORTED |= SCIENTIFIC_TASKS
SUPPORTED.add('LIFE_07')
SUPPORTED.add('CHEM_07')
SUPPORTED.add('LIFE_16')
from benchmarks.naturebench_active.adapters.resp import TASK as RESP_TASK
SUPPORTED.add(RESP_TASK)
from benchmarks.naturebench_active.adapters.pending import TASKS as PENDING_TASKS
SUPPORTED.update(PENDING_TASKS)
from benchmarks.naturebench_active.adapters.plm_interact import TASK as PLM_TASK
SUPPORTED.add(PLM_TASK)


def molecular_score(task, records, values, source):
    if task == 'CHEM_02':
        names = ['align_predictions_with_ground_truth', 'calculate_classification_metrics', 'calculate_regression_metrics']
        constants = ['CLASSIFICATION_INSTANCES', 'REGRESSION_INSTANCES']
    else:
        names = ['compute_rmse', 'compute_auc_single', 'compute_auc_multi']
        constants = ['REGRESSION_INSTANCES', 'SINGLE_TASK_CLASSIFICATION_INSTANCES', 'MULTI_TASK_CLASSIFICATION_INSTANCES']
    native = native_functions(source, names, constants)
    instances = {}
    for instance in sorted({r['instance'] for r in records}):
        pairs = [(r,v) for r,v in zip(records,values) if r['instance'] == instance]
        targets = sorted(pairs[0][0]['label'])
        if any(set(r['label']) != set(targets) or set(p) != set(targets) for r,p in pairs):
            raise ValueError('Molecular prediction target schema mismatch for ' + instance)
        truth = pd.DataFrame([r['label'] for r,_ in pairs], columns=targets).apply(pd.to_numeric, errors='coerce')
        pred = pd.DataFrame([p for _,p in pairs], columns=targets).astype(float)
        if not np.isfinite(pred.values).all():
            raise ValueError('Nonfinite molecular predictions')
        classification = instance not in native['REGRESSION_INSTANCES']
        if classification and ((pred.values < 0) | (pred.values > 1)).any():
            raise ValueError('Classification predictions must be probabilities')
        if task == 'CHEM_02':
            # The native merge key is replaced by unique curated row IDs so
            # duplicate V structures retain their original evaluation weight.
            ids = [r['sample_id'] for r,_ in pairs]
            truth['smiles'] = ids; pred['smiles'] = ids
            if classification:
                metrics = native['calculate_classification_metrics'](pred, truth, targets)
            else:
                metrics = native['calculate_regression_metrics'](pred, truth, targets, native['REGRESSION_INSTANCES'][instance]['metric'])
        elif instance in native['REGRESSION_INSTANCES']:
            metrics = native['compute_rmse'](pred.values.ravel(), truth.values.ravel())
        elif instance in native['SINGLE_TASK_CLASSIFICATION_INSTANCES']:
            metrics = native['compute_auc_single'](pred.values.ravel(), truth.values.ravel())
        else:
            metrics = native['compute_auc_multi'](pred.values, truth)
        instances[instance] = dict(count=len(pairs), metrics=metrics)
    return dict(instances=instances, native_metric_functions=names)


def native_functions(source, names, constants=()):
    tree = ast.parse(source.read_bytes())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('Missing native metric function')
    namespace = dict(np=np, pd=pd, pearsonr=pearsonr, spearmanr=spearmanr, ValidationError=ValueError)
    for name in ('roc_auc_score', 'average_precision_score', 'accuracy_score', 'precision_score',
                 'recall_score', 'f1_score', 'matthews_corrcoef', 'mean_squared_error', 'mean_absolute_error', 'r2_score'):
        namespace[name] = getattr(skmetrics, name)
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    namespace[target.id] = ast.literal_eval(n.value)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def scalar_score(task, records, values, source):
    names = ['calculate_cyclase_metrics', 'calculate_ubonodin_metrics'] if task == 'LIFE_02' else ['calculate_metrics']
    native = native_functions(source, names, ['GT_LABEL_COL'])
    instances = {}
    for instance in sorted({r['instance'] for r in records}):
        pairs = [(r, v) for r, v in zip(records, values) if r['instance'] == instance]
        rows, predictions = zip(*pairs)
        targets = set(rows[0]['label'])
        if len(targets) != 1 or any(set(r['label']) != targets or set(p) != targets for r,p in pairs):
            raise ValueError('Scalar prediction target schema mismatch for ' + instance)
        target = next(iter(targets))
        y = np.array([r['label'][target] for r in rows], dtype=float)
        p = np.array([v[target] for v in predictions], dtype=float)
        if not np.isfinite(p).all():
            raise ValueError('Nonfinite scientific predictions')
        ids = [r['sample_id'] for r in rows]
        if task == 'LIFE_02':
            if instance == 'cyclase_peptide_pair':
                if np.any((p < 0) | (p > 1)):
                    raise ValueError('Classification predictions must be probabilities')
                metrics = native['calculate_cyclase_metrics']((p >= .5).astype(int), y)
            else:
                metrics = native['calculate_ubonodin_metrics'](p, y)
        elif task == 'LIFE_14':
            metrics = native['calculate_metrics'](pd.DataFrame(dict(name=ids,prediction=p)),
                         pd.DataFrame({'name':ids, target:y}), instance)
        elif task == 'LIFE_10':
            metrics = native['calculate_metrics'](pd.DataFrame(dict(variant=ids,score=p)),
                                                 pd.DataFrame(dict(variant=ids,score=y)))
        else:
            if task == 'LIFE_15' and np.any((p < 0) | (p > 1)):
                raise ValueError('Classification predictions must be probabilities')
            metrics = native['calculate_metrics'](p, y)
        instances[instance] = dict(count=len(rows), metrics=metrics)
    return dict(instances=instances, native_metric_functions=names)


def aligned_predictions(records, predictions):
    expected = [r['sample_id'] for r in records]
    actual = [r['sample_id'] for r in predictions]
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual):
        raise ValueError('Duplicate sample IDs in scoring input')
    if set(actual) != set(expected):
        raise ValueError('Predictions must cover every validation sample exactly once')
    by_id = {r['sample_id']: r['prediction'] for r in predictions}
    return [by_id[sid] for sid in expected]


def score(task, records, predictions, source_root, training_records=None):
    if task not in SUPPORTED:
        raise NotImplementedError('Native validation metric adapter pending: ' + task)
    values = aligned_predictions(records, predictions)
    source = Path(source_root) / task / 'evaluation/evaluator.py'
    source_bytes = source.read_bytes()
    if task in PENDING_TASKS:
        from benchmarks.naturebench_active.scoring.pending import score as pending_score
        result=pending_score(task,records,values,source)
        result.update(native_metric_source=str(source),native_metric_sha256=hashlib.sha256(source_bytes).hexdigest(),test_truth_read=False)
        return result
    if task=='CHEM_07':
        from benchmarks.naturebench_active.scoring.linker import linker_score
        result=linker_score(task,records,values,source,training_records)
        result.update(native_metric_source=str(source),native_metric_sha256=hashlib.sha256(source_bytes).hexdigest(),test_truth_read=False)
        return result
    if task != 'LIFE_12':
        method = molecular_score if task in {'CHEM_02', 'CHEM_03'} else scalar_score
        if task in STRUCTURE_TASKS:
            from benchmarks.naturebench_active.scoring.structure import structure_score
            method=structure_score
        if task in SCIENTIFIC_TASKS:
            from benchmarks.naturebench_active.scoring.scientific import scientific_score
            method=scientific_score
        if task=='LIFE_07':
            from benchmarks.naturebench_active.scoring.cryo import cryo_score
            method=cryo_score
        if task=='LIFE_16':
            from benchmarks.naturebench_active.adapters.noncoding import score as noncoding_score
            method=lambda task,records,values,source:noncoding_score(records,values,source)
        if task==RESP_TASK:
            from benchmarks.naturebench_active.adapters.resp import score as resp_score
            method=lambda task,records,values,source:resp_score(records,values,source)
        if task==PLM_TASK:
            from benchmarks.naturebench_active.adapters.plm_interact import score as plm_score
            method=lambda task,records,values,source:plm_score(records,values,source)
        result = method(task, records, values, source)
        result.update(native_metric_source=str(source), native_metric_sha256=hashlib.sha256(source_bytes).hexdigest(),
                      test_truth_read=False)
        return result
    # Extract just the pure native metric function: neither importing the full
    # evaluator nor invoking its ground-truth loader is permitted here.
    namespace = native_functions(source, ['calculate_metrics'])
    truth, predicted = [], []
    for row, value in zip(records, values):
        if set(value) != {'pHopt'} or not np.isfinite(float(value['pHopt'])):
            raise ValueError('Each pH prediction must contain exactly one finite pHopt')
        # The native merge uses Accession. Curated sample IDs ensure one-to-one
        # alignment even if the native accession is not unique after curation.
        truth.append({'Accession': row['sample_id'], 'pHopt': float(row['label']['pHopt']),
                      'Sample Weight': float(row['label']['Sample Weight'])})
        predicted.append({'Accession': row['sample_id'], 'pHopt_pred': float(value['pHopt'])})
    metrics = namespace['calculate_metrics'](pd.DataFrame(predicted), pd.DataFrame(truth))
    return dict(instances={'phopt': dict(count=len(records), metrics=metrics)},
                native_metric_source=str(source), native_metric_sha256=hashlib.sha256(source_bytes).hexdigest(),
                native_metric_function='calculate_metrics', test_truth_read=False)
