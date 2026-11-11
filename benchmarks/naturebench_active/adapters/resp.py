"""RESP prediction contract, native tensor decoding and metric adaptation."""
import numpy as np

TASK = 'LIFE_01'
INSTANCE = 'atezolizumab'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'
OUTPUT_CONTRACT = '''
Predict ordered experimental binding bins for Atezolizumab variants.
Each prediction must be exactly {"probabilities": [p_RH01, p_RH02, p_RH03]}.
Values must be finite numbers in [0,1] and sum to 1 (absolute tolerance 1e-6).
The predicted class is argmax (ties choose the lowest class). Primary metric:
unweighted multiclass MCC; secondary: ROC-AUC of RH03 versus RH01/RH02.
Labels contain class (0,1,2) and confidence_weight; the weight is for optional
training use only.
Inputs contain sequence, chothia_sequence and an input-only pretrained embedding.
This is a retrospective finite-pool prediction task. Queries cannot measure
new designed sequences, and model scores are not measured affinities or KD.
'''


def tensor(path):
    """Decode an organizer-supplied RESP tensor without loading model objects."""
    import torch

    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError('Expected a tensor in '+str(path))
    return value.detach().numpy()


def labels(value, count):
    if value.shape != (count, 4) or not np.isfinite(value).all():
        raise ValueError('Invalid RESP label tensor shape or values')
    category = value[:, 2]
    if not np.isin(category, [0, 1, 2]).all() or not np.array_equal(value[:, 0], category > 0) or not np.array_equal(value[:, 1], category > 1):
        raise ValueError('RESP ordinal bits disagree with class')
    if np.any(value[:, 3] <= 0) or np.any(value[:, 3] > 1):
        raise ValueError('Invalid RESP confidence weights')
    return category.astype(int), value[:, 3]


def probabilities(prediction):
    if not isinstance(prediction, dict) or set(prediction) != {'probabilities'}:
        raise ValueError('RESP requires exactly a probabilities field')
    values = prediction['probabilities']
    if not isinstance(values, list) or len(values) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError('RESP needs three numeric class probabilities')
    p = np.asarray(values, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or abs(p.sum()-1) > 1e-6:
        raise ValueError('Invalid RESP probability simplex')
    return p


def score(records, values, source):
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    if not records or any(r['instance'] != INSTANCE for r in records):
        raise ValueError('Unknown or empty RESP instance')
    truth = [r['label']['class'] for r in records]
    prediction = [probabilities(p) for p in values]
    native = native_functions(source, ['calculate_metrics'])
    result = native['calculate_metrics'](truth, prediction)
    return dict(instances={INSTANCE: dict(count=len(records), metrics=result)},
                native_metric_functions=['calculate_metrics'],
                metric_origin='RSI4S metric implementation')
