"""Pure RESP metrics extracted by the host; independent Test scoring is terminal-only.

This implementation follows Table 1 and src/model_training in the RESP paper
repository. It does not import the upstream training script or any model weights.
"""
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def calculate_metrics(truth, probabilities):
    y = np.asarray(truth)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != (len(y), 3) or set(y.tolist()) != {0, 1, 2}:
        raise ValueError('RESP evaluation requires all three true classes and N by 3 predictions')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or np.any(abs(p.sum(axis=1)-1) > 1e-6):
        raise ValueError('Invalid RESP probability simplex')
    return {'MCC': float(matthews_corrcoef(y, p.argmax(axis=1))),
            'ROC_AUC_RH03': float(roc_auc_score(y == 2, p[:, 2]))}


def run_evaluation():
    raise RuntimeError('RESP uses active validation and maintenance.resp.score for terminal Test scoring')
