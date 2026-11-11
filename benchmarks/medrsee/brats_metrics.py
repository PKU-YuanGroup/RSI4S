"""BraTS custom1000_251_v1 region Dice, shared with the completed DSW smoke."""
import numpy as np


def region_dice(prediction, label):
    scores = {}
    for name, classes in [('WT',[1,2,4]), ('TC',[1,4]), ('ET',[4])]:
        p, y = np.isin(prediction, classes), np.isin(label, classes)
        denominator = int(p.sum()) + int(y.sum())
        scores[name] = 2 * int(np.logical_and(p,y).sum()) / denominator if denominator else 1.0
    return dict(metric='region_dice', score=float(np.mean(list(scores.values()))), regions=scores)
