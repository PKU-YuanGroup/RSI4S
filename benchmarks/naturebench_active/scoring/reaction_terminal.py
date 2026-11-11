"""Use native loaders and metrics for the experiment's fixed split 0 scope."""
import json
import math
from pathlib import Path

from benchmarks.naturebench_active.adapters.multi_properties import (
    INSTANCES, SINGLE_INSTANCES, REACTION_INSTANCES, evaluation_protocol, require_split_zero,
)


def validate_submission(run, summary):
    """Reject scope mismatches before any native ground-truth access."""
    context = json.loads((Path(run)/'final/final_input/native_identifiers.json').read_text())
    require_split_zero(context.get('repeated_split'))
    output = summary['native_output']
    expected = {name + '/predictions.csv' for name in SINGLE_INSTANCES}
    expected.update(name + '/predictions_split_0.csv' for name in REACTION_INSTANCES)
    if set(output['files']) != expected or set(output['instances']) != set(INSTANCES):
        raise ValueError('Fixed split 0 requires exactly seven complete native instance outputs')
    if any(type(n) is not int or n <= 0 for n in output['instances'].values()):
        raise ValueError('Fixed split 0 requires positive instance row counts')
    protocol = evaluation_protocol()
    if output.get('evaluation_protocol', protocol) != protocol:
        raise ValueError('Frozen output declares another evaluation protocol')
    return protocol


def run_evaluation(native):
    if (set(native.SINGLE_SPLIT_INSTANCES) != set(SINGLE_INSTANCES) or
            set(native.MULTI_SPLIT_INSTANCES) != set(REACTION_INSTANCES) or native.NUM_SPLITS != 10):
        raise ValueError('Native property evaluation scope changed')
    result = {name: native.evaluate_single_instance(name) for name in SINGLE_INSTANCES}
    for name in REACTION_INSTANCES:
        predictions = native.load_and_validate_multi(name, 0)
        truth = native.load_ground_truth_multi(name, 0)
        value = float(native.compute_r2(predictions, truth))
        if not math.isfinite(value):
            raise ValueError('Nonfinite native split 0 R-squared: ' + name)
        result[name] = {'R-squared': round(value, 6)}
    return result
