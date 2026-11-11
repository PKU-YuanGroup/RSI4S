"""BIO-07 input and prediction contract; no raw data or Test truth access."""
import math
import numbers

TASK = 'LIFE_03'
CROSS_SPECIES = tuple('cross_species_'+species for species in ('mouse', 'fly', 'worm', 'yeast', 'ecoli'))
SHARED_INSTANCE = CROSS_SPECIES[0]
INSTANCES = (*CROSS_SPECIES, 'bernett', 'mutation_effect', 'virus_human')
PAIR_FIELDS = ('query', 'text')
MUTATION_FIELDS = ('wild_seq', 'mutant_seq', 'participant_sequence')


def probability(prediction):
    if not isinstance(prediction, dict) or set(prediction) != {'prediction_score'}:
        raise ValueError('PLM-interact predictions require exactly prediction_score')
    value = prediction['prediction_score']
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError('prediction_score must be numeric')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('prediction_score must be finite and in [0, 1]')
    return value


def score(records, values, source):
    import numpy as np
    from benchmarks.naturebench_active.scoring.metrics import native_functions
    native = native_functions(source, ['calculate_metrics'], ['MUTATION_INSTANCES'])
    instances = {}
    for instance in sorted({r['instance'] for r in records}):
        if instance not in INSTANCES:
            raise ValueError('Unknown PLM-interact instance: '+instance)
        pairs = [(r, p) for r, p in zip(records, values) if r['instance'] == instance]
        if any(set(r['label']) != {'label'} or type(r['label']['label']) is not int
               or r['label']['label'] not in (0, 1) for r, _ in pairs):
            raise ValueError('Expected binary acquired interaction labels')
        y = np.asarray([r['label']['label'] for r, _ in pairs])
        p = np.asarray([probability(v) for _, v in pairs])
        instances[instance] = dict(count=len(pairs), metrics=native['calculate_metrics'](p, y, instance))
    return dict(instances=instances, native_metric_functions=['calculate_metrics'],
                validation_scope='Shared human Val is stored once under cross_species_mouse; '
                'it is not measured cross-species Test performance.')
