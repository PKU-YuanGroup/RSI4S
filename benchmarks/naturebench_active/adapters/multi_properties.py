"""Experiment contract for property prediction with fixed reaction split 0."""
TASK = 'CHEM_05'
SINGLE_INSTANCES = ('esol', 'freesolv', 'lipophilicity', 'fluorescence', 'stability')
REACTION_INSTANCES = ('buchwald_hartwig', 'suzuki')
INSTANCES = SINGLE_INSTANCES + REACTION_INSTANCES
PROTOCOL = 'multi_properties_fixed_split0_v1'


def require_split_zero(value):
    if type(value) is not int or value != 0:
        raise ValueError('CHEM_05 requires fixed reaction split 0')


def check_package(manifest):
    if manifest.get('task_id') == TASK:
        require_split_zero(manifest.get('protocol', {}).get('repeated_split'))


def evaluation_protocol():
    return dict(name=PROTOCOL, task=TASK, repeated_split=0,
                evaluated_reaction_splits=[0], native_reaction_splits=list(range(10)),
                reaction_aggregation='single_split_r_squared',
                experiment_coverage_complete=True, native_coverage_complete=False,
                native_reaction_split_coverage=dict(evaluated=1, total=10),
                other_instances='unchanged native single-split metrics')
