"""Recorded workflow boundaries; these labels impose no scheduling limits."""

TRANSITIONS = {
    'workflow_start': 'initialization',
    'resource_queue_start': 'resource_queue',
    'gpu_acquired': 'environment_preparation',
    'research_start': 'research',
    'exploration_end': 'research_closure',
    'meta_final_review_start': 'final_selection',
    'meta_final_review_end': 'finalization',
    'chain_distillation_start': 'chain_distillation',
    'chain_distillation_end': 'finalization',
    'final_test_start': 'final_inference_scoring',
    'final_test_end': 'finalization',
    'experiment_end': 'terminal_preparation',
    'terminal_scoring_start': 'terminal_scoring',
    'terminal_scoring_end': 'terminal_closure',
    'retention_start': 'retention',
    'retention_end': 'workflow_closure',
    'workflow_end': 'completed',
}


def phase_after(event, previous='unattributed'):
    return TRANSITIONS.get(event.get('event'), previous)
