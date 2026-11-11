"""One research clock shared by every Task and Meta stage."""
from __future__ import annotations

import math

MAX_ROUNDS = 40
STAGE_POLICY = 'shared_research_budget_v1'

ROUND_LIMIT_GUIDANCE = (
    'At most 40 Task rounds, including T00, are allowed; a lower configured limit still applies. '
    'This is an upper bound. Entries in future_stages describe possible next stages; '
    'they are subject to the shared budget and experiment termination.'
)

TASK_TIMING = (
    'All Task and Meta stages share the experiment budget. remaining_solve_seconds '
    'reports the shared research time left. research_elapsed_seconds and stage_elapsed_seconds '
    'report elapsed wall time across research and in the current stage. '
    'The host refreshes these values on every model request; get_time queries the same clock. '
    'There are no per-stage time limits or closing reserves. '
    'Choose when to hand off within that shared budget; handoff never resets it. '
    'evaluate returns feedback without handing off. A final response ends this Task stage. '
    'A successfully recorded request_labels also ends this Task and hands off to Meta to decide '
    'the data request, without requiring a scored baseline. Save notes before requesting; '
    'the host preserves an unscored workspace snapshot instead of running a closing evaluation. '
    'Invalid requests and ordinary fetch_data/annotation_status reads do not hand off. '
    'There is no tool-call count limit. Tool use consumes shared time and model calls. '
    'The host saves and attempts to score your current artifacts if time remains.'
) + ' ' + ROUND_LIMIT_GUIDANCE
TASK0_TIMING = (
    'This is Task0 (T00), the first Task stage of an iterative scientific research process. '
    'T00 starts from the original task input without a preceding Meta plan. '
    'This is a time-limited scientific research task. Task0 should hand off quickly so Meta0 and successor Tasks '
    'can begin improving the Agent and workflow. Use get_time() with no arguments to check '
    'elapsed research time, time spent in this stage and remaining shared time; use these facts '
    'to keep T00 brief and leave room for subsequent evolution. '
    'Prioritize short experimental cycles to obtain the first usable experimental output. '
    'Once you have a runnable baseline with a valid evaluation, save the code, results and '
    'concise notes, then promptly return a final response to hand off to Meta0 (M00). '
    'If a valid evaluation is blocked, '
    'you may instead hand off a reproduced failure, the known blocker and supporting evidence, '
    'explicitly stating that a valid evaluation has not been obtained. '
    'The initial Agent, method and workflow are starting points; they do not need to be fully '
    'developed in this stage. Handoff ends this stage while research continues: later Tasks '
    'can inherit selected artifacts and checkpoints and pursue unfinished experiments. '
) + TASK_TIMING
META_TIMING = (
    'All Task and Meta stages share the experiment budget. host_budget reports the current '
    'shared remaining_solve_seconds, research_elapsed_seconds and stage_elapsed_seconds, '
    'with no per-stage time limits or closing reserves. Elapsed values measure wall time; '
    'get_time queries the same live clock. '
    'Checks and publication consume that same budget; handoff never resets it. '
    'A final response validates the selected Task and any permitted Meta update, then hands off. '
    'Unchanged Agent source retains its version; changing source, prompts, tools or configuration '
    'can publish an updated version. Task artifact_parent selects scientific file reuse separately. '
    'Meta self-evolution availability is reported by the interface. '
    'There is no tool-call count limit. A time-budget failure does not publish unfinished edits. '
    'The host appends the current budget to every model request.'
) + ' ' + ROUND_LIMIT_GUIDANCE


def retire_stage_limits(config):
    """Keep obsolete settings as provenance; they cannot cap new research stages."""
    config['stage_policy'] = STAGE_POLICY
    if 'stage_seconds' in config:
        config['ignored_stage_seconds'] = config.pop('stage_seconds')


def meta_selection_seconds(config):
    """Closing review is untimed by default; an explicit integer opts into a cap."""
    if 'meta_selection_reserve_seconds' in config:
        raise ValueError('meta_selection_reserve_seconds was removed; use meta_selection_seconds after research')
    value = config.get('meta_selection_seconds')
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError('meta_selection_seconds must be null or a positive integer')
    return value


def final_inference_seconds(config):
    """NatureBench frozen Test prediction is untimed unless explicitly capped."""
    default = None if config.get('naturebench_active') else config.get('medrsee_final_seconds', 3600)
    value = config.get('final_inference_seconds', default)
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
        raise ValueError('final_inference_seconds must be positive and finite, or null for no time limit')
    return value


def terminal_scoring_seconds(config):
    """Native terminal scoring is untimed unless explicitly capped for diagnosis."""
    value = config.get('terminal_scoring_seconds')
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
        raise ValueError('terminal_scoring_seconds must be positive and finite, or null for no time limit')
    return value


def evolution_round_limit(config):
    """Count the initial Task0 as round one, matching the existing run IDs."""
    rounds = config.get('rounds', MAX_ROUNDS)
    if type(rounds) is not int or not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f'rounds must be an integer in 1..{MAX_ROUNDS} (including Task0)')
    return rounds


def stage_timing(config, kind, run_id):
    return (TASK0_TIMING if run_id == 'T00' else TASK_TIMING) if kind == 'task' else META_TIMING


def research_budget(config):
    wall = config['budget'].get('wall_seconds')
    seconds = wall if wall is not None else config['budget']['solve_seconds']
    clock = 'wall-clock' if wall is not None else 'effective solve-time'
    result = {'shared_by_all_task_and_meta_stages': True,
            'stage_policy': STAGE_POLICY,
            'per_stage_time_limits': False,
            'description': f'All stages share one {seconds / 3600:g}-hour ({seconds}-second) {clock} research race.',
            'solve_seconds': config['budget']['solve_seconds'],
            'wall_seconds': wall,
            'judge_reserve_seconds': 0,
            'final_validity_judge': 'disabled',
            'judge_time_in_research_budget': False,
            'task_stages': evolution_round_limit(config), 'meta_stages': evolution_round_limit(config) - 1,
            'startup_planning_stages': 0, 'total_meta_sessions': evolution_round_limit(config) - 1,
            'round_limit_includes_task0': True,
            'round_limit_is_target': False,
            'stage_counts_are_maxima': True,
            'future_stages_are_optional': True,
            'max_tool_calls_per_session': None}
    if 'meta_selection_seconds' in config:
        result['final_meta_selection'] = {'when': 'after_research', 'read_only': True,
            'seconds': meta_selection_seconds(config), 'time_in_research_budget': False,
            'api_budget': 'same cumulative ledger and limits'}
    return result


def _evolution_sequence(config):
    sequence = []
    rounds = evolution_round_limit(config)
    for round_index in range(rounds):
        sequence.append(('task', f'T{round_index:02d}'))
        if round_index + 1 < rounds:
            sequence.append(('meta', f'M{round_index:02d}'))
    return sequence


def session_context(config, kind, index, *, data_request_handoff=False):
    sequence = _evolution_sequence(config)
    if data_request_handoff and kind == 'meta' and index + 1 == evolution_round_limit(config):
        sequence.append(('meta', f'M{index:02d}'))
    offset = 2 * index + (kind == 'meta')
    if offset < 0 or offset >= len(sequence) or sequence[offset][0] != kind:
        raise ValueError('Invalid stage position')
    return {'run_id': sequence[offset][1], 'kind': kind, 'round_number': index + 1,
            'future_stages': [name for _, name in sequence[offset + 1:]]}
