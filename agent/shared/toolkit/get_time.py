"""Read the host's research or post-research session clock without changing it."""


def tool_info():
    return {
        'name': 'get_time',
        'description': 'Read current research_elapsed_seconds and stage_elapsed_seconds (wall time), '
                       'remaining_solve_seconds (the shared research budget), and the current session. '
                       'The same timing is refreshed in host_budget on every model request. '
                       'This read-only query does not reset the clock, hand off, or interrupt a running tool. '
                       'Final selection and chain distillation report research_closed=true and '
                       'remaining_solve_seconds=0. Use stage_elapsed_seconds and, when a stage limit '
                       'is configured, stage_remaining_seconds for their separate session clocks.',
        'input_schema': {'type': 'object', 'properties': {}, 'required': []},
    }
