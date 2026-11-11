"""Capabilities for a single organizer-owned simulator episode."""

def tools(role):
    shared = [{'name': 'simulator_observe',
        'description': 'Read the current permitted observation, usage and terminal status. Does not spend actions or reset the episode. No train/validation/test split exists for this fixed scenario.',
        'input_schema': {'type': 'object', 'properties': {}, 'required': []}}]
    if role == 'task':
        shared += [
            {'name': 'simulator_action',
             'description': 'Perform one simulator action. See task action schema. ChemGym uses action="step", arguments={"action":0..40}; X-ray uses wh, get_sample_information, umv, pd_exposure, dscan, or0, or1, finish. Actions spend the current episode budget. Agents cannot reset; the host protocol defines episode boundaries between Task generations.',
             'input_schema': {'type': 'object', 'properties': {'action': {'type': 'string'}, 'arguments': {'type': 'object'}}, 'required': ['action']}},
            {'name': 'simulator_run',
             'description': 'Run a workspace Python policy with next_action(observation,state). The host supplies the current simulator_observe object and initially state=None. Return {action,arguments,state} or {stop:true,state}. Each policy step runs in a fresh science sandbox; only JSON state and workspace files persist. Host forwards actions to the same live episode. Errors/timeouts preserve consumed actions; no reset. See PROBLEM_DIR for details.',
             'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'timeout_s': {'type': 'number'}}, 'required': ['path']}},
        ]
    return shared
