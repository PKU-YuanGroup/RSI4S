"""Agent-controlled internal validation; the host does not choose a holdout."""


def validate_policy(policy):
    expected = {'policy': 'agent_selected_v1'}
    if policy is not None and policy != expected:
        raise ValueError('Internal validation is Agent-selected; organizer-fixed seed/fraction policies are no longer supported')
    return expected
