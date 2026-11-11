"""Minimal scientific task tools."""
from toolkit.bash import tool_info as bash_info
from toolkit.read_tool_result import tool_info as read_result_info
from toolkit.get_time import tool_info as time_info
from protocol import dispatch as dispatch_tool
from toolkit.evidence import tools as evidence_tools
from toolkit.simulator import tools as simulator_tools


def describe(context=None):
    annotation=(context or {}).get('annotation') or {}
    active=annotation.get('protocol')=='naturebench-active'
    return [bash_info(), read_result_info(), time_info(), {
        'name': 'evaluate',
        'description': ('Fit saved model.py on authorized training labels and evaluate the current validation cohort in isolation. Loss tasks use labeled validation_loss without a generation probe; other tasks predict on unlabeled inputs. Returns measured metrics, actual coverage and a fitted artifact for Meta selection. Does not end this Task stage.' if active else
                        'Snapshot current task outputs and return independent task scores (validation only for active-evidence tasks).'),
        'input_schema': {'type': 'object', 'properties': {}, 'required': []},
    }] + (evidence_tools('task',annotation.get('protocol')) if annotation else []) + (simulator_tools('task') if (context or {}).get('simulator_episode') else [])


def dispatch(action, context=None):
    if (context or {}).get("annotation") or (context or {}).get("simulator_episode"):
        return dispatch_tool(action, describe(context))
    return dispatch_tool(action, describe())
