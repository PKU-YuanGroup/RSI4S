"""HyperAgents-style developer tools; execution stays in the host sandbox."""
from toolkit.bash import tool_info as bash_info
from toolkit.editor import tool_info as editor_info
from toolkit.read_tool_result import tool_info as read_result_info
from toolkit.get_time import tool_info as time_info
from protocol import dispatch as dispatch_tool
from toolkit.evidence import tools as evidence_tools
from toolkit.simulator import tools as simulator_tools


def history_file_tools():
    return [
        {'name': 'list_history_files',
         'description': 'List metadata for a completed research session: model requests/responses, original prompts, tool outputs and full command logs. No file content is loaded. Inspect only runs relevant to the current decision. Default 20 files; limit 1..100.',
         'input_schema': {'type': 'object', 'properties': {'run_id': {'type': 'string'},
             'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}}, 'required': ['run_id']}},
        {'name': 'read_history_file',
         'description': 'Read one file listed by list_history_files, without execution or changes. Default 4096 UTF-8 bytes; limit 4..50000. Returned text also has a 24 KiB JSON byte cap; always use next_offset. Read the smallest relevant range; avoid duplicate or unchanged content and stop when the decision is supported.',
         'input_schema': {'type': 'object', 'properties': {'run_id': {'type': 'string'},
             'path': {'type': 'string'}, 'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}},
             'required': ['run_id', 'path']}},
    ]


def history_tools():
    paging = {'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}}
    result = [{'name': 'list_history',
        'description': 'Read a compact index of completed runs and immutable Agent versions in original order, not a ranking. Default 20 rows; limit 1..1000 is an upper bound and pages also have a byte cap. Follow next_offset and next_version_offset separately. next_evaluation_offset/next_artifact_offset within a run continue through read_record(section=evaluations/artifacts). Omitted details remain in exact records.',
        'input_schema': {'type': 'object', 'properties': {**paging, 'version_offset': {'type': 'integer'}}, 'required': []}}]
    for name, key, description in [
        ('read_record', 'run_id', 'Read exact record JSON text by default (4096 characters). Optional section=evaluations or artifacts returns compact entry pages (default 20); offset/limit then count entries, limit 1..1000. Every candidate remains reachable in original order.'),
        ('read_version', 'version', 'Read immutable Agent metadata and a paginated file manifest (default 20 files); offset/limit count file entries. section=manifest reads exact manifest JSON. path reads a source text file.'),
        ('read_artifact', 'artifact', 'Read artifact metadata and a paginated file manifest (default 20 files); offset/limit count file entries. section=manifest reads exact manifest JSON. path reads an existing text file. No execution or edits.')]:
        properties = {key: {'type': 'string'}, **paging}
        if name != 'read_record':
            properties['path'] = {'type': 'string'}
        properties['section'] = {'type': 'string', 'enum': ['evaluations', 'artifacts'] if name == 'read_record' else ['manifest']}
        result.append({'name': name, 'description': description + ' Text offsets count characters (limit 1..50000). All pages have a byte cap, so always continue with returned next_offset, never offset+limit. Read only relevant pages.',
                       'input_schema': {'type': 'object', 'properties': properties, 'required': [key]}})
    return result


def describe(context=None):
    context = context or {}
    evidence = evidence_tools('meta',context['annotation'].get('protocol')) if context.get('annotation') else []
    if context.get('final_selection_only'):
        return history_tools() + history_file_tools() + [read_result_info(), time_info()] + [t for t in evidence
            if t['name'] in ('select_final_artifact', 'finish_experiment')]
    return [bash_info(), editor_info(), read_result_info(), time_info()] + history_file_tools() + (simulator_tools('meta') if context.get('simulator_episode') else []) + [t for t in evidence
        if t['name'] not in ('select_final_artifact', 'finish_experiment')]


def dispatch(action, context=None):
    if (context or {}).get("annotation") or (context or {}).get("simulator_episode"):
        return dispatch_tool(action, describe(context))
    return dispatch_tool(action, describe())
