"""Resumable HyperAgents-style text/tool conversation."""
import json
from pathlib import Path
import tools
from protocol import get_tooluse_prompt, parse_response


def step(state, context, response):
    if not state:
        template = Path(__file__).with_name('prompt.txt').read_text()
        instruction = template.replace('{repo_path}', context.get('repo_path', '<development repository>'))
        inputs = {key: context[key] for key in ('task_id', 'task', 'problem_path', 'workspace', 'agent_source_dir', 'output_dir', 'annotation', 'simulator_episode')
                  if key in context}
        instruction = instruction.replace('{inputs}', json.dumps(inputs, ensure_ascii=False))
        state = {'messages': [
            {'role': 'system', 'content': get_tooluse_prompt(tools.describe(context))},
            {'role': 'user', 'content': instruction}], 'waiting': 'llm'}
        if context.get('kind') == 'meta':
            state['messages'].append({'role': 'user', 'content': json.dumps({
                key: context[key] for key in ('eval_path', 'interface_path') if key in context})})
    elif state['waiting'] == 'llm' and response.get('error'):
        state['messages'].append({'role': 'user', 'content': response['error']})
    elif state['waiting'] == 'llm':
        text = response['text']
        state['messages'].append({'role': 'assistant', 'content': text})
        try:
            action = parse_response(text)
            if 'response' in action:
                request = {'op': 'finish', 'summary': action['response']}
            else:
                request = tools.dispatch(action, context)
            state['waiting'] = 'tool'
            state['last_action'] = action
            return {'state': state, 'request': request}
        except (ValueError, KeyError, TypeError) as exc:
            state['messages'].append({'role': 'user', 'content': str(exc)})
    else:
        if (context.get('session_info', {}).get('run_id') == 'T00'
                and state.get('last_action', {}).get('tool_name') == 'evaluate'
                and response.get('valid_primary_score') is True):
            state.setdefault('t00_first_valid_artifact', response.get('artifact') or 'the recorded evaluation')
        state['messages'].append({'role': 'user', 'content': '<json>\n' + json.dumps({
            'tool_name': state.get('last_action', {}).get('tool_name', 'finish'),
            'call_id': response.get('_tool_result', {}).get('call_id'), 'tool_output': response}, ensure_ascii=False) + '\n</json>'})
    state['waiting'] = 'llm'
    messages = state['messages']
    if state.get('t00_first_valid_artifact'):
        reminder = Path(__file__).with_name('handoff_prompt.txt').read_text()
        reminder = reminder.replace('{artifact}', str(state['t00_first_valid_artifact']))
        messages = messages + [{'role': 'user', 'content': reminder}]
    return {'state': state, 'request': {'op': 'llm', 'messages': messages}}
