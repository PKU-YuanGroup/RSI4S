"""Resumable HyperAgents-style text/tool conversation."""
import json
from pathlib import Path
import tools
from protocol import get_tooluse_prompt, parse_response


def step(state, context, response):
    if 'messages' not in state:
        # A task chain's closing session distills Agents and notes for the next task.
        prompt = 'distill_prompt.txt' if context.get('chain_distillation') else 'prompt.txt'
        template = Path(__file__).with_name(prompt).read_text()
        instruction = template.replace('{repo_path}', context.get('repo_path', '<development repository>'))
        inputs = {key: value for key, value in context.items()
                  if key not in ('resources', 'research_budget', 'session_info', 'source_files')}
        instruction = instruction.replace('{inputs}', json.dumps(inputs, ensure_ascii=False))
        if context.get('final_selection_only'):
            instruction = ('Research has ended. Read the historical trajectories and verified validation '
                           'evidence, compare eligible existing artifacts, and choose the final submission '
                           'with a reason. Only history reads, get_time and selection are allowed. Do not modify '
                           'Agents or scientific artifacts, train, query labels, fetch data or re-evaluate. '
                           'Use list_history_files and read_history_file for completed model/tool '
                           'transcripts on demand; full history is not automatically loaded. '
                           'Minimize history reads: use metadata and concise notes first, then only '
                           'relevant files and ranges needed to resolve a concrete uncertainty. '
                           'Do not replay whole conversations or reread unchanged evidence; stop '
                           'when the comparison is adequately supported. Account for recorded '
                           'fitting overlap when interpreting validation scores. '
                           'Use finish_experiment to select and complete this review. Test is unavailable.')
        descriptions = tools.describe(context)
        state = {**state, 'messages': [
            {'role': 'system', 'content': get_tooluse_prompt(descriptions)},
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
            if ('response' in action and not state.get('tool_calls') and not state.get('handoff_check')
                    and not context.get('final_selection_only')):
                # A final response before any tool call is usually a plan, not a decision. Ask once; a repeated
                # final response finishes as before. Final selection already requires a selected artifact.
                state['handoff_check'] = True
                state['messages'].append({'role': 'user', 'content': (
                    'Your message is a final response. A final response ends this Meta stage and is recorded as '
                    'its decision, but no tool has been used in this stage yet. To finish now, send your final '
                    'response again. Otherwise continue with exactly one tool call.')})
            else:
                if 'response' in action:
                    request = {'op': 'finish', 'summary': action['response']}
                else:
                    request = tools.dispatch(action, context)
                    state['tool_calls'] = state.get('tool_calls', 0) + 1
                state['waiting'] = 'tool'
                state['last_action'] = action
                return {'state': state, 'request': request}
        except (ValueError, KeyError, TypeError) as exc:
            state['messages'].append({'role': 'user', 'content': str(exc)})
    else:
        state['messages'].append({'role': 'user', 'content': '<json>\n' + json.dumps({
            'tool_name': state.get('last_action', {}).get('tool_name', 'finish'),
            'call_id': response.get('_tool_result', {}).get('call_id'), 'tool_output': response}, ensure_ascii=False) + '\n</json>'})
    state['waiting'] = 'llm'
    return {'state': state, 'request': {'op': 'llm', 'messages': state['messages']}}
