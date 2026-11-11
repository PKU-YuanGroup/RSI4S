"""Deterministic prompt deduplication and session-scoped tool result recovery.

Never summarize scientific content with a model. Raw records are immutable;
display changes are explicit and recoverable through a capability, not a path.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from rsi.common import encoded, write_json
from rsi.context_policy import AGENT_TRANSPORT_BYTES, validate_policy
from rsi.history_view import FACT_BYTES, PAGE_BYTES, evaluation_view, json_bytes, object_view, text_page, text_prefix
from rsi.research_rules import RESEARCH_RULES


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def compact_budget(resources):
    result = {'remaining_solve_seconds': max(0, int(resources['remaining_solve_seconds'])),
              'session': {k: v for k, v in (resources.get('session') or {}).items()
                          if k in ('run_id', 'kind', 'phase', 'round_number') and v is not None}}
    if resources.get('final_selection_only'):
        result.update(research_closed=True, final_selection_only=True)
    elif resources.get('chain_distillation'):
        result.update(research_closed=True, chain_distillation=True)
    if resources.get('session_calls_left') is not None:
        result['session_calls_left'] = resources['session_calls_left']
    for key in ('research_elapsed_seconds', 'stage_elapsed_seconds',
                'stage_remaining_seconds', 'stage_action_remaining_seconds'):
        if resources.get(key) is not None:
            result[key] = max(0, int(resources[key]))
    # Preserve a configured finite API budget; unlimited accounting stays in logs.
    budget = resources.get('budget', {})
    limits = budget.get('limits', {})
    for limit, used in (('api_calls', 'calls'), ('api_tokens', 'charged_tokens')):
        if limits.get(limit) is not None:
            result[limit + '_remaining'] = max(0, limits[limit] - budget.get(used, 0))
    return result


def measure_messages(messages):
    """Per-HTTP-attempt measurements, including retries; not token estimates."""
    categories = Counter()
    for message in messages:
        text = message['content']
        category = {'system': 'system_rules_and_tools', 'assistant': 'assistant_history'}.get(
            message['role'], 'task_context_and_other')
        stripped = text.strip()
        if message['role'] == 'user':
            try:
                value = json.loads(stripped[6:-7] if stripped.startswith('<json>') and stripped.endswith('</json>') else stripped)
            except (ValueError, TypeError):
                value = None
            if isinstance(value, dict):
                if set(value) == {'host_budget'}:
                    category = 'host_current_budget'
                elif 'tool_output' in value:
                    category = 'tool_results'
        categories[category] += len(text)
    return {'input_characters': sum(categories.values()), 'input_characters_by_category': dict(categories)}


class PromptContext:
    def __init__(self, instruction, initial_resources, tool_results=None, final_submission=None, policy=None,
                 experience=None):
        self.policy = validate_policy(policy)
        self.resource_fingerprints = set()
        self.tool_messages = {}
        self.history = (ConversationHistory(tool_results, trigger=self.policy['archive_trigger_chars'],
                                           recent=self.policy['archive_recent_chars'])
                        if tool_results is not None and self.policy['archive_history'] else None)
        self.recent_scores, self.recent_errors = [], []
        self.best_score = None
        fixed = {'host_instruction': instruction,
                              'execution_contract': 'Bash runs with pipefail. All child processes are cleaned up '
                              'whenever a tool call ends, even if the shell exits successfully. Run work in the '
                              'foreground or wait for children in that same call; nohup/background jobs do not '
                              'persist.',
                              'scientific_validity_rules': RESEARCH_RULES,
                              'experiment': initial_resources.get('experiment', {}),
                              'hardware': initial_resources.get('hardware', {}),
                              'context_settings': self.policy,
                              'context_policy': 'Only the last host_budget is current. Tool results have call_id; '
                              'read_tool_result recovers exact session-local records by character offset. '
                              'Display and archival settings come from your versioned context_policy.json. '
                              'Read pages have byte caps: '
                              'limit is an upper bound, so always use returned next_offset. '
                              'Old conversation may be archived with a history_archive call_id; '
                              'read_tool_result(field=json) recovers its exact messages and previous '
                              'archive link. Agent state and requests share an 8 MiB transport limit; '
                              'model input remains subject to the provider context capacity. '
                              'Raw logs and scores remain recorded. Clipped display is not the full output.'}
        if final_submission is not None:
            fixed['final_submission'] = final_submission
        if initial_resources.get('final_selection_only'):
            fixed['execution_contract'] = ('Research is closed. Only history reads, get_time, model calls and '
                'selection of an existing verified artifact are allowed. No shell, Python tools, '
                'file edits, training, queries, data fetching, evaluation or Agent publication.')
        self.fixed = encoded(fixed)
        if experience:
            # Rendered chain notes stay readable Markdown after the static host rules.
            self.fixed += '\n\n' + experience
        self.observe_resources(initial_resources)

    def observe_resources(self, value):
        # Match exact host-produced objects, not words in arbitrary research text.
        self.resource_fingerprints.add(fingerprint(value))

    def observe_tool(self, request, display):
        self.tool_messages[fingerprint(display)] = (request, display)
        call_id = display.get('_tool_result', {}).get('call_id')
        if request.get('op') == 'evaluate' and 'attempt' in display:
            score = evaluation_view(display)
            score['call_id'] = call_id
            self.recent_scores = (self.recent_scores + [score])[-3:]
            value = score.get('aggregate_improvement')
            if (score.get('valid_primary_score') and type(value) in (int, float) and
                    (self.best_score is None or value > self.best_score['aggregate_improvement'])):
                self.best_score = score
        if display.get('error') or display.get('returncode') or display.get('reason'):
            error = {k: display[k] for k in ('error', 'returncode', 'reason') if k in display}
            error['call_id'] = call_id
            if isinstance(display.get('stdout'), str):
                error['stdout_tail'] = display['stdout'][-2000:]
                error['stdout_tail_is_excerpt'] = len(display['stdout']) > 2000
            self.recent_errors = (self.recent_errors + [error])[-3:]

    def render(self, messages, resources):
        started = time.monotonic()
        self.observe_resources(resources)
        output, categories = [], Counter()
        removed_states = removed_state_chars = removed_echo_chars = 0
        for message in messages:
            content = message['content']
            category = 'assistant_history' if message['role'] == 'assistant' else (
                'system_and_tools' if message['role'] == 'system' else 'task_context_and_other')
            value = None
            if message['role'] == 'user':
                candidate = content.strip()
                tagged = candidate.startswith('<json>') and candidate.endswith('</json>')
                try:
                    value = json.loads(candidate[6:-7] if tagged else candidate)
                except (ValueError, TypeError):
                    pass
                if self.policy['deduplicate_resources'] and isinstance(value, dict) and set(value) == {'resources'} and (
                        fingerprint(value['resources']) in self.resource_fingerprints):
                    removed_states += 1
                    removed_state_chars += len(content)
                    continue
                if isinstance(value, dict) and 'tool_output' in value:
                    category = 'tool_results'
                    known = self.tool_messages.get(fingerprint(value['tool_output']))
                    if self.policy['deduplicate_tool_inputs'] and known and 'tool_input' in value:
                        request, _ = known
                        arguments = {k: v for k, v in request.items() if k != 'op'}
                        if value.get('tool_name') == request['op'] and value['tool_input'] == arguments:
                            # The original executed request is also in the immutable result record.
                            value = {k: v for k, v in value.items() if k != 'tool_input'}
                            content = '<json>\n' + encoded(value) + '\n</json>'
                            removed_echo_chars += max(0, len(message['content']) - len(content))
            output.append({'role': message['role'], 'content': content})
            categories[category] += len(content)
        history_metrics = {'archived_history_messages': 0, 'history_archive': None,
                           'history_archived_characters': 0, 'new_history_archive': None}
        if self.history:
            facts = {
                'best_scored_submission': self.best_score,
                'recent_evaluations': self.recent_scores, 'recent_errors': self.recent_errors}
            output, history_metrics = self.history.render(
                output, facts if self.policy['include_scientific_facts'] else {})
            # Categories must describe the actual request after archival.
            categories = Counter(measure_messages(output)['input_characters_by_category'])
        # Keep existing message positions; the static host rules share the system prefix.
        if output and output[0]['role'] == 'system':
            output[0] = {**output[0], 'content': output[0]['content'] + '\n\n' + self.fixed}
            categories['host_fixed_rules'] += 2
        else:
            output.insert(0, {'role': 'system', 'content': self.fixed})
        categories['host_fixed_rules'] += len(self.fixed)
        dynamic = encoded({'host_budget': compact_budget(resources)})
        output.append({'role': 'user', 'content': dynamic})
        categories['host_current_budget'] += len(dynamic)
        return output, {'input_characters': sum(len(m['content']) for m in messages),
                        'rendered_characters': sum(len(m['content']) for m in output),
                        'removed_resource_messages': removed_states,
                        'removed_resource_characters': removed_state_chars,
                        'removed_echo_characters': removed_echo_chars,
                        'category_characters': dict(categories),
                        'render_seconds': time.monotonic() - started,
                        'unit': 'Unicode characters, not tokens', 'semantic_summary_calls': 0,
                        'context_policy': self.policy, **history_metrics}


class ConversationHistory:
    """Archive old turns at soft size thresholds, retaining exact recoverable evidence.

    Never truncate task instructions, system messages, or the most recent turn.
    These are display thresholds in characters, not API/token/tool-call limits.
    Agent-owned state and the original API audit are not rewritten.
    """
    def __init__(self, archive, trigger=120000, recent=60000):
        self.archive, self.trigger, self.recent = archive, trigger, recent
        self.prefix = []
        self.latest = None
        self.branch = 0

    def render(self, messages, facts):
        first = next((i for i, m in enumerate(messages) if m['role'] == 'assistant'), len(messages))
        pinned = messages[:first]
        history = [m for m in messages[first:] if m['role'] != 'system']
        pinned += [m for m in messages[first:] if m['role'] == 'system']
        signatures = [fingerprint(m) for m in history]
        if signatures[:len(self.prefix)] != self.prefix:
            # Evolved agents may rewrite/prune their own state. Never reuse an
            # archive marker for messages that are not the current prefix.
            self.prefix, self.latest = [], None
            self.branch += 1
        cut = len(self.prefix)
        remaining = sum(len(m['content']) for m in history[cut:])
        new_archive = None
        start = cut
        if remaining > self.trigger:
            start, chars = len(history), 0
            while start > cut and (chars < self.recent or len(history) - start < 4):
                start -= 1
                chars += len(history[start]['content'])
            # Keep complete assistant/tool-result turns together.
            while start > cut and history[start]['role'] != 'assistant':
                start -= 1
            if start > cut:
                candidate = self.marker(history[start:], facts, start, 'pending-history-' + 'x' * 96)
                saved = sum(len(m['content']) for m in history[cut:start])
                # A summary must not cost more than the block it replaces.
                if saved <= len(encoded(candidate)):
                    start = cut
            if start > cut:
                new_archive = self.archive.archive_history(history[cut:start], self.latest, self.branch)
                self.latest = new_archive
                cut = start
                self.prefix = signatures[:cut]
        if not cut:
            return messages, {'archived_history_messages': 0, 'history_archive': None,
                              'history_archived_characters': 0, 'new_history_archive': None}
        marker = self.marker(history[cut:], facts, cut, self.latest)
        archived_chars = sum(len(m['content']) for m in history[:cut])
        if len(encoded(marker)) >= archived_chars:
            marker['scientific_facts'] = {'details': 'Read the archive for exact evidence.'}
        if len(encoded(marker)) >= archived_chars:
            return messages, {'archived_history_messages': 0, 'history_archive': None,
                              'history_archived_characters': 0, 'new_history_archive': new_archive}
        output = pinned + [{'role': 'user', 'content': encoded(marker)}] + history[cut:]
        return output, {'archived_history_messages': cut, 'history_archive': self.latest,
                        'history_archived_characters': archived_chars,
                        'history_marker_characters': len(encoded(marker)),
                        'new_history_archive': new_archive}

    @staticmethod
    def marker(retained, facts, cut, archive_id):
        visible = set()
        for message in retained:
            if message['role'] != 'user':
                continue
            text = message['content'].strip()
            try:
                value = json.loads(text[6:-7] if text.startswith('<json>') and text.endswith('</json>') else text)
                shown = value.get('tool_output', value) if isinstance(value, dict) else {}
                call_id = shown.get('_tool_result', {}).get('call_id') if isinstance(shown, dict) else None
                if call_id:
                    visible.add(call_id)
            except (ValueError, TypeError):
                pass
        compact = dict(facts)
        seen = set(visible)
        best = facts.get('best_scored_submission')
        if isinstance(best, dict):
            compact['best_scored_submission'] = None if best.get('call_id') in visible else evaluation_view(best)
            seen.add(best.get('call_id') or fingerprint(best))
        for key in ('recent_evaluations', 'recent_errors'):
            if key not in facts:
                continue
            rows = []
            for item in facts[key]:
                identity = item.get('call_id') or fingerprint(item)
                if identity in seen:
                    continue
                seen.add(identity)
                rows.append(evaluation_view(item) if key == 'recent_evaluations' else object_view(item, 2500)[0])
            compact[key] = rows
        if json_bytes(compact) > FACT_BYTES:
            bounded = {'details_omitted': 'Read referenced tool records or conversation archives for full evidence.'}
            for key, value in compact.items():
                if isinstance(value, list):
                    bounded[key] = []
                    for item in value:
                        if json_bytes({**bounded, key: bounded[key] + [item]}) <= FACT_BYTES:
                            bounded[key].append(item)
                elif json_bytes({**bounded, key: value}) <= FACT_BYTES:
                    bounded[key] = value
            compact = bounded
        return {'history_archive': archive_id, 'archived_messages': cut,
                  'readback': 'read_tool_result with call_id=history_archive, field=json; '
                              'follow previous_archive for older turns. Archived text is not '
                              'a semantic summary. Raw evidence remains available.',
                  'scientific_facts': compact}


ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
PROGRESS = re.compile(r'\d{1,3}%\|')


def clean_log(text):
    """Only terminal noise and consecutive identical warnings/progress; not numbers/code."""
    text = ANSI.sub('', text)
    text = re.sub(r'\x00+', lambda m: f'[omitted {len(m[0])} NUL characters]', text)
    result = []
    repeats = 0
    previous = None
    for line in text.splitlines(keepends=True):
        eligible = bool(re.search(r'\b\w*Warning\b|\bWARNING\b', line) or PROGRESS.search(line))
        if eligible and line == previous:
            repeats += 1
            continue
        if repeats:
            result.append(f'[previous line repeated {repeats} additional times]\n')
            repeats = 0
        result.append(line)
        previous = line if eligible else None
    if repeats:
        result.append(f'[previous line repeated {repeats} additional times]\n')
    return ''.join(result)


def preview(text, limit=12000, byte_limit=PAGE_BYTES):
    if (limit is None or len(text) <= limit) and (byte_limit is None or json_bytes(text) <= byte_limit):
        return text, None
    half = len(text) if limit is None else limit // 2
    # Preserve the diagnostic tail even when Unicode or JSON escapes make the
    # ordinary character-limited preview exceed the tool-result byte budget.
    budget = max(1, (byte_limit - 256) // 2) if byte_limit is not None else None
    head = text[:half] if budget is None else text_prefix(text[:half], budget)
    tail = text[-half:] if budget is None else text_prefix(text[-half:][::-1], budget)[::-1]
    return (head + '\n[display clipped; use read_tool_result for exact text]\n' + tail,
            {'original_chars': len(text), 'shown_ranges': [[0, len(head)], [len(text) - len(tail), len(text)]],
             'offset_unit': 'characters; end exclusive'})


class ToolResults:
    """Only IDs created in this session can be read; caller paths are never accepted."""
    def __init__(self, directory, run_id, policy=None):
        self.policy = validate_policy(policy)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.records = {}

    def publish(self, number, request, raw, stdout_path=None):
        started = time.monotonic()
        call_id = f'{self.run_id}-{number:04d}'
        path = self.directory / f'{call_id}.json'
        if call_id in self.records or path.exists():
            raise ValueError('Tool result ID cannot be reused')
        write_json(path, {'call_id': call_id, 'request': request, 'response': raw})
        path.chmod(0o444)
        stdout = None
        if stdout_path is not None:
            stdout = self.directory / f'{call_id}.stdout'
            # This path is supplied by the host sandbox, never by a model request.
            stdout.write_bytes(Path(stdout_path).read_bytes())
            stdout.chmod(0o444)
        self.records[call_id] = (path, stdout)
        display = dict(raw) if isinstance(raw, dict) else {'result': raw}
        op = request['op']
        display_changes = []
        if op in ('bash', 'python', 'editor') and isinstance(display.get('stdout'), str):
            text = stdout.read_bytes().decode(errors='replace') if stdout else display['stdout']
            failed = bool(display.get('returncode') or display.get('reason'))
            # Preserve raw diagnostics and editor output exactly in the archive.
            # The Agent policy controls display clipping, including failures.
            # A metric such as "Mean Absolute Error" is not an exception. Match
            # tracebacks, exception-class messages and explicit error records.
            diagnostic = failed or bool(re.search(
                r'Traceback \(most recent|\b\w+(?:Error|Exception):|\bERROR\b|'
                r'(?m:^\s*(?:Error|Exception):)|(?i:out of memory)', text))
            cleaned = (text if op == 'editor' or diagnostic or not self.policy['clean_terminal_output']
                       else clean_log(text))
            if cleaned != text:
                display_changes.append('terminal_noise_or_identical_warning_lines')
            display_limit = self.policy['tool_display_bytes']
            rendered, clipped = preview(cleaned, self.policy['stdout_preview_chars'],
                                        None if display_limit is None else max(1, display_limit - 2048))
            display['stdout'] = rendered
            if clipped:
                display['stdout_display'] = {**clipped, 'field': 'stdout',
                                            'ranges_refer_to': 'cleaned display' if cleaned != text else 'raw stdout'}
                display_changes.append('head_and_tail_preview')
            # Fixed environment is in host rules; abnormalities retain resource diagnostics.
            for key in ('uid', 'log', 'resource_limits', 'peak_sampled_rss_bytes'):
                if not diagnostic or key in ('uid', 'log'):
                    display.pop(key, None)
            display_changes.append('execution_metadata_in_raw_record')
        display['_tool_result'] = {'call_id': call_id, 'read_tool': 'read_tool_result'}
        if display_changes:
            display['_tool_result']['display_changes'] = display_changes
        display_limit = self.policy['tool_display_bytes']
        if display_limit is not None and json_bytes(display) > display_limit:
            # The exact response above is immutable. Only its prompt-facing view
            # changes, with an explicit readback capability and no fake scores.
            display = evaluation_view(raw) if op == 'evaluate' and isinstance(raw, dict) else {}
            if op != 'evaluate' and isinstance(raw, dict):
                status, clipped = object_view({k: raw[k] for k in ('error', 'returncode', 'reason', 'status') if k in raw}, min(4096, display_limit // 4))
                display.update(status)
                if clipped:
                    display['status_details_omitted'] = True
            display.update(display_truncated=True,
                           readback='read_tool_result(call_id, field=json, offset=0); follow next_offset only as needed.',
                           _tool_result={'call_id': call_id, 'read_tool': 'read_tool_result'})
            if json_bytes(display) > display_limit - 256:
                display = {'display_truncated': True,
                           '_tool_result': {'call_id': call_id, 'read_tool': 'read_tool_result'}}
            if op != 'evaluate':
                display['preview_is_complete_json'] = False
                display['preview_json'] = text_prefix(encoded(raw), max(1, display_limit - json_bytes(display) - 256))
            display_changes.append('byte_bounded_result_preview')
        if display_changes:
            display['_tool_result']['display_changes'] = display_changes
        if json_bytes(display) > AGENT_TRANSPORT_BYTES:
            display = {'error': 'Tool display exceeds the 8 MiB Agent transport limit. '
                       'Exact output is available through read_tool_result.',
                       'code': 'agent_transport_limit', 'limit_bytes': AGENT_TRANSPORT_BYTES,
                       '_tool_result': {'call_id': call_id, 'read_tool': 'read_tool_result'}}
            display_changes.append('transport_limit_error')
        return display, {'call_id': call_id, 'raw_record': str(path),
                         'raw_record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'raw_stdout': str(stdout) if stdout else None,
                         'raw_stdout_sha256': hashlib.sha256(stdout.read_bytes()).hexdigest() if stdout else None,
                         'raw_characters': len(encoded(raw)), 'display_characters': len(encoded(display)),
                         'display_changes': display_changes, 'render_seconds': time.monotonic() - started}

    def read(self, request):
        call_id = request['call_id']
        if not isinstance(call_id, str) or call_id not in self.records:
            raise ValueError('Unknown tool result in this session')
        offset, limit = request.get('offset', 0), request.get('limit', 8000)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50000:
            raise ValueError('Use a nonnegative character offset and limit in 1..50000')
        path, stdout = self.records[call_id]
        field = request.get('field', 'json')
        if field == 'json':
            text = path.read_text()
        elif field == 'stdout':
            if stdout:
                text = stdout.read_bytes().decode(errors='replace')
            else:
                response = json.loads(path.read_text())['response']
                text = response.get('stdout') if isinstance(response, dict) else None
                if not isinstance(text, str):
                    raise ValueError('This result has no stdout')
        else:
            raise ValueError('field must be json or stdout')
        return {'call_id': call_id, 'field': field, **text_page(text, offset, limit)}

    def archive_history(self, messages, previous, branch):
        """Register an immutable conversation block for the existing read capability."""
        content = {'messages': messages, 'previous_archive': previous, 'branch': branch}
        call_id = f'{self.run_id}-history-{fingerprint(content)}'
        if call_id not in self.records:
            path = self.directory / f'{call_id}.json'
            if path.exists():
                raise ValueError('Conversation archive ID cannot be overwritten')
            write_json(path, {'call_id': call_id, 'request': {'op': 'archive_history'},
                              'response': content})
            path.chmod(0o444)
            self.records[call_id] = (path, None)
        return call_id
