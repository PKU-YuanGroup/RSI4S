"""OpenAI-compatible HTTP API with durable, conservative per-task budgets."""
from __future__ import annotations

import http.client
import json
import signal
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from rsi.common import encoded, write_json
from rsi.providers import API_REQUEST_TIMEOUT_SECONDS, TOKEN_STALL_SECONDS, TRANSPORT_PROTOCOL
from rsi.api_stream import ChatStream, StreamProtocolError


# Provider-side serving errors that the endpoint reports with HTTP 400 although the request is
# valid and a later attempt can succeed. Observed from the Qwen endpoint as
# `InvalidParameter.PreemptionPlan` ("Master preemption is disabled for this model deployment"),
# sometimes in an SSE `data:` body on a non-stream 400. Every other 400 stays non-retryable.
TRANSIENT_PROVIDER_CODES = frozenset({'InvalidParameter.PreemptionPlan'})
# Wording of in-body provider errors (often HTTP 200, after the stream began) that names a
# transient serving condition, so the same request may succeed on another attempt.
TRANSIENT_PROVIDER_MESSAGES = ('timeout', 'timed out', 'overload', 'rate limit', 'try again',
                               'temporarily unavailable')


def provider_error(raw):
    """Return (code, message) from a plain JSON or SSE `data:` provider error body, else None."""
    candidates = [raw.strip()]
    candidates += [line[5:].strip() for line in raw.splitlines() if line.startswith('data:')]
    for text in candidates:
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            continue
        error = value.get('error') if isinstance(value, dict) else None
        if isinstance(error, dict) and isinstance(error.get('code'), str):
            return error['code'], str(error.get('message', ''))
    return None


class BudgetExceeded(RuntimeError):
    pass


class APIRequestError(RuntimeError):
    def __init__(self, message, *, retryable=False, http_status=None):
        super().__init__(message)
        self.retryable, self.http_status = retryable, http_status


class APIContextExceeded(APIRequestError):
    """The provider rejected this input size; the Agent can compose another request."""


@contextmanager
def request_deadline(seconds):
    """Bound the whole HTTP exchange, including a response that keeps trickling."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Model HTTP calls must run in the worker main thread')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('Cannot replace an existing real-time alarm')
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError('API request absolute deadline exceeded')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class TransportTimeout(TimeoutError):
    def __init__(self, kind):
        super().__init__({'token_stall': 'No token within the stall limit',
                          'research_deadline': 'Research deadline reached during API request',
                          'absolute_fallback': 'Non-SSE response exceeded the absolute limit'}[kind])
        self.kind = kind


class TokenWatchdog:
    """One SIGALRM timer: token stall (or the non-SSE limit) and the research deadline.

    Socket timeouts cannot enforce the stall rule because heartbeats reset them.
    """

    def __init__(self, stall_seconds, research_deadline):
        self.stall_seconds, self.research_deadline = stall_seconds, research_deadline
        self.limit_at, self.limit_kind = None, None

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Model HTTP calls must run in the worker main thread')
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError('Cannot replace an existing real-time alarm')
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.expired)
        self.token()
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)

    def expired(self, signum, frame):
        raise TransportTimeout(self.kind)

    def arm(self):
        now = time.monotonic()
        self.kind = self.limit_kind
        at = self.limit_at
        if self.research_deadline <= at:
            at, self.kind = self.research_deadline, 'research_deadline'
        if at <= now:
            raise TransportTimeout(self.kind)
        signal.setitimer(signal.ITIMER_REAL, at - now)

    def token(self):
        self.limit_at, self.limit_kind = time.monotonic() + self.stall_seconds, 'token_stall'
        self.arm()

    def absolute(self, at):
        self.limit_at, self.limit_kind = at, 'absolute_fallback'
        self.arm()


class Budget:
    def __init__(self, limits, journal):
        self.limits = limits
        self.journal = journal
        self.calls = 0
        self.charged_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unknown_usage_calls = 0

    def reserve(self, messages, max_output):
        # UTF-8 bytes upper-bound ordinary byte-tokenized text. Extra room covers
        # role/chat wrappers. Every HTTP attempt reserves separately; ambiguous
        # failures stay charged even when an explicitly configured retry succeeds.
        reserved = len(encoded(messages).encode()) + 1024 + max_output
        if self.limits['api_calls'] is not None and self.calls >= self.limits['api_calls']:
            raise BudgetExceeded('API call budget exhausted')
        if self.limits['api_tokens'] is not None and self.charged_tokens + reserved > self.limits['api_tokens']:
            raise BudgetExceeded('Insufficient tokens to reserve complete API request')
        self.calls += 1
        self.charged_tokens += reserved
        self.journal.add('api_reserved', call=self.calls, reserved_tokens=reserved,
                         charged_tokens=self.charged_tokens)
        return reserved

    def settle(self, reserved, usage):
        valid = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                                             for k in ('prompt_tokens', 'completion_tokens'))
        if valid:
            prompt, completion = usage['prompt_tokens'], usage['completion_tokens']
            self.charged_tokens += prompt + completion - reserved
            self.prompt_tokens += prompt
            self.completion_tokens += completion
        else:
            self.unknown_usage_calls += 1
        self.journal.add('api_settled', call=self.calls, usage=usage if valid else None,
                         totals=self.snapshot())

    def snapshot(self):
        return {'calls': self.calls, 'charged_tokens': self.charged_tokens,
                'reported_prompt_tokens': self.prompt_tokens,
                'reported_completion_tokens': self.completion_tokens,
                'unknown_usage_calls': self.unknown_usage_calls,
                'limits': self.limits, 'cost_usd': None,
                'cost_note': 'Token usage recorded; no unverified API prices assumed.'}


class APIOutputTruncated(RuntimeError):
    """A billed response hit its output cap before completing an Agent action."""


def bounded_agent_response(content, finish_reason):
    """Keep one complete JSON action and restore a provider-stripped end tag.

    Decode JSON before finding its end: literal closing tags inside a command
    string must never truncate that command. Raw provider responses stay logged.
    """
    start = content.find('<json>')
    tagged = start >= 0
    body_start = start + len('<json>') if tagged else len(content)-len(content.lstrip())
    candidate = content[body_start:]
    whitespace = len(candidate)-len(candidate.lstrip())
    try:
        value, end = json.JSONDecoder().raw_decode(candidate.lstrip())
    except ValueError:
        value = None
    if isinstance(value, dict) and ('response' in value or 'tool_name' in value):
        result = content[:body_start+whitespace+end]
        return result + ('\n</json>' if tagged else '')
    if finish_reason == 'length':
        raise APIOutputTruncated('Model output reached the per-response token limit before a complete action. '
                                 'Split large code/file edits into smaller calls and return one complete JSON action.')
    return content


class APIClient:
    def __init__(self, credential_path, model, budget, directory, journal, max_output='model_max',
                 request_timeout=API_REQUEST_TIMEOUT_SECONDS, max_retries=0, retry_delay=2, stop_after_action=False,
                 token_stall_seconds=TOKEN_STALL_SECONDS):
        # Construct only at the live-model stage. No credentials in configuration
        # snapshots or child environments. No implicit model/provider fallback.
        from rsi.providers import load_credentials, generation_settings
        self.key, base = load_credentials(credential_path)
        base = base.rstrip('/')
        self.provider_settings = generation_settings(model)
        if not isinstance(self.key, str) or not self.key:
            raise ValueError('Credential api_key is empty')
        if not base.startswith(('https://', 'http://127.0.0.1:', 'http://localhost:')):
            raise ValueError('API must use HTTPS, or loopback for tests')
        self.url = base if base.endswith('/chat/completions') else base + '/chat/completions'
        self.model, self.budget, self.directory, self.journal = model, budget, Path(directory), journal
        self.max_output = 'model_max' if max_output is None else max_output
        if self.max_output != 'model_max' and (type(self.max_output) is not int or self.max_output < 1):
            raise ValueError('max_output must be model_max, null or a positive integer')
        if request_timeout <= 0 or token_stall_seconds <= 0 or type(max_retries) is not int or max_retries < 0 or retry_delay < 0:
            raise ValueError('Invalid API timeout/retry configuration')
        self.request_timeout, self.max_retries, self.retry_delay = request_timeout, max_retries, retry_delay
        self.token_stall_seconds = token_stall_seconds
        self.stop_after_action = stop_after_action
        self.model_limits = None
        # Set by the runtime: callable(seconds) -> seconds actually returned to the research
        # clock. None leaves failed attempts on the caller's clock (tests, standalone use).
        self.infrastructure_clock = None

    def _resolve_max_output(self, remaining, *, require_context=False):
        """Use the provider's actual maximum, not its smaller omitted-field default.

        Resolve once per experiment; this metadata GET generates no model tokens.
        Fail explicitly if unavailable instead of silently restoring an 8K cap.
        """
        if self.model_limits is not None:
            if require_context:
                from rsi.judge_context import judge_limits
                judge_limits(self.model_limits)
            return
        if self.max_output != 'model_max' and not require_context:
            return
        url = self.url.removesuffix('/chat/completions') + '/models'
        req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + self.key})
        try:
            with request_deadline(min(self.request_timeout, remaining)):
                with urllib.request.urlopen(req, timeout=min(self.request_timeout, remaining)) as response:
                    payload = json.load(response)
            rows = payload.get('data') if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError('Model metadata requires a data list')
            match = next((row for row in rows if isinstance(row, dict) and row.get('id') == self.model), None)
            maximum = match.get('max_output_tokens') if match else None
            from rsi.providers import documented_limits
            documented = documented_limits(self.model) if match else None
            if (type(maximum) is not int or maximum < 1) and documented is not None:
                resolved = {**documented, 'provider_metadata_source': url}
                maximum = resolved['max_output_tokens']
            else:
                if type(maximum) is not int or maximum < 1:
                    raise ValueError('Selected model has no valid max_output_tokens metadata')
                resolved = {'model': self.model, 'max_output_tokens': maximum,
                            'context_window': match.get('context_window'), 'source': url}
            if require_context:
                from rsi.judge_context import judge_limits
                judge_limits(resolved)
            write_json(self.directory / 'model_limits.json', resolved)
            self.journal.add('model_limits_resolved', **resolved)
            self.model_limits = resolved
            if self.max_output == 'model_max':
                self.max_output = maximum
        except Exception as exc:
            detail = str(exc).replace(self.key, '[REDACTED]')
            status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            retryable = (status in (408, 429) or status >= 500) if status else isinstance(
                exc, (TimeoutError, ConnectionError, urllib.error.URLError,
                      http.client.IncompleteRead, http.client.RemoteDisconnected, json.JSONDecodeError))
            self.journal.add('model_limits_error', http_status=status, retryable=retryable, message=detail)
            raise APIRequestError('Cannot resolve provider output maximum: ' + detail,
                                  retryable=retryable, http_status=status) from None

    def _retry(self, operation, role, version, remaining):
        # Post-exploration validity review has no research deadline. HTTP
        # attempts still obey the token timer and the configured retry count.
        if remaining is None and role != 'judge':
            raise ValueError('Only the final judge may run outside the research deadline')
        if remaining is not None and remaining <= 0:
            raise BudgetExceeded('Solve time exhausted before API request')
        deadline = time.monotonic() + remaining if remaining is not None else float('inf')
        for attempt in range(self.max_retries + 1):
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise BudgetExceeded('Solve time exhausted before API request')
            attempt_started = time.monotonic()
            try:
                return operation(seconds)
            except APIRequestError as exc:
                delay = self.retry_delay * (2 ** attempt)
                if exc.retryable and remaining is not None:
                    # A failed provider/transport attempt is host infrastructure time.
                    deadline += self._exclude_infrastructure_time(
                        time.monotonic() - attempt_started, 'failed_attempt', role, version, exc)
                if not exc.retryable or attempt == self.max_retries or deadline - time.monotonic() <= delay:
                    raise
                self.journal.add('api_retry', previous_call=self.budget.calls, role=role,
                                 version=version, retry=attempt + 1, delay_seconds=delay,
                                 reason=str(exc))
                time.sleep(delay)
                if remaining is not None:
                    # Returned after the sleep: only time that actually elapsed is returned.
                    deadline += self._exclude_infrastructure_time(delay, 'retry_backoff', role, version, exc)

    def _exclude_infrastructure_time(self, seconds, interval, role, version, exc):
        """Return time lost to a failed attempt or its backoff to the research clock, when wired.

        Agent code does not run while the host waits on the provider: each Agent step is a fresh
        sandboxed process whose descendants are terminated, so the returned time carries no work."""
        clock = self.infrastructure_clock
        if clock is None or seconds <= 0:
            return 0
        credited = clock(seconds)
        self.journal.add('api_infrastructure_time_excluded', interval=interval, role=role, version=version,
                         call=self.budget.calls, seconds=seconds, credited_seconds=credited,
                         http_status=exc.http_status, reason=str(exc)[:300])
        return credited

    def judge_limits(self):
        """Resolve full judge capacity independently of Agent output settings."""
        from rsi.judge_context import judge_limits
        self._retry(lambda seconds: self._resolve_max_output(seconds, require_context=True),
                    'judge', 'model_limits', None)
        return judge_limits(self.model_limits)

    def complete(self, messages, role, version, remaining, *, purpose=None):
        if purpose not in (None, 'complete_task_history') or (purpose and role != 'meta'):
            raise ValueError('Complete-history API settings require a Meta review request')
        def request(seconds):
            deadline = time.monotonic() + seconds
            self._resolve_max_output(seconds, require_context=role == 'judge')
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise BudgetExceeded('Solve time exhausted during model metadata request')
            if purpose:
                return self._complete_once(messages, role, version, seconds, purpose=purpose)
            return self._complete_once(messages, role, version, seconds)
        return self._retry(request, role, version, remaining)

    @staticmethod
    def _transport_fields(stream, last_token_seconds, timeout_kind, absolute_fallback):
        return dict(protocol=TRANSPORT_PROTOCOL, text_events=stream.text_events if stream is not None else 0,
                    last_token_seconds=last_token_seconds, timeout_kind=timeout_kind,
                    absolute_fallback=absolute_fallback)

    def _complete_once(self, messages, role, version, seconds, *, purpose=None):
        from rsi.providers import agent_wire_messages
        messages = agent_wire_messages(messages, self.provider_settings, role, purpose)
        maximum = self.max_output
        if purpose == 'complete_task_history':
            maximum = min(maximum, self.provider_settings['complete_history_max_output'])
        if role == 'judge':
            from rsi.judge_context import judge_limits, require_context
            limits = judge_limits(self.model_limits)
            size = require_context(messages, limits)
            self.journal.add('judge_request_sized', version=version, **size)
            maximum = limits['max_output_tokens']
        request = {'model': self.model, 'messages': messages,
                   self.provider_settings['output_parameter']: maximum,
                   'stream': self.provider_settings['stream'], 'stream_options': self.provider_settings['stream_options'],
                   **self.provider_settings['sampling_parameters'],
                   **self.provider_settings['request_options']}
        # HyperAgents uses tagged JSON tool calls and normal final answers.
        # The independent validity judge retains its original JSON-only format.
        if purpose == 'complete_task_history':
            request['response_format'] = {'type': 'json_object'}
            # Review is a Meta model call and retains the same thinking policy
            # as ordinary research and final selection for every provider.
        elif role not in ('meta', 'task'):
            request['response_format'] = {'type': 'json_object'}
        elif self.stop_after_action:
            # The leading literal newline cannot occur unescaped inside a valid
            # JSON string, so code that contains "</json>" remains expressible.
            request['stop'] = ['\n</json>']
        reserved = self.budget.reserve(messages, maximum)
        call = self.budget.calls
        write_json(self.directory / f'{call:04d}.request.json', request)
        from rsi.context import measure_messages
        self.journal.add('api_start', call=call, role=role, version=version, model=self.model,
                         **measure_messages(messages))
        started = time.monotonic()
        usage = None
        status = None
        body = bytearray()
        stream = None
        first_byte_seconds = None
        response_headers = {}
        progress_at = None
        last_token_seconds = None
        timeout_kind = None
        absolute_fallback = False
        research_deadline = started + seconds
        try:
            req = urllib.request.Request(self.url, data=encoded(request).encode(),
                                         headers={'Authorization': 'Bearer ' + self.key,
                                                  'Content-Type': 'application/json'})
            with TokenWatchdog(self.token_stall_seconds, research_deadline) as watchdog:
                try:
                    response = urllib.request.urlopen(req)
                except urllib.error.HTTPError as exc:
                    response = exc  # Preserve error bodies under the same deadline.
                with response:
                    status = response.status
                    response_headers = {key: value for key, value in response.headers.items()
                                        if key.lower() in ('content-type', 'transfer-encoding', 'server',
                                                           'date', 'x-request-id', 'request-id')}
                    content_type = response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                    if request['stream'] and status < 400 and content_type == 'text/event-stream':
                        stream = ChatStream()
                    else:
                        # The token rule cannot apply without SSE: keep the old limit.
                        absolute_fallback = True
                        watchdog.absolute(started + self.request_timeout)
                    try:
                        while chunk := response.read1(65536):
                            if first_byte_seconds is None:
                                first_byte_seconds = time.monotonic() - started
                            body.extend(chunk)
                            if stream is not None:
                                text_events = stream.text_events
                                stream.feed(chunk)
                                now = time.monotonic()
                                if stream.text_events > text_events:
                                    last_token_seconds = now - started
                                    watchdog.token()
                                if stream.events and (progress_at is None or now - progress_at >= 30):
                                    self.journal.add('api_stream_progress', call=call, role=role,
                                                     seconds=now - started, received_bytes=len(body),
                                                     stream_events=stream.events, text_events=stream.text_events,
                                                     content_characters=sum(map(len, stream.content)),
                                                     reasoning_characters=sum(map(len, stream.reasoning)))
                                    progress_at = now
                                if stream.done or stream.error is not None:
                                    break
                    except http.client.IncompleteRead as exc:
                        # Keep any undelivered bytes for the redacted audit log;
                        # a truncated response must never be accepted as an answer.
                        body.extend(exc.partial)
                        raise
            raw = body.decode(errors='replace').replace(self.key, '[REDACTED]')
            if status >= 400:
                write_json(self.directory / f'{call:04d}.error.json', {'http_status': status, 'body': raw})
            try:
                payload = stream.payload() if stream is not None else json.loads(raw)
            except StreamProtocolError as exc:
                raise APIRequestError(str(exc), retryable=True, http_status=status) from None
            except (ValueError, TypeError):
                found = provider_error(raw) if status >= 400 else None
                if found and found[0] in TRANSIENT_PROVIDER_CODES:
                    raise APIRequestError(f'Provider error (HTTP {status}): {found[1][:2000]}',
                                          retryable=True, http_status=status) from None
                raise APIRequestError(f'HTTP {status}: invalid JSON response',
                                      retryable=status < 400 or status in (408, 429) or status >= 500,
                                      http_status=status) from None
            if not isinstance(payload, dict):
                raise APIRequestError('API response must be an object', retryable=True, http_status=status)
            usage = payload.get('usage')
            # Credential redaction is defense in depth against an endpoint that
            # includes a request header in its diagnostic response.
            safe = json.loads(encoded(payload).replace(self.key, '[REDACTED]'))
            write_json(self.directory / f'{call:04d}.response.json', safe)
            if status >= 400 or payload.get('error') is not None:
                error = payload.get('error', {})
                message = error.get('message', encoded(error)) if isinstance(error, dict) else str(error)
                code = error.get('code') if isinstance(error, dict) else None
                context_error = code in ('context_length_exceeded', 'context_window_exceeded',
                                          'prompt_too_long', 'input_too_long')
                diagnostic = message.lower()
                context_error |= ('context length' in diagnostic or 'context window' in diagnostic) and (
                    'exceed' in diagnostic or 'too long' in diagnostic or 'maximum' in diagnostic)
                if status in (400, 413, 422) and context_error:
                    raise APIContextExceeded(f'Provider context limit (HTTP {status}): {message[:2000]}',
                                             http_status=status)
                from rsi.providers import provider_error_block
                block = provider_error_block(self.provider_settings, status, message)
                if block:
                    # A contract-declared refusal of this sample (see blocked_errors): retried.
                    self.journal.add('api_provider_block', call=call, role=role, version=version, block=block,
                                     http_status=status, error_code=code)
                    raise APIRequestError(f'Provider safety filter rejected the call ({block}, HTTP {status}); '
                                          f'not delivered to the Agent: {message[:2000]}',
                                          retryable=True, http_status=status)
                transient = status in (408, 429) or status >= 500 or code in TRANSIENT_PROVIDER_CODES or (
                    status < 400 and any(s in message.lower() for s in TRANSIENT_PROVIDER_MESSAGES))
                raise APIRequestError(f'Provider error (HTTP {status}): {message[:2000]}',
                                      retryable=transient, http_status=status)
            choices = payload.get('choices')
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            message = choice.get('message')
            content = message.get('content') if isinstance(message, dict) else None
            # Always-on provider reasoning can consume the whole review limit and
            # leave empty content: that is truncation, not a transport failure.
            truncated = purpose == 'complete_task_history' and choice.get('finish_reason') == 'length'
            if not truncated and (not isinstance(content, str) or not content.strip()):
                raise APIRequestError('API response has no nonempty text content', retryable=True, http_status=status)
            self.journal.add('api_end', call=call, role=role, version=version,
                             seconds=time.monotonic() - started,
                             response_transport='sse' if stream is not None else 'json',
                             first_byte_seconds=first_byte_seconds,
                             stream_events=stream.events if stream is not None else None,
                             finish_reason=choice.get('finish_reason'),
                             **self._transport_fields(stream, last_token_seconds, None, absolute_fallback))
            if truncated:
                raise APIOutputTruncated('Complete-history evidence notes reached the output limit; '
                                         'retry this segment with shorter, complete JSON notes.')
            content = content.replace(self.key, '[REDACTED]')
            if self.stop_after_action and role in ('meta', 'task') and purpose is None:
                bounded = bounded_agent_response(content, choice.get('finish_reason'))
                self.journal.add('api_action_bounded', call=call, raw_characters=len(content),
                                 delivered_characters=len(bounded), finish_reason=choice.get('finish_reason'))
                return bounded
            return content
        except APIOutputTruncated:
            self.journal.add('api_output_truncated', call=call, role=role, max_output_tokens=maximum)
            raise
        except Exception as exc:
            if isinstance(exc, TransportTimeout):
                timeout_kind = exc.kind
            if not isinstance(exc, APIRequestError):
                exc = APIRequestError(f'{type(exc).__name__}: {exc}',
                                      retryable=timeout_kind != 'research_deadline' and isinstance(exc, (
                                          TimeoutError, ConnectionError, urllib.error.URLError,
                                          http.client.IncompleteRead, StreamProtocolError)),
                                      http_status=status)
            error = str(exc).replace(self.key, '[REDACTED]')
            self.journal.add('api_error', call=call, role=role, error_type=type(exc).__name__,
                             http_status=status, seconds=time.monotonic() - started,
                             message=error, retryable=exc.retryable,
                             **self._transport_fields(stream, last_token_seconds, timeout_kind, absolute_fallback))
            error_type = APIContextExceeded if isinstance(exc, APIContextExceeded) else APIRequestError
            raise error_type(f'API request failed: {error}', retryable=exc.retryable, http_status=status) from None
        finally:
            transport = dict(
                requested_stream=request['stream'], response_transport='sse' if stream is not None else 'json',
                http_status=status, first_byte_seconds=first_byte_seconds, received_bytes=len(body),
                response_headers=response_headers,
                stream_events=stream.events if stream is not None else None,
                stream_done=stream.done if stream is not None else None,
                finish_reason=stream.finish_reason if stream is not None else None,
                token_stall_seconds=self.token_stall_seconds, absolute_fallback_seconds=self.request_timeout,
                **self._transport_fields(stream, last_token_seconds, timeout_kind, absolute_fallback))
            write_json(self.directory / f'{call:04d}.transport.json',
                       json.loads(encoded(transport).replace(self.key, '[REDACTED]')))
            if body:
                (self.directory / f'{call:04d}.response.txt').write_text(
                    body.decode(errors='replace').replace(self.key, '[REDACTED]'))
            self.budget.settle(reserved, usage)
