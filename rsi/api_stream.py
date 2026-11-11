"""Assemble a complete text chat response from SSE without exposing partial actions."""
import json


class StreamProtocolError(ValueError):
    pass


class ChatStream:
    def __init__(self):
        self.pending = b''
        self.data = []
        self.done = False
        self.error = None
        self.metadata = {}
        self.content = []
        self.reasoning = []
        self.finish_reason = None
        self.usage = None
        self.events = 0
        self.text_events = 0

    def feed(self, chunk):
        self.pending += chunk
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            line = line.removesuffix(b'\r')
            if not line:
                if self.data:
                    self._event(b'\n'.join(self.data))
                    self.data = []
            elif line.startswith(b'data:'):
                self.data.append(line[5:].removeprefix(b' '))
            elif line.startswith(b':') or line.split(b':', 1)[0] in (b'event', b'id', b'retry'):
                continue
            elif line.strip():
                raise StreamProtocolError('Invalid SSE field')

    def _event(self, raw):
        if self.done:
            raise StreamProtocolError('Unexpected data after stream completion')
        if raw.strip() == b'[DONE]':
            self.done = True
            return
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise StreamProtocolError('Invalid JSON stream event') from exc
        if not isinstance(event, dict):
            raise StreamProtocolError('Stream event must be an object')
        self.events += 1
        if event.get('error') is not None:
            self.error = event
            return
        for key in ('id', 'model'):
            if key in event:
                if key in self.metadata and event[key] != self.metadata[key]:
                    raise StreamProtocolError('Inconsistent stream identity')
                self.metadata[key] = event[key]
        # MiMo timestamps each chunk at emission time, even within one id.
        # Preserve the first timestamp; only id/model identify the response.
        for key in ('created', 'system_fingerprint'):
            if event.get(key) is not None:
                self.metadata.setdefault(key, event[key])
        if event.get('usage') is not None:
            self.usage = event['usage']
        choices = event.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamProtocolError('Expected at most one streamed choice')
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                raise StreamProtocolError('Unexpected streamed choice index')
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                raise StreamProtocolError('Invalid streamed delta')
            if delta.get('tool_calls') or delta.get('function_call'):
                raise StreamProtocolError('Native function calls are not part of the text action protocol')
            has_text = False
            for key, parts in (('content', self.content), ('reasoning_content', self.reasoning)):
                value = delta.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise StreamProtocolError('Streamed text must be a string')
                    if value and self.finish_reason is not None:
                        raise StreamProtocolError('Text after finish reason')
                    parts.append(value)
                    has_text = has_text or bool(value)
            # Only nonempty text is model progress; heartbeats, empty deltas,
            # usage and finish_reason events must not extend the token timer.
            self.text_events += has_text
            finish = choice.get('finish_reason')
            if finish is not None:
                if not isinstance(finish, str) or not finish:
                    raise StreamProtocolError('Invalid finish reason')
                if self.finish_reason is not None and self.finish_reason != finish:
                    raise StreamProtocolError('Conflicting finish reasons')
                self.finish_reason = finish

    def payload(self):
        if self.error is not None:
            return self.error
        if not self.done or self.finish_reason is None:
            raise StreamProtocolError('Incomplete SSE response: finish reason and DONE are required')
        if self.pending.strip() or self.data:
            raise StreamProtocolError('Trailing incomplete SSE event')
        message = dict(role='assistant', content=''.join(self.content))
        if self.reasoning:
            message['reasoning_content'] = ''.join(self.reasoning)
        return dict(self.metadata, object='chat.completion',
                    choices=[dict(index=0, message=message, finish_reason=self.finish_reason)],
                    usage=self.usage)
