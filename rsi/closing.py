"""Independent finalization steps with a durable fallback summary."""
import sys

from rsi.common import encoded, write_json


class Finalizer:
    def __init__(self, directory, journal):
        self.directory, self.journal = directory, journal
        self.errors = []
        self.interrupted = False

    def failed(self, operation, exc):
        self.interrupted |= isinstance(exc, KeyboardInterrupt)
        failure = dict(operation=operation, error=f'{type(exc).__name__}: {exc}')
        self.errors.append(failure)
        try:
            self.journal.add('finalization_failed', **failure)
        except (Exception, KeyboardInterrupt) as logging_error:
            self.interrupted |= isinstance(logging_error, KeyboardInterrupt)
            self.errors.append(dict(operation='failure_log', error=str(logging_error)))

    def call(self, operation, function, default=None):
        try:
            return function()
        except (Exception, KeyboardInterrupt) as exc:
            self.failed(operation, exc)
            return default

    def event(self, name, **fields):
        return self.call('event:' + name, lambda: self.journal.add(name, **fields))

    def update(self, summary):
        summary['cleanup_errors'] = self.errors
        if self.errors:
            summary['status'] = 'failed'
            summary['error'] = summary.get('error') or 'Finalization failed: ' + self.errors[0]['error']
        return summary

    def save(self, summary):
        for name in ('summary.json', 'summary.recovery.json'):
            try:
                write_json(self.directory/name, self.update(summary))
                return
            except (Exception, KeyboardInterrupt) as exc:
                self.failed('save:' + name, exc)
        self.update(summary)
        print('Unable to persist final summary: ' + encoded(self.errors), file=sys.stderr)
