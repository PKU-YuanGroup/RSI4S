"""Evaluation phases share the existing deadline without separate quotas."""
import time

from rsi.common import write_json


DEFAULT_PLAN = dict(policy='shared_deadline_v1')


def phase_plan(value=None):
    if value is not None and value != DEFAULT_PLAN:
        raise ValueError('Evaluation phases share one deadline; phase weights and caps were removed')
    return dict(DEFAULT_PLAN)


class PhaseBudget:
    """Give each operation the remaining time and record actual consumption."""
    def __init__(self, deadline, directory, plan=None, *, clock=time.monotonic):
        self.plan, self.clock, self.deadline = phase_plan(plan), clock, deadline
        self.path = directory/'phase_timing.json'
        self.started = clock()
        self.total = max(0, deadline-self.started)
        self.report = dict(plan=self.plan, initial_seconds=self.total,
                           phases={}, scope='Existing evaluation deadline; no research clock extension')
        write_json(self.path, self.report)

    def run(self, name, function):
        remaining = self.deadline-self.clock()
        seconds = max(0, remaining)
        record = dict(status='running', allowed_seconds=seconds,
                      remaining_seconds=seconds)
        self.report['phases'][name] = record
        write_json(self.path, self.report)
        started = self.clock()
        try:
            if seconds <= 0:
                raise TimeoutError('No evaluation time available for '+name)
            result = function(seconds)
            if self.clock() > self.deadline:
                raise TimeoutError('Evaluation deadline exceeded during '+name)
            record['status'] = 'completed'
            return result
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['seconds'] = self.clock()-started
            write_json(self.path, self.report)
