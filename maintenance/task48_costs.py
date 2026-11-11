"""Observed stage costs, with gaps and sampling assumptions made explicit."""

from rsi.workflow import phase_after


def stage_spans(events):
    spans = []
    phase, start = 'unattributed', None
    last = None
    for event in events or []:
        now = event.get('time')
        if not isinstance(now, (float, int)):
            continue
        if last is not None and now < last:
            raise ValueError('Workflow event timestamps move backwards')
        last = now
        successor = phase_after(event, phase)
        if successor == phase:
            continue
        if start is not None:
            spans.append((start, now, phase))
        start, phase = now, successor
    # An unclosed phase has no invented end. Its telemetry stays unattributed.
    return spans


class CostAccumulator:
    def __init__(self, job, events, workflow, token_report, max_interval=15.):
        self.job, self.spans, self.tokens = job, stage_spans(events), token_report
        self.max_interval = max_interval
        self.stages, self.gaps, self.previous_end = {}, [], None
        self.lower = workflow.get('started_at', job.get('started_at'))
        self.upper = workflow.get('ended_at', job.get('ended_at'))
        self.total_seconds = workflow.get('wall_seconds')
        if self.total_seconds is None and self.lower is not None and self.upper is not None:
            self.total_seconds = max(0, self.upper-self.lower)
        for start, end, phase in self.spans:
            self.get(phase)['wall_seconds'] += end-start
        if self.total_seconds is not None:
            self.get('unattributed')['wall_seconds'] = max(0, self.total_seconds-sum(r['wall_seconds'] for r in self.stages.values()))

    def get(self, phase):
        return self.stages.setdefault(phase, dict(phase=phase, wall_seconds=0., observed_seconds=0.,
            resource_samples=0, cpu_core_seconds_sampled=None, gpu_resident_seconds_sampled=None,
            gpu_observed_seconds=0., peak_rss_bytes=None, peak_gpu_memory_mib=None,
            cross_stage_intervals=0, api_cost=None,
            api_cost_reason='No verified billing receipt or price schedule supplied'))

    def add(self, sample, current):
        end, dt = sample['time'], sample['interval_seconds']
        if dt <= 0:
            return
        start = end-dt
        if self.lower is not None:
            start = max(start, self.lower)
        if self.upper is not None:
            end = min(end, self.upper)
        if self.previous_end is not None:
            start = max(start, self.previous_end)
        if end <= start:
            return
        self.previous_end = end
        if dt > self.max_interval:
            self.gaps.append(dict(start=start, end=end, reason='Long sampling interval', seconds=end-start))
            return
        if self.lower is None and current.get('status') != 'running':
            return
        parts = [(max(start,a), min(end,b), p) for a,b,p in self.spans if a < end and b > start]
        covered = sum(b-a for a,b,_ in parts)
        if end-start-covered > 1e-6:
            parts.append((0., end-start-covered, 'unattributed'))
        for a,b,phase in parts:
            row = self.get(phase); seconds = b-a
            row['observed_seconds'] += seconds
            row['resource_samples'] += 1
            row['cross_stage_intervals'] += len(parts) > 1
            cpu = current.get('cpu_percent')
            if cpu is not None:
                row['cpu_core_seconds_sampled'] = (row['cpu_core_seconds_sampled'] or 0) + cpu/100*seconds
            memory = current.get('gpu_memory_mib')
            if memory is not None:
                row['gpu_observed_seconds'] += seconds
                row['gpu_resident_seconds_sampled'] = (row['gpu_resident_seconds_sampled'] or 0) + (seconds if memory > 0 else 0)
                row['peak_gpu_memory_mib'] = max(row['peak_gpu_memory_mib'] or 0, memory)
            if current.get('rss_bytes') is not None:
                row['peak_rss_bytes'] = max(row['peak_rss_bytes'] or 0, current['rss_bytes'])

    def finish(self):
        for phase, usage in self.tokens.get('by_phase', {}).items():
            self.get(phase)['api_usage'] = usage
        for row in self.stages.values():
            row['unsampled_seconds'] = max(0, row['wall_seconds']-row['observed_seconds'])
            row['sampling_policy'] = 'CPU uniform within interval; GPU resident at endpoint, not compute time; intervals >15s excluded'
        rows = list(self.stages.values())
        totals = dict(workflow_seconds=self.total_seconds, resource_samples=sum(r['resource_samples'] for r in rows),
                      sampling_gaps=self.gaps, api_cost=self.tokens.get('api_cost'),
                      api_cost_reason=self.tokens.get('api_cost_reason', 'No verified billing receipt or price schedule supplied'))
        for key in ('cpu_core_seconds_sampled', 'gpu_resident_seconds_sampled', 'peak_rss_bytes', 'peak_gpu_memory_mib'):
            values = [r[key] for r in rows if r[key] is not None]
            totals[key] = ((max(values) if key.startswith('peak_') else sum(values)) if values else None)
        totals['unsampled_seconds'] = sum(r['unsampled_seconds'] for r in rows) if self.total_seconds is not None else None
        return rows, totals


def costs(job, events, samples, workflow, token_report, max_interval=15.):
    accumulator = CostAccumulator(job, events, workflow, token_report, max_interval)
    for sample in samples:
        current = next((r for r in sample.get('jobs', []) if r['id'] == job['id']), None)
        if current is not None:
            accumulator.add(sample, current)
    return accumulator.finish()
