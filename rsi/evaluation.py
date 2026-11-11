"""Local adapter to the unmodified official service; one timer for all rounds."""
from __future__ import annotations

import json
import math
import secrets
import threading
import time
import sys
import urllib.error
import urllib.request
from pathlib import Path

import eval_service
from rsi.common import ROOT
from rsi.sandbox import Sandbox

# Host time lost to failed provider attempts (safety blocks, dropped streams, outages, token
# stalls, backoff) is returned to the shared research clock, at most this much per run so a long
# outage still ends the run. Introduced 2026-11-07; earlier runs counted that time as research.
INFRASTRUCTURE_EXCLUSION_LIMIT_SECONDS = 3600


class Evaluation:
    # Class defaults: protocol adapters with their own __init__ inherit them.
    infrastructure_excluded_seconds = 0.0        # returned to the research clock
    infrastructure_post_research_seconds = 0.0   # returned to post-research stage clocks

    def __init__(self, task, data_dir, run_dir, seconds, gpu, wall_seconds=None, resource_limits=None):
        self.task, self.directory = task, Path(run_dir)
        self.wall_seconds = wall_seconds
        self.resource_limits = resource_limits
        self.wall_deadline = None
        self.tracker = eval_service.ScoreTracker()
        self.token = secrets.token_urlsafe(32)
        self.batch = str(self.directory.resolve())
        primary, issues = self.tracker.register_task(
            task, Path(data_dir), timeout=seconds, out_dir=self.directory,
            batch_name=self.batch, eval_token=self.token)
        if issues or not primary:
            raise ValueError(f'Invalid official task metadata: {issues}')
        self.state = self.tracker.get_task(task, self.batch)
        self.server = eval_service.create_server('127.0.0.1', 0, self.tracker, secrets.token_urlsafe(32))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.gpu = gpu
        self.eval_index = 0
        # Keep deadlines tied to immutable output paths, so a late HTTP worker
        # cannot accidentally inherit a subsequent stage's time allowance.
        self.score_deadlines = {}
        self.scoring_seconds = {}
        # Change only execution placement: the original scoring routine still
        # runs verbatim inside the restricted evaluator subprocess.
        self.original_runner = eval_service.run_evaluator
        eval_service.run_evaluator = self._run_confined

    def _run_confined(self, task_dir, output_dir):
        deadline = self.score_deadlines.get(str(Path(output_dir).resolve()), float('inf'))
        self.eval_index += 1
        base = self.directory / 'evaluator' / str(self.eval_index)
        extra_reads, extra_env = self.scorer_access(task_dir, output_dir)
        sandbox = Sandbox(sys.executable, Path(task_dir) / 'problem', base / 'scratch',
                          base / 'logs', self.gpu, uid=1200000 + __import__('os').getpid(),
                          read_extra=[Path(task_dir) / 'evaluation', Path(output_dir),
                                      ROOT / 'eval_service.py', ROOT / '_evaluator_runner.py', *extra_reads],
                          env_extra={'OUTPUT_DIR': str(output_dir), **extra_env}, limits=self.resource_limits)
        code = (f'import sys,json,importlib.util; '
                f's=importlib.util.spec_from_file_location("eval_service",{str(ROOT / "eval_service.py")!r}); '
                'm=importlib.util.module_from_spec(s); sys.modules["eval_service"]=m; s.loader.exec_module(m); '
                f'from pathlib import Path; r=m.run_evaluator(Path({str(task_dir)!r}),'
                f'Path({str(output_dir)!r})); print("RSI_EVALUATION_RESULT="+json.dumps(r))')
        # Snapshot read permissions for the dedicated evaluator UID. Scientific
        # processes have already terminated and cannot race this copy.
        for base_path, dirs, files in __import__('os').walk(output_dir):
            Path(base_path).chmod(0o755)
            for filename in files:
                (Path(base_path) / filename).chmod(0o444)
        seconds = min(3650, self.remaining(), deadline - time.monotonic())
        if seconds <= 0:
            raise TimeoutError('Scoring deadline exhausted')
        outcome = sandbox.run(code, seconds)
        self.scoring_seconds[str(Path(output_dir).resolve())] = outcome['seconds']
        if outcome['returncode']:
            raise RuntimeError(f"Evaluator returncode={outcome['returncode']}, reason={outcome.get('reason')}: "
                               + outcome['stdout'][-1500:])
        marker = 'RSI_EVALUATION_RESULT='
        # Read the trusted log, not its display-tail (some tasks have many scores).
        output = Path(outcome['log']).read_text()
        if marker not in output:
            raise RuntimeError('Evaluator produced no result marker')
        return json.loads(output.rsplit(marker, 1)[1].strip())

    def scorer_access(self, task_dir, output_dir):
        """Trusted evaluator-only dependencies; never granted to scientific code."""
        return [], {}

    def start(self):
        # Same time origin as official /start_timer, after GPU allocation/setup.
        self.state.start_time = time.time()
        if self.wall_seconds is not None:
            self.wall_deadline = time.monotonic() + self.wall_seconds

    def wall_remaining(self):
        return max(0, self.wall_deadline - time.monotonic()) if self.wall_deadline is not None else float('inf')

    def remaining(self):
        return max(0, min(self.state.timeout - self.state.get_effective_elapsed(), self.wall_remaining()))

    def exclude_infrastructure_time(self, seconds, *, research=True):
        """Return seconds lost to failed provider attempts, within the per-run limit.

        Research time moves every research clock together: the official solve timer (as a pause),
        the wall cap and any protocol ledger deadline. Post-research stages own their clocks;
        their share is only counted here. Returns the seconds actually returned."""
        used = self.infrastructure_excluded_seconds + self.infrastructure_post_research_seconds
        credited = max(0.0, min(float(seconds), INFRASTRUCTURE_EXCLUSION_LIMIT_SECONDS - used))
        if credited <= 0:
            return 0.0
        if research:
            with self.state.lock:
                self.state.total_paused += credited
            if self.wall_deadline is not None:
                self.wall_deadline += credited
            self.extend_research_deadline(credited)
            self.infrastructure_excluded_seconds += credited
        else:
            self.infrastructure_post_research_seconds += credited
        return credited

    def extend_research_deadline(self, seconds):
        """Protocols that store their own research deadline move it here."""

    def infrastructure_time(self):
        return dict(research_seconds=self.infrastructure_excluded_seconds,
                    post_research_seconds=self.infrastructure_post_research_seconds,
                    limit_seconds=INFRASTRUCTURE_EXCLUSION_LIMIT_SECONDS)

    def evaluate(self, round_dir, timeout=None):
        seconds = min(self.remaining(), timeout) if timeout is not None else self.remaining()
        if seconds <= 0:
            raise TimeoutError('Solve deadline exhausted before evaluation')
        self.state.out_dir = Path(round_dir)
        self.score_deadlines[str((Path(round_dir) / 'workspace/output').resolve())] = time.monotonic() + seconds
        req = urllib.request.Request(self.url + '/evaluate',
                                     data=json.dumps({'eval_token': self.token}).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            # The scorer stops at its deadline. Brief transport/cleanup grace
            # lets the official handler record failure and resume its timer.
            with urllib.request.urlopen(req, timeout=min(3660, seconds + 2, self.wall_remaining())) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            result = {'error': json.load(exc).get('error', 'official evaluation failed')}
        measured = self.scoring_seconds.get(str((Path(round_dir)/'workspace/output').resolve()))
        if measured is not None:
            result['host_scoring_seconds'] = measured
        return result

    def results(self):
        return self.tracker.all_results()[self.batch][self.task]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        eval_service.run_evaluator = self.original_runner


def valid_score(result, primary):
    """Require finite measured metrics; active validation may cover a subset."""
    raw = result.get('raw_scores', {})
    active = result.get('benchmark_protocol') == 'naturebench-active'
    if active:
        if not isinstance(raw, dict) or not raw or set(raw)-set(primary):
            return False
        primary = {instance:primary[instance] for instance in raw}
    for instance, info in primary.items():
        value = eval_service._find_metric_value(raw.get(instance), info['metric'])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
    aggregate = result.get('aggregate_improvement')
    if active and result.get('selection_metric') == 'self_reported_loss_diagnostic':
        return bool(primary) and result.get('training_audit', {}).get('status') == 'passed'
    return bool(primary) and type(aggregate) in (float, int) and math.isfinite(aggregate)
