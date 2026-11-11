"""Interactive host capabilities and terminal scores for repeated fixed episodes.

Each Task stage works on one episode of the same fixed scenario; a terminal
episode hands off to Meta and the next Task receives a fresh copy.
"""
import json
import math
import time
from pathlib import Path

from rsi.common import Journal, digest, write_json
from rsi.evaluation import Evaluation
from rsi.sandbox import assert_plain_tree
from rsi.store import safe_path
from benchmarks.simulator.client import EpisodeClient

EPISODE_PROTOCOL = 'repeated_fixed_episode_v1'
# Primary metric and orientation (+1 higher is better, -1 lower is better).
PRIMARY = {'CHEM_01': ('episode_return', 1), 'MAT_01': ('relative_angle_degrees', -1)}
SELECTION_RULE = 'best_completed_valid_episode_v1'


def primary_value(task, episode):
    """Finite primary score of a completed episode, else None."""
    key = PRIMARY[task][0]
    value = (episode.get('metrics') or {}).get(key) if episode.get('terminal') is True else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def best_episode(task, episodes):
    """Best completed episode with a finite primary score; the earliest wins ties."""
    sign = PRIMARY[task][1]
    valid = [e for e in episodes if primary_value(task, e) is not None]
    if not valid:
        return None
    return min(valid, key=lambda e: (-sign * primary_value(task, e), e['episode_index']))


class SimulatorEvaluation(Evaluation):
    def __init__(self, task, data_dir, run_dir, seconds, gpu, wall_seconds=None,
                 resource_limits=None, data_root=None, python=None, journal=None):
        self.episode_protocol = EPISODE_PROTOCOL
        self.episode_index = 0
        self.episode_history = []
        self.simulator_options = dict(data_root=data_root, python=python)
        self.task, self.directory = task, Path(run_dir)
        self.simulator = EpisodeClient(task, self.episode_directory(), **self.simulator_options)
        self.simulator_log = Journal(Path(run_dir)/'simulator/actions.jsonl')
        self.host_journal = journal
        self.closed = False
        self.trajectory_hash = digest(self.simulator.current)
        try:
            super().__init__(task, data_dir, run_dir, seconds, gpu, wall_seconds, resource_limits)
        except BaseException:
            self.simulator.close()
            raise

    @property
    def episode_done(self):
        return self.simulator.current['terminal']

    def episode_directory(self):
        return self.directory/'simulator'/'episodes'/f'{self.episode_index:04d}'

    def simulator_status(self):
        return dict(json.loads(json.dumps(self.simulator.current)), available=not self.simulator.failed,
                    protocol=self.episode_protocol, episode_index=self.episode_index)

    def current_record(self):
        return dict(self.simulator_status(), trajectory_sha256=self.trajectory_hash,
                    independent_test=False)

    def archive_episode(self):
        if any(x['episode_index'] == self.episode_index for x in self.episode_history):
            return
        record = self.current_record()
        self.episode_history.append(record)
        write_json(self.episode_directory()/'result.json', record)
        write_json(self.directory/'simulator/episodes.json', self.episode_history)

    def episodes(self):
        """Archived episodes plus the current one if it has not been archived yet."""
        archived = any(x['episode_index'] == self.episode_index for x in self.episode_history)
        return list(self.episode_history) + ([] if archived else [self.current_record()])

    def reported_episode(self):
        """The campaign result: the best completed valid episode, else the latest episode."""
        episodes = self.episodes()
        selected = best_episode(self.task, episodes)
        return dict(selected or episodes[-1], selection=SELECTION_RULE, selected=selected is not None,
                    completed_episodes=sum(e['terminal'] is True for e in episodes),
                    valid_episodes=sum(primary_value(self.task, e) is not None for e in episodes))

    def prepare_next_task(self):
        """Host-only boundary: keep unfinished state; renew a completed episode.

        The same global clock, API ledger, score history and Agent lineage remain.
        A crashed/ambiguous simulator is never silently restarted.
        """
        if self.simulator.failed or self.simulator.proc.poll() is not None:
            raise RuntimeError('Unavailable simulator cannot be renewed')
        if not self.episode_done:
            return False
        if self.remaining() <= 0:
            raise TimeoutError('Research budget exhausted before next episode')
        self.archive_episode()
        next_index = self.episode_index + 1
        successor = EpisodeClient(self.task, self.directory/'simulator/episodes'/f'{next_index:04d}',
                                  **self.simulator_options)
        try:
            self.simulator.close()
        except BaseException:
            successor.close()
            raise
        self.episode_index = next_index
        self.simulator = successor
        self.trajectory_hash = digest([self.trajectory_hash, 'new_fixed_episode',
                                       self.episode_index, self.simulator.current])
        if self.host_journal:
            self.host_journal.add('simulator_episode_started', episode_index=self.episode_index,
                protocol=self.episode_protocol, initial_usage=self.simulator.current['usage'],
                remaining_seconds=self.remaining())
        return True

    def export_trace(self, workspace):
        """Only action requests and permitted responses enter Agent artifacts."""
        workspace = Path(workspace)
        assert_plain_tree(workspace)
        source = self.directory/'simulator/actions.jsonl'
        path = workspace/'simulator_trajectory.jsonl'
        path.write_text(source.read_text() if source.exists() else '')
        return path.name

    def simulator_action(self, action, arguments, timeout, attribution):
        attribution = dict(attribution, episode_index=self.episode_index)
        request = dict(op='action', action=action, arguments=arguments)
        try:
            reply = self.simulator.call(request, min(timeout, self.remaining()))
        except Exception as exc:
            reply = dict(error=f'{type(exc).__name__}: {exc}', result=self.simulator_status())
        else:
            reply['result'] = self.simulator_status()
        self.trajectory_hash = digest([self.trajectory_hash, request, reply, attribution])
        self.simulator_log.add('simulator_action', request=request, response=reply,
                            trajectory_hash=self.trajectory_hash, **attribution)
        if self.host_journal:
            self.host_journal.add('simulator_action', action=action, arguments=arguments,
                usage=self.simulator.current['usage'], terminal=self.episode_done,
                error=reply.get('error'), trajectory_hash=self.trajectory_hash, **attribution)
        return reply

    def run_policy(self, science, workspace, path, seconds, attribution):
        """Run next_action(observation,state) in the ordinary science sandbox.

        A fresh subprocess selects each action. Only explicit JSON state persists;
        the simulator stays in the separate trusted process throughout.
        """
        workspace = Path(workspace)
        policy = safe_path(workspace, path)
        if not policy.is_file() or policy.suffix != '.py':
            raise ValueError('path must name a workspace Python policy')
        deadline = time.monotonic() + min(seconds, self.remaining())
        state, count = None, 0
        while not self.episode_done:
            if deadline <= time.monotonic():
                raise TimeoutError('Policy time exhausted; episode progress is retained')
            assert_plain_tree(workspace)
            exchange = workspace/'.simulator_policy'
            exchange.mkdir(exist_ok=True)
            inp, out = exchange/'input.json', exchange/'decision.json'
            write_json(inp, dict(observation=self.simulator_status(), state=state))
            out.unlink(missing_ok=True)
            code = (f'import importlib.util,json; from pathlib import Path; '
                f's=importlib.util.spec_from_file_location("policy",{str(policy)!r}); '
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m); '
                f'i=json.loads(Path({str(inp)!r}).read_text()); '
                'r=m.next_action(i["observation"],i["state"]); '
                f'Path({str(out)!r}).write_text(json.dumps(r,allow_nan=False))')
            outcome = science.run(code, deadline-time.monotonic())
            assert_plain_tree(workspace)
            if outcome['returncode'] or not out.is_file():
                return dict(error='Policy failed', process=outcome, actions=count,
                            result=self.simulator_status())
            if out.stat().st_size > 1024*1024:
                raise ValueError('Policy JSON state exceeds 1 MiB')
            choice = json.loads(out.read_text())
            if not isinstance(choice, dict):
                raise ValueError('Policy must return an object')
            state = choice.get('state')
            if choice.get('stop') is True:
                break
            reply = self.simulator_action(choice['action'], choice.get('arguments', {}),
                                       deadline-time.monotonic(), attribution)
            count += 1
            if reply.get('error'):
                return dict(reply, actions=count)
        return dict(result=self.simulator_status(), actions=count, state=state)

    def _run_confined(self, task_dir, output_dir):
        # Scores come only from the trusted live episode, never Agent files.
        current = self.simulator_status()
        metrics = current['metrics']
        key = PRIMARY[self.task][0]
        if not current['terminal'] or metrics is None or metrics.get(key) is None:
            raise ValueError('No completed simulator result with a finite primary score')
        return {self.task: {key: metrics[key], 'simulator_metrics': metrics,
                           'trajectory_sha256': self.trajectory_hash,
                           'protocol': self.episode_protocol, 'episode_index': self.episode_index,
                           'independent_test': False}}

    def evaluate(self, round_dir, timeout=None):
        if not self.episode_done:
            return dict(episode_pending=True, simulator=self.simulator_status(),
                        trajectory_sha256=self.trajectory_hash,
                        note='Continue the existing episode; no terminal score yet.')
        (Path(round_dir)/'workspace/output').mkdir(parents=True, exist_ok=True)
        return dict(super().evaluate(round_dir, timeout), simulator=self.simulator_status(),
                    trajectory_sha256=self.trajectory_hash, independent_test=False,
                    protocol=self.episode_protocol, episode_index=self.episode_index)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.archive_episode()
        write_json(self.directory/'simulator/result.json', self.reported_episode())
        try:
            self.simulator.close()
        finally:
            super().close()
