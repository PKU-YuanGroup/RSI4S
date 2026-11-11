"""Confinement and host capability bridge for executable M/T source bundles."""
from __future__ import annotations

import json
import itertools
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from rsi.api import BudgetExceeded, APIOutputTruncated, APIRequestError, APIContextExceeded
from rsi.agent_sources import source_revision
from rsi.chain import DISTILLATION_SECONDS, check_bundle, distillation_timing, render_experience, sha256_text
from rsi.context import PromptContext, ToolResults, compact_budget
from rsi.context_policy import AGENT_TRANSPORT_BYTES, load_policy
from rsi.common import encoded, write_json
from rsi.host_io import HostIODeadline, run_io
from rsi.history import OPS as META_HISTORY, publish as publish_history
from rsi.evaluation import valid_score
from rsi.development import Development, bash
from rsi.sandbox import Sandbox, assert_plain_tree
from rsi.schedule import stage_timing, research_budget, meta_selection_seconds
from rsi.submission import submission_contract
from rsi.store import hashes, read_text, safe_path, seal
from benchmarks.evidence import TASK_OPS as TASK_EVIDENCE_OPS, META_OPS as META_EVIDENCE_OPS, OPS as ALL_EVIDENCE_OPS

HISTORY = {'list_history', 'read_record', 'read_version', 'read_artifact'}
TASK_OPS = HISTORY | TASK_EVIDENCE_OPS | {'simulator_observe', 'simulator_action', 'simulator_run', 'bash', 'python', 'evaluate', 'finish', 'llm', 'read_tool_result', 'get_time'}
META_OPS = HISTORY | META_HISTORY | META_EVIDENCE_OPS | {'simulator_observe', 'bash', 'editor', 'create_candidate', 'read_file', 'write_file', 'check_candidate', 'submit_candidate', 'finish', 'llm', 'read_tool_result', 'get_time'}
FINAL_SELECTION_OPS = HISTORY | META_HISTORY | {'llm', 'read_tool_result', 'finish', 'get_time',
                                 'select_final_artifact', 'finish_experiment'}
DISTILLATION_OPS = HISTORY | META_HISTORY | {'bash', 'editor', 'llm', 'read_tool_result', 'finish', 'get_time'}


class StageDeadlineReached(RuntimeError):
    pass


class AgentTransportExceeded(RuntimeError):
    pass


class AgentProcessFailed(RuntimeError):
    """The confined Agent step crashed; nothing it produced is published."""


class ProcessAgent:
    """Reload a complete source bundle in a new confined process for every step.

    The Agent owns its resumable state and control flow, but never host credentials
    or the capability dispatcher. No generated code is imported into the host.
    """
    def __init__(self, config, source, directory, gpu, journal, identity, remaining, attribution=None,
                 mutable_source=False):
        self.source, self.directory = Path(source), Path(directory)
        self.mutable_source = mutable_source
        self.directory.mkdir(parents=True, exist_ok=True)
        public = self.directory / 'public_interface'
        public.mkdir(exist_ok=True)
        self.sandbox = Sandbox(config['python'], public, self.directory / 'scratch',
                               self.directory / 'processes', gpu,
                               read_extra=[] if mutable_source else [self.source],
                               write_extra=[self.source] if mutable_source else [],
                               limits=config.get('sandbox_resources'))
        self.journal, self.identity, self.remaining = journal, identity, remaining
        self.attribution = attribution or {}
        self.expected = hashes(self.source)
        self.source_revisions = set()
        self.index = 0

    def step(self, state, context, response):
        files = hashes(self.source)
        if not self.mutable_source and files != self.expected:
            raise RuntimeError('Agent source changed during execution')
        if self.remaining() <= 0:
            raise BudgetExceeded('Agent step has no allotted time left')
        revision = source_revision(files) if self.mutable_source else None
        if revision is not None and revision not in self.source_revisions:
            run_io('agent_source_revision', time.monotonic()+self.remaining(),
                   directory=str(self.directory), current=str(self.source),
                   files=files, base_version=self.identity)
            self.source_revisions.add(revision)
        self.index += 1
        write_json(self.sandbox.workspace / 'input.json', {'state': state, 'context': context, 'response': response})
        output = self.sandbox.workspace / 'result.json'
        if output.exists():
            output.unlink()
        code = f'''import sys,json,importlib.util,os
from pathlib import Path
source=Path({str(self.source)!r})
sys.path.insert(0,str(source))
coverage={{}}
def trace(frame,event,arg):
    path=Path(frame.f_code.co_filename)
    if event=='line' and path.is_relative_to(source):
        coverage.setdefault(str(path.relative_to(source)),set()).add(frame.f_lineno)
    return trace
sys.settrace(trace)
spec=importlib.util.spec_from_file_location('entry',source/'entry.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inputs=json.loads(Path('input.json').read_text())
result=module.step(**inputs)
sys.settrace(None)
Path('result.json').write_text(json.dumps({{'result':result,'coverage':{{k:sorted(v) for k,v in coverage.items()}}}}))
'''
        seconds = min(30, self.remaining())
        if seconds <= 0:
            raise BudgetExceeded('Agent step has no allotted time left')
        outcome = self.sandbox.run(code, seconds)
        if self.mutable_source:
            try:
                after, source_error = hashes(self.source), None
            except Exception as exc:
                after, source_error = None, f'{type(exc).__name__}: {exc}'
            self.journal.add('agent_source_observed', version=self.identity, step=self.index,
                             source=str(self.source), before=files, after=after, error=source_error,
                             source_revision=revision, process=outcome, **self.attribution)
        if outcome['returncode'] or not output.is_file():
            self.journal.add('agent_source_failed', version=self.identity, step=self.index,
                             files=files, source_revision=revision, source=str(self.source),
                             allotted_seconds=seconds, process=outcome, **self.attribution)
            detail = (f"returncode={outcome['returncode']}, reason={outcome.get('reason')}, "
                      f"allotted_seconds={seconds:.3f}, log={outcome.get('log')}: "
                      + outcome['stdout'][-2500:])
            if outcome.get('reason') == 'timeout' and self.remaining() <= 0:
                raise BudgetExceeded('Agent step reached allotted deadline: ' + detail)
            raise AgentProcessFailed('Agent process failed: ' + detail)
        assert_plain_tree(self.sandbox.workspace)
        if output.stat().st_size > AGENT_TRANSPORT_BYTES:
            rejected = self.directory.parent / 'protocol-errors' / f'{self.directory.name}-output-{self.index:04d}.json'
            rejected.parent.mkdir(exist_ok=True)
            output.replace(rejected)
            rejected.chmod(0o444)
            self.journal.add('agent_transport_rejected', version=self.identity, step=self.index,
                             bytes=rejected.stat().st_size, limit_bytes=AGENT_TRANSPORT_BYTES,
                             raw_result=str(rejected))
            raise AgentTransportExceeded('Agent state and request exceeded the combined 8 MiB transport limit. '
                                         'The previous accepted state is unchanged; revise the returned state/request.')
        result = json.loads(output.read_text())
        value = result['result']
        if not isinstance(value, dict) or set(value) != {'state', 'request'} or not isinstance(value['request'], dict):
            raise ValueError('step must return exactly state and request')
        self.journal.add('agent_source_executed', version=self.identity, step=self.index,
                         source=str(self.source), files=files, coverage=result['coverage'], process=outcome,
                         source_revision=revision,
                         **self.attribution)
        return value


class Session:
    def __init__(self, config, store, client, evaluation, gpu, journal, directory, *, research_started=None,
                 inherited_experience=None):
        self.config, self.store, self.client = config, store, client
        self.evaluation, self.gpu, self.journal = evaluation, gpu, journal
        self.directory = Path(directory)
        # Notes distilled after an earlier chain task; the host shows them to research sessions.
        self.inherited_experience = inherited_experience
        self.check_index = 0
        self.session_info = None
        self.tool_calls = 0
        self.stage_deadline = self.action_deadline = None
        self.finalization_reserve = 0
        self.finalizing = False
        self.research_closed = False
        self.selection_only = False
        self.distillation = False
        self.startup_seconds = {}
        self.research_started = time.monotonic() if research_started is None else research_started
        self.research_finished = None
        self.stage_started = None
        # Failed provider attempts return their time to whichever clock the call ran on.
        if client is not None:
            client.infrastructure_clock = self.exclude_infrastructure_time

    def exclude_infrastructure_time(self, seconds):
        research = not (self.post_research or self.research_closed)
        credited = self.evaluation.exclude_infrastructure_time(seconds, research=research)
        if self.stage_deadline is not None:
            self.stage_deadline += credited
        if self.action_deadline is not None:
            self.action_deadline += credited
        return credited

    def close_research(self):
        """Freeze elapsed research time before the separate final-selection clock."""
        if self.research_finished is None:
            self.research_finished = time.monotonic()
        self.research_closed = True

    @property
    def post_research(self):
        """Final selection and chain distillation run on their own clocks after research."""
        return self.selection_only or self.distillation

    def finish_without_submission(self, reason, identity, run_id, phase):
        """Accept an explicit final-review finish only when no validated choice exists."""
        if not self.selection_only or not self.research_closed:
            return False
        eligible, comparison = self.evaluation.selection_candidates(self.store.records())
        if eligible:
            return False
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Explain why no validated artifact is available')
        outcome = dict(status='no_submission', reason=reason, meta_version=identity,
                       run_id=run_id, phase=phase, outside_research_budget=True,
                       eligible=0, comparison=comparison, time=time.time())
        write_json(self.evaluation.directory/'meta_no_submission.json', outcome)
        self.journal.add('meta_no_submission', **outcome)
        return True

    def remaining(self):
        seconds = float('inf') if self.post_research else self.evaluation.remaining()
        deadline = self.stage_deadline if self.finalizing else self.action_deadline
        return max(0, min(seconds, deadline-time.monotonic())) if deadline is not None else seconds

    def evaluation_remaining(self):
        """Research evaluation uses the same remaining budget as Agent actions."""
        return self.evaluation.remaining()

    def deadline_status(self):
        if not self.post_research and self.evaluation.remaining() <= 0:
            return 'budget_exhausted'
        if self.stage_deadline is not None and self.remaining() <= 0:
            return 'stage_deadline'
        return None

    def check_start_time(self, kind):
        needed = max(1, 1.2*self.startup_seconds.get(kind, 0))
        if not self.post_research and self.evaluation.remaining() < needed:
            raise BudgetExceeded('Insufficient research time to start another stage')
        if self.remaining() < needed:
            raise StageDeadlineReached('Insufficient stage time for startup')

    def io_failure(self, exc):
        status = self.deadline_status() or 'failed'
        scope = {'budget_exhausted': 'research', 'stage_deadline': 'stage'}.get(status, 'operation')
        return status, str(exc), dict(category='deadline', operation=exc.operation,
                                      expires=exc.expires, scope=scope)

    def resources(self, calls_left):
        now = time.monotonic()
        research_end = now if self.research_finished is None else self.research_finished
        result = {'remaining_solve_seconds': 0 if self.research_closed else self.evaluation.remaining(),
                'research_elapsed_seconds': max(0, research_end - self.research_started),
                'stage_elapsed_seconds': max(0, now - self.stage_started) if self.stage_started is not None else None,
                'research_closed': self.research_closed,
                'final_selection_only': self.selection_only,
                'chain_distillation': self.distillation,
                'remaining_action_seconds': self.remaining(),
                'experiment': research_budget(self.config),
                'session': self.session_info,
                'shared_wall_deadline_utc': (
                    datetime.fromtimestamp(time.time() + self.evaluation.wall_remaining(), timezone.utc).isoformat()
                    if self.evaluation.wall_deadline is not None else None),
                'session_calls_left': calls_left,
                'max_tool_calls': None,
                'tool_calls_used': self.tool_calls,
                'tool_calls_left': None,
                'hardware': {'gpu': self.gpu, 'cuda_device': 'cuda:0' if self.gpu is not None else None,
                             'gpu_slots_per_gpu': self.config.get('gpu_slots_per_gpu', 2),
                             'sandbox_resources': self.config.get('sandbox_resources', {}),
                             'gpu_is_shared': self.config.get('gpu_slots_per_gpu', 2) > 1,
                             'memory_enforcement': (
                                 'sampled process-tree RSS watchdog; not a hard cgroup cap'
                                 if self.config.get('sandbox_resources', {}).get('memory_limit_bytes') else
                                 'No per-sandbox memory cap; sampled process-tree RSS is reported')},
                'budget': self.client.budget.snapshot()}
        if self.stage_deadline is not None:
            result.update(stage_remaining_seconds=max(0,self.stage_deadline-time.monotonic()),
                          stage_action_remaining_seconds=max(0,self.action_deadline-time.monotonic()),
                          host_finalization_reserve_seconds=self.finalization_reserve)
        return result

    def check(self, candidate, context, *, deadline=None):
        self.check_index += 1
        before = self.store.syntax(candidate)
        proc = ProcessAgent(self.config, self.store.candidate(candidate),
                            self.directory / 'checks' / f'{self.check_index:04d}', self.gpu,
                            self.journal, candidate, self.remaining if deadline is None else
                            lambda: max(0, deadline-time.monotonic()))
        result = proc.step({}, {**context, 'context_policy': load_policy(self.store.candidate(candidate)),
                               'resources': self.resources(1)}, None)
        kind = self.store.candidates[candidate]['kind']
        if result['request'].get('op') not in (TASK_OPS if kind == 'task' else META_OPS):
            raise ValueError('Candidate requested an invalid capability')
        if result['request']['op'] == 'llm':
            self.prepare_messages(result['request'].get('messages'))
        if hashes(self.store.candidate(candidate)) != before:
            raise RuntimeError('Candidate changed while being checked')
        checks = {'ok': True, 'files': before, 'initial_request': result['request']['op'],
                  'checks': ['plain_tree', 'python_syntax', 'isolated_import', 'step_protocol', 'capability'],
                  'directory': str(proc.directory)}
        self.journal.add('candidate_checked', candidate=candidate, checks=checks)
        return checks

    @staticmethod
    def validate_messages(messages):
        if not isinstance(messages, list) or not messages:
            raise ValueError('llm requires messages')
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != {'role', 'content'} or message['role'] not in ('system', 'user', 'assistant') or not isinstance(message['content'], str):
                raise ValueError(f'Invalid LLM message at index {index}: require role '
                                 '(system/user/assistant) and string content only')

    @staticmethod
    def prepare_messages(messages):
        """Project private state annotations out of the API protocol, without editing state.

        Evolved agents may annotate messages (e.g. _bound_note). Only private
        underscore-prefixed keys are stripped; unsupported API fields, roles,
        and non-string content are still rejected rather than silently lost.
        """
        prepared = ([{k: v for k, v in message.items() if not (isinstance(k, str) and k.startswith('_'))}
                     if isinstance(message, dict) else message for message in messages]
                    if isinstance(messages, list) else messages)
        Session.validate_messages(prepared)
        return prepared

    def run(self, kind, identity, meta_version, run_id, context, max_calls, artifact_parent=None, allow_meta=False, session_info=None):
        selection_only = bool(context.get('final_selection_only'))
        distillation = bool(context.get('chain_distillation'))
        if selection_only and (kind != 'meta' or not self.research_closed or allow_meta):
            raise ValueError('Final selection requires closed research and a read-only Meta session')
        if distillation and (kind != 'meta' or not self.research_closed or selection_only or artifact_parent):
            raise ValueError('Chain distillation requires closed research and a Meta development session')
        if self.research_closed and not (selection_only or distillation):
            raise ValueError('Research is closed; only final Meta selection or chain distillation is allowed')
        self.session_info = session_info or {'run_id': run_id, 'kind': kind,
            'phase': 'final_selection' if selection_only else 'chain_distillation' if distillation else
                     ('task_execution' if kind == 'task' else 'evolution')}
        self.stage_started = time.monotonic()
        self.tool_calls = 0
        # Research always reads the live shared clock; never freeze its remainder
        # into a new session clock or subtract a per-stage closing reserve.
        limit = (meta_selection_seconds(self.config) if selection_only else
                 self.config.get('chain_distillation_seconds', DISTILLATION_SECONDS) if distillation else None)
        self.selection_only = selection_only
        self.distillation = distillation
        self.stage_deadline = time.monotonic()+limit if limit is not None else None
        if selection_only:
            self.evaluation.meta_selection = None
        self.finalization_reserve = 0
        self.action_deadline = self.stage_deadline
        self.finalizing = False
        try:
            if session_info:
                context = {**context, 'session_info': session_info}
            return self._run(kind, identity, meta_version, run_id, context, max_calls, artifact_parent, allow_meta)
        finally:
            self.session_info = None
            self.stage_started = None
            self.stage_deadline = self.action_deadline = None
            self.finalization_reserve = 0
            self.finalizing = False
            self.selection_only = False
            self.distillation = False

    def _run(self, kind, identity, meta_version, run_id, context, max_calls, artifact_parent, allow_meta):
        phase = ('final_selection' if self.selection_only else 'chain_distillation' if self.distillation else
                 ('task_execution' if kind == 'task' else 'evolution'))
        base = self.directory / run_id
        base.mkdir(parents=True, exist_ok=False)
        workspace = base / 'workspace'
        artifacts, evaluations, trace, visible = [], [], [], []
        submitted = {}
        protocol_feedback = None
        state, response, status, error, summary = {}, None, 'call_allocation', None, None
        failure = development = None
        task_source = source_archive = source_archive_error = None
        begin = time.monotonic()
        cost_before = self.client.budget.snapshot()
        calls = rejected_messages = 0
        dirty = experiment_finished = False
        handoff = None
        deadline_handoff = False

        def snapshot(deadline):
            attempt = base / 'attempts' / f'{len(artifacts):04d}'
            files = run_io('snapshot', deadline, source=str(workspace), target=str(attempt/'workspace'))
            artifact = {'id': run_id + '-A' + str(len(artifacts)), 'workspace': str(attempt / 'workspace'),
                        'files': files, 'task_version': identity, 'meta_version': meta_version}
            artifacts.append(artifact)
            write_json(attempt / 'artifact.json', artifact)
            return attempt, artifact

        def evaluate(*, automatic=False):
            nonlocal dirty
            deadline = time.monotonic() + self.evaluation_remaining()
            files = run_io('hashes', deadline, source=str(workspace))
            cohort = None
            if hasattr(self.evaluation, 'evidence'):
                cohort = self.evaluation.validation_cohort(deadline=deadline)
            if hasattr(self.evaluation, 'simulator_status'):
                episode = self.evaluation.simulator_status()
                cohort = {'simulator_revision': episode['revision'], 'simulator_episode_index': episode['episode_index']}
            if automatic and artifacts and evaluations and files == artifacts[-1]['files'] and cohort == evaluations[-1].get('validation_cohort') and not hasattr(self.evaluation, 'reuse_evaluation'):
                dirty = False
                self.journal.add('evaluation_reused', run_id=run_id, artifact=artifacts[-1]['id'],
                                 reason='Identical workspace and validation membership')
                return dict(evaluations[-1], reused=True)
            attempt, artifact = snapshot(deadline)
            try:
                history = self.store.records()
                self.evaluation.historical_artifacts = [a for r in history for a in r.get('artifacts', [])] + artifacts[:-1]
                if hasattr(self.evaluation, 'record_execution'):
                    self.evaluation.record_execution(
                        dict(run_id=run_id, artifact_parent=artifact_parent, trace=trace), history)
                feedback = None
                if automatic and evaluations and hasattr(self.evaluation, 'reuse_evaluation'):
                    feedback = self.evaluation.reuse_evaluation(
                        attempt, evaluations[-1], deadline=time.monotonic()+self.evaluation_remaining())
                if feedback is None:
                    feedback = self.evaluation.evaluate(attempt, timeout=self.evaluation_remaining())
                else:
                    self.journal.add('evaluation_reused', run_id=run_id, artifact=artifact['id'],
                                     reason='Automatic handoff with identical verified scientific inputs',
                                     source_artifact=evaluations[-1].get('artifact'))
            except Exception as exc:
                feedback = {'error': f'{type(exc).__name__}: {exc}'}
            feedback.update({'artifact': artifact['id'], 'task_version': identity, 'meta_version': meta_version,
                             'snapshot': str(attempt), 'valid_primary_score': valid_score(feedback, self.evaluation.state.primary_table)})
            evaluations.append(feedback)
            write_json(attempt / 'score.json', feedback)
            # The artifact is sealed by the bounded snapshot worker. Replay outputs
            # remain separate and are never reused as scientific artifacts.
            self.journal.add('evaluation', run_id=run_id, **feedback)
            dirty = False
            return feedback

        def publish_successors(summary, *, deadline=None, allow_empty_experience=False):
            # Validate remaining required and optional updates before publishing.
            staged = []
            for successor_kind, candidate, parent in development.candidates(summary, exclude=submitted,
                    deadline=deadline if deadline is not None else time.monotonic()+self.remaining()):
                info = self.store.candidates[candidate]
                checks = self.check(candidate, context, deadline=deadline)
                staged.append((successor_kind, candidate, parent, info, checks))
            if self.distillation:
                # A chain handoff is all-or-nothing: notes and final Agent sizes pass before publication.
                experience = development.experience(required=not allow_empty_experience)
                for _, candidate, *_ in staged:
                    check_bundle(self.store.candidate(candidate))
                if not any(item[0] == 'meta' for item in staged):
                    check_bundle(self.store.version(context['current_meta'])[0] / 'source')
            for successor_kind, candidate, parent, info, checks in staged:
                published = self.store.publish(successor_kind, self.store.candidate(candidate),
                            base=info['base_version'], sources=info['source_versions'],
                            author=identity, rationale=info['rationale'], checks=checks,
                            source_run=info.get('source_run'))
                submitted[successor_kind] = {
                    'version': published, 'artifact_parent': parent,
                    'source_run': info.get('source_run'),
                    'decision': 'retain' if published == context['current_' + successor_kind] else 'update',
                    'reason': str(summary)}
            if self.distillation:
                submitted['experience'] = dict(text=experience, sha256=sha256_text(experience),
                                               characters=len(experience))

        try:
            # Even a deadline reached during setup must leave a checkout to transfer.
            if not self.distillation:
                self.check_start_time(kind)
            context = {**context, 'final_submission': submission_contract(
                context['task_id'], self.config.get('medrsee_final_seconds', 3600))}
            if hasattr(self.evaluation,'submission_contract'):
                context['final_submission']=self.evaluation.submission_contract()
            # Distillation has no evidence or simulator tools; it only edits Agents and notes.
            if hasattr(self.evaluation,'evidence') and not self.distillation:
                context = {**context,'annotation':self.evaluation.annotation_status()}
            if hasattr(self.evaluation, 'simulator_status') and not self.distillation:
                context = {**context, 'simulator_episode': self.evaluation.simulator_status()}
            source, manifest = self.store.version(identity)
            policy = load_policy(source / 'source')
            context = {**context, 'context_policy': policy}
            agent_source = source / 'source'
            if kind == 'task':
                task_source = base / 'agent' / 'source'
                run_io('copy', time.monotonic()+self.remaining(),
                       source=str(agent_source), target=str(task_source))
                agent_source = task_source
                context = {**context, 'agent_source_dir': str(task_source)}
            proc = ProcessAgent(self.config, agent_source, base / 'agent', self.gpu,
                                self.journal, identity, self.remaining,
                                attribution=dict(run_id=run_id, agent_kind=kind, phase=phase),
                                mutable_source=kind == 'task')
            if kind == 'task':
                if artifact_parent:
                    artifact = self.store.artifact(artifact_parent, deadline=time.monotonic()+self.remaining())
                    self.check_start_time(kind)
                    run_io('copy', time.monotonic()+self.remaining(), source=artifact['workspace'], target=str(workspace))
                else:
                    workspace.mkdir()
                self.journal.add('artifact_reuse', run_id=run_id, artifact_parent=artifact_parent,
                                 agent_base=manifest['base_version'], auto_inherit=False)
                science = Sandbox(self.config['python'], context['problem_path'], workspace,
                                   base / 'science', self.gpu, limits=self.config.get('sandbox_resources'),
                                   audit_sources=hasattr(self.evaluation, 'evidence'),
                                   write_extra=[task_source])
                context = {**context, 'workspace': str(workspace),
                           'output_dir': str(workspace / 'output')}
            else:
                if hasattr(self.evaluation, 'evidence') and not self.distillation:
                    _, candidates = self.evaluation.selection_candidates(self.store.records())
                    context = {**context, 'final_selection_candidates': candidates,
                               'current_final_selection': getattr(self.evaluation, 'meta_selection', None)}
                if self.selection_only:
                    # No writable Agent checkout, scientific sandbox or data view.
                    public = base / 'selection_interface'
                    write_json(public / 'context.json', context)
                    write_json(public / 'history.json', {'runs': [self.store.record_index(r) for r in self.store.records()]})
                    (public / 'README.md').write_text(
                        'Research is closed. Use list_history, read_record, read_version and read_artifact '
                        'to inspect existing evidence; list_history_files and read_history_file provide '
                        'completed model conversations and full command logs on demand. '
                        'Then use select_final_artifact or finish_experiment. '
                        'If no eligible artifact exists, use finish with a summary explaining the missing submission. '
                        'Only read-only history access, get_time and selection are allowed. Test is unavailable.\n')
                    seal(public)
                    context = {**context, 'eval_path': str(public / 'history.json'),
                               'interface_path': str(public / 'README.md')}
                else:
                    notes = None
                    if self.distillation:
                        notes = self.inherited_experience['text'] if self.inherited_experience else ''
                    development = Development(self.config, self.store, base, context, self.gpu,
                                              source / 'source', allow_meta, experience=notes)
                    context = {**context, **development.context()}
            visible = [r['run_id'] for r in self.store.records()]
            context = {**context, 'agent_version': identity, 'meta_version': meta_version,
                       'visible_history': visible, 'artifact_parent': artifact_parent,
                       'source_files': manifest['files'], 'kind': kind}
            if not self.distillation:
                from rsi.task_trajectories import complete_history_context
                context = complete_history_context(context, self.store,
                                                   deadline=time.monotonic()+self.remaining())
            write_json(base / 'context.json', context)
            tool_results = ToolResults(base / 'tool-results', run_id, policy=policy)
            if kind == 'task':
                timing = stage_timing(self.config, kind, run_id)
            elif self.distillation:
                timing = distillation_timing(context['chain'], context['mode'],
                                             self.config.get('chain_distillation_seconds', DISTILLATION_SECONDS))
            elif context.get('final_selection_only'):
                selection_timing = (
                    'Please complete it promptly. '
                    if self.stage_deadline is None else
                    'This review has its own configured time limit after the research budget. ')
                timing = ('FINAL META REVIEW. Compare historical research and select the final artifact. '
                          'No successor is required or published in this session. Research is closed. '
                          'Only read-only history access, get_time and selection are allowed: no shell, Python tools, '
                          'file edits, training, label queries, data fetching or re-evaluation. '
                          'Use select_final_artifact or finish_experiment with an eligible existing artifact '
                          'ID and a comparison reason. If eligible_artifacts is zero, use finish with a summary '
                          'explaining why no validated submission is available; this closes the review without Test. ' + selection_timing +
                          'API usage still counts against the same cumulative ledger and limits. Test is '
                          'unavailable. Candidate scores with different cohort hashes may not be directly '
                          'comparable. list_history exposes the run index; list_history_files and '
                          'read_history_file expose completed model/tool transcripts. Full histories and '
                          'candidate score tables are available on demand, not appended to every prompt.\n'
                          + encoded({'current_final_selection': context.get('current_final_selection'),
                                     'eligible_artifacts': context.get('final_selection_candidates', {}).get('eligible', 0),
                                     'visible_research_runs': visible}))
            else:
                timing = stage_timing(self.config, kind, run_id)
            experience = (render_experience(self.inherited_experience)
                          if self.inherited_experience and phase in ('task_execution', 'evolution') else None)
            prompt_context = PromptContext(timing, self.resources(max_calls), tool_results,
                                           final_submission=context['final_submission'], policy=policy,
                                           experience=experience)

            self.startup_seconds[kind] = time.monotonic() - begin
            dirty = kind == 'task'
            self.journal.add('session_start', run_id=run_id, agent_kind=kind, version=identity, meta_version=meta_version,
                             max_calls=max_calls, max_tool_calls=None, visible_history=visible,
                             session_info=self.session_info, phase=phase, context_policy=policy)
            for turn in itertools.count():
                if not self.post_research and (self.evaluation.remaining() <= 0 or self.evaluation.state.should_skip):
                    raise BudgetExceeded('Shared solve time exhausted or official skip requested')
                if self.remaining() <= 0:
                    raise StageDeadlineReached('Configured stage action deadline reached')
                current_context = {**context, 'resources': self.resources(None if max_calls is None else max_calls - calls)}
                if protocol_feedback:
                    current_context['agent_protocol_error'] = protocol_feedback
                prompt_context.observe_resources(current_context['resources'])
                try:
                    result = proc.step(state, current_context, response)
                except AgentTransportExceeded as exc:
                    protocol_feedback = {'error': str(exc), 'code': 'agent_transport_limit',
                                         'limit_bytes': AGENT_TRANSPORT_BYTES}
                    response = {**protocol_feedback, 'rejected_tool_result':
                                response.get('_tool_result') if isinstance(response, dict) else None}
                    continue
                protocol_feedback = None
                state, request = result['state'], result['request']
                op = request.get('op')
                if op not in ('llm', 'finish'):
                    # Failed/denied tool attempts count; API calls, retries and
                    # host finalization do not count as Agent tool calls.
                    self.tool_calls += 1
                allowed = TASK_OPS if kind == 'task' else META_OPS
                if context.get('final_selection_only'):
                    allowed = FINAL_SELECTION_OPS
                elif self.distillation:
                    allowed = DISTILLATION_OPS
                stdout_path = None
                if op not in allowed:
                    response = {'error': f'Capability denied for {kind}: {op}'}
                elif op == 'llm':
                    if max_calls is not None and calls >= max_calls:
                        break
                    try:
                        prepared = self.prepare_messages(request.get('messages'))
                        from rsi.task_trajectories import history_review_purpose
                        purpose = history_review_purpose(context, request, prepared)
                    except ValueError as exc:
                        # A malformed request is a protocol error that the agent
                        # can repair in the same session and cumulative budget.
                        detail = str(exc) + '. Repair the message list and retry; no API request was sent.'
                        response = {'error': detail, 'text': detail}
                        rejected_messages += 1
                        rejected_path = base / 'protocol-errors' / f'{rejected_messages:04d}.json'
                        write_json(rejected_path, {'request': request, 'error': detail})
                        rejected_path.chmod(0o444)
                        self.journal.add('agent_request_rejected', run_id=run_id, version=identity,
                                         op='llm', error=detail, raw_request=str(rejected_path))
                        continue
                    messages, context_metrics = prompt_context.render(
                        prepared, self.resources(None if max_calls is None else max_calls - calls))
                    context_metrics['private_message_fields_removed'] = sum(
                        len(original) - len(clean) for original, clean in zip(request['messages'], prepared))
                    self.validate_messages(messages)
                    calls += 1
                    audit = base / 'prompt-audit' / f'{calls:04d}.json'
                    write_json(audit, {'original_messages': request['messages'], 'metrics': context_metrics})
                    self.journal.add('context_rendered', run_id=run_id, version=identity,
                                     next_api_call=self.client.budget.calls + 1,
                                     raw_prompt=str(audit), **context_metrics)
                    try:
                        options = {'purpose': purpose} if purpose else {}
                        response = {'text': self.client.complete(messages, kind, identity, self.remaining(), **options)}
                    except APIOutputTruncated as exc:
                        response = {'error': str(exc), 'text': str(exc)}
                    except APIContextExceeded as exc:
                        response = {'error': str(exc), 'code': 'model_context_limit',
                                    'text': str(exc), 'repair_instruction': 'Revise the model request to fit provider capacity.'}
                        self.journal.add('agent_context_rejected', run_id=run_id, version=identity,
                                         call=self.client.budget.calls, error=str(exc))
                    # Raw requests/responses are already recorded with call ids.
                    self.journal.add('agent_llm', run_id=run_id, version=identity, call=self.client.budget.calls)
                    continue
                else:
                    try:
                        if op == 'get_time':
                            response = compact_budget(self.resources(None))
                        elif op == 'read_tool_result':
                            response = tool_results.read(request)
                        elif op in HISTORY | META_HISTORY:
                            response = self.store.query(request, identity, deadline=time.monotonic()+self.remaining())
                        elif op == 'bash':
                            target = science if kind == 'task' else development.sandbox
                            response = bash(target, request['command'],
                                            min(3600 if kind == 'task' else 120, self.remaining()))
                            stdout_path = response.get('log')
                            if kind == 'task':
                                dirty = True
                        elif op == 'editor':
                            response = development.editor({k: v for k, v in request.items() if k != 'op'},
                                                          min(30, self.remaining()))
                            stdout_path = response.get('log')
                        elif op == 'python':
                            seconds = request.get('timeout_s', 120)
                            if type(seconds) not in (int, float) or not 0 < seconds <= 3600:
                                raise ValueError('timeout_s must be 1..3600')
                            response = science.run(request['code'], min(seconds, self.remaining()))
                            stdout_path = response.get('log')
                            dirty = True
                        elif op == 'evaluate':
                            response = evaluate()
                        elif op in ('simulator_observe', 'simulator_action', 'simulator_run'):
                            if not hasattr(self.evaluation, 'simulator_status'):
                                raise ValueError('Simulator tools require an interactive task')
                            attribution = dict(run_id=run_id, task_version=identity,
                                               meta_version=meta_version)
                            if op == 'simulator_observe':
                                response = self.evaluation.simulator_status()
                            elif op == 'simulator_action':
                                dirty = True
                                response = self.evaluation.simulator_action(request['action'],
                                    request.get('arguments', {}), self.remaining(), attribution)
                            else:
                                seconds = request.get('timeout_s', 120)
                                if type(seconds) not in (int, float) or not 0 < seconds <= 3600:
                                    raise ValueError('timeout_s must be in (0,3600]')
                                dirty = True
                                response = self.evaluation.run_policy(science, workspace,
                                    request['path'], min(seconds, self.remaining()), attribution)
                            if kind == 'task':
                                response['trajectory_file'] = self.evaluation.export_trace(workspace)
                        elif op in ALL_EVIDENCE_OPS:
                            if not hasattr(self.evaluation,'evidence'):
                                raise ValueError('Evidence tools are available only for active-label tasks')
                            ids = request.get('sample_ids')
                            if request.get('sample_ids_file'):
                                if ids is not None:
                                    raise ValueError('Choose sample_ids or sample_ids_file, not both')
                                path = safe_path(workspace,request['sample_ids_file'])
                                ids = path.read_text().splitlines()
                            response = self.evaluation.evidence_dispatch(op, {**request, 'sample_ids': ids},
                                kind, run_id, self.remaining(), tool_call=self.tool_calls)
                            if (kind == 'task' and op == 'request_labels'
                                    and response.get('status') == 'pending_meta_decision'
                                    and not response.get('error')):
                                handoff = {'reason': 'annotation_request', 'run_id': run_id,
                                           'request_tool_call': self.tool_calls,
                                           **{key: response[key] for key in ('request_path', 'request_id')
                                              if key in response}}
                                summary = ('Task handed off to Meta to decide an additional data/label request. '
                                           'Request reason: ' + request['reason'])
                        elif op == 'create_candidate':
                            if request['kind'] == 'meta' and not allow_meta:
                                raise ValueError('Meta evolution is disabled for this session')
                            response = {'candidate': self.store.create(request['kind'], request['base_version'],
                                        request.get('source_versions', []), request.get('rationale', ''),
                                        source_run=request.get('source_run'))}
                        elif op == 'read_file':
                            response = read_text(safe_path(self.store.candidate(request['candidate']), request['path']),
                                                 request.get('offset', 0), request.get('limit', 16000))
                        elif op == 'write_file':
                            self.store.edit(request['candidate'], request['path'], request['content'])
                            response = {'written': request['path']}
                        elif op == 'check_candidate':
                            response = self.check(request['candidate'], context)
                        elif op == 'submit_candidate':
                            candidate = request['candidate']
                            info = self.store.candidates[candidate]
                            if info['kind'] == 'meta' and not allow_meta:
                                raise ValueError('Meta evolution is disabled')
                            checks = self.check(candidate, context)
                            previous = self.store.version(info['base_version'])[1]['files']
                            changes = [p for p in sorted(previous.keys() | checks['files'].keys())
                                       if previous.get(p) != checks['files'].get(p)]
                            parent = request.get('artifact_parent')
                            if parent:
                                if info['kind'] != 'task':
                                    raise ValueError('Only Task may reuse scientific artifacts')
                                self.store.artifact(parent, deadline=time.monotonic()+self.remaining())
                            published = self.store.publish(info['kind'], self.store.candidate(candidate),
                                          base=info['base_version'], sources=info['source_versions'], author=identity,
                                          rationale=info['rationale'], checks=checks, source_run=info.get('source_run'))
                            submitted[info['kind']] = {
                                'version': published, 'artifact_parent': parent,
                                'source_run': info.get('source_run'),
                                'decision': 'retain' if published == context['current_' + info['kind']] else 'update',
                                'reason': request.get('reason', info['rationale'])}
                            response = {**submitted[info['kind']], 'changed_files': changes,
                                        'changed_python_files': [p for p in changes if p.endswith('.py')]}
                        elif op in ('select_final_artifact', 'finish_experiment'):
                            if not hasattr(self.evaluation, 'evidence'):
                                raise ValueError('Explicit final selection requires an active-label task')
                            decision = dict(artifact=request.get('artifact'), reason=request.get('reason'),
                                            meta_version=identity, run_id=run_id, phase=phase,
                                            outside_research_budget=True, selected_at=time.time())
                            artifact, comparison = self.evaluation.select_submission(self.store.records(), decision)
                            if artifact is None:
                                if (op == 'finish_experiment' and request.get('artifact') is None
                                        and self.finish_without_submission(request.get('reason'), identity, run_id, phase)):
                                    response = dict(status='no_submission', reason=request['reason'])
                                    self.journal.add('agent_action', run_id=run_id, version=identity,
                                                     turn=turn, request=request, response=response, tool_call=self.tool_calls)
                                    experiment_finished = True
                                    status, summary = 'finished', request['reason']
                                    break
                                raise ValueError(comparison['error'])
                            current = run_io('hashes', time.monotonic()+self.remaining(), source=artifact['workspace'])
                            if current != artifact['files']:
                                raise ValueError('Selected artifact changed since validation')
                            self.evaluation.meta_selection = decision
                            write_json(self.evaluation.directory/'meta_selection.json', decision)
                            self.journal.add('meta_final_selection', **decision)
                            response = dict(status='selected', **decision, comparison=comparison)
                            if op == 'finish_experiment':
                                experiment_finished = True
                                status, summary = 'finished', request['reason']
                        elif op == 'finish':
                            if kind == 'meta' and context.get('final_selection_only'):
                                if (not getattr(self.evaluation, 'meta_selection', None)
                                        and not self.finish_without_submission(request.get('summary'), identity, run_id, phase)):
                                    raise ValueError('Select a validated artifact before finishing final review')
                            elif kind == 'meta':
                                publish_successors(request.get('summary', ''))
                            summary = request.get('summary', '')
                            status = 'finished'
                            break
                    except BudgetExceeded:
                        raise
                    except (ValueError, TypeError, KeyError, OSError, RuntimeError, SyntaxError) as exc:
                        response = {'error': f'{type(exc).__name__}: {exc}'[:4000]}
                        if op == 'finish' and self.selection_only:
                            response['repair_instruction'] = ('Research is closed. Select an eligible existing artifact when available; '
                                                              'if none exist, finish with a summary explaining the missing submission.')
                        elif op == 'finish' and kind == 'meta':
                            response['repair_instruction'] = (
                                'Successor publication failed. This session remains open. '
                                'Repair the required successor code/submission, then finish again; '
                                'the same shared clock and API ledger continue.')
                record = {'turn': turn, 'request': request, 'response': response,
                          'tool_call': self.tool_calls if op != 'finish' else None}
                trace.append(record)
                self.journal.add('agent_action', run_id=run_id, version=identity, **record)
                response, tool_metrics = tool_results.publish(
                    len(tool_results.records) + 1, request, response, stdout_path=stdout_path)
                prompt_context.observe_tool(request, response)
                self.journal.add('tool_result_displayed', run_id=run_id, version=identity, **tool_metrics)
                if handoff is not None:
                    status = 'finished'
                    self.journal.add('annotation_request_handoff', **handoff)
                    break
                if experiment_finished:
                    break
                if (kind == 'task' and op in ('simulator_action', 'simulator_run')
                        and getattr(self.evaluation, 'episode_done', False)):
                    status = 'finished'
                    summary = 'Simulator episode terminal; snapshot and score, then hand off to Meta.'
                    self.journal.add('simulator_terminal_handoff', run_id=run_id,
                                     episode_index=self.evaluation.episode_index)
                    break
        except StageDeadlineReached as exc:
            status, error = 'stage_deadline', str(exc)
            failure = dict(category='deadline', scope='stage', expires=self.action_deadline)
        except HostIODeadline as exc:
            status, error, failure = self.io_failure(exc)
        except BudgetExceeded as exc:
            status = ('stage_deadline' if self.stage_deadline is not None
                      and self.remaining() <= 0 and (self.post_research or self.evaluation.remaining() > 0) else 'budget_exhausted')
            error = str(exc)
        except KeyboardInterrupt as exc:
            status, error = 'interrupted', str(exc)
            raise
        except APIRequestError as exc:
            status, error = self.deadline_status() or 'api_error', str(exc)
            failure = dict(category='infrastructure' if status == 'api_error' else 'deadline', operation='model_request',
                           retryable=exc.retryable, http_status=exc.http_status)
        except AgentProcessFailed as exc:
            status, error = 'failed', f'{type(exc).__name__}: {exc}'
            failure = dict(category='agent_process', operation='agent_step', retryable=False)
        except Exception as exc:
            status = 'failed'
            error = f'{type(exc).__name__}: {exc}'
        finally:
            self.finalizing = True
            if not self.post_research and status in ('finished', 'call_allocation', 'stage_deadline') and self.evaluation.remaining() <= 0:
                status = 'budget_exhausted'
                error = error or 'Shared solve deadline reached'
            if self.distillation and development is not None and status == 'stage_deadline':
                # Model/tool work has stopped and sandbox descendants have been reaped.
                # Use the saved checkout, without another model call or an Agent finish.
                # Only bounded host checks/publication run after the action deadline;
                # keep the expired stage clock intact for the record and checks' context.
                try:
                    summary = 'M-chain time limit reached; transfer the saved Agents and notes.'
                    publish_successors(summary, deadline=time.monotonic()+30, allow_empty_experience=True)
                    deadline_handoff = True
                    self.journal.add('chain_deadline_handoff', run_id=run_id, submitted=submitted)
                except Exception as exc:
                    status, error = 'failed', f'Deadline handoff validation failed: {type(exc).__name__}: {exc}'
                    failure = dict(category='handoff_validation', scope='stage')
                    self.journal.add('chain_deadline_handoff_failed', run_id=run_id, error=error)
            elif development is not None and not self.distillation and status == 'stage_deadline' and self.remaining() > 0:
                try:
                    publish_successors('Stage deadline: publish completed, validated changes only.')
                except Exception as exc:
                    error = (error + '; ' if error else '') + f'Publication deferred: {type(exc).__name__}: {exc}'
                    self.journal.add('stage_publication_deferred', run_id=run_id, error=error)
            if kind == 'task':
                # A data request yields before further fitting/scoring. Preserve
                # unfinished scientific work without presenting it as evaluated.
                try:
                    if handoff is not None and self.remaining() > 0:
                        attempt, artifact = snapshot(time.monotonic() + self.remaining())
                        artifact['evaluation_status'] = 'not_evaluated'
                        write_json(attempt / 'artifact.json', artifact)
                        handoff['artifact'] = artifact['id']
                        self.journal.add('data_request_snapshot', run_id=run_id,
                                         artifact=artifact['id'], evaluated=False)
                    elif dirty and handoff is None and status != 'interrupted' and self.remaining() > 0:
                        evaluate(automatic=True)
                    elif dirty:
                        # Preserve the live workspace for diagnosis; never perform an
                        # unbounded copy after research time has expired. Earlier
                        # validated immutable artifacts remain eligible for Meta.
                        self.journal.add('terminal_snapshot_skipped', run_id=run_id,
                                         workspace=str(workspace), reason='No research time remaining')
                except HostIODeadline as exc:
                    status, error, failure = self.io_failure(exc)
                except Exception as exc:
                    # Persist the failed Task and its earlier valid artifacts so
                    # Meta can diagnose it. A snapshot failure must not escape
                    # finally and abort the entire research loop.
                    snapshot_error = f'{type(exc).__name__}: {exc}'
                    error = (error + '; ' if error else '') + 'Terminal snapshot failed: ' + snapshot_error
                    if status not in ('interrupted', 'budget_exhausted'):
                        status = 'failed'
                    self.journal.add('artifact_snapshot_failed', run_id=run_id, error=snapshot_error)
                if task_source is not None and task_source.exists():
                    try:
                        source_archive = self.store.archive_task_source(run_id, identity)
                    except Exception as exc:
                        source_archive_error = f'{type(exc).__name__}: {exc}'
                        self.journal.add('agent_source_archive_failed', run_id=run_id,
                                         source=str(task_source), error=source_archive_error)
            record = {'run_id': run_id, 'kind': kind, 'phase': phase,
                      'task_version': identity if kind == 'task' else submitted.get('task', {}).get('version'),
                      'meta_version': meta_version, 'executed_version': identity, 'status': status, 'error': error,
                      'failure': failure, 'handoff': handoff,
                      'agent_source_archive': source_archive,
                      'agent_source_archive_error': source_archive_error,
                      'summary': summary, 'artifacts': artifacts, 'evaluations': evaluations, 'trace': trace,
                      'experiment_finished': experiment_finished,
                      'final_selection': getattr(self.evaluation, 'meta_selection', None),
                      'submitted': submitted, 'deadline_handoff': deadline_handoff,
                      'artifact_parent': artifact_parent, 'visible_history': visible,
                      'budget_before': cost_before, 'budget_after': self.client.budget.snapshot(),
                      'session_info': self.session_info,
                      'tool_calls': self.tool_calls, 'max_tool_calls': None,
                      'api_calls': calls, 'wall_seconds': time.monotonic() - begin}
            self.store.save_record(record)
            write_json(base / 'result.json', record)
            if not self.post_research:
                try:
                    archive = publish_history(self.store.root, base, record)
                    self.journal.add('research_history_published', run_id=run_id, **archive)
                except Exception as exc:
                    # Preserve the completed scientific record and explicitly
                    # report any incomplete archive; never silently drop history.
                    self.journal.add('research_history_failed', run_id=run_id,
                                     error=f'{type(exc).__name__}: {exc}')
            self.journal.add('session_end', run_id=run_id, status=status, error=error, submitted=submitted,
                             failure=failure,
                             tool_calls=self.tool_calls, max_tool_calls=None, phase=phase,
                             budget=self.client.budget.snapshot())
        return record
