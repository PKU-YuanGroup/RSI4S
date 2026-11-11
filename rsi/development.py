"""Editable Agent checkout for HyperAgents-style bash/editor development."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from rsi.chain import BUNDLE_BYTES, EXPERIENCE, EXPERIENCE_CHARS, check_experience
from rsi.common import write_json
from rsi.sandbox import Sandbox, assert_plain_tree
from rsi.store import hashes, seal


def distillation_readme(repo, allow_meta):
    return (
        '# Chain distillation interface\n\n'
        'Research for this task has closed and its final artifact is selected. This session prepares the '
        'starting point for a later, different task in the same task chain. Only the Agents and notes below '
        'cross the task boundary; scientific artifacts, data, labels, histories and scores do not.\n\n'
        f'Editable checkout: {repo}\n'
        '- repo/task: required Task Agent for the next task, initialized from the current published version. '
        'Check the latest completed Task source archive in history.json for useful self-edits missing here. '
        'Preserve reusable capabilities and make minimal changes to generalize obvious task dependencies; '
        'unchanged source is valid.\n'
        '- repo/meta: Meta Agent for the next task. '
        + ('Meta self-evolution is enabled. You may modify its code, prompts, tools and control flow as needed, '
           'including distill_prompt.txt. Focus on evidence-backed, reusable improvements and removal of task '
           'dependencies. Preserve evolution, final selection and distillation; unchanged source is also valid.\n'
           if allow_meta else 'Meta self-evolution is disabled; the current Meta is carried unchanged.\n') +
        f'- repo/{EXPERIENCE}: the chain\'s general research experience, prefilled with notes inherited from '
        f'earlier tasks. Explicit finish requires nonempty notes, at most {EXPERIENCE_CHARS} characters. Keep a few concise, '
        'evidence-backed lessons; merge or correct only where needed. Useful inherited notes may stay unchanged.\n'
        '- repo/submission.json: optional base_version, source_versions and Task source_run provenance. '
        'artifact_parent must stay null.\n\n'
        f'Each Agent bundle is limited to {BUNDLE_BYTES} bytes of .py/.txt/.json/.md files (at most 150000 '
        'bytes each). Keep general improvements to prompts, tools, control flow and context_policy.json; '
        'generalize fixed task paths and output requirements, and remove sample IDs, label lists and cached results. '
        'Keep reusable domain tools; do not redesign Agents or prescribe the next task\'s scientific solution. '
        'Never write data values, labels, sequences, sample identifiers or Test information.\n\n'
        'history.json lists published versions, completed records, Task source archives and artifact '
        'workspaces. HISTORY_DIR holds the complete research trajectories with exact model requests, tool '
        'outputs and command logs; list_history_files and read_history_file page through them. Task problem '
        'data is not mounted in this session. Start with the current checkout and latest Task summary; '
        'read further evidence only to resolve a concrete uncertainty, without replaying all trajectories.\n\n'
        'A final answer runs the usual Agent checks, verifies the notes and bundle sizes, then seals the '
        'handoff. Failed checks return as tool feedback. Bash commands have a 120-second cap; this session '
        'has its own time limit after research and before frozen Test inference. Call get_time() with no '
        'arguments at the start, periodically during work and before submitting. Use stage_elapsed_seconds '
        'and stage_remaining_seconds; remaining_solve_seconds is zero because research is closed. '
        'Leave time for checks and repairs. At the time limit the host stops model/tool work and '
        'automatically checks and transfers the saved checkout and notes, without another model call '
        'or a final response. Timeout alone does not block the chain; notes may still be empty in '
        'this case. Save useful changes as you work. Invalid saved sources and other failures still '
        'stop the chain.\n')


def bash(sandbox, command, timeout):
    if not isinstance(command, str) or not command.strip():
        raise ValueError('bash requires a nonempty command string')
    # The existing supervisor enforces UID, filesystem/network/GPU isolation,
    # output bounds, wall time and cleanup of all background descendants.
    return sandbox.run(f'import os;os.execv("/bin/bash", ["bash", "-o", "pipefail", "-c", {command!r}])', timeout)


class Development:
    def __init__(self, config, store, base, context, gpu, source, allow_meta, experience=None):
        """`experience` is the chain notes to revise in a distillation session, else None."""
        self.store, self.source, self.allow_meta = store, Path(source), allow_meta
        self.distillation = experience is not None
        self.current_meta = context['current_meta']
        self.workspace = Path(base) / 'workspace'
        self.repo = self.workspace / 'repo'
        self.repo.mkdir(parents=True)
        self.public = Path(base) / 'development_interface'
        self.public.mkdir()
        self.required = ['task']
        self.optional = ['meta'] if allow_meta else []
        self.submission = {}
        for kind in ('task', 'meta'):
            identity = context['current_' + kind]
            version, _ = store.version(identity)
            shutil.copytree(version / 'source', self.repo / kind)
            self.submission[kind] = {'base_version': identity, 'source_versions': [], 'artifact_parent': None}
        write_json(self.repo / 'submission.json', self.submission)
        if self.distillation:
            (self.repo / EXPERIENCE).write_text(experience)
        records = store.records()
        source_archives = [record['agent_source_archive'] for record in records
                           if record.get('agent_source_archive')]
        # The host's immutable records describe history; defer expensive content
        # verification until a particular artifact is reused or submitted.
        artifacts = [a for r in records for a in r['artifacts']]
        if config.get('reuse_artifacts') and artifacts and not self.distillation:
            valid = {e.get('artifact') for r in records for e in r.get('evaluations', [])
                     if e.get('valid_primary_score')}
            reusable = [a for a in artifacts if a['id'] in valid] or artifacts
            self.submission['task']['artifact_parent'] = reusable[-1]['id']
            write_json(self.repo / 'submission.json', self.submission)
        versions = store.root / 'versions'
        history = store.root / 'records'
        write_json(self.public / 'history.json', {
            'versions_path': str(versions), 'records_path': str(history),
            'research_history_path': str(store.root/'history'),
            'task_source_archives': [{k: archive[k] for k in ('run_id', 'path', 'base_version')}
                                     for archive in source_archives],
            'artifacts': [{k: a[k] for k in ('id', 'workspace', 'task_version', 'meta_version') if k in a}
                          for a in artifacts],
            'runs': [store.record_index(record) for record in records],
            'details': 'Full records include the artifact manifests, exact evaluations and raw traces. Read by run_id.',
        })
        write_json(self.public / 'context.json', context)
        (self.public / 'README.md').write_text(distillation_readme(self.repo, allow_meta) if self.distillation else (
            '# Agent development interface\n\n'
            f'Editable checkout: {self.repo}\n'
            f"Current explicit Task artifact_parent: {self.submission['task']['artifact_parent']}. "
            'This selection controls scientific file reuse independently of Agent versions.\n'
            'Required selection: task. A final answer submits the checkout as your chosen next Task. '
            'Unchanged source retains the existing version; no file change is required. '
            'The final summary records the reason for this choice. '
            'Code, prompts, tools, configuration and control flow are editable.\n'
            + ('Meta self-evolution is enabled in this session. Unchanged Meta source retains its version. '
               'Edit repo/meta to publish a successor for the next Meta stage.\n'
               if allow_meta else 'Meta self-evolution is disabled in this session; only Task is published.\n') +
            'Use bash to inspect files and execute local tests. '
            'A normal final answer triggers independent checks and publication; '
            'check failures return as tool feedback for repair. Task and Meta may both remain unchanged.\n'
            'Tool calls have no count limit; they share the experiment time and model budgets. '
            'Each bundle contains entry.py, tools.py, prompt.txt and optional helpers. '
            'entry.step(state, context, response) returns {state, request}; '
            'each request uses one host operation. State is passed to the next fresh process.\n'
            'Host operations: both roles llm, finish, read_tool_result, list_history, read_record, read_version, read_artifact; '
            'Task also bash, python, evaluate; Meta also bash, editor, create_candidate, read_file, '
            'write_file, check_candidate, submit_candidate. llm accepts role/content messages and returns {text}.\n'
            'The initial model-facing tools are bash/editor/read_tool_result for Meta and bash/evaluate/read_tool_result for Task.\n'
            'read_tool_result accepts call_id, field (json or stdout), offset and limit. Only results from '
            'this session are accessible; next_offset pages through exact content. call_id identifies '
            'the executed request and raw result.\n'
            'Fixed host rules are in the system prefix; the latest host_budget is appended by the host. '
            'Each Agent bundle owns context_policy.json. It controls automatic archival, display sizes, '
            'deduplication and supplemental scientific facts. Settings take effect when that version runs. '
            'archive_history=false disables host conversation archival. tool_display_bytes and '
            'stdout_preview_chars accept null to disable their display caps. Retrieval and message '
            'composition can be implemented in entry.py and tools.py. Raw evidence remains immutable. '
            'The policy does not change tool permissions, budgets, model capacity, the 8 MiB Agent '
            'state/request transport limit or history read-page limits. '
            'An oversized Agent return is rejected without replacing the last accepted state. '
            'The next step receives context.agent_protocol_error and an error response; '
            'entry.py can revise its state/request. Provider context-length errors also return '
            'to the Agent as model_context_limit feedback. '
            'list_history is a paginated compact index; read_record supports character offset/limit '
            'for full records. File hashes remain in the complete manifests.\n'
            'history.json lists read-only source versions, completed records and artifact paths. '
            'task_source_archives lists completed Tasks\' self-modified Agent sources. '
            'Choose any recorded archive or published version as your starting point; no Task archive '
            'is automatically selected. Copy the chosen files into repo/task, then modify or retain them. '
            'HISTORY_DIR contains read-only completed research archives with manifests, exact API '
            'requests/responses, original prompts, tool outputs and full command logs. '
            'Use list_history_files/run_id and read_history_file for bounded pages, or process '
            'HISTORY_DIR with your own code during research. The platform supplies access only; '
            'retrieval, indexing, summaries and memory strategies are yours to develop. '
            'File listing returns metadata only; the platform does not put whole histories in prompts. '
            'context.json contains task inputs and the current version identities. '
            'Train/problem data are read-only at PROBLEM_DIR/DATA_DIR. '
            'Credentials, hidden evaluation files, evaluator and budgets are unavailable to shell commands. '
            'Model calls and official scoring must pass through the host.\n'
            'Optional: edit repo/submission.json to record base_version, source_versions and explicit '
            'Task artifact_parent. Changing metadata never copies files: copy the selected Agent source '
            'into the checkout yourself. Scientific artifacts are never inherited implicitly. '
            'Optionally set task.source_run to the completed Task run_id to record source provenance. '
            'This field may be omitted or null; it does not copy or select files. '
            'A Meta artifact_parent must remain null.\n'
            'Files persist; shell variables/processes do not. Meta commands have a 120-second cap. '
            'All research stages share one global time budget, with no per-stage limit or budget reset. '
            'Each model call receives updated host_budget values. Checks/publication use the same clock. '
            'Run foreground Python scripts for syntax checks and step/response replay; '
            'these local checks are not an official scientific evaluation.\n'
        ))
        if context.get('annotation'):
            with (self.public/'README.md').open('a') as stream:
                stream.write('\nMedRSEE evidence tools: both roles may fetch_data and annotation_status. '
                    'Only Meta may query_labels; Task may request_labels and select_validation. '
                    'A successful Task request_labels ends that Task and hands off to Meta; '
                    'task_handoff identifies the request and its unscored workspace snapshot. '
                    'Queries share one cumulative budget across all generations. '
                    'All iterative scores are validation scores. T is exposed only to final frozen inference.\n')
        if context.get('simulator_episode'):
            with (self.public/'README.md').open('a') as stream:
                stream.write('\nInteractive task: both roles may simulator_observe; only Task may '
                    'simulator_action and simulator_run. Episode lifecycle follows the host final_submission '
                    'protocol in context.json. There is no Agent reset, label pool or independent Test. '
                    'Meta edits the Task strategy using the historical actions and observations; '
                    'it cannot operate the simulator. Read the active protocol before planning '
                    'whether the next Task continues unfinished state or receives a new fixed episode.\n')
        seal(self.public)
        problem = context['problem_path']
        if self.distillation:
            # General notes are distilled from recorded trajectories, not from task data.
            problem = Path(base) / 'unmounted_problem'
            problem.mkdir()
            seal(problem)
        self.sandbox = Sandbox(config['python'], problem, self.workspace,
                               Path(base) / 'development_commands', gpu,
                               read_extra=[self.public, versions, history, store.root/'history', self.source,
                                           *(archive['path'] for archive in source_archives),
                                           *(a['workspace'] for a in artifacts)],
                               env_extra={'REPO_PATH': str(self.repo), 'EVAL_PATH': str(self.public / 'history.json'),
                                          'HISTORY_DIR': str(store.root/'history')},
                               limits=config.get('sandbox_resources'))
        store.journal.add('development_opened', repo_path=str(self.repo), required=self.required, optional=self.optional,
                          visible_versions=[p.name for p in versions.iterdir()],
                          visible_records=[r['run_id'] for r in records])

    def context(self):
        return {'repo_path': str(self.repo), 'eval_path': str(self.public / 'history.json'),
                'interface_path': str(self.public / 'README.md'), 'history_path': str(self.store.root/'history')}

    def editor(self, arguments, timeout):
        # Execute the versioned editor inside confinement, never import it into
        # the privileged host. Persist its upstream undo stack across processes.
        code = f'''import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('editor', {str(self.source / 'toolkit/editor.py')!r})
editor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(editor)
history=Path('.editor_history.json')
if history.exists(): editor.file_history._history=json.loads(history.read_text())
result=editor.tool_function(**json.loads({json.dumps(arguments)!r}))
history.write_text(json.dumps(editor.file_history._history))
print(result)
'''
        return self.sandbox.run(code, timeout)

    def candidates(self, summary, exclude=(), deadline=None):
        # Validate before any privileged filesystem traversal/copy. New processes
        # are fully reaped before reaching this path.
        assert_plain_tree(self.workspace, ignore=('.tmp', '__pycache__'))
        spec = json.loads((self.repo / 'submission.json').read_text())
        if not isinstance(spec, dict) or set(spec) != {'task', 'meta'}:
            raise ValueError('submission.json must contain task and meta objects')
        result = []
        for kind in self.required + self.optional:
            if kind in exclude:
                continue
            info = spec[kind]
            required = {'base_version', 'source_versions', 'artifact_parent'}
            if (not isinstance(info, dict) or not required <= set(info)
                    or set(info) - required - {'source_run'}):
                raise ValueError('Invalid successor metadata')
            if not isinstance(info['source_versions'], list):
                raise ValueError('source_versions must be a list')
            if kind != 'task' and info.get('source_run') is not None:
                raise ValueError('Only Task may use a completed Task source archive')
            parent = info['artifact_parent']
            if parent is not None and self.distillation:
                raise ValueError('A chain handoff carries no scientific artifacts; set artifact_parent to null')
            if parent is not None:
                if kind != 'task' or not isinstance(parent, str):
                    raise ValueError('Only Task can reuse an artifact id')
                self.store.artifact(parent, deadline=deadline)
            folder = self.repo / kind
            if not folder.is_dir():
                raise ValueError('Missing successor directory: ' + kind)
            files = hashes(folder)
            for name in files:
                path = folder / name
                if path.suffix not in ('.py', '.txt', '.json', '.md') or path.stat().st_size > 150000:
                    raise ValueError('Successors contain source/text files only, max 150000 bytes per file')
                path.read_text()  # Validate UTF-8 before publishing a source patch.
            if kind == 'meta' and files == self.store.version(self.current_meta)[1]['files']:
                continue
            candidate = self.store.create(kind, info['base_version'], info['source_versions'], str(summary),
                                          source_run=info.get('source_run'))
            destination = self.store.candidate(candidate)
            shutil.rmtree(destination)
            shutil.copytree(folder, destination, ignore=shutil.ignore_patterns('__pycache__'))
            # Writable checkout files are owned by the shell UID and mode 0600;
            # published versions/check subprocesses need read access.
            for path in [destination, *destination.rglob('*')]:
                path.chmod(0o755 if path.is_dir() else 0o644)
            self.store.journal.add('development_staged', candidate=candidate, files=files,
                                   artifact_parent=parent, source_run=info.get('source_run'))
            result.append((kind, candidate, parent))
        return result

    def experience(self, *, required=True):
        """Read the revised chain notes after candidates() verified the checkout is a plain tree."""
        path = self.repo / EXPERIENCE
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'repo/{EXPERIENCE} is missing')
        return check_experience(path.read_text(), required=required)
