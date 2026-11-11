"""Ordered task chains: a later task starts from the Agents and notes distilled after an earlier one.

Only the two Agent source bundles and one research-experience note cross a task
boundary. Scientific artifacts, data, labels, histories and Test results never do.
Each handoff is sealed by the host and verified again before the next task runs.
"""
from __future__ import annotations
from benchmarks.task_names import public_name

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from rsi.common import encoded, record_path, write_json
from rsi.sandbox import assert_plain_tree
from rsi.store import hashes, seal

FORMAT = 1
SCHEDULING_POLICY = 'chain_sequential_v1'
HANDOFF = 'chain_handoff'
EXPERIENCE = 'research_experience.md'
DISTILLATION_RUN = 'M-chain'
DISTILLATION_SECONDS = 1800
EXPERIENCE_CHARS = 20000
BUNDLE_BYTES = 256 * 1024
SOURCE_SUFFIXES = ('.py', '.txt', '.json', '.md')

DISTILLATION_TIMING = (
    'CHAIN DISTILLATION. Research for this task has closed and its final artifact selection is complete; '
    'this session cannot change either. A later, different scientific task in the same task chain starts '
    'from the Task and Meta Agents submitted here and receives repo/research_experience.md as inherited '
    'notes. Nothing else crosses the task boundary: no scientific artifacts, data, labels, histories or '
    'scores. Keep this handoff lightweight: preserve working Agents and reusable capabilities by default. '
    'Check the latest completed Task source archive for useful self-edits missing from repo/task. '
    'Make minimal changes to remove obvious task dependencies, such as fixed data paths, sample IDs, '
    'label lists and old output requirements. Keep useful domain tools; read task requirements from '
    'the next task\'s context. Do not redesign the Agents or plan the next scientific solution. '
    'Read the current checkout and latest Task summary first, then only the history needed to resolve '
    'a concrete uncertainty. When self-evolution is enabled, you may modify Meta code, prompts, tools '
    'and control flow as needed, including distill_prompt.txt. Focus on evidence-backed, reusable '
    'improvements and removal of task dependencies. Keep concise, general research notes; merge or correct them only '
    'where supported by evidence. Neither Agent source nor inherited notes need to change just to '
    'demonstrate progress. An explicit finish requires nonempty experience notes. Never copy data values, '
    'labels, sequences, sample identifiers or held-out Test information into the notes or Agents. '
    'repo/task is required; repo/meta may change only when Meta self-evolution is enabled. Scientific '
    'artifact_parent must stay null. Tools: bash and editor in the development checkout, read-only history '
    'and get_time; no training, evaluation, data fetching, label queries or final selection. A final response '
    'checks and seals the handoff; failed checks return as tool feedback for repair. At the time limit, '
    'the host stops model/tool work and automatically transfers the saved checkout and notes after '
    'bounded validation, without waiting for a final response or further model calls. Timeout alone '
    'is not a failure and does not block the next task. Save useful changes as you work; empty notes '
    'are allowed only for this deadline transfer. Other failures, including invalid saved sources, '
    'stop the chain without reverting to old Agents or notes. '
    'Call get_time with no arguments at the start, periodically while reading or editing, and before '
    'submitting. Use stage_elapsed_seconds and stage_remaining_seconds for this session; '
    'remaining_solve_seconds is zero because research has closed. Leave time for submission checks '
    'and repairs, and finish early when ready. This session has its own time limit after research '
    'and before frozen Test inference; API usage stays '
    'in the same cumulative ledger. Test is never feedback.'
)


def _fail_unless(condition, message):
    if not condition:
        raise ValueError(message)


def chain_settings(config):
    """Validate one run's chain position; None for an ordinary single-task run."""
    chain = config.get('chain')
    if chain is None:
        return None
    _fail_unless(isinstance(chain, dict) and {'name', 'tasks', 'position'} <= set(chain)
                 and not set(chain) - {'name', 'tasks', 'position', 'inherit'},
                 'chain requires name, tasks, position and optional inherit')
    name, tasks, position = chain['name'], chain['tasks'], chain['position']
    _fail_unless(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name),
                 'Chain name must be 1-64 letters, digits, dot, underscore or hyphen')
    _fail_unless(isinstance(tasks, list) and tasks and len(set(tasks)) == len(tasks)
                 and all(isinstance(task, str) and task for task in tasks),
                 'Chain tasks must be distinct task IDs')
    _fail_unless(type(position) is int and 1 <= position <= len(tasks), 'Chain position is outside the chain')
    _fail_unless(config.get('task') == tasks[position-1], 'Chain position does not name this task')
    inherit = chain.get('inherit')
    _fail_unless((inherit is None) == (position == 1),
                 'The first chain task starts from the initial Agents; every later task inherits its predecessor')
    seconds = config.get('chain_distillation_seconds', DISTILLATION_SECONDS)
    _fail_unless(seconds is None or (type(seconds) is int and seconds >= 1),
                 'chain_distillation_seconds must be null or a positive integer')
    return dict(name=name, tasks=list(tasks), position=position,
                inherit=None if inherit is None else str(record_path(inherit, area='runs')),
                distillation_seconds=seconds)


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def check_experience(text, *, required=True):
    _fail_unless(isinstance(text, str) and '\0' not in text, 'research_experience.md must be UTF-8 text')
    _fail_unless(len(text) <= EXPERIENCE_CHARS,
                 f'research_experience.md is limited to {EXPERIENCE_CHARS} characters; '
                 'merge and condense the notes instead of appending')
    _fail_unless(not required or text.strip(),
                 'Write general research experience in repo/research_experience.md before finishing')
    return text


def check_bundle(source):
    """Handoff bundles stay source/text sized; task data and payloads do not travel."""
    source = Path(source)
    files = hashes(source)
    total = 0
    for name in files:
        path = source / name
        size = path.stat().st_size
        _fail_unless(path.suffix in SOURCE_SUFFIXES and size <= 150000,
                     'Chain handoff Agents contain source/text files only, max 150000 bytes per file: ' + name)
        path.read_text()  # Reject non-UTF-8 payloads.
        total += size
    _fail_unless(total <= BUNDLE_BYTES,
                 f'Each chain handoff Agent is limited to {BUNDLE_BYTES} bytes of source text '
                 f'({total} found); remove task-specific data, ID lists, logs and cached results')
    return files


def _host_sealed(root):
    for path in [root, *root.rglob('*')]:
        info = path.lstat()
        _fail_unless(not path.is_symlink() and info.st_uid == os.geteuid() and not info.st_mode & 0o222,
                     'Chain handoff must remain host-owned and read-only')


def check_handoff_status(manifest):
    """Accept an explicit finish or a host-validated transfer at the time limit."""
    distillation = manifest.get('distillation') if isinstance(manifest, dict) else None
    _fail_unless(isinstance(distillation, dict) and (
        (manifest.get('status') == 'distilled' and distillation.get('distilled') is True
         and distillation.get('status') == 'finished') or
        (manifest.get('status') == 'deadline_transfer' and distillation.get('distilled') is False
         and distillation.get('status') == 'stage_deadline' and distillation.get('deadline_handoff') is True)),
        'Chain requires a successfully completed distillation or a validated deadline transfer; '
        'retained or unfinished handoffs are refused')


def load_handoff(chain, mode):
    """Verify the predecessor's sealed handoff before any inherited Agent is published."""
    if chain is None or chain['inherit'] is None:
        return None
    directory = Path(chain['inherit']) / HANDOFF
    _fail_unless(directory.is_dir(), 'Chain predecessor has no sealed handoff: ' + str(directory))
    assert_plain_tree(directory)
    _host_sealed(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    check_handoff_status(manifest)
    expected = dict(name=chain['name'], tasks=chain['tasks'], position=chain['position'] - 1)
    _fail_unless(manifest.get('format') == FORMAT and manifest.get('chain') == expected
                 and public_name(manifest.get('task_id')) == chain['tasks'][chain['position'] - 2]
                 and manifest.get('mode') == mode,
                 'Chain handoff belongs to a different chain, position, task or mode')
    for kind in ('task', 'meta'):
        if hashes(directory / kind) != manifest[kind]['files']:
            raise RuntimeError('Chain handoff Agent integrity violation: ' + kind)
    text = (directory / EXPERIENCE).read_text()
    if sha256_text(text) != manifest['experience']['sha256']:
        raise RuntimeError('Chain handoff research experience integrity violation')
    return dict(directory=str(directory), manifest=manifest,
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), experience=text)


def lineage(handoff):
    """Earlier chain steps in order, each bound to its sealed manifest digest."""
    if handoff is None:
        return []
    manifest = handoff['manifest']
    return manifest['lineage'] + [dict(position=manifest['chain']['position'], task_id=manifest['task_id'],
                                       run_directory=manifest['run_directory'], status=manifest['status'],
                                       manifest_sha256=handoff['manifest_sha256'])]


def agent_context(chain, handoff):
    """Compact chain facts visible to Agents; inherited notes are rendered by the host."""
    previous = lineage(handoff)
    return dict(name=chain['name'], position=chain['position'], length=len(chain['tasks']),
                previous_tasks=[public_name(step['task_id']) for step in previous],
                inherited_agents=handoff is not None,
                inherited_experience_sha256=sha256_text(handoff['experience']) if handoff else None,
                distillation=('After research and the final artifact selection, a separate M-chain session '
                              'outside the research budget prepares a lightweight handoff: preserve Agent '
                              'capabilities, generalize obvious task dependencies and maintain concise '
                              'research experience for the next task. Only those Agents and '
                              'notes cross the task boundary. Meta successors must keep running that session '
                              '(context.chain_distillation with distill_prompt.txt) as well as final selection.'))


def inherited_experience(handoff):
    if handoff is None or not handoff['experience'].strip():
        return None
    return dict(from_tasks=[public_name(step['task_id']) for step in lineage(handoff)], text=handoff['experience'],
                sha256=sha256_text(handoff['experience']))


def render_experience(experience):
    """Host-rendered block shared by every research session of the inheriting task."""
    return ('# Inherited research experience\n'
            'Meta wrote these notes after completing earlier tasks in this task chain ('
            + ', '.join(experience['from_tasks']) + '). They come from different tasks and data: check them '
            'against this task\'s instructions, data and evidence. Host rules and this task\'s contract take '
            'precedence.\n\n' + experience['text'].strip())


def distillation_timing(chain, mode, seconds):
    """Host instruction for M-chain; `chain` is the Agent-visible chain context."""
    limit = ('This M-chain session has no hard wall-clock limit configured.' if seconds is None else
             f'This M-chain work phase has a hard wall-clock limit of {seconds} seconds ({seconds / 60:g} minutes), '
             'including history reads, model calls, edits and normal submission checks. '
             'Only bounded host validation and sealing may follow the deadline; model/tool work does not resume.')
    return DISTILLATION_TIMING + '\n' + limit + '\n' + encoded({
        'chain': {k: chain[k] for k in ('name', 'position', 'length')},
        'meta_self_evolution': mode == 'coevolve', 'experience_char_limit': EXPERIENCE_CHARS,
        'agent_bundle_byte_limit': BUNDLE_BYTES, 'session_seconds': seconds})


def _remove(path):
    for item in [path, *path.rglob('*')]:
        if not item.is_symlink():
            item.chmod(0o700 if item.is_dir() else 0o600)
    shutil.rmtree(path)


def seal_handoff(directory, store, chain, *, mode, task_id, release, task_version, meta_version,
                 experience, distillation, inherited):
    """Write the next task's starting point once; the host never replaces it."""
    directory = Path(directory)
    target = directory / HANDOFF
    if target.exists():
        raise RuntimeError('Chain handoff cannot be replaced')
    status = 'deadline_transfer' if distillation.get('deadline_handoff') is True else 'distilled'
    check_handoff_status(dict(status=status, distillation=distillation))
    check_experience(experience, required=status != 'deadline_transfer')
    staging = Path(tempfile.mkdtemp(prefix='.chain-handoff-', dir=directory))
    try:
        bundles = {}
        for kind, identity in (('task', task_version), ('meta', meta_version)):
            source, info = store.version(identity)
            if info['kind'] != kind:
                raise ValueError('Chain handoff Agent kind mismatch')
            check_bundle(source / 'source')
            shutil.copytree(source / 'source', staging / kind, ignore=shutil.ignore_patterns('__pycache__'))
            if hashes(staging / kind) != info['files']:
                raise RuntimeError('Chain handoff Agent changed while sealing')
            bundles[kind] = dict(version=identity, files=info['files'])
        (staging / EXPERIENCE).write_text(experience)
        manifest = dict(format=FORMAT, status=status, task_id=task_id, mode=mode,
                        chain=dict(name=chain['name'], tasks=chain['tasks'], position=chain['position']),
                        run_directory=str(directory), release=release, **bundles,
                        experience=dict(path=EXPERIENCE, sha256=sha256_text(experience), characters=len(experience)),
                        distillation=distillation, lineage=lineage(inherited), created_at=time.time())
        write_json(staging / 'manifest.json', manifest)
        seal(staging)
        staging.rename(target)
    finally:
        if staging.exists():
            _remove(staging)
    return dict(path=str(target), manifest_sha256=hashlib.sha256((target / 'manifest.json').read_bytes()).hexdigest(),
                status=status, task_version=task_version, meta_version=meta_version,
                experience_sha256=sha256_text(experience), experience_characters=len(experience))
