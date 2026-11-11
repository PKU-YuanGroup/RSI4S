"""Host-owned submission instructions shared by every Task and Meta generation."""
from pathlib import Path
import re

from rsi.tasks import MEDRSEE_TASKS, SIMULATOR_TASKS, task_directory


MEDRSEE_EXECUTION_FRAGMENTS = {
    'MEDRSEE_BATCH_READER': 'medrsee-batch-reader.md',
    'MEDRSEE_IMAGE_READER': 'medrsee-image-reader.md',
    'MEDRSEE_VOLUME_PREFLIGHT': 'medrsee-volume-preflight.md',
}


def medrsee_execution_instructions(task_id, problem=None):
    """Expand only the shared guidance requested by this task's execution file."""
    if task_id not in MEDRSEE_TASKS:
        raise ValueError('MedRSEE execution instructions require a MedRSEE task')
    problem = Path(problem) if problem is not None else task_directory('', task_id) / 'problem'
    instructions = (problem / 'execution.md').read_text()
    if not instructions.strip():
        raise ValueError('Empty execution instructions: ' + task_id)

    def include(match):
        name = match.group(1)
        if name not in MEDRSEE_EXECUTION_FRAGMENTS:
            raise ValueError('Unknown execution instruction fragment: ' + name)
        path = Path(__file__).with_name('interfaces') / MEDRSEE_EXECUTION_FRAGMENTS[name]
        return path.read_text().strip()

    return re.sub(r'\{\{([^{}]+)\}\}', include, instructions).strip()


def submission_ready(task_id, workspace):
    """Readiness for early Task0 handoff, in addition to a valid validation score."""
    if task_id not in MEDRSEE_TASKS:
        return True
    from benchmarks.medrsee.submission import declaration
    try:
        source = Path(workspace)
        files = {p.relative_to(source).as_posix(): None for p in source.rglob('*') if p.is_file()}
        declaration(task_id, dict(workspace=str(source), files=files))
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def submission_contract(task_id, final_seconds=3600):
    protocol = 'medrsee-frozen' if task_id in MEDRSEE_TASKS else 'naturebench-output'
    if task_id in SIMULATOR_TASKS:
        protocol = 'simulator-repeated'
    template = Path(__file__).with_name('interfaces') / (protocol + '.md')
    instructions = template.read_text().replace('{{TASK_ID}}', task_id)
    instructions = instructions.replace('{{FINAL_SECONDS}}', str(final_seconds))
    if task_id in MEDRSEE_TASKS:
        instructions += '\n\n' + medrsee_execution_instructions(task_id)
    return {'protocol': protocol, 'task_id': task_id, 'instructions': instructions}
