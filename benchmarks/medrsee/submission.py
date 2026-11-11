"""Validate Meta's explicit scientific submission; the host never ranks models."""
import hashlib
import json
from pathlib import Path

from rsi.evaluation import valid_score
from rsi.common import digest
from rsi.store import safe_path
from .membership import connect


def validation_cohort(pool):
    """Fingerprint actual authorized Val membership, not the changing train pool."""
    identity = hashlib.sha256()
    count = 0
    with connect(pool) as db:
        for row in db.execute("SELECT sample_id FROM samples WHERE role='val' AND acquired=1 ORDER BY ordinal"):
            identity.update(json.dumps(row[0], ensure_ascii=True).encode() + b'\n')
            count += 1
    return dict(samples=count, sha256=identity.hexdigest())


def declaration(task, artifact):
    source = Path(artifact['workspace'])
    if (source/'frozen_model.json').stat().st_size > 1024 * 1024:
        raise ValueError('Frozen declaration exceeds 1 MiB')
    frozen = json.loads((source/'frozen_model.json').read_text())
    if not isinstance(frozen, dict):
        raise ValueError('Frozen declaration must be an object')
    if frozen.get('task') != task or not isinstance(frozen.get('method'), str) or not frozen['method']:
        raise ValueError('Require frozen task/method declaration')
    entry = frozen.get('inference_entry', 'predict.py')
    if not isinstance(entry, str) or Path(entry).suffix != '.py':
        raise ValueError('Frozen inference_entry must name a saved Python file')
    checkpoints = frozen.get('checkpoints')
    if not isinstance(checkpoints, list) or not checkpoints or any(not isinstance(p, str) for p in checkpoints):
        raise ValueError('List checkpoint/preprocessing files in frozen_model.json checkpoints')
    from .training import provenance
    _, fitting_files = provenance(source, frozen)
    dependencies = frozen.get('source_dependencies', [])
    if not isinstance(dependencies, list) or any(not isinstance(name, str) or not name for name in dependencies):
        raise ValueError('source_dependencies must list saved method source files')
    training_entry = frozen.get('training_entry')
    if training_entry is not None:
        if not isinstance(training_entry, str) or not training_entry:
            raise ValueError('training_entry must name a saved fitting program')
        dependencies = [*dependencies, training_entry]
    for name in ['frozen_model.json', entry, *checkpoints, *fitting_files, *dependencies]:
        path = safe_path(source, name)
        relative = path.relative_to(source.resolve())
        if any(part in ('output', '.tmp', '__pycache__') for part in relative.parts):
            raise ValueError('Required frozen file is excluded from final inference: ' + name)
        if not path.is_file() or relative.as_posix() not in artifact['files']:
            raise ValueError('Missing saved frozen file: ' + name)
    return frozen


def selection_candidates(task, runs, primary):
    """Return validated candidates and comparison context, without ranking them."""
    artifacts = {a['id']: a for run in runs for a in run.get('artifacts', [])}
    report = dict(policy='explicit_meta_selection', artifacts=len(artifacts), eligible=0,
                  excluded=[], validation_cohorts=[], candidates=[], test_used_for_selection=False)
    eligible, cohorts, unknown = {}, {}, 0
    scored_ids = set()
    for run in runs:
        for feedback in run.get('evaluations', []):
            identity = feedback.get('artifact')
            scored_ids.add(identity)
            artifact = artifacts.get(identity)
            reason = None
            proof = feedback.get('inference_verification', {})
            if artifact is None:
                reason = 'Evaluation has no saved artifact'
            elif (feedback.get('error') or feedback.get('valid_primary_score') is not True
                  or isinstance(feedback.get('aggregate_improvement'), bool) or not valid_score(feedback, primary)):
                reason = 'No successful finite validation score'
            elif feedback.get('snapshot') and Path(feedback['snapshot']).resolve() != Path(artifact['workspace']).resolve().parent:
                reason = 'Evaluation and artifact snapshot differ'
            elif proof.get('status') != 'completed' or proof.get('source_files_sha256') != digest(artifact['files']):
                reason = 'Missing matching host inference verification'
            elif feedback.get('training_audit', {}).get('status') != 'passed':
                reason = 'Missing successful declared training membership audit'
            else:
                try:
                    declaration(task, artifact)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    reason = str(exc)
            if reason:
                report['excluded'].append(dict(artifact=identity, reason=reason))
                continue
            cohort = feedback.get('validation_cohort')
            if cohort is None:
                unknown += 1
            else:
                cohorts[cohort['sha256']] = cohort
            item = dict(artifact=identity, attempt=feedback.get('attempt'),
                        aggregate_improvement=feedback['aggregate_improvement'],
                        raw_scores=feedback['raw_scores'], validation_cohort=cohort,
                        validation_fitting_overlap=feedback['training_audit'].get('validation_overlap'))
            eligible[identity] = artifact
            report['candidates'].append(item)
    report['excluded'].extend(dict(artifact=identity, reason='No validation evaluation')
                              for identity in artifacts if identity not in scored_ids)
    report.update(eligible=len(eligible), validation_cohorts=list(cohorts.values()),
                  unknown_validation_cohorts=unknown,
                  same_validation_membership=len(cohorts) == 1 and unknown == 0)
    return eligible, report


def select_submission(task, runs, primary, decision):
    from benchmarks.selection import select
    return select(*selection_candidates(task, runs, primary), decision)
