"""Explicit Meta selection of host-fitted and independently validated programs."""
from pathlib import Path

from rsi.common import digest
from rsi.evaluation import valid_score
from benchmarks.naturebench_active.common import sha256


def selection_candidates(task, runs, primary):
    artifacts={a['id']:a for run in runs for a in run.get('artifacts',[])}
    eligible={};candidates=[];excluded=[];cohorts={}
    for run in runs:
        for feedback in run.get('evaluations',[]):
            identity=feedback.get('artifact');artifact=artifacts.get(identity)
            reason=None
            if artifact is None or feedback.get('error') or not valid_score(feedback,primary):
                reason='No successful finite validation score'
            elif feedback.get('inference_verification',{}).get('source_files_sha256')!=digest(artifact['files']):
                reason='Missing matching host inference proof'
            elif feedback.get('training_audit',{}).get('status')!='passed':
                reason='Missing isolated host fitting proof'
            else:
                model=Path(feedback['fitted_artifact']['path'])
                expected=feedback['fitted_artifact']['files']
                if {p.name:sha256(p) for p in model.iterdir()}!=expected:
                    reason='Validated fitted artifact changed'
                elif expected['model.py']!=artifact['files'].get('model.py'):
                    reason='Fitted program differs from the submitted program'
            if reason:
                excluded.append(dict(artifact=identity,reason=reason));continue
            eligible[identity]=artifact
            cohort=feedback['validation_cohort'];cohorts[cohort['sha256']]=cohort
            candidates.append(dict(artifact=identity,raw_scores=feedback['raw_scores'],validation_cohort=cohort,
                validation_coverage=feedback.get('validation_coverage'),
                generation_validation=feedback.get('generation_validation'),
                selection_metric=feedback.get('selection_metric'), metric_comparison=feedback.get('metric_comparison'),
                aggregate_improvement=feedback['aggregate_improvement'],
                aggregate_scope=feedback.get('aggregate_scope','Development validation on a fixed unit scale, not relative to any published value; not a Test improvement')))
    report=dict(policy='explicit_meta_selection',eligible=len(eligible),artifacts=len(artifacts),candidates=candidates,
                excluded=excluded,validation_cohorts=list(cohorts.values()),same_validation_membership=len(cohorts)==1,
                automatic_scalar_ranking=False,
                self_reported_loss_diagnostic=any(c['selection_metric']=='self_reported_loss_diagnostic' for c in candidates),
                test_used_for_selection=False)
    return eligible,report


def select_submission(task, runs, primary, decision):
    from benchmarks.selection import select
    return select(*selection_candidates(task, runs, primary), decision)
