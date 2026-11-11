"""Choose only the eligible artifact explicitly selected by Meta."""


def select(eligible, report, decision):
    report['meta_decision'] = decision
    if not eligible:
        error = ('Meta selected an ineligible artifact: ' + str(decision.get('artifact'))
                 if isinstance(decision, dict) and decision.get('artifact') is not None else
                 'Unable to explicitly select a final artifact')
        report.update(status='no_submission', error=error + '; no eligible validated artifact is available')
        return None, report
    if not isinstance(decision, dict) or not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
        report['error'] = 'Meta must explicitly select a validated artifact with a comparison reason'
        return None, report
    identity = decision.get('artifact')
    if identity not in eligible:
        report['error'] = 'Meta selected an ineligible artifact: ' + str(identity)
        return None, report
    report.update(next(item for item in report['candidates'] if item['artifact'] == identity))
    return eligible[identity], report
