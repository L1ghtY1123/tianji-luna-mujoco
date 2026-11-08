"""A visual limitation is evidence; selecting a recovery motion is a proposal."""
from ..core.decisions import JudgmentRequest


def visibility_requests(subject=None, *, failed_skill=None, required_judgment=None):
    subject = subject or {}
    label = 'the subject required by the failed operation'
    if subject.get('id'):
        appearance = ' '.join(str(subject[key]) for key in ('color', 'kind') if subject.get(key))
        label = 'the ' + (appearance or 'object') + ' with permanent instance ID ' + subject['id']
    requirement = 'the required visual judgment'
    if required_judgment:
        requirement = 'evaluating this failed factual question: "' + required_judgment + '"'
    if failed_skill in {'generate_grasps', 'plan_grasp'}:
        requirement = ('estimating the target extent and exposed contact surfaces needed for grasp planning '
            'from the current observations')
    scope = ('Use ALL three CURRENT RGB views for the failed operation. One adequate view is sufficient. '
        'Judge missing visual information only; low confidence, CAD mismatch, API or planning errors alone '
        'do not establish a visual limitation. '
        'Use dated references when supplied to identify the subject; determine current visibility from CURRENT RGB only. '
        'Do not substitute another similar-looking object or robot part.')
    wording = (
            'None of the CURRENT RGB views gives an adequate view of ' + label + ' for '
            + requirement + ', because it is outside the frame, cut off, or hidden. '
            'If ANY current view already shows enough of this subject to make that judgment, answer false. '
            'Judge visibility from the images, not from the fact that an operation failed. '
            'The exact cause of a visible information gap may be uncertain. Do not demand hidden surfaces '
            'or a complete silhouette when the judgment does not need them.')
    return [JudgmentRequest(name='recovery_view_limited', instructions=wording + ' ' + scope)]


def visibility_assessment(judgments):
    eligibility = judgments.get('recovery_view_limited', {'status': 'unknown'})
    # Gate the required visual fact directly, not confidence in its exact cause.
    value = {'yes': 'view_limited', 'no': 'sufficient'}.get(eligibility.get('status'), 'unknown')
    return {'status': 'unknown' if value == 'unknown' else 'known', 'value': value,
            'eligibility': eligibility}


def visibility_allows_motion(answer):
    return answer.get('status') == 'known' and answer.get('value') == 'view_limited'
