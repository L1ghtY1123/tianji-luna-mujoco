"""A visual limitation is evidence; selecting a recovery motion is a proposal."""
from .decisions import JudgmentRequest

VISIBILITY_FIELDS = {'recovery_not_visible': 'not_visible', 'recovery_cropped': 'cropped',
                     'recovery_occluded': 'occluded'}


def visibility_requests():
    scope = ('Use ALL three CURRENT RGB views for the failed operation. One adequate view is sufficient. '
        'Judge missing visual information only; low confidence, CAD mismatch, API or planning errors alone '
        'do not establish a visual limitation. These limitations may overlap and are independent.')
    wording = {
        'recovery_not_visible': 'The subject required by the failed operation is absent from every current image, preventing that visual judgment.',
        'recovery_cropped': 'Image boundaries cut off visual information required by the failed operation, and no other current view supplies it.',
        'recovery_occluded': 'The robot or another object hides visual information required by the failed operation, and no other current view supplies it.'}
    return [JudgmentRequest(name=name, instructions=text + ' ' + scope) for name, text in wording.items()]


def visibility_assessment(judgments):
    diagnostics = {name: judgments.get(name, {'status': 'unknown'}) for name in VISIBILITY_FIELDS}
    causes = [cause for name, cause in VISIBILITY_FIELDS.items() if diagnostics[name].get('status') == 'yes']
    sufficient = all(answer.get('status') == 'no' for answer in diagnostics.values())
    return {'status': 'known' if causes or sufficient else 'unknown',
            'value': causes[0] if causes else 'sufficient' if sufficient else 'unknown',
            'causes': causes, 'diagnostics': diagnostics}


def visibility_allows_motion(answer):
    return answer.get('status') == 'known' and answer.get('value') in {'not_visible', 'cropped', 'occluded'}
