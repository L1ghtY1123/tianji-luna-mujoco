"""Task-dependent visual evidence gaps, independent of camera count or action type."""
from .decision_conditions import condition


def sufficiency_question():
    return condition('observation_sufficient',
        'The combined CURRENT views provide enough evidence to confirm OR rule out all the visual '
        'conditions asked in this request for the active task and state. A clearly visible failed '
        'action still has sufficient evidence. Occlusion, an unseen relevant area, unresolved object '
        'identity, or conflicting views make the evidence insufficient. One useful view may suffice; '
        'three cameras alone do not prove sufficiency. Historical images and action records cannot '
        'fill missing CURRENT pixels. Do not assess kinematic reachability or hidden depth from RGB.')


def assessment(judgments, expected, observation_id):
    missing = [name for name in expected if judgments.get(name, 'unknown') == 'unknown']
    failed = [name for name, value in expected.items()
              if judgments.get(name) not in (value, 'unknown', None)]
    if judgments.get('observation_sufficient') != 'yes':
        missing.append('observation_sufficient')
    needs_observation = bool(missing)
    return {'observation_id': observation_id, 'required_criteria': dict(expected),
        'judgments': dict(judgments), 'missing_evidence': missing, 'failed_criteria': failed,
        'needs_observation': needs_observation,
        'status': 'unknown' if needs_observation else 'rejected' if failed else 'confirmed',
        'policy': 'task_criteria_and_current_evidence_not_camera_count'}
