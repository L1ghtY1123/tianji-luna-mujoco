"""Task-dependent visual evidence gaps, independent of camera count or action type."""
from .decision_conditions import condition


def sufficiency_question(criteria=()):
    scope = ', '.join(criteria) if criteria else 'the other named outcome questions in this request'
    return condition('observation_sufficient',
        f'The combined CURRENT views provide enough evidence to confirm OR rule out ONLY: {scope}. '
        'Do not assess this sufficiency question itself or add conditions. Use exactly the scope and '
        'evidence rules of those named questions. A clearly visible failed action still has sufficient '
        'evidence. Occlusion matters only when it prevents judging one of those conditions across ALL '
        'CURRENT views; one useful view is enough. Do not require all cameras to see the object, a '
        'complete object silhouette, hidden surfaces, or identities that the outcome questions exclude. '
        'When a named question asks for a change, compare its labeled BEFORE reference with CURRENT; '
        'this is valid comparison evidence, although BEFORE alone cannot prove the current outcome. '
        'Action records alone cannot prove visual success. Do not assess kinematic reachability or hidden '
        'depth from RGB. Request another observation only for a missing or ambiguous required visual fact.')


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
