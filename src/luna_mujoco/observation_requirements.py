"""Task-dependent visual evidence gaps, independent of camera count or action type."""




def assessment(judgments, expected, observation_id):
    missing = [name for name in expected if judgments.get(name, 'unknown') == 'unknown']
    failed = [name for name, value in expected.items()
              if judgments.get(name) not in (value, 'unknown', None)]
    # A definite violated requirement already disproves a conjunction. Preserve
    # other missing evidence without disguising that contradiction as uncertainty.
    needs_observation = bool(missing) and not failed
    return {'observation_id': observation_id, 'required_criteria': dict(expected),
        'judgments': dict(judgments), 'missing_evidence': missing, 'failed_criteria': failed,
        'needs_observation': needs_observation,
        'status': 'rejected' if failed else 'unknown' if missing else 'confirmed',
        'policy': 'task_criteria_and_current_evidence_not_camera_count'}
