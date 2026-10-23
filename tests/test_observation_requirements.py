from luna_mujoco.observation_requirements import assessment, sufficiency_question


def test_visible_failure_is_distinct_from_missing_evidence():
    assert 'confirm OR rule out' in sufficiency_question()['instructions']
    result = assessment({'on_support': 'no', 'observation_sufficient': 'yes'}, {'on_support': 'yes'}, 'frame')
    assert result['status'] == 'rejected' and not result['needs_observation']
    assert result['failed_criteria'] == ['on_support']


def test_missing_any_task_criterion_requires_supplement_even_with_three_cameras():
    result = assessment({'on_support': 'yes', 'unwrapped': 'unknown', 'observation_sufficient': 'yes'},
                        {'on_support': 'yes', 'unwrapped': 'yes'}, 'frame')
    assert result['needs_observation'] and result['status'] == 'unknown'
    assert result['missing_evidence'] == ['unwrapped']


def test_success_requires_sufficient_evidence_not_only_positive_result():
    result = assessment({'on_support': 'yes', 'observation_sufficient': 'unknown'}, {'on_support': 'yes'}, 'frame')
    assert result['needs_observation'] and result['status'] == 'unknown'
    result = assessment({'on_support': 'yes', 'observation_sufficient': 'yes'}, {'on_support': 'yes'}, 'frame')
    assert not result['needs_observation'] and result['status'] == 'confirmed'
