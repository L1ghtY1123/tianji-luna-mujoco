"""Verified inventory memory survives occlusion, without becoming fresh pixels."""
from copy import deepcopy

import pytest

from luna_mujoco.runtime.support import support_memory


def record():
    return {'id': 'green', 'status': 'observed', 'last_seen_observation_id': 'initial',
        'last_observed_position': [.5, .53, .59], 'initial_source': {
            'initial_parent_id': 'box', 'task_source_evidence': {
                'source': 'openai_decisions', 'status': 'yes', 'target_id': 'green',
                'source_id': 'box', 'source_kind': 'bin', 'observation_id': 'initial',
                'request_hash': 'verified-initial-source'}}}


def test_verified_source_is_retained_with_its_original_evidence_time():
    previous = record()
    original = deepcopy(previous)
    memory = support_memory({'id': 'green', 'visibility': 'unseen'}, previous, 0)
    assert memory['verified_initial_support']['support_entity_id'] == 'box'
    assert memory['verified_initial_support']['observation_id'] == 'initial'
    assert memory['last_seen_observation_id'] == 'initial'
    assert memory['last_observed_position'] == [.5, .53, .59]
    assert memory['visibility'] == 'unseen'
    assert memory['failed_attempt_count'] == 0 and memory['transfers'] == []
    assert previous == original


@pytest.mark.parametrize('fault', ['proposal', 'unknown', 'wrong_target', 'missing_hash'])
def test_unverified_initial_relation_is_not_presented_as_verified_memory(fault):
    previous = record()
    proof = previous['initial_source']['task_source_evidence']
    if fault == 'proposal': proof['source'] = 'openai_responses_proposal'
    elif fault == 'unknown': proof['status'] = 'unknown'
    elif fault == 'wrong_target': proof['target_id'] = 'pink'
    else: proof.pop('request_hash')
    assert support_memory({'id': 'green'}, previous, 0)['verified_initial_support'] is None


def test_manipulation_history_is_visible_instead_of_silently_reusing_initial_location():
    previous = record()
    previous['transfers'] = [{'destination_id': 'table', 'observation_id': 'after-place'}]
    memory = support_memory({'id': 'green'}, previous, 2)
    assert memory['transfers'] == previous['transfers']
    assert memory['failed_attempt_count'] == 2
    assert memory['verified_initial_support']['observation_id'] == 'initial'


def test_repeated_assessment_does_not_recursively_copy_old_memory():
    entity = {'id': 'green', 'observed_support': {'source': 'openai_decisions',
        'support_entity_id': 'box', 'observation_id': 'previous', 'request_hash': 'previous-proof',
        'support_memory': {'older_nested_memory': 'not copied'}}}
    memory = support_memory(entity, record(), 0)
    assert 'support_memory' not in memory['previous_support_assessment']


def test_support_binding_can_be_an_explicit_model_authored_recovery_attempt():
    from luna_mujoco.skills import Registry
    assert Registry().contracts['identify_target'].recovery_retry_safe


def test_production_controller_allows_only_initial_responses(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.runtime.controller import PolicyController
    from luna_mujoco.visual_api import VisualInventoryError
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.compiler.budget.reserve()
    with pytest.raises(VisualInventoryError, match='responses_budget_exhausted'):
        runner.compiler.budget.reserve()


def test_single_call_schema_cannot_request_responses_replanning():
    import json

    from test_agent_model_layer import CATALOG

    from luna_mujoco.agent.schema import InitialPlan, response_schema
    schema = response_schema(InitialPlan, CATALOG)
    assert '"replan"' not in json.dumps(schema)
    assert '#/$defs/Replan' not in json.dumps(schema)
