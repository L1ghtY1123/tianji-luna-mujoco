"""Condition probabilities are distinct from classification confidence."""
import math

import pytest

from luna_mujoco.decision_conditions import condition, parse_answers
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import choice


@pytest.mark.parametrize('probability,expected', [(1., 'yes'), (.99, 'yes'), (.8, 'yes'),
    (.79, 'unknown'), (.5, 'unknown'), (.21, 'unknown'), (.2, 'no'), (.19, 'no'), (0., 'no')])
def test_condition_has_explicit_uncertain_band(probability, expected):
    questions = [condition('placed', 'A pink workpiece is on the table.')]
    body = {'answers': [{'name': 'placed', 'type': 'predicate', 'probability': probability}]}
    judgments, raw = parse_answers(body, questions, .8)
    assert judgments == {'placed': expected}
    assert raw == body['answers'] and 'confidence' not in raw[0]


@pytest.mark.parametrize('probability', [True, None, '0.99', -.1, 1.1, math.nan, math.inf])
def test_invalid_condition_never_passes(probability):
    with pytest.raises(VisualInventoryError, match='invalid_response'):
        parse_answers({'answers': [{'name': 'placed', 'type': 'predicate', 'probability': probability}]},
                      [condition('placed', '')], .8)


def test_joint_request_keeps_identity_classification_separate_from_role_probability():
    questions = [condition('role', 'Pink workpiece visible.'), choice('original_id', '', ['p1', 'p2', 'unknown'])]
    body = {'answers': [{'type': 'predicate', 'name': 'role', 'probability': .99},
                        {'type': 'choice', 'name': 'original_id', 'choice': 'p1', 'confidence': .52}]}
    judgments, raw = parse_answers(body, questions, .8)
    assert judgments == {'role': 'yes', 'original_id': 'unknown'}
    assert raw == body['answers']


def test_choice_response_cannot_impersonate_a_current_predicate():
    with pytest.raises(VisualInventoryError, match='invalid_response'):
        parse_answers({'answers': [{'name': 'placed', 'type': 'choice', 'choice': 'yes', 'confidence': .99}]},
                      [condition('placed', '')], .8)


@pytest.mark.parametrize('mutation', ['missing', 'outside', 'false_footprint', 'changed_tcp'])
def test_numeric_release_evidence_cannot_be_replaced_with_command_success(mutation):
    from test_demo_release_result import release_inputs

    from luna_mujoco.demo_transfer_verification import release_result_payload
    capture, binding, reference = release_inputs()
    reference['demo_task']['workflow'] = 'unpack_transfer'
    endpoint = binding['evidence']['release_endpoint']
    if mutation == 'missing':
        binding['evidence'].pop('release_endpoint')
    elif mutation == 'false_footprint':
        endpoint['footprint_inside_region'] = False
    elif mutation == 'outside':
        binding['release_tcp'][0][3] = .7
        endpoint['actual_reference_world_m'][0] = .7
    else:
        endpoint['actual_reference_world_m'][0] += .03
    with pytest.raises(ValueError, match='numeric_position'):
        release_result_payload(capture, binding, reference)


@pytest.mark.parametrize('visible,held,expected', [(.99, 0., True), (.99, .2, True),
    (.99, .5, False), (.99, 1., False), (.79, 0., False), (0., 0., False)])
def test_table_presence_and_gripper_occupancy_are_independent_observations(visible, held, expected):
    import httpx
    from test_demo_release_result import release_inputs

    from luna_mujoco.demo_perception import DemoVisualInventory
    from luna_mujoco.demo_transfer_verification import verify_release_result

    capture, binding, reference = release_inputs()
    reference['demo_task']['workflow'] = 'unpack_transfer'
    rows = [{'type': 'predicate', 'name': name, 'probability': probability}
            for name, probability in [('place_result', visible), ('workpiece_still_held', held)]]
    client = DemoVisualInventory(api_key='offline', client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={'answers': rows}))))
    result = verify_release_result(client, capture, binding, reference)
    assert (result['visual_evidence']['status'] == 'at_destination') is expected
    assert result['visual_evidence']['answers'] == rows
