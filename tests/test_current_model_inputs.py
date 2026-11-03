"""Actual question-scoped imagery and current graph/release boundaries."""
import json

from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms
from test_demo_release_result import decisions, fused_answers, release_inputs

from luna_mujoco.demo_transfer_verification import release_result_payload, verify_release_result
from luna_mujoco.release_decision_input import scoped_release_payload
from luna_mujoco.runner import Runner


def test_release_without_reference_crop_does_not_claim_a_reference_image():
    capture, binding, reference = release_inputs()
    target = reference['entities'][binding['target_id']]
    target.pop('bbox', None)
    target.pop('reference_camera_id', None)
    target.update(color='black', kind='object', description='black workpiece')
    payload = release_result_payload(capture, binding, reference)
    payload = scoped_release_payload(payload, binding['destination_id'])
    text = json.dumps(payload, ensure_ascii=False)
    assert 'shown in REFERENCE' not in text
    assert 'reference crop predates' not in text
    assert 'black object' in text
    for question in payload['questions']:
        assert sum(part['type'] == 'input_image' for part in question['instructions']) == 3


def test_release_of_last_object_does_not_insert_global_inventory_check():
    calls = []
    capture, binding, reference = release_inputs(final=True)
    result = verify_release_result(decisions(fused_answers(), calls), capture, binding, reference)
    assert len(calls) == 1
    assert {q['name'] for q in calls[0]['questions']} == {'place_result', 'workpiece_still_held'}
    assert result['demo_inventory_check']['status'] == 'not_requested'


def test_model_call_image_count_includes_question_scoped_images(tmp_path):
    runner = Runner(SensorRobot(), VisionAlgorithms(), None, tmp_path)
    runner.run_id = 'current'
    runner.state['status'] = 'running'
    request = {'input': [], 'questions': [{'name': 'place_result', 'instructions': [
        {'type': 'input_image', 'image_url': str(i)} for i in range(3)]}]}
    runner.record_visual_decision({'request': request, 'response': {'answers': []}},
                                 experiment=None, run_id='current')
    assert runner.events[-1]['data']['image_count'] == 3


def test_retired_transport_skills_are_not_exposed_by_the_catalog():
    from luna_mujoco.skills import Registry
    registry = Registry()
    assert not {'plan_transport', 'execute_transport', 'drop', 'verify_drop'} & registry.contracts.keys()
    assert {'resolve_placement', 'plan_place', 'execute_place', 'verify_place'} <= registry.contracts.keys()
