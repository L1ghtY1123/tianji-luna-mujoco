"""Question-local images prevent historical identity from becoming current location."""
import json
from copy import deepcopy

import pytest
from test_demo_release_result import decisions, release_inputs

from luna_mujoco.demo_transfer_verification import verify_release_result


def scoped_inputs():
    capture, binding, reference = release_inputs()
    binding['placement_zone'] = {
        'support_entity_id': binding['destination_id'],
        'intent': {'relation': 'on', 'reference_ids': [], 'relative': 'none', 'zone': 'whole'},
        'semantic_scope': 'entire_region',
        'original_requirement': {'instruction': 'place, then process, then return', 'completion_pose': 'observation'},
        'current_transfer': {'scope': 'declared_work_area'},
    }
    reference['entities'][binding['destination_id']] = {'kind': 'table', 'color': 'white'}
    return capture, binding, reference


def answers(place=.99, held=.01):
    return [{'type': 'predicate', 'name': name, 'probability': value}
            for name, value in [('place_result', place), ('workpiece_still_held', held)]]


def test_current_images_are_bound_to_each_question_and_shared_crop_is_reference_only():
    calls = []
    verify_release_result(decisions(answers(), calls), *scoped_inputs())
    payload = calls[0]
    shared = payload['input'][0]['content']
    assert len([p for p in shared if p['type'] == 'input_image']) == 1
    assert 'completion_pose' not in json.dumps(payload)
    assert 'then process' not in json.dumps(payload)
    assert 'REFERENCE identity only' in json.dumps(shared)
    for question in payload['questions']:
        instructions = question['instructions']
        assert isinstance(instructions, list)
        assert len([p for p in instructions if p['type'] == 'input_image']) == 3
        camera_order = ['1', '3', '2'] if question['name'] == 'workpiece_still_held' else ['1', '2', '3']
        assert [p['text'] for p in instructions if p.get('text', '').startswith('CURRENT RGB camera')] == [
            'CURRENT RGB camera ' + camera for camera in camera_order]
    place, held = payload['questions']
    assert 'pink cylinder' in place['instructions'][0]['text']
    assert 'on the white table' in place['instructions'][0]['text']
    assert 'current_gripper_feedback' not in json.dumps(place)
    assert 'current_gripper_feedback' in json.dumps(held)


@pytest.mark.parametrize('arm,side,camera', [('A', 'left', '2'), ('B', 'right', '1')])
def test_retention_question_binds_the_executing_arm_to_its_actual_wrist(arm, side, camera):
    capture, binding, reference = scoped_inputs()
    binding['arm'] = arm
    calls = []
    verify_release_result(decisions(answers(), calls), capture, binding, reference)
    question = next(q for q in calls[0]['questions'] if q['name'] == 'workpiece_still_held')
    assert f'{side} gripper (arm {arm}, wrist camera {camera})' in question['instructions'][0]['text']
    assert question['instructions'][-2]['text'] == 'CURRENT RGB camera ' + camera
    assert len([p for p in question['instructions'] if p['type'] == 'input_image']) == 3


@pytest.mark.parametrize('place,held,status', [(.99,.01,'at_destination'), (.1,.01,'rejected'),
    (.99,.99,'rejected'), (.5,.01,'unknown'), (.99,.5,'unknown')])
def test_rejection_is_not_missing_visual_evidence(place, held, status):
    result = verify_release_result(decisions(answers(place, held), []), *scoped_inputs())
    assert result['visual_evidence']['status'] == status
    assessment = result['observation_assessment']
    if status == 'rejected':
        assert assessment['failed_criteria']
        assert not assessment['needs_observation']
    if status == 'unknown':
        assert assessment['missing_evidence']


def test_extra_language_constraints_are_not_discarded_without_entire_region_authorization():
    capture, binding, reference = scoped_inputs()
    binding['placement_zone']['semantic_scope'] = 'candidate_selection'
    binding['placement_zone']['original_requirement']['instruction'] = 'left of the cup, handle facing the robot'
    calls = []
    verify_release_result(decisions(answers(), calls), capture, binding, reference)
    assert 'left of the cup, handle facing the robot' in json.dumps(calls[0])


def test_scope_preparation_does_not_modify_source_evidence():
    inputs = scoped_inputs()
    binding = deepcopy(inputs[1])
    verify_release_result(decisions(answers(), []), *inputs)
    assert inputs[1] == binding


async def test_skill_returns_failure_for_a_decisive_visual_rejection(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController

    class RejectedVision(DemoVision):
        def verify_release(self, *args, **kwargs):
            result = super().verify_release(*args, **kwargs)
            result['visual_evidence']['status'] = 'rejected'
            result['observation_assessment'] = {'status': 'rejected', 'needs_observation': False,
                'failed_criteria': ['place_result'], 'missing_evidence': []}
            return result

    class InspectResult(TestPolicyController):
        result = None

        async def execute(self, skill, context, arguments=None):
            result = await super().execute(skill, context, arguments)
            if skill == 'verify_place':
                self.result = result
            return result

    runner = InspectResult(DemoRobot(), RejectedVision(1), None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.result['status'] == 'failure'
    assert runner.result['reason'] == 'visual_placement_rejected'
    assert not runner.result['success']
    assert not runner.task_progress.completed_ids
