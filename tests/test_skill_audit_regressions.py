import pytest

from luna_mujoco.skills import Registry
from luna_mujoco.task_progress import TaskProgress


def test_visible_unfinished_required_member_cannot_be_reported_complete():
    progress = TaskProgress('move all cubes', 'collection')
    progress.observe([{'id': 'cube', 'kind': 'cube'}], 'frame')
    progress.assess_membership({'cube': 'required'})
    assert progress.terminal_status('complete') == 'needs_attention'
    assert not progress.completed_ids


def test_unknown_grasp_allows_jev_to_choose_new_result_feedback():
    facts = {'observed': True, 'lifted': True, 'result_capture_ready': True, 'verified': False,
             'grasp_completed': False, 'grasp_evidence_pending': False, 'recovery_needed': False}
    Registry().validate_selection('verify_grasp', facts)
    with pytest.raises(ValueError, match='unknown_skill'):
        Registry().validate_selection('observe_result', facts)


@pytest.mark.parametrize('evidence,expected', [({'status': 'unknown'}, 0), ({'executed': False}, 0), ({}, 3)])
def test_unknown_evidence_or_rejected_action_does_not_consume_motion_retry_budget(evidence, expected):
    progress = TaskProgress('move cube')
    for _ in range(3):
        progress.fail('cube', 'observe_geometry', 'same_failure', evidence)
    assert progress.attempts.get('cube', 0) == expected
    assert progress.failure['consecutive_same_code_count'] == 3


async def test_retired_inspection_cannot_call_visual_assistance(tmp_path):
    from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms

    from luna_mujoco.runner import Runner
    runner = Runner(SensorRobot(), VisionAlgorithms(), None, tmp_path)
    with pytest.raises(ValueError, match='unknown_skill:inspect_scene'):
        await runner.execute('inspect_scene', {})


async def test_new_observation_never_inherits_old_container_floor(tmp_path):
    from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms

    from luna_mujoco.runner import Runner

    class MissingFloor(VisionAlgorithms):
        def observe(self, *args, **kwargs):
            result = super().observe(*args, **kwargs)
            result['objects'][1].pop('placement_reference', None)
            return result

    robot = SensorRobot()
    before = VisionAlgorithms().observe(robot.capture_rgbd())
    before['objects'][1]['placement_reference']['observation_id'] = before['observation_id']
    runner = Runner(robot, MissingFloor(), None, tmp_path)
    runner.state.update(generation=1, task='grasp_place', attributes={'kind': 'cube', 'color': 'red'})
    state = {'observation': before, 'visual_plan': {'fine': 'initialize_inventory'}}
    after = await runner.observe(state, checkpoint='fine', capture=robot.capture_rgbd(), purpose='geometry')
    assert 'placement_reference' not in after['objects'][1]


def test_public_placement_reference_exposes_its_source_frame():
    from luna_mujoco.placement import public_observation
    observation = {'source': 'rgbd', 'observation_id': 'new', 'objects': [
        {'id': 'bin', 'kind': 'bin', 'placement_reference': {'observation_id': 'old', 'floor_z_m': .6}}]}
    assert public_observation(observation)['objects'][0]['placement_reference']['observation_id'] == 'old'
