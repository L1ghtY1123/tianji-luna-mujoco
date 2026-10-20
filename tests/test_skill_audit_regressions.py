import pytest

from jev_mujoco.skills import Registry
from jev_mujoco.task_progress import TaskProgress


def test_visible_unfinished_required_member_cannot_be_reported_complete():
    progress = TaskProgress('move all cubes', 'collection', route='table_box')
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


async def test_same_capture_geometry_preserves_visual_semantics_without_api_repeat(tmp_path):
    from test_rgbd_workflow import SensorRobot, VisionAlgorithms

    from jev_mujoco.runner import Runner

    class Visual:
        calls = 0
        enabled = True

        def inspect_scene(self, capture, objects, instruction, **kwargs):
            self.calls += 1
            return {'observation_id': capture['observation_id'], 'objects': [
                {'id': obj['id'], 'label': 'visible test object'} for obj in objects]}

    class Views(VisionAlgorithms):
        def observe(self, *args, source_hint=None, **kwargs):
            result = super().observe(*args, source_hint=source_hint, **kwargs)
            for camera, view in result['views'].items():
                if camera not in source_hint.get('camera_ids', ['1', '2', '3']):
                    view['instance_masks'] = {}
            return result

    runner = Runner(SensorRobot(), Views(), None, tmp_path)
    runner.state.update(generation=1, task='grasp_place', attributes={'kind': 'cube', 'color': 'red'})
    state = {'visual_plan': {'initial': 'initialize_inventory', 'fine': 'initialize_inventory'}}
    runner.visual_assistance = Visual()
    capture = runner.simulation.capture_rgbd()
    await runner.observe(state, checkpoint='initial', capture=capture, camera_ids=['3'], purpose='scene')
    await runner.execute('inspect_scene', state)
    state['visual_plan'] = {'fine': 'initialize_inventory'}
    await runner.observe(state, checkpoint='fine', capture=capture, purpose='geometry')
    assert state['visual_scene']['observation_id'] == capture['observation_id']
    assert state['observation']['objects'][0]['visual_semantics']['label'] == 'visible test object'
    assert runner.visual_assistance.calls == 1
    state['visual_plan'] = {'fine': 'initialize_inventory'}
    await runner.observe(state, checkpoint='fine', capture=runner.simulation.capture_rgbd(), purpose='geometry')
    assert 'visual_scene' not in state
    assert 'visual_semantics' not in state['observation']['objects'][0]


@pytest.mark.parametrize('task,expected', [('grasp', 'held'), ('grasp_place', 'completed')])
async def test_action_status_refreshes_labels_only_for_analyzed_current_capture(tmp_path, task, expected):
    import asyncio

    from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

    from jev_mujoco.runner import Runner

    class RecordedRunner(Runner):
        def record_annotation(self, capture, labels):
            from copy import deepcopy
            revisions.append(deepcopy(labels))

    revisions = []
    runner = RecordedRunner(SensorRobot(), VisionAlgorithms(), lambda record: Choices(task), tmp_path)
    await runner.start('把红色方块放进蓝色盒子')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert {row['status'] for row in runner.state['object_labels']['items'] if row['id'] == 'source'} == {expected}
    assert runner.algorithms.checkpoints == ['initial', 'fine'] + (['post_place'] if task == 'grasp_place' else [])


async def test_new_observation_never_inherits_old_container_floor(tmp_path):
    from test_rgbd_workflow import SensorRobot, VisionAlgorithms

    from jev_mujoco.runner import Runner

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
    from jev_mujoco.placement import public_observation
    observation = {'source': 'rgbd', 'observation_id': 'new', 'objects': [
        {'id': 'bin', 'kind': 'bin', 'placement_reference': {'observation_id': 'old', 'floor_z_m': .6}}]}
    assert public_observation(observation)['objects'][0]['placement_reference']['observation_id'] == 'old'
