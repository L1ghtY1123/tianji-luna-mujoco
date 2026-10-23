import copy

import pytest
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from luna_mujoco.runner import Runner
from luna_mujoco.task_progress import TaskProgress


class VisualIds(VisionAlgorithms):
    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        result['visual_inventory'] = {'membership': {result['objects'][0]['id']: 'required'},
                                      'progress': {}, 'evidence': {'source': 'openai_decisions'}}
        return result

    def associate(self, *args, **kwargs):
        raise AssertionError('Decisions IDs must not pass through the old spatial matcher')


async def test_visual_ids_bypass_old_matcher_and_retain_initial_reference(tmp_path):
    algorithms = VisualIds()
    runner = Runner(SensorRobot(), algorithms, lambda record: Choices(), tmp_path)
    runner.state.update(task='grasp_place', generation=1, instruction='搬运红色物体',
                        attributes={'kind': 'cube', 'color': 'red'})
    runner.task_progress = TaskProgress('搬运红色物体')
    context = {'visual_plan': {'initial': 'initialize_inventory'}}
    first = await runner.observe(context, purpose='scene', recovery=True)
    context['visual_plan'] = {'initial': 'associate_inventory'}
    await runner.observe(context, purpose='scene', recovery=True)
    assert algorithms.source_hints[-1]['visual_reference']['observation_id'] == first['observation_id']
    assert runner.task_progress.membership[first['objects'][0]['id']] == 'required'


def progress_with_released_object():
    progress = TaskProgress('把红色物体放到黄色区域')
    progress.records = {'red': {'id': 'red', 'status': 'selected', 'destination_id': 'yellow'}}
    progress.membership = {'red': 'required'}
    progress.current_id = 'red'
    progress.record_release('red', feedback_id='release-1')
    return progress


@pytest.mark.parametrize('status', ['unknown', 'elsewhere'])
def test_release_without_visual_destination_confirmation_is_not_complete(status):
    progress = progress_with_released_object()
    result = progress.confirm_visual_transfer('red', {'status': status, 'destination_id': 'yellow',
        'observation_id': 'after-release', 'source': 'openai_decisions'})
    assert result is False
    assert not progress.completed_ids
    assert progress.records['red']['release_feedback_id'] == 'release-1'


def test_visual_confirmation_is_destination_bound_and_idempotent():
    progress = progress_with_released_object()
    evidence = {'status': 'at_destination', 'destination_id': 'blue',
                'observation_id': 'after-release', 'source': 'openai_decisions'}
    assert not progress.confirm_visual_transfer('red', evidence)
    evidence['destination_id'] = 'yellow'
    assert progress.confirm_visual_transfer('red', evidence)
    first = copy.deepcopy(progress.records['red']['completed_stages'])
    assert progress.confirm_visual_transfer('red', evidence)
    assert progress.records['red']['completed_stages'] == first
    assert progress.completed_ids == ['red']


@pytest.mark.parametrize('visual_status,completed', [('at_destination', True), ('elsewhere', False), ('unknown', False)])
@pytest.mark.parametrize('has_remaining', [False, True])
async def test_release_builds_next_motion_map_only_after_visual_confirmation(
        tmp_path, visual_status, completed, has_remaining):
    from test_release_verification import release_case

    runner, context = release_case(tmp_path)
    runner.run_id = 'visual-release-test'
    runner.state['current_step'] = {'skill': 'verify_place'}
    context['observation']['visual_inventory'] = {'progress': {}}
    if has_remaining:
        runner.task_progress.records['next'] = {'id': 'next', 'status': 'pending'}
        runner.task_progress.membership['next'] = 'required'
    calls = []

    async def observe(context, *, checkpoint, purpose):
        calls.append(('observe', checkpoint, purpose))
        observation = {'observation_id': 'last-release-image', 'visual_inventory': {'progress': {'target': {
            'status': visual_status, 'destination_id': 'configured_region', 'source': 'openai_decisions',
            'observation_id': 'last-release-image'}}}}
        context['observation'] = observation
        return observation

    async def build(context, exclusion_state):
        calls.append(('build', exclusion_state))

    runner.observe, runner.build_scene = observe, build
    result = await runner.execute('verify_place', context, {})
    assert result['success'] is completed
    assert bool(runner.task_progress.completed_ids) is completed
    assert calls == [('observe', 'post_place', 'scene')] + (
        [('build', 'observation')] if completed and has_remaining else [])
    assert result['verification_scope'] == 'release_and_visual_destination'
    assert runner.task_progress.records['target']['release_feedback_id']


def test_one_release_cannot_complete_two_route_stages():
    progress = progress_with_released_object()
    progress.route = 'box_table_box'
    progress.records['red']['initial_source'] = {'measured_relationships': [
        {'relation': 'above_observed_bin_floor', 'object_id': 'source-box', 'visible_xy_fraction': 1.}]}
    evidence = {'status': 'at_destination', 'destination_id': 'yellow',
                'observation_id': 'after-release', 'source': 'openai_decisions'}
    assert progress.confirm_visual_transfer('red', evidence)
    evidence['observation_id'] = 'another-image'
    progress.confirm_visual_transfer('red', evidence)
    assert not progress.completed_ids
    assert len(progress.records['red']['completed_stages']) == 1
    progress.select({'id': 'red'}, destination_id='blue')
    evidence['destination_id'] = 'blue'
    assert not progress.confirm_visual_transfer('red', evidence)
    progress.record_release('red', feedback_id='release-2')
    assert progress.confirm_visual_transfer('red', evidence)
    assert progress.completed_ids == ['red']


def test_successful_segmentation_prompt_is_reused_literally():
    progress = TaskProgress('move contents')
    progress.initialize_scene_inventory({'observation_id': 'initial', 'objects': [
        {'id': 'box', 'kind': 'bin', 'color': 'blue', 'segmentation_prompt': 'cube'}]})
    assert progress.scene_queries()[0]['prompt'] == 'cube'


def test_visual_acceptance_cannot_silently_lose_an_object_in_geometry(monkeypatch):
    from test_perception import FrozenSegmenter, capture_fixture

    from luna_mujoco import perception as module
    capture, masks = capture_fixture()
    class Inventory:
        def resolve(self, capture, detections, **kwargs):
            return {'groups': [[{**d, 'entity_id': str(i)}] for i, d in enumerate(detections)],
                    'membership': {}, 'evidence': {}}
    estimate = module._estimate_object
    def estimate_one(group, *args, **kwargs):
        if group[0]['entity_id'] == '1':
            raise module.PerceptionError('instance_visible_depth_insufficient:test')
        return estimate(group, *args, **kwargs)
    monkeypatch.setattr(module, '_estimate_object', estimate_one)
    from geometry_perception_fixture import ScriptedDiscovery
    perception = module.Perception(FrozenSegmenter(masks), visual_inventory=Inventory(),
        visual_discovery=ScriptedDiscovery(('0', '1')))
    result = perception.observe(capture, source_hint={'kind': 'cube', 'camera_ids': ['3'],
        'visual_decision': {'source': 'jev', 'purpose': 'initialize_inventory'}})
    assert '1' in result['scene_entities']
    assert result['mapping_unresolved'][0]['previous_id'] == '1'





async def test_confirmed_release_map_retry_reuses_confirmation_without_new_image(tmp_path):
    from test_release_verification import release_case

    runner, context = release_case(tmp_path)
    runner.run_id = 'visual-map-retry'
    runner.state['current_step'] = {'skill': 'verify_place'}
    context['observation']['visual_inventory'] = {'progress': {}}
    runner.task_progress.records['next'] = {'id': 'next', 'status': 'pending'}
    runner.task_progress.membership['next'] = 'required'
    calls = {'observe': 0, 'build': 0}
    async def observe(context, **kwargs):
        calls['observe'] += 1
        assert calls['observe'] == 1, 'Already confirmed release must not spend another visual request'
        result = {'observation_id': 'confirmed-image', 'visual_inventory': {'progress': {'target': {
            'status': 'at_destination', 'destination_id': 'configured_region',
            'source': 'openai_decisions', 'observation_id': 'confirmed-image'}}}}
        context['observation'] = result
        return result
    async def build(context, *args, **kwargs):
        calls['build'] += 1
        if calls['build'] == 1:
            raise ValueError('map_service_error')
    runner.observe, runner.build_scene = observe, build
    with pytest.raises(ValueError, match='map_service_error'):
        await runner.execute('verify_place', context, {})
    result = await runner.execute('verify_place', context, {})
    assert result['success']
    assert calls == {'observe': 1, 'build': 2}
    assert len(runner.task_progress.records['target']['completed_stages']) == 1
