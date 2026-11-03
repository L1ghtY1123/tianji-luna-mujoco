import copy
from types import SimpleNamespace

import pytest
from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms

from luna_mujoco.runner import Runner
from luna_mujoco.runtime.inventory import PolicyTaskProgress
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
    runner = Runner(SensorRobot(), algorithms, None, tmp_path)
    runner.state.update(task='grasp_place', generation=1, instruction='搬运红色物体',
                        attributes={'kind': 'cube', 'color': 'red'})
    runner.task_progress = TaskProgress('搬运红色物体')
    context = {'visual_plan': {'initial': 'initialize_inventory'}}
    first = await runner.observe(context, purpose='scene', recovery=True)
    context['visual_plan'] = {'initial': 'associate_inventory'}
    await runner.observe(context, purpose='scene', recovery=True)
    assert algorithms.source_hints[-1]['visual_reference']['observation_id'] == first['observation_id']
    assert runner.task_progress.membership[first['objects'][0]['id']] == 'required'


def progress_with_released_object(destination='yellow'):
    progress = PolicyTaskProgress('move the selected entity')
    progress.records = {'red': {'id': 'red', 'status': 'selected', 'destination_id': 'yellow'}}
    progress.membership = {'red': 'required'}
    progress.current_id = 'red'
    progress.configure_task(SimpleNamespace(destination_ref=destination, processing=[]), ['red'])
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
    first = copy.deepcopy(progress.records['red']['transfers'])
    assert progress.confirm_visual_transfer('red', evidence)
    assert progress.records['red']['transfers'] == first
    assert progress.completed_ids == ['red']


@pytest.mark.parametrize('visual_status,completed', [('at_destination', True), ('elsewhere', False), ('unknown', False)])
@pytest.mark.parametrize('has_remaining', [False, True])
async def test_registered_release_verification_never_selects_next_map(
        tmp_path, visual_status, completed, has_remaining):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController
    class InspectVerification(TestPolicyController):
        async def execute(self, skill, context, arguments=None):
            if skill != 'verify_place':
                return await super().execute(skill, context, arguments)
            maps = [entry for entry in self.algorithms.map_order if entry[0] == 'map']
            result = await super().execute(skill, context, arguments)
            assert [entry for entry in self.algorithms.map_order if entry[0] == 'map'] == maps
            assert result['success'] is completed
            assert result['verification_scope'] == 'release_and_visual_destination'
            self.verified.append(self.state['target_id'])
            return result
    count = 2 if has_remaining else 1
    algorithms = DemoVision(count, progress_status=visual_status)
    runner = InspectVerification(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.verified = []
    await runner.start('move all pink objects')
    await runner.worker
    assert len(runner.verified) == (count if completed else 1), runner.state.get('error')
    assert bool(runner.task_progress.completed_ids) is completed


def test_one_release_cannot_confirm_two_explicit_transfers():
    progress = progress_with_released_object(destination='blue')
    progress.records['red']['initial_source'] = {'measured_relationships': [
        {'relation': 'above_observed_bin_floor', 'object_id': 'source-box', 'visible_xy_fraction': 1.}]}
    evidence = {'status': 'at_destination', 'destination_id': 'yellow',
                'observation_id': 'after-release', 'source': 'openai_decisions'}
    assert progress.confirm_visual_transfer('red', evidence)
    evidence['observation_id'] = 'another-image'
    progress.confirm_visual_transfer('red', evidence)
    assert not progress.completed_ids
    assert len(progress.records['red']['transfers']) == 1
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
    manifest = ScriptedDiscovery(('0', '1')).discover(capture, instruction='fixture')
    perception = module.Perception(FrozenSegmenter(masks), visual_inventory=Inventory())
    result = perception.observe(capture, source_hint={'kind': 'cube', 'camera_ids': ['3'],
        'scene_entities': manifest['entities'], 'visual_reference': manifest['reference'],
        'visual_decision': {'source': 'responses_program', 'purpose': 'initialize_inventory'}})
    assert '1' in result['scene_entities']
    assert result['mapping_unresolved'][0]['previous_id'] == '1'


async def test_later_explicit_map_failure_cannot_replay_confirmed_release(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController
    class FailLaterMap(TestPolicyController):
        async def execute(self, skill, context, arguments=None):
            if skill == 'build_scene' and self.task_progress.completed_ids:
                raise ValueError('map_service_error')
            return await super().execute(skill, context, arguments)
    algorithms = DemoVision(2)
    runner = FailLaterMap(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.state['status'] == 'needs_attention'
    assert runner.state['error_code'] == 'map_service_error'
    assert runner.task_progress.completed_ids == ['pink-1']
    assert len(runner.task_progress.records['pink-1']['transfers']) == 1
    assert algorithms.release_calls == 1
