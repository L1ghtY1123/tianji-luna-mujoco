from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_observation_motion import ObservationAlgorithms, context

from jev_mujoco.perception import PerceptionError
from jev_mujoco.runner import SkillSelectionError
from jev_mujoco.task_progress import TaskProgress


class RouteAlgorithms(ObservationAlgorithms):
    def observe(self, capture, **kwargs):
        observation = super().observe(capture, **kwargs)
        observation['visual_inventory']['membership'] = {'black': 'unknown'}
        return observation


def route_context(tmp_path):
    algorithms = RouteAlgorithms()
    runner, state = context(tmp_path, algorithms)
    runner.task_progress = TaskProgress('桌面到盒子', route='table_box')
    runner.state['route'] = 'table_box'
    return runner, state, algorithms


async def test_formal_route_map_requires_reached_observation_pose(tmp_path):
    runner, state, algorithms = route_context(tmp_path)
    with pytest.raises(ValueError, match='observation_pose_required'):
        await runner.build_scene(state, 'selected')
    assert algorithms.map_calls == []


async def test_formal_route_map_rejects_motion_after_observation_capture(tmp_path):
    runner, state, algorithms = route_context(tmp_path)
    state['observation_pose_capture_id'] = state['observation']['observation_id']
    state['observation_pose_reference'] = deepcopy(state['observation'])
    runner.simulation.state['arms']['A']['joints_rad'][0] += .2
    with pytest.raises(ValueError, match='observation_pose_analysis_required'):
        await runner.build_scene(state, 'selected')
    assert algorithms.map_calls == []


async def test_verified_place_cannot_return_using_map_that_omits_released_object(tmp_path):
    runner, state, algorithms = route_context(tmp_path)
    state['scene'] = {'scene_token': 'prior-grasp-map', 'target_id': 'source',
                      'observation_id': 'prior', 'generation': 1, 'exclusion_state': 'selected'}
    state['scene_observation'] = deepcopy(state['observation']) | {'observation_id': 'prior'}
    state['pending_transfer'] = {'verified': True, 'target_id': 'source'}
    state['requires_observation_pose'] = True
    with pytest.raises(ValueError, match='observation_motion_requires_current_map'):
        await runner.move_to_observe(state)
    assert algorithms.map_calls == []
    assert algorithms.moves == []
    assert not runner.state['facts'].get('scene_ready')
    assert state['pending_transfer']['target_id'] == 'source'


async def test_task_can_finish_without_rebuilding_map_after_last_place(tmp_path):
    runner, state, _ = route_context(tmp_path)
    state['pending_transfer'] = {'verified': True, 'target_id': 'source'}
    runner.task_progress.select({'id': 'source'}, 'bin')
    runner.task_progress.complete('source', observation_id='after')
    result = await runner.execute('assess_task', state, {'judgment': 'complete'})
    assert result['judgment'] == 'complete'
    assert runner.state['status'] == 'succeeded'


async def test_geometry_after_return_can_refresh_inventory_before_selecting_target(tmp_path):
    runner, state, _ = route_context(tmp_path)
    runner.state['target_id'] = None
    state['latest_capture'] = runner.simulation.capture_rgbd()
    await runner.observe(state, checkpoint='fine', capture=state['latest_capture'], purpose='geometry')
    assert runner.task_progress.snapshot()['unresolved_membership']
    assert runner.task_progress.snapshot()['objects'] == []
    assert not runner.state['facts'].get('views_ready')


async def test_new_visual_semantics_reopens_only_pending_membership_assessment(tmp_path):
    runner, state, _ = route_context(tmp_path)
    await runner.observe(state, purpose='scene')
    runner.state['facts']['task_assessed'] = True
    source = state['observation']['objects'][0]['id']
    runner.visual_assistance = SimpleNamespace(enabled=True, inspect_scene=lambda observation, *args, **kwargs: {
        'observation_id': observation['observation_id'], 'objects': [{'id': source, 'inside_id': 'bin'}]})
    await runner.inspect_scene(state)
    assert runner.state['facts']['task_assessed'] is False
    assert runner.task_progress.snapshot()['unresolved_membership'][0]['visual_semantics']['inside_id'] == 'bin'
    runner.state['facts']['task_assessed'] = True
    await runner.inspect_scene(state)
    assert runner.state['facts']['task_assessed'] is True  # An identical answer is not new evidence.


async def test_jev_retry_grasp_marks_observation_pose_required_without_moving(tmp_path):
    runner, state, algorithms = route_context(tmp_path)
    state['observation_pose_capture_id'] = 'old-pose'
    runner.task_progress.failure = {'code': 'grasp_verification_failed', 'attempt_count': 1}
    await runner.recover_task(state, 'retry_grasp')
    assert state['requires_observation_pose']
    assert 'observation_pose_capture_id' not in state
    assert algorithms.moves == []


async def test_scene_observation_rejects_repeated_same_capture_but_accepts_new_frame(tmp_path):
    runner, state, _ = route_context(tmp_path)
    state['latest_capture'] = runner.simulation.capture_rgbd()
    state['observation_pose_capture_id'] = state['latest_capture']['observation_id']
    await runner.execute('observe_scene', state, {'visual_plan': {'initial': 'initialize_inventory'}})
    count = state['analysis_count']
    with pytest.raises(SkillSelectionError, match='visual_input_already_analyzed'):
        await runner.execute('observe_scene', state)
    assert state['analysis_count'] == count
    state['latest_capture'] = runner.simulation.capture_rgbd()
    state['observation_pose_capture_id'] = state['latest_capture']['observation_id']
    await runner.execute('observe_scene', state, {'visual_plan': {'initial': 'associate_inventory'}})
    assert state['analysis_count'] == count + 1


@pytest.mark.parametrize('visible', [False, True])
@pytest.mark.parametrize('destination_kind', ['bin', 'marker'])
async def test_placed_object_requires_current_observation_before_next_map(tmp_path, visible, destination_kind):
    runner, state, algorithms = route_context(tmp_path)
    state['observation_pose_capture_id'] = state['observation']['observation_id']
    state['observation_pose_reference'] = deepcopy(state['observation'])
    runner.task_progress.records['placed'] = {'id': 'placed', 'status': 'completed', 'destination_kind': destination_kind}
    if visible:
        state['observation']['objects'].append({'id': 'placed', 'kind': 'cube'})
        await runner.build_scene(state, 'selected')
        assert algorithms.map_calls
    else:
        with pytest.raises(ValueError, match='placed_object_not_observed'):
            await runner.build_scene(state, 'selected')
        assert algorithms.map_calls == []


@pytest.mark.parametrize('existing_scene', [False, True])
async def test_scene_camera_scope_depends_only_on_existing_scene(tmp_path, monkeypatch, existing_scene):
    runner, state, algorithms = route_context(tmp_path)
    if existing_scene:
        state['scene'] = {'scene_token': 'previous'}
    cameras = []
    original = algorithms.observe

    def capture_scope(capture, **kwargs):
        cameras.append(kwargs['source_hint']['camera_ids'])
        return original(capture, **kwargs)

    monkeypatch.setattr(algorithms, 'observe', capture_scope)
    await runner.observe(state, purpose='scene', camera_ids=['3'])
    assert cameras == ([['1', '2', '3']] if existing_scene else [['3']])


async def test_visual_identity_unknown_preserves_frozen_inventory(tmp_path, monkeypatch):
    runner, state, algorithms = route_context(tmp_path)
    await runner.observe(state, purpose='scene')
    original_ids = set(runner.task_progress.scene_entities)
    state['visual_plan'] = {'initial': 'associate_inventory'}

    def uncertain(*args, **kwargs):
        raise PerceptionError('visual_identity_unknown', details={'status': 'unknown'})

    monkeypatch.setattr(algorithms, 'observe', uncertain)
    with pytest.raises(PerceptionError, match='visual_identity_unknown'):
        await runner.observe(state, purpose='scene')
    assert set(runner.task_progress.scene_entities) == original_ids
    assert {obj['id'] for obj in state['observation']['objects']} == {'black'}


async def test_fixed_inventory_retains_unseen_collision_entity(tmp_path, monkeypatch):
    runner, state, algorithms = route_context(tmp_path)
    await runner.observe(state, purpose='scene')
    state['visual_plan'] = {'initial': 'associate_inventory'}
    original = algorithms.observe

    def empty(capture, **kwargs):
        result = original(capture, **kwargs)
        result['objects'] = []
        result['visual_inventory']['membership'] = {}
        for view in result['views'].values():
            view['instance_masks'] = {}
        return result

    monkeypatch.setattr(algorithms, 'observe', empty)
    result = await runner.observe(state, purpose='scene')
    assert result['objects'] == []
    assert result['mapping_unresolved'] == [{'previous_id': 'black', 'status': 'unseen', 'candidates': [],
                                           'reason': 'fixed_collision_entity_not_observed'}]
    assert 'black' in runner.task_progress.scene_entities
    assert result['association_evidence']['source'] == 'openai_decisions'
