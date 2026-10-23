from copy import deepcopy

import numpy as np
import pytest
from test_observation_motion import ObservationAlgorithms, context

from luna_mujoco.observation_motion import source_view_evidence


@pytest.mark.parametrize('cameras', [['1'], ['2'], ['3'], ['1', '3']])
def test_one_measured_camera_can_supply_geometry(cameras):
    points = np.array([[x, y, .7] for x in np.linspace(.4, .44, 8) for y in np.linspace(0, .04, 8)])
    source = {'id': 'source', 'points_world_m': points, 'coverage': {'camera_ids': cameras},
              'observation_provenance': {'observation_id': 'current'}}
    evidence = source_view_evidence({'observation_id': 'current', 'source': 'rgbd'}, source)
    assert evidence['views_ready']
    assert evidence['policy'] == 'current_measured_geometry'


async def test_observation_motion_requires_explicit_current_map(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    with pytest.raises(ValueError, match='observation_motion_requires_current_map'):
        await runner.move_to_observe(state)
    assert algorithms.map_calls == [] and algorithms.moves == []


async def test_motion_returns_capture_without_sam3_or_rebuild(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    state['scene'] = {'scene_token': 'map', 'observation_id': state['observation']['observation_id'],
                      'generation': 1, 'exclusion_state': 'observation'}
    state['scene_observation'] = deepcopy(state['observation'])
    result = await runner.move_to_observe(state)
    assert algorithms.map_calls == []
    assert result['semantic_processing'] is False
    assert 'fine_capture' in state
    assert not runner.state['facts']['scene_ready']
    evidence = runner.snapshot_for_decision(state)['perception_state']['source_views']
    assert not evidence['views_ready'] and evidence['geometry_analysis_pending']
    assert evidence['observation_id'] == state['fine_capture']['observation_id']


async def test_unexpected_change_cannot_directly_rebuild_manipulation_scene(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    state['requires_observation_pose'] = True
    with pytest.raises(ValueError, match='changed_environment_requires_observation_pose'):
        await runner.build_scene(state, 'selected')


async def test_fresh_unchanged_analysis_can_reuse_capture(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    capture = runner.simulation.capture_rgbd()
    await runner.observe(state, checkpoint='fine', capture=capture)
    await runner.observe(state, checkpoint='fine', capture=capture)
    assert state['observation_count'] == 1
    assert state['analysis_count'] == 1


async def test_geometry_uses_latest_capture_after_transfer(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    old = runner.simulation.capture_rgbd()
    new = deepcopy(old) | {'observation_id': 'after-release', 'time': old.get('time', 0) + 1}
    state.update(fine_capture=old, post_place_capture=new, latest_capture=new)
    seen = []
    async def observe(context, **kwargs):
        seen.append(kwargs['capture']['observation_id'])
        return context['observation']
    runner.observe = observe
    await runner.observe(state, checkpoint='fine', capture=state['latest_capture'], purpose='geometry')
    assert seen == ['after-release']


async def test_accepted_transfer_uses_current_visual_progress_without_old_change_reference(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    original = algorithms.observe
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        result['visual_inventory']['progress'] = {'black': {
            'source': 'openai_decisions', 'status': 'elsewhere', 'observation_id': result['observation_id']}}
        return result
    algorithms.observe = observe
    current = observe(runner.simulation.capture_rgbd())
    state['inventory_observation'] = deepcopy(current) | {'observation_id': 'accepted'}
    state['transfer_before'] = deepcopy(current) | {'observation_id': 'old-transfer'}
    state['transfer_visual_evidence'] = {'target_id': 'previous-target'}
    result = await runner.observe(state, checkpoint='fine', purpose='geometry')
    evidence = result['visual_inventory']['progress']['black']
    assert evidence['observation_id'] == result['observation_id']
    assert evidence['status'] == 'elsewhere'


async def test_scene_observation_preserves_unselected_obstacle_identity(tmp_path):
    algorithms = ObservationAlgorithms()
    runner, state = context(tmp_path, algorithms)
    obstacle = deepcopy(state['observation']['objects'][0]) | {'id': 'obstacle', 'color': 'white'}
    state['observation']['objects'].append(obstacle)
    original = algorithms.observe
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        result['objects'].append(deepcopy(obstacle))
        return result
    algorithms.observe = observe
    result = await runner.observe(state, checkpoint='fine', purpose='geometry')
    assert 'obstacle' in {obj['id'] for obj in result['objects']}
    assert result['visual_inventory']['evidence']['source'] == 'openai_decisions'


async def test_rebuild_after_observation_motion_requires_analysis_of_its_capture(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    state['observation_pose_capture_id'] = 'new-pose-frame'
    state['scene_refresh_required'] = True
    with pytest.raises(ValueError, match='observation_pose_analysis_required'):
        await runner.build_scene(state, 'selected')


async def test_rebuild_preserves_grasps_generated_from_the_same_current_capture(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    state['scene'] = {'scene_token': 'old'}
    grasps = {'poses': []}
    state.update(grasps=grasps, grasps_observation_id=state['observation']['observation_id'])
    runner.state['facts']['grasps_ready'] = True
    await runner.build_scene(state, 'selected')
    assert state['grasps'] is grasps and runner.state['facts']['grasps_ready']
    assert not runner.state['facts']['arms_evaluated']


async def test_scene_discovery_does_not_claim_target_geometry_analysis(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    state['observation_purpose'] = 'scene'
    await runner.execute('identify_target', state, {'target_id': runner.state['target_id']})
    assert not runner.state['facts']['views_ready']
    assert state['view_evidence']['geometry_analysis_pending']


async def test_first_scene_capture_does_not_require_a_previous_capture(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    seen = []
    async def observe(context, **kwargs):
        seen.append(kwargs['capture'])
        return context['observation']
    runner.observe = observe
    await runner.execute('observe_scene', state)
    assert seen == [None]


async def test_scene_analysis_after_motion_uses_new_pose_frame(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    state.update(post_place_capture={'observation_id': 'old'}, pending_transfer={},
                 latest_capture={'observation_id': 'new'}, observation_pose_capture_id='new')
    seen = []
    async def observe(context, **kwargs):
        seen.append(kwargs['capture']['observation_id'])
        return context['observation']
    runner.observe = observe
    await runner.execute('observe_scene', state)
    assert seen == ['new']


async def test_new_capture_invalidates_observation_motion_map_fact(tmp_path):
    runner, state = context(tmp_path, ObservationAlgorithms())
    await runner.build_scene(state, 'observation')
    assert runner.state['facts']['observation_map_ready']
    capture = runner.simulation.capture_rgbd()
    capture['observation_id'] = 'new-measurement'
    await runner.observe(state, capture=capture, purpose='geometry')
    assert not runner.state['facts']['observation_map_ready']


@pytest.mark.parametrize('joints_changed', [False, True])
async def test_later_capture_at_observation_pose_can_rebuild(tmp_path, joints_changed):
    runner, state = context(tmp_path, ObservationAlgorithms())
    reference = deepcopy(state['observation']) | {'observation_id': 'pose-frame', 'time': 1.}
    state.update(fine_capture={'observation_id': 'another-fine-frame'}, observation_pose_reference=reference,
                 observation_pose_capture_id='pose-frame', scene_refresh_required=True)
    state['observation']['time'] = 2.
    if joints_changed:
        state['observation']['arms']['A']['joints_rad'][0] += .1
        with pytest.raises(ValueError, match='observation_pose_analysis_required'):
            await runner.build_scene(state, 'selected')
    else:
        await runner.build_scene(state, 'selected')
        assert runner.state['facts']['scene_ready']
