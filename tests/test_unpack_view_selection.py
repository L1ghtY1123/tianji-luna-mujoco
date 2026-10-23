"""The unpack visibility check authorizes one measured grasp view or one return."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from test_unpack_perception_integration import runner_and_context

from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.demo_scene import demo_region
from luna_mujoco.scene_update import prepare_scene_update


def test_requested_pink_release_center():
    assert demo_region()['center_world_xy_m'] == pytest.approx([.50160, .11646])


def test_pink_release_targets_front_not_offset_cloud_centroid():
    import json
    from pathlib import Path

    from luna_mujoco.demo_placement import fixed_release_goal
    from luna_mujoco.geometry import front_tcp_pose

    fixture = json.loads((Path(__file__).parent / 'fixtures/luna_loaded_release.json').read_text())
    old = fixture['goal']['release_reference']
    center = [.50160, .11646]
    polygon = [[x, y] for x, y in [(.34, -.04), (.66, -.04), (.66, .28), (.34, .28)]]
    layout = {'region_polygon_world_xy_m': polygon, 'cells': [
        {'slot_id': 1, 'center_world_xy_m': center, 'polygon_world_xy_m': polygon}]}
    destination = {'id': 'table', 'placement_reference': {'floor_z_m': old['floor_z_m']}}
    holding = fixture['holding']
    holding['arm'] = 'A'
    holding['measurement_provenance']['measured_reference_tool_m'] = [.2, -.1, .05]
    goal = fixed_release_goal(destination, holding,
        {'object_id': holding['instance_id'], 'slot_id': 1, 'layout': layout},
        old['attachment_spheres_tool_m'], old['gripper_spheres_tool_m'],
        clearance_m=.01, observation_id='current')
    np.testing.assert_allclose(front_tcp_pose(goal['T_world_tcp'], holding['arm'])[:2, 3], center)
    assert goal['release_reference']['position_constraint'] == 'finger_front_world_xy'


async def test_visible_black_updates_obstacle_then_always_returns_before_final_confirmation(tmp_path):
    runner, context, _ = runner_and_context(tmp_path)
    sequence = []
    async def move(ctx):
        sequence.append('move')
        ctx['fine_capture'] = await runner.capture_observation(ctx)
    def publish(scene, *args):
        sequence.append('publish')
        return {'scene': deepcopy(scene), 'measurement': {}}
    runner.move_to_observe = AsyncMock(side_effect=move)
    runner.algorithms.publish_local_object = publish
    result = await runner._step('verify_unpack', context)
    assert result['success']
    assert context['unpacked_grasp_target_id'] == result['output_entity_id']
    assert context['unpacked_grasp_capture'] is context['post_unpack_capture']
    assert runner.state['facts']['grasp_view_ready']
    runner.move_to_observe.assert_awaited_once()
    assert sequence == ['publish', 'move']
    assert runner.state['facts']['observation_pose_ready']


@pytest.mark.parametrize('visible_after_return', [True, False])
async def test_hidden_black_returns_once_then_checks_new_capture(tmp_path, visible_after_return):
    runner, context, _ = runner_and_context(tmp_path)
    original = runner.verify_unpacking
    checks = []

    def verify(capture, **kwargs):
        checks.append(capture['observation_id'])
        if len(checks) == 1 or not visible_after_return:
            return {'success': False, 'status': 'unknown'}
        return original(capture, **kwargs)

    async def move(ctx):
        capture = await runner.capture_observation(ctx)
        ctx['fine_capture'] = capture
        return {'observation_id': capture['observation_id']}

    runner.verify_unpacking = verify
    runner.move_to_observe = AsyncMock(side_effect=move)
    runner.move_to_observe.__name__ = 'move_to_observe'
    runner.algorithms.omit_unpack_obstacle = Mock(side_effect=lambda scene, *args: deepcopy(scene))
    runner.algorithms.omit_unpack_obstacle.__name__ = 'omit_unpack_obstacle'
    result = await runner._step('verify_unpack', context)
    assert result['success'] is visible_after_return
    assert len(checks) == len(set(checks)) == 2
    runner.move_to_observe.assert_awaited_once()
    runner.algorithms.omit_unpack_obstacle.assert_called_once()
    assert bool(runner.demo_task.get('product_entities')) is visible_after_return


def test_grasp_view_binds_current_joints_generation_capture_and_product():
    capture = {'observation_id': 'current', 'generation': 1,
               'arms': {a: {'joints_rad': np.zeros(7).tolist()} for a in ('A', 'B')}}
    context = {'unpacked_grasp_capture': capture, 'unpacked_grasp_target_id': 'black',
               'post_unpack_capture': capture}
    runner = SimpleNamespace(unpack_task=True, state={'target_id': 'black', 'generation': 1},
        current_lifecycle=lambda: {'phase': 'unpacked_confirmed'},
        require_observation_pose=Mock(),
        simulation=SimpleNamespace(robot_snapshot=lambda: deepcopy(capture)))
    DemoRunner.require_grasp_view(runner, context, capture)
    runner.require_observation_pose.assert_called_once_with(context, capture)
    moved = deepcopy(capture)
    moved['arms']['B']['joints_rad'][0] = .1
    with pytest.raises(ValueError, match='grasp_view_changed'):
        DemoRunner.require_grasp_view(runner, context, moved)
    with pytest.raises(ValueError, match='grasp_view_changed'):
        DemoRunner.require_grasp_view(runner, context, capture | {'observation_id': 'old'})


def test_unpack_return_removes_only_active_workpiece_and_reuses_background_grids():
    scene = {'scene_token': 'old', 'generation': 1, 'target_id': 'pink', 'observation_id': 'initial'}
    grids = [SimpleNamespace(name='semantic_instance::other')]
    model = {'cuboid': {'table': {'dims': [1, 1, 1]}, 'local_object::pink': {},
                       'action_envelope::pink': {}, 'local_object::other': {}},
             'voxel': {'semantic_instance::other': {}}}
    before = deepcopy(model)
    request = {'scene_token': 'old', 'generation': 1, 'target_id': 'pink',
               'operation': 'unpack_view_motion', 'exclusion_state': 'released',
               'source': 'user_authorized_unpack_visibility_fallback', 'action_id': 'mock-1'}
    identity, new_grids, updated = prepare_scene_update(scene, grids, model, request)
    assert new_grids[0] is grids[0]
    assert updated['cuboid'] == {'table': before['cuboid']['table'], 'local_object::other': {}}
    assert identity['layer_evidence']['background_remapped'] is False
    assert model == before
    with pytest.raises(ValueError, match='unpack_view_motion_binding_invalid'):
        prepare_scene_update(scene, grids, model, request | {'target_id': 'other'})


def test_unpack_omission_passes_actual_geometry_service_boundary():
    from test_scene_update import mapped

    from luna_mujoco.geometry_service import run_operation

    runtime, _ = mapped()
    background = runtime.geometry_grids[1]
    result = run_operation(runtime, '/perception/update', {
        'scene_token': 'old', 'generation': 1, 'target_id': 'one',
        'operation': 'unpack_view_motion', 'exclusion_state': 'released',
        'source': 'user_authorized_unpack_visibility_fallback', 'action_id': 'mock-1'})
    assert runtime.geometry_grids == [background]
    assert 'table' in runtime.geometry_model['cuboid']
    assert result['scene_token'] != 'old'


def test_post_vla_proposals_remove_robot_height_and_merge_descriptions_not_ids(monkeypatch):
    from luna_mujoco import demo_perception

    monkeypatch.setattr(demo_perception, '_remove_own_robot_surface',
        lambda groups, *args: (groups, {}))
    perception = demo_perception.DemoPerception()
    perception.demo_tables = {'primary_table_id': 'table', 'tables': {'table': {
        'surface_height_m': .56, 'model': {'pose': [.5, 0, .5], 'dims': [.6, .8, .1]}}}}

    def proposal(offset, z, prompt):
        mask = np.zeros((10, 10), bool)
        mask[:6, offset:offset + 5] = True
        pixels = np.column_stack(np.nonzero(mask))
        return {'camera_id': '3', 'kind': 'object', 'requested_color': 'black', 'prompt': prompt,
            'mask': mask, 'silhouette_mask': mask, 'pixels_rc': pixels,
            'pixels': np.zeros((len(pixels), 3)), 'points': np.tile([.5, 0, z], (len(pixels), 1))}

    raw = [proposal(0, .59, 'black object'), proposal(0, .59, 'unpacked object'),
           proposal(5, .59, 'black object'), proposal(5, .85, 'robot false positive')]
    result = perception._prepare_identity_detections(raw, {}, {},
        {'demo_task': {'item_lifecycle': {'black': {'phase': 'unpacked_confirmed'}}}})
    assert len(result) == 2
    assert result[0]['prompt_proposals'] == ['black object', 'unpacked object']
    assert all('entity_id' not in row for row in result)
    assert all(row['identity_proposal_filter']['identity_assignment'] is False for row in result)
