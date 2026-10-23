"""Result checks stay at the action pose; the next grasp needs a fresh observation map."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.demo_unpack_workflow import UnpackWorkflow


def test_mock_action_does_not_publish_a_black_obstacle_from_simulator_pose():
    lifecycle = {'phase': 'table_released', 'table_release_feedback_id': 'release'}
    receipt = {'action_id': 'mock', 'commanded_pose': {
        'position': [.56, -.22, .595], 'yaw_deg': 30., 'pitch_deg': 90.}}
    runner = SimpleNamespace(current_lifecycle=lambda: lifecycle,
        region={'region_polygon_world_xy_m': [[.3, -.1], [.6, .2]], 'center_world_xy_m': [.45, .05]},
        check_generation=Mock(), state={'generation': 1, 'target_id': 'pink'}, run_id='run',
        simulation=SimpleNamespace(mock_unpack=Mock(return_value=receipt)),
        sync_lifecycle=Mock(), emit=Mock(), call_algorithm=AsyncMock(),
        algorithms=SimpleNamespace(publish_action_envelope=Mock()))
    scene = {'scene_token': 'retained'}
    context = {'vla_pose_execution': {'success': True},
               'observation': {'table': {'surface_height_m': .572}}, 'scene': scene}
    result = asyncio.run(UnpackWorkflow._mock_unpack(runner, context))
    assert result['success'] and lifecycle['phase'] == 'unpack_pending'
    assert context['scene'] is scene
    runner.call_algorithm.assert_not_awaited()


@pytest.mark.parametrize('mutation', ['missing', 'wrong_primary', 'wrong_source', 'rotated'])
def test_unpack_crop_requires_the_cached_measured_destination_table(mutation):
    from luna_mujoco.demo_lifecycle_perception import unpack_table_support
    from luna_mujoco.visual_api import VisualInventoryError

    table = {'model': {'pose': [.46, 0, .41, 1, 0, 0, 0], 'dims': [.6, .8, .3]},
             'surface_height_m': .56, 'source': 'initial_inventory_ransac_table'}
    observation = {'demo_tables': {'primary_table_id': 'main', 'source_observation_id': 'initial',
                                   'tables': {'main': table}}}
    if mutation == 'missing':
        observation.clear()
    elif mutation == 'wrong_primary':
        observation['demo_tables']['primary_table_id'] = 'other'
    elif mutation == 'wrong_source':
        table['source'] = 'simulation_truth'
    else:
        table['model']['pose'][3:] = [.707, 0, 0, .707]
    with pytest.raises(VisualInventoryError, match='measured_table_support_required'):
        unpack_table_support(observation, 'main')


def test_mock_result_must_be_verified_before_returning_to_observation_pose():
    move = AsyncMock()
    runner = SimpleNamespace(unpack_skills=UnpackWorkflow.unpack_skills, unpack_task=True,
        state={'target_id': 'pink'},
        current_lifecycle=lambda: {'phase': 'unpack_pending'}, move_to_observe=move)
    with pytest.raises(ValueError, match='unpack_verification_required_before_observation_motion'):
        asyncio.run(DemoRunner.execute(runner, 'move_to_observe', {'pending_transfer': {'verified': True}}))
    move.assert_not_awaited()


def test_between_process_return_does_not_require_a_selected_target():
    move = AsyncMock(return_value={'success': True})
    lifecycle = Mock(side_effect=AssertionError('next target is not selected until after observation'))
    runner = SimpleNamespace(unpack_skills=UnpackWorkflow.unpack_skills, unpack_task=True,
        state={'target_id': None}, current_lifecycle=lifecycle, move_to_observe=move)
    context = {'pending_transfer': {'verified': True}}
    assert asyncio.run(DemoRunner.execute(runner, 'move_to_observe', context)) == {'success': True}
    move.assert_awaited_once_with(context)
    lifecycle.assert_not_called()
    assert context['use_retained_scene_for_return'] is True


def test_grasp_planning_rejects_robot_away_from_dual_observation_pose():
    runner = SimpleNamespace(require_grasp_view=Mock(side_effect=ValueError('mapping_requires_observation_pose')))
    with pytest.raises(ValueError, match='mapping_requires_observation_pose'):
        asyncio.run(DemoRunner.plan_grasp(runner, {}, 'pink', 'A', {}))


def test_grasp_planning_rejects_map_from_a_different_observation():
    runner = SimpleNamespace(require_grasp_view=Mock())
    context = {'observation_pose_capture_id': 'fresh', 'scene': {'observation_id': 'old'},
               'observation': {'observation_id': 'fresh'}}
    with pytest.raises(ValueError, match='grasp_requires_current_observation_scene'):
        asyncio.run(DemoRunner.plan_grasp(runner, {'observation_id': 'fresh'}, 'pink', 'A', context))
