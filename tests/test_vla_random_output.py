from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def simulation_for_sampling():
    from luna_mujoco.simulation import Simulation
    simulation = object.__new__(Simulation)
    simulation._scene = {'table': {'position': [.572, -.026, .532], 'size': [.6, .8, .08]}}
    simulation._unpack_rng = np.random.default_rng(42)
    return simulation


def test_vla_default_requests_internal_random_output_and_hides_simulation_geometry():
    from luna_mujoco.robot_skills.unpack import VLAUnpack
    receipt = {'completed': True, 'is_visual_evidence': False, 'action_id': 'effect',
               'after': {'position': [1, 2, 3]}, 'before': {}, 'physical_object_id': 'sim',
               'commanded_pose': {'position': [1, 2, 3]}, 'commanded_size_m': [1, 1, 1]}
    simulation = SimpleNamespace(unpack_at_pose=Mock(return_value=receipt))
    result = VLAUnpack(simulation).execute({'floor_z_m': .572}, generation=4, target_id='input', owner_id='run')
    assert simulation.unpack_at_pose.call_args.kwargs['output_pose'] is None
    assert result == {'completed': True, 'is_visual_evidence': False, 'action_id': 'effect'}
    assert receipt['after']['position'] == [1, 2, 3]


def test_random_output_varies_inside_supported_table_and_known_right_workspace():
    from luna_mujoco.geometry import scene_object_rotation
    from luna_mujoco.workspace import WorkspaceAtlas
    simulation = simulation_for_sampling()
    outputs = [simulation._unpack_output({'id': 'input'}, None) for _ in range(20)]
    assert len({tuple(item['position']) for item in outputs}) == 20
    table = simulation._scene['table']
    for item in outputs:
        half = np.abs(scene_object_rotation(item)) @ (np.array(item['size']) / 2)
        assert np.all(np.abs(np.array(item['position'][:2]) - table['position'][:2]) + half[:2] <= np.array(table['size'][:2]) / 2)
        assert item['position'][2] == pytest.approx(.572 + half[2] + .001)
        assert WorkspaceAtlas().query('B', [item['position']])[0]['reachable']
        assert item['yaw_deg'] == 210 and item['pitch_deg'] == 90


def test_random_output_refuses_table_without_reachable_samples():
    simulation = simulation_for_sampling()
    simulation._scene['table']['position'] = [10., 10., .532]
    with pytest.raises(ValueError, match='right_workspace'):
        simulation._unpack_output({'id': 'input'}, None)


def test_explicit_simulation_output_still_requires_workspace_and_table_support():
    simulation = simulation_for_sampling()
    with pytest.raises(ValueError, match='output'):
        simulation._unpack_output({'id': 'input'}, {'position': [.8, .3, .595], 'yaw_deg': 210.})


def test_random_output_stays_in_configured_easy_right_table_region():
    from luna_mujoco.vla_configuration import vla_unpack_configuration
    region = vla_unpack_configuration()['output_region']
    assert region['center_world_xy_m'] == [.44, -.30]
    assert region['size_xy_m'] == [.04, .04]
    simulation = simulation_for_sampling()
    points = np.asarray([simulation._unpack_output({'id': 'input'}, None)['position'] for _ in range(100)])
    assert np.all(points[:, 1] < -.13)
    assert np.all(np.abs(points[:, :2] - region['center_world_xy_m']) <= np.asarray(region['size_xy_m']) / 2)


def test_default_collision_clearance_is_five_millimeters():
    from luna_mujoco.planning_settings import PlanningSettings
    assert PlanningSettings().trajectory_collision_clearance_m == .005
