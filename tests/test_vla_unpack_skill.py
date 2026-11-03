from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_vla_unpack_uses_configured_pose_and_returns_only_an_action_receipt():
    from luna_mujoco.robot_skills.unpack import UnpackOutputPose, VLAUnpack

    receipt = {'completed': True, 'is_visual_evidence': False, 'after': {'position': [1, 2, 3]}}
    simulation = SimpleNamespace(unpack_at_pose=Mock(return_value=receipt))
    pose = UnpackOutputPose(position=(.44, -.30, .595), yaw_deg=90.)
    skill = VLAUnpack(simulation, pose)
    result = skill.execute({'floor_z_m': .572}, generation=4, target_id='input', owner_id='run')
    simulation.unpack_at_pose.assert_called_once_with(
        {'floor_z_m': .572}, output_pose={'position': [.44, -.30, .595], 'yaw_deg': 90.},
        generation=4, target_id='input', owner_id='run', task_id=None)
    assert result == {'completed': True, 'is_visual_evidence': False}
    assert 'success' not in result


def test_explicit_output_pose_does_not_depend_on_input_pose():
    from luna_mujoco.simulation import Simulation

    simulation = object.__new__(Simulation)
    simulation._scene = {'table': {'position': [.572, -.026, .532], 'size': [1., 1., .08]}}
    first = simulation._unpack_output({'id': 'input', 'position': [.4, .1, .627], 'yaw_deg': 0}, {'position': [.44, -.30, .595], 'yaw_deg': 210.})
    second = simulation._unpack_output({'id': 'input', 'position': [.5, 0, .627], 'yaw_deg': 180}, {'position': [.44, -.30, .595], 'yaw_deg': 210.})
    assert first['position'] == second['position']
    assert first['yaw_deg'] == second['yaw_deg'] == 210.


def test_fixed_output_can_keep_the_input_xy_and_yaw():
    from luna_mujoco.simulation import Simulation

    simulation = object.__new__(Simulation)
    simulation._scene = {'table': {'position': [.572, -.026, .532], 'size': [1., 1., .08]}}
    output = simulation._unpack_output({'id': 'input', 'position': [.44, -.30, .627], 'yaw_deg': 210},
                                       {'position': [.44, -.30, .595], 'yaw_deg': 210})
    assert output['position'] == [.44, -.30, .595]


def test_configured_output_rejects_nonfinite_coordinates():
    from luna_mujoco.robot_skills.unpack import UnpackOutputPose

    with pytest.raises(ValueError):
        UnpackOutputPose(position=(float('nan'), -.2, .595), yaw_deg=90.)
