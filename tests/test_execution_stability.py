from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_trajectory_tracking_guard import moving_simulation

from luna_mujoco.simulation import Simulation


def test_endpoint_requires_continuous_position_and_velocity_stability():
    simulation = moving_simulation()
    simulation._data.qpos[:] = .2
    simulation._data.ctrl[:] = .2
    simulation._data.time = 1.1
    Simulation._update_trajectories(simulation)
    assert not simulation._commands['A'].done
    simulation._data.time = 1.15
    simulation._data.qvel[:] = .1
    Simulation._update_trajectories(simulation)
    simulation._data.time = 1.2
    simulation._data.qvel[:] = 0
    Simulation._update_trajectories(simulation)
    simulation._data.time = 1.25
    Simulation._update_trajectories(simulation)
    assert not simulation._commands['A'].done
    simulation._data.time = 1.29
    Simulation._update_trajectories(simulation)
    assert simulation._commands['A'].done
    assert simulation._commands['A'].result['stable_duration_s'] >= .08


class PhysicsCondition:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def wait(self, timeout):
        self.data.time += .05


def test_unsettled_gripper_times_out_instead_of_returning_success():
    data = SimpleNamespace(time=0., qvel=np.array([.1]), qpos=np.array([.2]))
    condition = PhysicsCondition(data)
    simulation = SimpleNamespace(
        _data=data, _condition=condition, _motion_locks={'A': condition}, _command_revision=0,
        _model=SimpleNamespace(joint=lambda name: SimpleNamespace(range=[0., .8])),
        _grip_v={'A': 0}, _grip_q={'A': 0}, _desired_grip={'A': 0.},
        _check_arm=Mock(), _check_generation=Mock(), _gripper_fraction=lambda arm: .25)
    with pytest.raises(TimeoutError, match='gripper'):
        Simulation.set_gripper(simulation, 'A', True, 1)


@pytest.mark.parametrize('position', [-.0613184682, .8002])
def test_stable_out_of_range_gripper_does_not_return_success(position):
    data = SimpleNamespace(time=0., qvel=np.array([0.]), qpos=np.array([position]))
    condition = PhysicsCondition(data)
    simulation = SimpleNamespace(
        _data=data, _condition=condition, _motion_locks={'B': condition}, _command_revision=0,
        _model=SimpleNamespace(joint=lambda name: SimpleNamespace(range=[0., .8])),
        _grip_v={'B': 0}, _grip_q={'B': 0}, _desired_grip={'B': 0.},
        _check_arm=Mock(), _check_generation=Mock(), _gripper_fraction=lambda arm: 0.)
    with pytest.raises(ValueError, match='gripper_position_outside_authored_limits'):
        Simulation.set_gripper(simulation, 'B', True, 1)
