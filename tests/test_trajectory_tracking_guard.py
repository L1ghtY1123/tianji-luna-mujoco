"""A blocked joint must stop execution before a later endpoint hides the error."""

from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.simulation import Simulation, Trajectory


def moving_simulation(error_rad=0.):
    commands = {arm: Trajectory(np.array([np.zeros(7), np.full(7, .2)]), 1., 1, 0.)
                for arm in ("A", "B")}
    actual = np.full(14, .1)
    actual[6] += error_rad
    return SimpleNamespace(
        _commands=commands,
        _data=SimpleNamespace(time=.5, qpos=actual, qvel=np.zeros(14), ctrl=np.full(14, .1)),
        _arm_q={"A": np.arange(7), "B": np.arange(7, 14)},
        _arm_v={"A": np.arange(7), "B": np.arange(7, 14)},
        _arm_act={"A": np.arange(7), "B": np.arange(7, 14)},
        _desired_q={arm: np.full(7, .1) for arm in commands},
        _desired_v={arm: np.full(7, .2) for arm in commands},
    )


def test_blocked_joint_stops_both_active_arms_at_measured_position():
    # Frozen second-grasp contact first crosses the existing 0.03 rad bound at
    # 1.264 s, before the object is pushed away. A later accurate endpoint must
    # not erase that execution failure.
    simulation = moving_simulation(-.0303918)
    simulation._data.qpos[7] += .002
    actual = simulation._data.qpos.copy()
    Simulation._update_trajectories(simulation)
    for arm, command in simulation._commands.items():
        assert command.done
        assert "trajectory_tracking_error" in command.error
        np.testing.assert_array_equal(simulation._desired_q[arm], actual[simulation._arm_q[arm]])
        np.testing.assert_array_equal(simulation._desired_v[arm], np.zeros(7))
    assert simulation._commands["A"].result["max_tracking_error_rad"] == pytest.approx(.0303918)
    assert simulation._commands["B"].result["max_tracking_error_rad"] == pytest.approx(.002)
    np.testing.assert_array_equal(simulation._data.qpos, actual)


def test_normal_servo_lag_keeps_trajectory_running_and_records_peak():
    # Maximum lag in the same exact trajectory without the wall contact.
    simulation = moving_simulation(.000484088)
    Simulation._update_trajectories(simulation)
    assert not any(command.done for command in simulation._commands.values())
    assert simulation._commands["A"].max_tracking_error_rad == pytest.approx(.000484088)


def test_tracking_compares_previous_applied_reference_before_advancing_command():
    simulation = moving_simulation()
    simulation._data.time = .8
    Simulation._update_trajectories(simulation)
    assert not any(command.done for command in simulation._commands.values())
    np.testing.assert_allclose(simulation._desired_q["A"], .16)


def test_completed_trajectory_preserves_peak_tracking_evidence():
    simulation = moving_simulation(.001)
    Simulation._update_trajectories(simulation)
    simulation._data.time = 1.1
    simulation._data.qpos[:] = .2
    simulation._data.ctrl[:] = .2
    for arm in simulation._desired_q:
        simulation._desired_q[arm][:] = .2
    Simulation._update_trajectories(simulation)
    assert simulation._commands["A"].result["max_tracking_error_rad"] == pytest.approx(.001)
