"""Position commands must remain physical and initialize at the requested pose."""
import mujoco
import numpy as np

from luna_mujoco.simulation import Simulation


def test_arm_ctrl_is_joint_position_and_dynamics_enforce_force_limits():
    sim = Simulation({"objects": []})
    try:
        before = sim._data.qpos.copy()
        sim._desired_q["A"] = before[sim._arm_q["A"]] + .01
        sim._apply_motor_controls()
        np.testing.assert_array_equal(sim._data.qpos, before)
        np.testing.assert_allclose(sim._data.ctrl[sim._arm_act["A"]], sim._desired_q["A"])
        for _ in range(100):
            sim._apply_motor_controls()
            mujoco.mj_step(sim._model, sim._data)
        assert np.linalg.norm(sim._data.qpos[sim._arm_q["A"]] - before[sim._arm_q["A"]]) > .001
        for indices in sim._arm_act.values():
            limits = sim._model.actuator_forcerange[indices]
            forces = sim._data.actuator_force[indices]
            assert np.all(forces >= limits[:, 0] - 1e-8)
            assert np.all(forces <= limits[:, 1] + 1e-8)
            assert np.all(sim._model.actuator_biastype[indices] == mujoco.mjtBias.mjBIAS_AFFINE)
    finally:
        sim.close()


def test_initial_joint_pose_is_set_before_physics_starts():
    pose = {
        "A": np.deg2rad([-30.116, 55.982, 21.322, -109.932, 14.629, -29.333, -60.589]),
        "B": np.deg2rad([30.116, 55.982, -21.322, -109.932, -14.629, -29.333, 60.589]),
    }
    sim = Simulation({"objects": []}, initial_joints=pose)
    try:
        snapshot = sim.robot_snapshot()
        for arm, joints in pose.items():
            np.testing.assert_allclose(snapshot["arms"][arm]["joints_rad"], joints)
            np.testing.assert_allclose(sim._desired_q[arm], joints)
    finally:
        sim.close()


def test_loading_new_demo_scene_reinitializes_pose_without_a_motion():
    sim = Simulation({"objects": []})
    pose = {"A": [.1] * 7, "B": [-.1] * 7}
    try:
        sim.configure({"objects": []}, initial_joints=pose)
        for arm, joints in pose.items():
            np.testing.assert_allclose(sim.robot_snapshot()["arms"][arm]["joints_rad"], joints)
            np.testing.assert_allclose(sim._desired_q[arm], joints)
        assert not sim._commands
        assert not sim._confirmed
    finally:
        sim.close()
