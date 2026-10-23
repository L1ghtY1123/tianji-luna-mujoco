"""Physical hold regression at a measured demo release configuration."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from luna_mujoco.simulation import Simulation


@pytest.mark.parametrize("arm", ["A", "B"])
def test_position_servo_holds_object_reference_under_robot_gravity(arm):
    simulation = Simulation({"objects": []})
    try:
        # A position servo must hold this extended pose under gravity. This is a
        # dynamics regression, not a task-level placement accuracy requirement.
        pose = np.array([
            .13492050766944885, -1.1217315196990967, .36990007758140564,
            -1.1179159879684448, -.25704047083854675, .2617356479167938,
            -.5910974740982056,
        ])
        if arm == "B":
            pose *= [-1, 1, -1, 1, -1, 1, -1]
        simulation._data.qpos[simulation._arm_q[arm]] = pose
        mujoco.mj_forward(simulation._model, simulation._data)
        simulation._hold_current_pose()
        reference = np.array([.00490611214, .00147980861, .21028605861, 1.])
        initial = np.array(simulation.robot_snapshot()["arms"][arm]["tcp"]) @ reference
        for _ in range(2500):
            simulation._apply_motor_controls()
            mujoco.mj_step(simulation._model, simulation._data)
        actual = np.array(simulation.robot_snapshot()["arms"][arm]["tcp"]) @ reference
        # Detect the known 1.75 mm robot-weight sag. This is an isolated
        # controller regression, never a task-level placement tolerance.
        assert np.linalg.norm(actual[:3] - initial[:3]) < .00005
        for indices in simulation._arm_act.values():
            limits = simulation._model.actuator_forcerange[indices]
            assert np.all(simulation._data.actuator_force[indices] >= limits[:, 0])
            assert np.all(simulation._data.actuator_force[indices] <= limits[:, 1])
    finally:
        simulation.close()


def test_native_gravity_compensation_is_robot_only_and_joint_force_limited():
    simulation = Simulation()
    try:
        model, data = simulation._model, simulation._data
        for body_id in simulation._object_ids.values():
            assert model.body_gravcomp[body_id] == 0
        for arm, indices in simulation._arm_act.items():
            joints = model.actuator_trnid[indices, 0]
            assert np.all(model.jnt_actgravcomp[joints])
            assert np.all(model.jnt_actfrclimited[joints])
            limits = model.actuator_forcerange[indices]
            np.testing.assert_array_equal(model.jnt_actfrcrange[joints], limits)
            simulation._desired_q[arm] = data.qpos[simulation._arm_q[arm]] + .5
        simulation._apply_motor_controls()
        mujoco.mj_forward(model, data)
        assert np.any(np.abs(data.qfrc_gravcomp) > .1)
        for arm, indices in simulation._arm_act.items():
            limits = model.actuator_forcerange[indices]
            # Individual actuator_force excludes native gravity compensation.
            # Total joint qfrc_actuator must respect the finite simulator
            # position-servo capacity, including native gravity compensation.
            force = data.qfrc_actuator[simulation._arm_v[arm]]
            assert np.all(force >= limits[:, 0] - 1e-10)
            assert np.all(force <= limits[:, 1] + 1e-10)
            np.testing.assert_array_equal(data.ctrl[indices], simulation._desired_q[arm])
        assert np.all(data.qfrc_applied == 0)
    finally:
        simulation.close()


@pytest.mark.parametrize("arm", ["A", "B"])
def test_loaded_public_gripper_close_keeps_both_arm_position_targets(arm):
    record = json.loads((Path(__file__).parent / "fixtures/luna_closure_load.json").read_text(encoding="utf-8"))
    if arm == "B":
        for item in [record["scene"]["table"], *record["scene"]["support_tables"], *record["scene"]["objects"]]:
            item["position"][1] *= -1
            if "yaw_deg" in item:
                item["yaw_deg"] *= -1
        signs = np.array([-1, 1, -1, 1, -1, 1, -1])
        record["initial_joints"] = {"A": np.asarray(record["initial_joints"]["B"]) * signs,
                                    "B": np.asarray(record["initial_joints"]["A"]) * signs}
    simulation = Simulation(record["scene"], initial_joints=record["initial_joints"])
    simulation.start()
    try:
        with simulation._condition:
            targets = {name: values.copy() for name, values in simulation._desired_q.items()}
        result = simulation.set_gripper(arm, True, simulation.robot_snapshot()["generation"])
        with simulation._condition:
            for name, expected in targets.items():
                np.testing.assert_array_equal(simulation._desired_q[name], expected)
                np.testing.assert_array_equal(simulation._data.ctrl[simulation._arm_act[name]], expected)
                # Same physical hold bound as the existing frozen loaded-close
                # regression; never changes a runtime task or trajectory threshold.
                assert np.max(np.abs(simulation._data.qpos[simulation._arm_q[name]] - expected)) < np.deg2rad(1)
            assert .05 < result["gripper_closed_fraction"] < .95
            assert np.all(simulation._data.qfrc_applied == 0)
            contacts = [(simulation._model.body(int(simulation._model.geom_bodyid[contact.geom1])).name,
                         simulation._model.body(int(simulation._model.geom_bodyid[contact.geom2])).name)
                        for contact in simulation._data.contact]
            assert any(any("pink_cylinder" in name for name in pair) and
                       any("grip" in name.lower() for name in pair) for pair in contacts)
    finally:
        simulation.close()

