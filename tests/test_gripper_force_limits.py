"""A loaded closure must respect the authored driver effort and arm position hold."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from luna_mujoco.model_builder import ASSET_ROOT, GRIP_DRIVERS
from luna_mujoco.runner import Runner
from luna_mujoco.simulation import Simulation


def test_gripper_actuator_and_total_joint_force_limits_match_own_urdf():
    root = ET.parse(ASSET_ROOT / "marvin_urdf/marvin_with_grippers.urdf").getroot()
    authored = {joint.get("name"): joint for joint in root.findall("joint")}
    simulation = Simulation({"objects": []})
    try:
        for arm, name in GRIP_DRIVERS.items():
            effort = float(authored[name].find("limit").get("effort"))
            actuator = simulation._grip_act[arm]
            joint = simulation._model.joint(name).id
            assert simulation._model.actuator_trnid[actuator, 0] == joint
            assert actuator not in simulation._arm_act["A"]
            assert actuator not in simulation._arm_act["B"]
            np.testing.assert_allclose(simulation._model.actuator_forcerange[actuator], [-effort, effort])
            np.testing.assert_allclose(simulation._model.jnt_actfrcrange[joint], [-effort, effort])
    finally:
        simulation.close()


def test_frozen_loaded_closure_does_not_overpower_wrist_position_servo():
    record = json.loads((Path(__file__).parent / "fixtures/luna_closure_load.json").read_text(encoding="utf-8"))
    simulation = Simulation(record["scene"], initial_joints=record["initial_joints"])
    try:
        model, data = simulation._model, simulation._data
        target = np.asarray(record["initial_joints"]["A"])
        for _ in range(500):
            simulation._apply_motor_controls()
            mujoco.mj_step(model, data)
        simulation._desired_grip["A"] = .8
        for _ in range(3000):
            simulation._apply_motor_controls()
            mujoco.mj_step(model, data)
        # Old hardcoded 200 Nm allowed ~44 Nm driver effort and displaced
        # Joint7 by 5.97 degrees. This detects that physical regression at the
        # unchanged one-degree trajectory-start guard, not a new task tolerance.
        assert np.max(np.abs(data.qpos[simulation._arm_q["A"]] - target)) < np.deg2rad(1)
        np.testing.assert_array_equal(simulation._desired_q["A"], target)
        np.testing.assert_array_equal(data.ctrl[simulation._arm_act["A"]], target)
        assert .05 < data.qpos[simulation._grip_q["A"]] < .75
        actuator = simulation._grip_act["A"]
        assert abs(data.actuator_force[actuator]) <= model.actuator_forcerange[actuator, 1] + 1e-9
        driver_joint = model.actuator_trnid[actuator, 0]
        driver_dof = model.jnt_dofadr[driver_joint]
        assert abs(data.qfrc_actuator[driver_dof]) <= model.jnt_actfrcrange[driver_joint, 1] + 1e-9
        assert np.all(data.qfrc_applied == 0)
    finally:
        simulation.close()


@pytest.mark.parametrize("arm", ["A", "B"])
def test_start_guard_retains_one_degree_limit_and_reports_measured_difference(arm):
    current = np.zeros(7)
    current[6] = np.deg2rad(-6.)
    runner = object.__new__(Runner)
    runner.state = {"arm": arm}
    runner.simulation = SimpleNamespace(robot_snapshot=lambda: {
        "generation": 3, "time": 231.7, "arms": {arm: {"joints_rad": current.tolist()}}})
    command = {"positions_rad": [[0.] * 7, [.01] * 7], "dt": .02}
    with pytest.raises(ValueError, match="active_arm_moved_since_trajectory_start") as caught:
        runner.prepare_command(command)
    error = caught.value
    assert error.code == "active_arm_moved_since_trajectory_start"
    assert error.details["maximum_drift_deg"] == pytest.approx(6.)
    assert error.details["threshold_deg"] == 1.
    assert error.details["joint_name"] == ("Joint7_L" if arm == "A" else "Joint7_R")
    assert error.details["actual_joints_rad"] == current.tolist()
    assert error.details["trajectory_start_joints_rad"] == [0.] * 7
    assert error.details["generation"] == 3 and error.details["feedback_time"] == 231.7
