"""Replay the actual 2e473 grasp that saturated Joint7 during loaded closure."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from luna_mujoco.cad_assets import cad_asset
from luna_mujoco.model_builder import GRIP_DRIVERS
from luna_mujoco.simulation import Simulation, Trajectory


@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("timestep", [.001, .0005])
@pytest.mark.parametrize("cad_contact_mode", [False, True])
def test_actual_loaded_closure_stays_inside_unchanged_start_guard_throughout(arm, timestep, cad_contact_mode):
    record = json.loads((Path(__file__).parent / "fixtures/luna_unpack_close_saturation.json")
                        .read_text(encoding="utf-8"))
    if arm == "B":
        signs = np.array([-1, 1, -1, 1, -1, 1, -1])
        for item in [record["scene"]["table"], *record["scene"]["support_tables"], *record["scene"]["objects"]]:
            item["position"][1] *= -1
            if "yaw_deg" in item:
                item["yaw_deg"] *= -1
            if "quaternion" in item:
                item["quaternion"] = (np.asarray(item["quaternion"]) * [1, -1, 1, -1]).tolist()
        record["initial_joints"] = {"A": np.asarray(record["initial_joints"]["B"]) * signs,
                                    "B": np.asarray(record["initial_joints"]["A"]) * signs}
        for phase in record["phases"].values():
            phase["positions_rad"] = np.asarray(phase["positions_rad"]) * signs
    if cad_contact_mode:
        # A remote CAD prop selects the production CAD collision pipeline while
        # keeping the frozen loaded grasp and its contact geometry unchanged.
        asset = cad_asset("blue_box")
        record["scene"]["objects"].append({"id": "cad_pipeline_probe", "kind": "cad", "asset_id": "blue_box",
                                          "color": "blue", "size": asset["size_m"], "position": [1.5, 1.5, .1],
                                          "yaw_deg": 0, "quaternion": [1, 0, 0, 0]})
    sim = Simulation(record["scene"], initial_joints=record["initial_joints"])
    try:
        model, data = sim._model, sim._data
        assert bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD) == cad_contact_mode
        model.opt.timestep = timestep
        for item in record["scene"]["objects"]:
            q = int(model.joint("free_" + item["id"]).qposadr[0])
            data.qpos[q:q + 3] = item["position"]
            data.qpos[q + 3:q + 7] = item["quaternion"]
        mujoco.mj_forward(model, data)
        timing = record["timing_s"]
        def steps(duration):
            for _ in range(round(duration / model.opt.timestep)):
                sim._update_trajectories()
                sim._apply_motor_controls()
                mujoco.mj_step(model, data)
        steps(timing["open_hold"])
        for name in ("approach", "grasp"):
            phase = record["phases"][name]
            sim._commands[arm] = Trajectory(np.asarray(phase["positions_rad"]), phase["dt"], 1, data.time)
            steps(timing[name])
            sim._commands.clear()
            if name == "approach":
                steps(timing["dwell_approach"])
        targets = {name: values.copy() for name, values in sim._desired_q.items()}
        sim._desired_grip[arm] = float(model.joint(GRIP_DRIVERS[arm]).range[1])
        maximum_error = 0.
        maximum_actuation = 0.
        finger_contact = False
        duration = timing["close"] + timing["dwell_grasp"] + timing["capture_reference"]
        for _ in range(round(duration / model.opt.timestep)):
            sim._apply_motor_controls()
            mujoco.mj_step(model, data)
            for name, target in targets.items():
                maximum_error = max(maximum_error, float(np.max(np.abs(data.qpos[sim._arm_q[name]] - target))))
            maximum_actuation = max(maximum_actuation, abs(float(data.actuator_force[sim._arm_act[arm][-1]])))
            for contact in data.contact:
                names = [model.body(int(model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
                finger_contact |= any("pink_cylinder_1" in name for name in names) and any(
                    name.startswith("gripL" if arm == "A" else "gripR") for name in names)
        # Strictest existing simulator start gate (.05 degrees), checked at
        # every physics step. The 1-degree runtime guard is not widened.
        assert np.degrees(maximum_error) < .05
        assert maximum_actuation > 18, "frozen contact must exercise the former wrist saturation"
        assert finger_contact
        assert .05 < data.qpos[sim._grip_q[arm]] < .75
        for name, target in targets.items():
            np.testing.assert_array_equal(sim._desired_q[name], target)
            np.testing.assert_array_equal(data.ctrl[sim._arm_act[name]], target)
        assert np.all(data.qfrc_applied == 0)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert not np.any(data.warning.number)
    finally:
        sim.close()
