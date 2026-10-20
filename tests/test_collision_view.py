import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from jev_mujoco.simulation import Simulation


def test_collision_view_uses_configured_spheres_and_never_changes_live_grippers():
    simulation = Simulation()
    try:
        opened = simulation.collision_snapshot()
        with simulation._condition:
            for index in range(simulation._model.njnt):
                joint = simulation._model.joint(index)
                if joint.name.startswith("grip"):
                    simulation._data.qpos[int(joint.qposadr[0])] = .4
            mujoco.mj_forward(simulation._model, simulation._data)
            before_q = simulation._data.qpos.copy()
            before_ctrl = simulation._data.ctrl.copy()
        result = simulation.collision_snapshot()
        np.testing.assert_array_equal(simulation._data.qpos, before_q)
        np.testing.assert_array_equal(simulation._data.ctrl, before_ctrl)
        assert result["spheres"] == opened["spheres"]
        assert result["simulation"]["geometry_poses"] == opened["simulation"]["geometry_poses"]
        assert result["gripper_state"] == "open"
        assert result["radius_policy"] == "original_yaml_plus_link_buffer"
        assert result["link_count"] == 38
        path = Path(__file__).resolve().parents[1] / "resources/curobo/marvin_gui_dual.yml"
        provenance = json.loads((path.parents[2] / "docs/algorithm-provenance.json").read_text())
        assert result["sphere_count"] == provenance["collision_config"]["spheres"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["collision_config"]["sha256"]
        config = yaml.safe_load(path.read_text())["robot_cfg"]["kinematics"]
        expected = [(link, sphere, config["collision_sphere_buffer"].get(link, 0.))
                    for link in config["collision_link_names"]
                    for sphere in config["collision_spheres"][link]]
        for sphere, (link, original, buffer) in zip(result["spheres"], expected, strict=True):
            assert sphere["link"] == link
            assert sphere["radius_m"] == pytest.approx(original["radius"] + buffer)
        assert result["simulation"]["generation"] == simulation.snapshot()["generation"]
    finally:
        simulation.close()


def test_collision_view_tracks_arm_pose_and_matches_manifest():
    simulation = Simulation()
    try:
        before = simulation.collision_snapshot()
        with simulation._condition:
            simulation._data.qpos[simulation._arm_q["A"][0]] = .3
            mujoco.mj_forward(simulation._model, simulation._data)
        after = simulation.collision_snapshot()
        before_by_id = {item["id"]: item for item in before["spheres"]}
        changed = [item for item in after["spheres"]
                   if item["center_m"] != before_by_id[item["id"]]["center_m"]]
        assert changed and any(item["link"].startswith("gripL_") for item in changed)
        assert all(item["link"].endswith("_L") or item["link"].startswith("gripL_") for item in changed)
        assert {item["id"] for item in after["simulation"]["geometry_poses"]} == {
            item["id"] for item in simulation.manifest()["geometries"]}
    finally:
        simulation.close()
