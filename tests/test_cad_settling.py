"""Real CAD contact dynamics must remain settled, not only pass an early window."""
import runpy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from luna_mujoco.cad_assets import cad_asset
from luna_mujoco.demo_scene import demo_observation_poses, demo_scene
from luna_mujoco.model_builder import ModelBuilder, normalize_scene
from luna_mujoco.simulation import Simulation


@pytest.mark.parametrize("objects", ["pink", "black_supports"])
def test_cad_objects_remain_below_existing_settle_speed_after_natural_contact(objects):
    scene = demo_scene(2)
    if objects == "black_supports":
        asset = cad_asset("black_core")
        for item, position in zip(scene["objects"][2:], ([.36, -.24, .595], [.481, -.625, .85]), strict=True):
            item.update(asset_id=asset["id"], color="black", size=asset["size_m"],
                        mass_kg=asset["demo_mass_kg"], position=position, yaw_deg=30, pitch_deg=90)
    sim = Simulation(scene, initial_joints=demo_observation_poses())
    try:
        model, data = sim._model, sim._data
        velocities = [int(model.joint("free_" + item["id"]).dofadr[0]) for item in scene["objects"] if not item.get("fixed", False)]
        maximum_speed = 0.
        contacting_objects = set()
        for _ in range(round(20 / model.opt.timestep)):
            sim._apply_motor_controls()
            mujoco.mj_step(model, data)
            if data.time >= 5:
                maximum_speed = max(maximum_speed, *(float(np.max(abs(data.qvel[v:v + 6]))) for v in velocities))
                for contact in data.contact:
                    for geom in (contact.geom1, contact.geom2):
                        name = model.body(int(model.geom_bodyid[geom])).name
                        if name.startswith("object_"):
                            contacting_objects.add(name.removeprefix("object_"))
        # Uses the unchanged public settle criterion for every step in a late
        # 15-second window. No velocity reset, sleeping or pose override.
        assert maximum_speed < .015
        assert {item['id'] for item in scene['objects'] if not item.get('fixed', False)} <= contacting_objects
        assert contacting_objects <= {item['id'] for item in scene['objects']}
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert not np.any(data.warning.number)
        for arm, target in sim._desired_q.items():
            assert np.max(abs(data.qpos[sim._arm_q[arm]] - target)) < np.deg2rad(.05)
            np.testing.assert_array_equal(data.ctrl[sim._arm_act[arm]], target)
        if objects == "black_supports":
            # Independent QA uses actual CAD vertices and the rounded cavity,
            # not distance to the commanded drop center. Natural rolling is valid.
            audit = runpy.run_path(str(Path(__file__).parents[1] / "scripts/audit_unpack_run.py"))
            snapshot = sim.snapshot()
            rows = audit["cad_objects"]({"simulation": snapshot}, {"right_bin": {"id": "blue_right_bin"}})
            assert next(row for row in rows if row["physical_id"] == "pink_cylinder_2")["inside_right_bin"]
            item = next(row for row in snapshot["objects"] if row["id"] == "pink_cylinder_1")
            transform = np.asarray(item["T_world_object"])
            vertices = audit["cad_vertices"]("black_core") @ transform[:3, :3].T + transform[:3, 3]
            table = scene["table"]
            lower = np.asarray(table["position"]) - np.asarray(table["size"]) / 2
            upper = np.asarray(table["position"]) + np.asarray(table["size"]) / 2
            assert np.all(vertices[:, :2] >= lower[:2]) and np.all(vertices[:, :2] <= upper[:2])
            assert abs(vertices[:, 2].min() - upper[2]) < .002  # Offline contact QA only.
    finally:
        sim.close()


def test_primitive_only_scene_keeps_native_collision_pipeline():
    builder = ModelBuilder()
    try:
        model = builder.build(normalize_scene({"objects": []}))
        assert not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    finally:
        builder.close()
