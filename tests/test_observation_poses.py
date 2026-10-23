"""The saved endpoints must provide actual RGBD and robot-model clearance."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from luna_mujoco.cameras import RgbdRenderer, clone_render_model, deproject_view
from luna_mujoco.model_builder import ARM_JOINTS, ModelBuilder, normalize_scene

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCENE = {
    "objects": [
        {"id": "source", "kind": "cube", "color": "red", "position": [.43, .15, .625],
         "size": [.05, .05, .05]},
        {"id": "destination", "kind": "bin", "color": "green", "position": [.65, -.15, .65],
         "size": [.18, .16, .10]},
    ]
}


@pytest.fixture(scope="module")
def endpoint():
    path = ROOT / "resources" / "observation_poses.json"
    assert path.exists(), "validated observation endpoint is missing"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["model_validated"] is True
    builder = ModelBuilder()
    try:
        model = builder.build(normalize_scene(REFERENCE_SCENE))
        data = mujoco.MjData(model)
        for arm, names in ARM_JOINTS.items():
            values = np.asarray(record["joints_rad"][arm])
            assert values.shape == (7,) and np.isfinite(values).all()
            for name, value in zip(names, values):
                joint = model.joint(name)
                assert joint.range[0] <= value <= joint.range[1]
                data.qpos[joint.qposadr[0]] = value
        mujoco.mj_forward(model, data)
        yield model, data, record
    finally:
        builder.close()


def test_observation_endpoint_clears_reference_table_peer_and_nonignored_self(endpoint):
    model, data, _ = endpoint
    config = yaml.safe_load((ROOT / "resources/curobo/marvin_gui_dual.yml").read_text())
    kin = config["robot_cfg"]["kinematics"]
    ignored = kin["self_collision_ignore"]
    links = list(kin["collision_spheres"])
    sphere_positions = {}
    sphere_radii = {}
    for name in links:
        spheres = kin["collision_spheres"][name]
        body = data.body(name)
        local = np.array([sphere["center"] for sphere in spheres])
        sphere_positions[name] = local @ body.xmat.reshape(3, 3).T + body.xpos
        sphere_radii[name] = np.array([sphere["radius"] for sphere in spheres])
        sphere_radii[name] += kin["collision_sphere_buffer"].get(name, 0.)
    for index, body1 in enumerate(links):
        centers1, radii1 = sphere_positions[body1], sphere_radii[body1]
        if body1 not in ("Link_Base", "Link_Stand", "Base_L", "Base_R"):
            offsets = np.abs(centers1 - [.55, 0., .56]) - [.35, .50, .04]
            signed = np.linalg.norm(np.maximum(offsets, 0.), axis=1) + np.minimum(offsets.max(axis=1), 0.)
            assert np.min(signed - radii1) >= .01, body1
        for body2 in links[index + 1:]:
            if body2 in ignored.get(body1, []) or body1 in ignored.get(body2, []):
                continue
            centers2, radii2 = sphere_positions[body2], sphere_radii[body2]
            left1 = body1.startswith("gripL") or (body1.startswith(("Link", "Base", "TCP")) and body1.endswith("_L"))
            left2 = body2.startswith("gripL") or (body2.startswith(("Link", "Base", "TCP")) and body2.endswith("_L"))
            distances = np.linalg.norm(centers1[:, None] - centers2[None, :], axis=2)
            distances -= radii1[:, None] + radii2[None, :]
            if left1 == left2:
                distances -= kin["self_collision_buffer"].get(body1, 0.)
                distances -= kin["self_collision_buffer"].get(body2, 0.)
                assert distances.min() > 0., (body1, body2, distances.min())
            else:
                assert distances.min() >= .01, (body1, body2, distances.min())
    for contact in data.contact:
        assert model.geom_group[contact.geom1] != 3 and model.geom_group[contact.geom2] != 3


def test_both_wrist_rgbd_views_and_head_see_source_destination_and_tabletop(endpoint):
    model, data, record = endpoint
    renderer = RgbdRenderer(clone_render_model(model))
    arms = {arm: {"joints_rad": joints} for arm, joints in record["joints_rad"].items()}
    try:
        capture = renderer.capture(data, 1, arms)
        for camera_id, view in capture["views"].items():
            rgb = view["rgb"].astype(float)
            red = (rgb[..., 0] > 1.7 * rgb[..., 1]) & (rgb[..., 0] > 1.7 * rgb[..., 2])
            green = (rgb[..., 1] > 1.5 * rgb[..., 0]) & (rgb[..., 1] > 1.3 * rgb[..., 2])
            assert np.count_nonzero(red & view["valid_depth"]) > 100, camera_id
            assert np.count_nonzero(green & view["valid_depth"]) > 100, camera_id
            points = deproject_view(view)
            table = (np.abs(points[:, 2] - .60) < .004) & (points[:, 0] > .30) & (points[:, 0] < .80)
            table &= np.abs(points[:, 1]) < .30
            assert np.count_nonzero(table) > 3000, camera_id
    finally:
        renderer.close()
