"""Open-bin construction and contact checks use real MuJoCo geometry."""
import mujoco
import numpy as np
import pytest

from jev_mujoco.model_builder import DEFAULT_SCENE, ModelBuilder, normalize_scene


def bin_scene(**changes):
    item = {"id": "blue_bin", "kind": "bin", "color": "blue",
            "position": [.55, .15, .85], "size": [.24, .18, .12], "yaw_deg": 0,
            "wall_thickness_m": .01, "mass_kg": .4}
    return normalize_scene({"objects": [{**item, **changes}]})


def test_bin_normalization_preserves_dimensions_and_supplies_physical_defaults():
    scene = normalize_scene({"objects": [{"id": "bin", "kind": "bin", "size": [.24, .18, .12]}]})
    assert scene["objects"][0]["wall_thickness_m"] == .01
    assert scene["objects"][0]["mass_kg"] == .4
    assert scene["objects"][0]["size"] == [.24, .18, .12]


@pytest.mark.parametrize("changes", [
    {"wall_thickness_m": 0}, {"wall_thickness_m": -.001},
    {"wall_thickness_m": .09}, {"wall_thickness_m": float("nan")},
    {"size": [.24, .18, .01]}, {"mass_kg": 0}, {"mass_kg": float("inf")},
])
def test_bin_rejects_empty_cavity_or_invalid_physical_values(changes):
    with pytest.raises(ValueError, match="bin"):
        bin_scene(**changes)


@pytest.fixture(scope="module")
def builder():
    instance = ModelBuilder()
    yield instance
    instance.close()


def test_bin_has_one_free_body_and_five_nonoverlapping_collision_boxes(builder):
    model = builder.build(bin_scene())
    body = model.body("object_blue_bin")
    indices = np.flatnonzero(model.geom_bodyid == body.id)
    assert len(indices) == 5
    assert model.body_jntnum[body.id] == 1
    assert model.jnt_type[model.body_jntadr[body.id]] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body_mass[body.id] == pytest.approx(.4)
    assert {model.geom(index).name for index in indices} == {
        "object_geom_blue_bin__bottom", "object_geom_blue_bin__wall_x_minus",
        "object_geom_blue_bin__wall_x_plus", "object_geom_blue_bin__wall_y_minus",
        "object_geom_blue_bin__wall_y_plus",
    }
    assert np.all(model.geom_contype[indices] != 0)
    # A point in the middle of the cavity must lie outside every solid box.
    point = np.array([0., 0., .01])
    assert all(np.any(np.abs(point - model.geom_pos[index]) > model.geom_size[index])
               for index in indices)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    manifest = builder.render_geometries(model, data)
    parts = [item for item in manifest if item.get("object_id") == "blue_bin"]
    assert len(parts) == 5
    assert all(item["object_kind"] == "bin" and item["wall_thickness_m"] == .01 for item in parts)
    assert {item["component"] for item in parts} == {"bottom", "wall_x_minus", "wall_x_plus", "wall_y_minus", "wall_y_plus"}


def test_scene_model_contains_all_three_calibrated_sensor_cameras(builder):
    model = builder.build(bin_scene())
    assert model.ncam == 3
    assert {model.camera(index).name for index in range(model.ncam)} == {"rgbd_1", "rgbd_2", "rgbd_3"}


def test_default_scene_has_a_visible_open_blue_bin_on_the_table(builder):
    scene = normalize_scene(DEFAULT_SCENE)
    bins = [item for item in scene["objects"] if item["kind"] == "bin" and item["color"] == "blue"]
    assert len(bins) == 1
    item = bins[0]
    table = scene["table"]
    for axis in (0, 1):
        assert abs(item["position"][axis] - table["position"][axis]) + item["size"][axis] / 2 < table["size"][axis] / 2
    model = builder.build(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    parts = [geom for geom in builder.render_geometries(model, data) if geom.get("object_id") == item["id"]]
    assert len(parts) == 5
    assert all(part["rgba"][2] > part["rgba"][0] and part["rgba"][3] == 1 for part in parts)


def test_solid_box_keeps_its_original_single_geom_and_mass(builder):
    scene = normalize_scene({"objects": [{"id": "solid", "kind": "box", "size": [.24, .18, .12]}]})
    model = builder.build(scene)
    body = model.body("object_solid")
    assert len(np.flatnonzero(model.geom_bodyid == body.id)) == 1
    assert model.geom("object_geom_solid").size.tolist() == [.12, .09, .06]
    assert model.body_mass[body.id] == pytest.approx(.08)


def test_bin_falls_onto_table_and_object_falls_through_open_top_onto_floor(builder):
    scene = bin_scene(position=[.55, .15, .74])
    scene["objects"].append({"id": "ball", "kind": "sphere", "color": "red", "position": [.55, .15, .91],
                             "size": [.03, .03, .03], "yaw_deg": 0})
    model = builder.build(scene)
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    bin_center = data.xpos[model.body("object_blue_bin").id]
    ball_center = data.xpos[model.body("object_ball").id]
    assert bin_center[2] == pytest.approx(.66, abs=.003)
    assert ball_center[2] == pytest.approx(.625, abs=.003)
    assert ball_center[2] < bin_center[2] + .06 - .03
