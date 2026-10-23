"""Markers are camera-measured placement regions, never dynamic graspable bodies."""

import importlib
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from luna_mujoco.model_builder import ModelBuilder, normalize_scene


def detect(capture, table):
    assert importlib.util.find_spec("luna_mujoco.table_markers"), "RGBD marker detector is missing"
    return importlib.import_module("luna_mujoco.table_markers").detect_table_markers(capture, table)


def marker_scene(**changes):
    marker = {"id": "editor_name", "kind": "marker", "color": "yellow",
              "position": [.55, -.25, 1.], "size": [.16, .12, .04], "yaw_deg": 20}
    return normalize_scene({"objects": [marker | changes]})


def capture_fixture():
    transform = np.diag([1., -1., -1., 1.])
    transform[:3, 3] = [.55, 0., 1.2]
    view = {"rgb": np.full((80, 100, 3), 130, np.uint8),
            "depth_m": np.full((80, 100), .6, np.float32),
            "valid_depth": np.ones((80, 100), bool), "T_world_camera": transform,
            "intrinsics": {"fx": 200., "fy": 200., "cx": 49.5, "cy": 39.5,
                           "width": 100, "height": 80}}
    view["rgb"][25:55, 30:70] = [242, 184, 31]
    capture = {"observation_id": "pixels-1", "generation": 4, "time": 12.,
               "views": {camera: deepcopy(view) for camera in ("1", "2", "3")}}
    return capture, {"surface_height_m": .6, "uncertainty_m": .003}


def test_normalized_marker_is_flat_and_fixed_to_the_tabletop():
    item = marker_scene()["objects"][0]
    assert item["position"] == pytest.approx([.55, -.25, .6001])
    assert item["size"] == [.16, .12, .0002]


@pytest.mark.parametrize("changes", [{"color": "gray"}, {"position": [1., 0., .6]},
                                     {"size": [.9, .1, .001]}])
def test_marker_rejects_ambiguous_neutral_color_or_off_table_extent(changes):
    with pytest.raises(ValueError, match="marker"):
        marker_scene(**changes)


def test_duplicate_marker_colors_are_rejected_before_scene_rebuild():
    scene = marker_scene()
    scene["objects"].append(scene["objects"][0] | {"id": "different_editor_id"})
    with pytest.raises(ValueError, match="marker.*color"):
        normalize_scene(scene)


@pytest.fixture(scope="module")
def builder():
    instance = ModelBuilder()
    yield instance
    instance.close()


def test_marker_model_is_visible_static_and_has_no_collision_contacts(builder):
    model = builder.build(marker_scene())
    body = model.body("object_editor_name")
    geom = model.geom("object_geom_editor_name")
    assert model.body_jntnum[body.id] == 0
    assert geom.contype == 0 and geom.conaffinity == 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = data.xpos[body.id].copy()
    for _ in range(50):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.xpos[body.id], initial)
    parts = [item for item in builder.render_geometries(model, data)
             if item.get("object_id") == "editor_name"]
    assert len(parts) == 1 and parts[0]["object_kind"] == "marker"
    assert parts[0]["rgba"][0] > .9


def test_detector_uses_pixels_and_calibration_for_region_with_no_editor_identity():
    capture, table = capture_fixture()
    marker, = detect(capture, table)
    assert marker["id"] == "marker-yellow" and marker["kind"] == "marker"
    assert marker["color"] == "yellow" and marker["graspable"] is False
    assert marker["visible_centroid_world"] == pytest.approx([.55, 0., .6], abs=1e-6)
    reference = marker["placement_reference"]
    assert reference["floor_z_m"] == pytest.approx(.6)
    polygon = np.array(reference["region_polygon_world_xy_m"])
    np.testing.assert_allclose(polygon.min(axis=0), [.4915, -.0435], atol=1e-6)
    np.testing.assert_allclose(polygon.max(axis=0), [.6085, .0435], atol=1e-6)
    assert set(marker["masks_by_camera"]) == {"1", "2", "3"}
    assert marker["coverage"]["sam3_camera_ids"] == []
    points = marker["points_world_m"]
    assert len(points) >= 100 and not points.flags.writeable
    for camera, pixel, point in zip(marker["point_sources"]["camera_ids"],
                                    marker["point_sources"]["pixels_rc"], points, strict=True):
        assert capture["views"][str(camera)]["rgb"][tuple(pixel)][0] == 242
        assert point[2] == pytest.approx(.6)


def test_marker_identity_survives_camera_change_and_pixel_motion():
    capture, table = capture_fixture()
    old, = detect(capture, table)
    capture["views"] = {"2": capture["views"]["2"]}
    capture["views"]["2"]["rgb"] = np.roll(capture["views"]["2"]["rgb"], 8, axis=1)
    new, = detect(capture, table)
    assert new["id"] == old["id"]
    assert new["visible_centroid_world"][0] == pytest.approx(.574, abs=1e-6)


def test_invalid_depth_or_above_table_color_cannot_create_marker():
    capture, table = capture_fixture()
    for view in capture["views"].values():
        view["depth_m"][25:55, 30:50] = .5
        view["valid_depth"][25:55, 50:70] = False
    assert detect(capture, table) == []


def test_two_visible_regions_with_one_color_raise_identity_ambiguity():
    capture, table = capture_fixture()
    for view in capture["views"].values():
        view["rgb"][:] = 130
        view["rgb"][10:30, 5:25] = [242, 184, 31]
        view["rgb"][50:70, 75:95] = [242, 184, 31]
    with pytest.raises(ValueError, match="table_marker_color_ambiguous:yellow"):
        detect(capture, table)


def test_cross_view_bridge_cannot_merge_two_same_color_regions_from_one_camera():
    capture, table = capture_fixture()
    capture["views"]["1"]["rgb"][25:55, 48:52] = 130
    with pytest.raises(ValueError, match="table_marker_color_ambiguous:yellow"):
        detect(capture, table)


def test_occlusion_cannot_be_filled_with_synthetic_marker_surface():
    capture, table = capture_fixture()
    for view in capture["views"].values():
        view["rgb"][30:50, 45:55] = [200, 25, 25]
        view["depth_m"][30:50, 45:55] = .54
    marker, = detect(capture, table)
    assert all(not mask[30:50, 45:55].any() for mask in marker["masks_by_camera"].values())
    assert len(marker["points_world_m"]) == 1000


def test_authored_marker_is_detectable_in_actual_head_camera_rgbd(builder):
    from luna_mujoco.cameras import RgbdRenderer, clone_render_model

    model = builder.build(marker_scene(position=[.6, -.2, .6], yaw_deg=0))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = RgbdRenderer(clone_render_model(model))
    try:
        capture = renderer.capture(data, 1, {"A": {"joints_rad": [0.] * 7},
                                             "B": {"joints_rad": [0.] * 7}})
        marker, = detect(capture, {"surface_height_m": .6, "uncertainty_m": .003})
        assert marker["id"] == "marker-yellow"
        assert "3" in marker["masks_by_camera"]
        assert marker["visible_centroid_world"][:2] == pytest.approx([.6, -.2], abs=.02)
    finally:
        renderer.close()


def test_actual_vertical_bin_base_is_not_a_table_marker_at_wrist_observation_pose(builder):
    from luna_mujoco.cameras import RgbdRenderer, clone_render_model
    from luna_mujoco.model_builder import ARM_JOINTS

    scene = normalize_scene({"objects": [{"id": "bin", "kind": "bin", "color": "blue",
        "position": [.64, .15, .65], "size": [.16, .16, .1]}]})
    model = builder.build(scene)
    data = mujoco.MjData(model)
    # Measured wrist configuration from the failing live RGBD capture. It exposes
    # the vertical wall's narrow lower strip inside the tabletop height band.
    joints = {"A": [.07736, .34148, .33051, -1.67505, 1.32028, -.86759, 1.05482],
              "B": [-.10373, .39974, -.32998, -1.73362, -1.14887, -.93105, -.91936]}
    for arm, values in joints.items():
        for name, value in zip(ARM_JOINTS[arm], values, strict=True):
            data.qpos[int(model.joint(name).qposadr[0])] = value
    mujoco.mj_forward(model, data)
    renderer = RgbdRenderer(clone_render_model(model))
    try:
        capture = renderer.capture(data, 1, {arm: {"joints_rad": values} for arm, values in joints.items()})
        assert detect(capture, {"surface_height_m": .6, "uncertainty_m": .003}) == []
    finally:
        renderer.close()
