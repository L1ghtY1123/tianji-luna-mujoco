"""Real model and scene-contract checks; simulator truth stays in offline QA."""

import mujoco
import numpy as np
import pytest

from luna_mujoco.model_builder import ModelBuilder, normalize_scene


@pytest.fixture
def observation_calibration(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibrate_demo_observation_poses

    return calibrate_demo_observation_poses


def test_normalization_retains_second_static_support():
    scene = normalize_scene({
        "objects": [],
        "support_tables": [{"id": "source_table", "position": [.39, .66, .56],
                            "size": [.34, .32, .08]}],
    })
    assert scene["support_tables"] == [
        {"id": "source_table", "position": [.39, .66, .56], "size": [.34, .32, .08]},
    ]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_demo_scene_count_source_fit_and_unmarked_destination(count):
    from luna_mujoco.demo_scene import demo_region, demo_scene

    scene = demo_scene(count)
    bins = [item for item in scene["objects"] if item.get("asset_id") == "blue_box"]
    objects = [item for item in scene["objects"] if item["color"] == "pink"]
    bin_item, right_bin = bins
    assert bin_item["kind"] == "cad"
    assert bin_item["color"] == "blue"
    assert len(objects) == count
    assert all(item["kind"] == "cad" and item["asset_id"] == "pink_wrapped" for item in objects)
    assert all(item["size"] == [.1110853500366211, .06278736877441407, .11] for item in objects)
    expected_positions = {
        1: [[.315, .525]],
        2: [[.315, .525], [.315, .685]],
        3: [[.355, .50], [.315, .57], [.275, .64]],
        4: [[.245, .49], [.365, .49], [.245, .61], [.365, .61]],
    }
    np.testing.assert_allclose([item["position"][:2] for item in objects], np.array(expected_positions[count]) + [.166, -.032])
    main = scene["table"]
    np.testing.assert_allclose(main["position"], [.572, -.026, .532])
    np.testing.assert_allclose(bin_item["position"], [.481, .573, .571])
    np.testing.assert_allclose(right_bin["position"], [.481, -.625, .571])
    assert main["size"][:2] == [.60, .80]
    assert main["position"][2] + main["size"][2] / 2 == pytest.approx(.572)
    assert scene["support_tables"] == []
    assert bin_item["size"] == pytest.approx([.410, .310, .145])
    assert right_bin["size"] == bin_item["size"]
    assert bin_item["yaw_deg"] == 90 and right_bin["yaw_deg"] == -90
    assert all(item['fixed'] for item in bins)
    for item in objects:
        offset = np.abs(np.array(item["position"][:2]) - bin_item["position"][:2])
        # Conservative inner rectangle at the bottom, accounting for bin yaw.
        assert np.all(offset + np.array(item["size"][:2]) / 2 < [.130, .185])
    region = demo_region()
    assert region["center_world_xy_m"] == pytest.approx([.50160, .11646])
    assert region["size_m"] == .32
    np.testing.assert_allclose(region["region_polygon_world_xy_m"],
                               [[.34160, -.04354], [.66160, -.04354], [.66160, .27646], [.34160, .27646]])
    assert all(item["kind"] != "marker" for item in scene["objects"])


@pytest.mark.parametrize("count", [0, 5, True, 1.5])
def test_demo_rejects_unsupported_counts(count):
    from luna_mujoco.demo_scene import demo_scene

    with pytest.raises(ValueError, match="object_count"):
        demo_scene(count)


def test_default_keeps_nearer_original_pink_part():
    from luna_mujoco.demo_scene import demo_scene

    objects = [item for item in demo_scene()["objects"] if item["color"] == "pink"]
    assert len(objects) == 1
    positions = np.array([item["position"][:2] for item in objects])
    np.testing.assert_allclose(positions, [[.481, .493]])
    # Even the full AABBs leave at least 7 cm within the conservative inner rectangle.
    for item in objects:
        offset = abs(np.asarray(item["position"][:2]) - [.481, .573])
        assert np.min(np.asarray([.130, .185]) - offset - np.asarray(item["size"][:2])/2) > .07


def test_source_row_and_four_object_grid_follow_explicit_configuration():
    from copy import deepcopy

    from luna_mujoco.demo_scene import demo_scene, load_demo_config

    config = load_demo_config()
    config["source_objects"]["row_start_offset_xy_m"] = [-.03, -.09]
    config["source_objects"]["row_step_xy_m"] = [.02, .06]
    config["source_objects"]["two_object_offsets_xy_m"] = [[-.02, -.08], [.02, .06]]
    config["source_objects"]["four_object_grid"] = {
        "x_offsets_from_bin_center_m": [-.06, .04],
        "y_offsets_from_bin_center_m": [-.10, .02],
    }
    before = deepcopy(config)
    rows = [[item["position"][:2] for item in demo_scene(count, config)["objects"] if item["color"] == "pink"]
            for count in (1, 2, 3)]
    grid = [item["position"][:2] for item in demo_scene(4, config)["objects"] if item["color"] == "pink"]
    for row, expected in zip(rows, ([[.315, .525]], [[.295, .525], [.335, .665]],
                                    [[.285, .515], [.305, .575], [.325, .635]]), strict=True):
        np.testing.assert_allclose(row, np.array(expected) + [.166, -.032])
    np.testing.assert_allclose(grid, np.array([[.255, .505], [.355, .505], [.255, .625], [.355, .625]]) + [.166, -.032])
    assert config == before


def test_support_table_and_cad_compile_without_free_support_joint():
    from luna_mujoco.demo_scene import demo_scene

    builder = ModelBuilder()
    try:
        model = builder.build(demo_scene(1))
        assert model.geom("table").bodyid == 0
        assert model.geom("table").contype != 0
        assert model.body("object_blue_source_bin").jntnum == 0
        visual = model.geom("object_visual_pink_cylinder_1")
        assert visual.type == mujoco.mjtGeom.mjGEOM_MESH
        assert visual.group == 0 and visual.contype == 0
        collision = model.geom("object_geom_pink_cylinder_1__cad_0")
        assert collision.group == 3 and collision.contype != 0
        mesh_id = int(visual.dataid[0])
        start = model.mesh_vertadr[mesh_id]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
        rotation = model.mesh_quat[mesh_id]
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, rotation)
        original_vertices = vertices @ matrix.reshape(3, 3).T + model.mesh_pos[mesh_id]
        np.testing.assert_allclose(np.ptp(original_vertices, axis=0),
                                   [.1110853500366211, .06278736877441406, .11], atol=1e-7)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert data.body("TCP_Link_L").xpos[1] > 0
        assert data.body("TCP_Link_R").xpos[1] < 0
    finally:
        builder.close()


def test_demo_simulation_settles_in_fixed_bin_and_reset_retains_it():
    from luna_mujoco.demo_scene import demo_scene
    from luna_mujoco.simulation import Simulation

    simulation = Simulation(scene=demo_scene(4))
    try:
        simulation.start()
        settled = simulation.settle(timeout_s=5)
        assert settled["settled"]
        assert settled["support_tables"] == []
        for item in settled["objects"]:
            if item["color"] != "pink":
                continue
            assert item["position"][2] - item["size"][2] / 2 == pytest.approx(.5035, abs=.003)
            assert item["asset_id"] == "pink_wrapped"
        baseline = simulation.confirm()
        simulation.configure({"objects": []})
        restored = simulation.reset()
        assert restored["support_tables"] == baseline["support_tables"]
        assert len(restored["objects"]) == 6
    finally:
        simulation.close()


def test_calibrated_observation_resource_maps_left_camera_and_has_joint_limits():
    import json

    from luna_mujoco.demo_scene import ROOT, demo_observation_poses, load_demo_config

    config = load_demo_config()
    record = json.loads((ROOT / config["observation_poses_resource"]).read_text())
    poses = demo_observation_poses()
    assert set(poses) == {"A", "B"}
    assert all(len(joints) == 7 for joints in poses.values())
    assert record["validation"]["physical_left_arm"] == "A"
    assert record["validation"]["source_camera_id"] == "2"
    np.testing.assert_allclose(np.degrees(poses["A"]),
                               [-30.116, 55.982, 21.322, -109.932, 14.629, -29.333, -60.589], atol=1e-10)
    np.testing.assert_allclose(np.degrees(poses["B"]),
                               [30.116, 55.982, -21.322, -109.932, -14.629, -29.333, 60.589], atol=1e-10)
    assert record["validation"]["mirror_fk"]["maximum_position_error_m"] < .0001
    assert record["validation"]["environment_truth_in_runtime"] is False


def test_observation_validation_describes_current_scene_without_full_bin_gate():
    import hashlib
    import json

    from luna_mujoco.demo_scene import ROOT, load_demo_config

    config = load_demo_config()
    record = json.loads((ROOT / config["observation_poses_resource"]).read_text())
    evidence = record["validation"]
    assert {k: v for k, v in evidence["scene_config"].items() if k != "destination"} == {
        k: v for k, v in config.items() if k != "destination"}
    for name, digest in record["source_sha256"].items():
        if name == "resources/demo-config.json":
            continue  # Destination intent changed; physical calibration geometry above is unchanged.
        assert (ROOT / name).is_file()
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert evidence["source_bin_coverage"]["full_envelope_inside_image"] == (
        evidence["source_bin_coverage"]["minimum_edge_margin_px"] >= 0)
    assert evidence["source_bin_coverage"]["acceptance_gate"] is False
    assert evidence["all_source_objects_visible"] is True
    assert evidence["endpoint_collision_clear"] is True
    assert record["model_validated"] is True
    mirror = evidence["mirror_fk"]
    assert mirror["scope"] == "robot_link_mirror_independent_of_scene_layout"
    assert mirror["robot_urdf_sha256"] == record["source_sha256"][
        "assets/marvin_urdf/marvin_with_grippers.urdf"]


def test_observation_resource_covers_each_configured_object_count():
    import json

    from luna_mujoco.demo_scene import ROOT, demo_scene, load_demo_config

    config = load_demo_config()
    record = json.loads((ROOT / config["observation_poses_resource"]).read_text())
    assert set(record["validation_by_object_count"]) == {"1", "2", "3", "4"}
    for count, evidence in record["validation_by_object_count"].items():
        assert evidence["source_object_count"] == int(count)
        assert len(evidence["source_pixels_by_object"]) == int(count)
        assert min(evidence["source_pixels_by_object"].values()) >= 100
        assert evidence["mujoco_robot_scene_contacts"] == 0
        assert evidence["mujoco_source_object_contacts"] == 0
        assert evidence["all_source_objects_visible"] is True
        expected = {item["id"]: item["position"] for item in demo_scene(int(count))["objects"]
                    if item["color"] == "pink"}
        assert set(evidence["source_positions_world_m"]) == set(expected)
        for object_id, position in expected.items():
            np.testing.assert_allclose(evidence["source_positions_world_m"][object_id], position)


def test_current_bin_frustum_clipping_is_explicitly_diagnostic(observation_calibration):
    from luna_mujoco.demo_scene import demo_observation_poses, demo_scene

    builder = ModelBuilder()
    try:
        model = builder.build(demo_scene())
        data = mujoco.MjData(model)
        observation_calibration.set_joints(model, data, demo_observation_poses())
        coverage = observation_calibration.source_bin_coverage(model, data)
        assert coverage["full_envelope_inside_image"] == (coverage["minimum_edge_margin_px"] >= 0)
        assert coverage["acceptance_gate"] is False
        assert "required_edge_margin_px" not in coverage
    finally:
        builder.close()


def test_user_extrinsics_are_copied_and_converted_once():
    import hashlib
    import json

    from luna_mujoco.cameras import load_calibration
    from luna_mujoco.demo_scene import ROOT

    calibration = load_calibration()
    for camera_id in ("1", "2", "3"):
        camera = calibration["cameras"][camera_id]
        provenance = camera["extrinsic_provenance"]
        path = ROOT / provenance["source_path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance["source_sha256"]
        source = json.loads(raw)
        expected = np.asarray(source["extrinsic"], dtype=float)
        expected[:3, 3] *= .001
        np.testing.assert_allclose(camera["T_parent_rgb_m"], expected, atol=1e-12)
    head = calibration["cameras"]["3"]
    assert head["parent_body"] == "world"
    np.testing.assert_allclose(np.asarray(head["T_parent_rgb_m"])[:3, 3],
                               [.12736032187563909, -.0034876335486242414, 1.199981040227405])
