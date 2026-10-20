"""Visible measurement contracts independent of any inferred object geometry."""


import numpy as np
import pytest
from geometry_perception_fixture import GeometryPerception
from test_perception import FrozenSegmenter, bin_cloud, capture_fixture, perception_module


def test_visible_centroid_and_sources_preserve_measured_top_without_cube_completion():
    capture, masks = capture_fixture()
    observed = GeometryPerception(FrozenSegmenter(masks)).observe(capture, checkpoint="initial")
    for instance in observed["objects"]:
        assert not {"size", "quaternion", "T_world_object", "points_object_m", "cavity"} & instance.keys()
        np.testing.assert_allclose(instance["visible_centroid_world"], instance["points_world_m"].mean(axis=0), atol=1e-6)
        assert instance["position"] == instance["visible_centroid_world"]
        assert instance["position"][2] == pytest.approx(.7)
        sources = instance["point_sources"]
        assert len(sources["camera_ids"]) == len(instance["points_world_m"])
        assert sources["pixels_rc"].shape == (len(instance["points_world_m"]), 2)
        assert instance["observation_provenance"]["checkpoint"] == "initial"
        assert "3" in instance["coverage"]["camera_ids"]


def test_bin_reference_uses_visible_floor_without_requiring_four_walls():
    module = perception_module()
    cloud = bin_cloud()
    points = np.vstack((cloud[:121], cloud[121:][cloud[121:, 1] < -.065]))
    group = [{"entity_id": "fixture-entity", "kind": "bin", "camera_id": "3", "points": points,
              "pixels": np.full((len(points), 3), [20, 20, 220], np.uint8), "score": .9}]
    instance = module._estimate_object(group, {"surface_height_m": .6, "uncertainty_m": .003})
    reference = instance["placement_reference"]
    assert reference["floor_z_m"] == pytest.approx(.61)
    np.testing.assert_allclose(reference["position_world_m"], [0, 0, .61], atol=1e-6)
    assert reference["measurement_sources"]["point_count"] >= 30
    higher_indices = reference["measurement_sources"]["higher_context_point_indices"]
    assert len(higher_indices) >= 10
    assert np.all(instance["points_world_m"][higher_indices, 2] > reference["floor_z_m"])
    assert not {"cavity", "size", "T_world_object"} & instance.keys()


def test_bin_without_horizontal_depth_is_detected_without_fabricated_floor():
    module = perception_module()
    x, z = np.meshgrid(np.linspace(-.04, .04, 11), np.linspace(.61, .7, 11))
    points = np.column_stack((x.ravel(), np.full(x.size, .05), z.ravel()))
    group = [{"entity_id": "fixture-entity", "kind": "bin", "camera_id": "3", "points": points,
              "pixels": np.full((len(points), 3), [20, 20, 220], np.uint8), "score": .9}]
    instance = module._estimate_object(group, {"surface_height_m": .6, "uncertainty_m": .003})
    assert instance["kind"] == "bin"
    assert "placement_reference" not in instance


def test_horizontal_rim_without_visible_interior_cannot_be_a_floor_reference():
    module = perception_module()
    x, y = np.meshgrid(np.linspace(-.07, .07, 71), np.linspace(-.07, .07, 71))
    ring = (np.abs(x) >= .055) | (np.abs(y) >= .055)
    points = np.column_stack((x[ring], y[ring], np.full(ring.sum(), .69)))
    assert module.measured_bin_reference(points, {"surface_height_m": .6, "uncertainty_m": .003}) is None


@pytest.mark.parametrize("higher_outlier", [False, True])
def test_partial_horizontal_rim_without_higher_bin_context_is_not_a_floor(higher_outlier):
    module = perception_module()
    x, y = np.meshgrid(np.linspace(-.07, .07, 141), np.linspace(.055, .065, 11))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .69)))
    if higher_outlier:
        points = np.vstack((points, [0., .06, .72]))
    assert module.measured_bin_reference(points, {"surface_height_m": .6, "uncertainty_m": .003}) is None


def test_horizontal_patch_alone_does_not_identify_its_role_as_bin_floor():
    module = perception_module()
    assert module.measured_bin_reference(
        bin_cloud()[:121], {"surface_height_m": .6, "uncertainty_m": .003}) is None


def test_floor_with_central_observation_gap_returns_nearest_supported_actual_sample():
    module = perception_module()
    x, y = np.meshgrid(np.arange(-.05, .051, .002), np.arange(-.04, .041, .002))
    visible = np.hypot(x, y) >= .012
    floor = np.column_stack((x[visible], y[visible], np.full(visible.sum(), .61)))
    higher = bin_cloud()[121:]
    points = np.vstack((floor, higher)).astype(np.float32)
    reference = module.measured_bin_reference(points, {"surface_height_m": .6, "uncertainty_m": .003})
    assert reference is not None
    selected = np.asarray(reference["position_world_m"])
    assert reference["floor_z_m"] == pytest.approx(.61, abs=1e-6)
    assert np.any(np.all(points == selected, axis=1))
    assert .0119 <= np.linalg.norm(selected[:2]) <= .015
    assert all(points[index, 2] > selected[2] for index in
               reference["measurement_sources"]["higher_context_point_indices"])


def test_sparse_floor_samples_do_not_become_a_supported_point_beside_a_gap():
    module = perception_module()
    x, y = np.meshgrid(np.arange(-.2, .201, .02), np.arange(-.2, .201, .02))
    visible = np.hypot(x, y) >= .04
    floor = np.column_stack((x[visible], y[visible], np.full(visible.sum(), .61)))
    points = np.vstack((floor, bin_cloud()[121:]))
    assert module.measured_bin_reference(points, {"surface_height_m": .6, "uncertainty_m": .003}) is None


@pytest.mark.parametrize("central_outlier", [False, True])
def test_small_horizontal_rim_does_not_supply_a_measured_floor(central_outlier):
    module = perception_module()
    x, y = np.meshgrid(np.linspace(-.025, .025, 51), np.linspace(-.025, .025, 51))
    ring = (np.abs(x) >= .02) | (np.abs(y) >= .02)
    points = np.column_stack((x[ring], y[ring], np.full(ring.sum(), .69)))
    if central_outlier:
        points = np.vstack((points, [0., 0., .69]))
    group = [{"entity_id": "fixture-entity", "kind": "bin", "camera_id": "3", "points": points,
              "pixels": np.full((len(points), 3), [20, 20, 220], np.uint8), "score": .9}]
    instance = module._estimate_object(group, {"surface_height_m": .6, "uncertainty_m": .003})
    assert instance["kind"] == "bin"
    assert "placement_reference" not in instance


@pytest.mark.parametrize("color,rgb", [("blue", [20, 20, 220]), ("green", [20, 220, 20]),
                                      ("red", [220, 20, 20])])
def test_head_open_box_semantics_and_measured_color_guide_wrist_bin_queries(color, rgb):
    capture, masks = capture_fixture()
    for camera_id, view in capture["views"].items():
        view["rgb"][0, 0, 0] = int(camera_id)
        view["rgb"][masks[1]] = rgb

    class SemanticDetector(FrozenSegmenter):
        def segment(self, image, text, timestamp_us):
            if text == f"{color} open box":
                return [{"mask": masks[1].copy(), "score": .9, "label": text}]
            if text == "red cube":
                return [{"mask": masks[0].copy(), "score": .9, "label": text}]
            return []

    observation = GeometryPerception(SemanticDetector(masks), kinds={'second': 'bin'},
        prompts={'first': 'red cube', 'second': f'{color} open box'}).observe(capture, checkpoint="fine",
        source_hint={"kind": "cube", "color": "red", "destination": {"color": color}})
    assert len(observation["objects"]) == 2
    cubes = [obj for obj in observation["objects"] if obj["kind"] == "cube"]
    bins = [obj for obj in observation["objects"] if obj["kind"] == "bin"]
    assert len(cubes) == 1 and cubes[0]["color"] == "red"
    assert len(bins) == 1 and bins[0]["color"] == color
    assert set(bins[0]["coverage"]["camera_ids"]) == {"1", "2", "3"}


@pytest.mark.parametrize("rgb,expected", [([12, 48, 23], "green"), ([48, 8, 7], "red"),
                                        ([7, 20, 48], "blue"), ([3, 3, 4], "black"),
                                        ([31, 33, 37], "gray"), ([150, 150, 150], "gray"),
                                        ([3, 9, 4], "black")])
def test_dim_saturated_color_is_preserved_without_relabeling_neutral_black(rgb, expected):
    module = perception_module()
    pixels = np.tile(np.asarray(rgb, np.uint8), (100, 1))
    assert module._color(pixels)[0] == expected


def test_public_allowlist_drops_poison_shape_fields_and_nested_reference_fields():
    module = perception_module()
    capture, masks = capture_fixture()
    observation = GeometryPerception(FrozenSegmenter(masks)).observe(capture)
    observation["objects"][0].update(size=[99] * 3, quaternion=[1, 0, 0, 0],
        T_world_object=np.eye(4).tolist(), cavity={"secret": "HIDDEN"},
        placement_reference={"position_world_m": [0, 0, .61], "floor_z_m": .61,
                             "source": "visible_rgbd_floor", "size": "HIDDEN"})
    public = module.public_observation(observation)
    first = public["objects"][0]
    assert not {"size", "quaternion", "T_world_object", "cavity", "point_sources"} & first.keys()
    assert first["placement_reference"]["floor_z_m"] == .61
    assert "HIDDEN" not in str(public)
