"""Inventory-only demo maps retain all listed tables without any background."""
from copy import deepcopy

import numpy as np
import pytest
from test_perception_mapping import wire
from test_rgbd_algorithms import observed_scene
from test_semantic_builder import builder, request

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.demo_geometry import complete_source_height
from luna_mujoco.perception_mapping import decode_mapping_payload, mapping_builder_payload


def table_evidence():
    table = {"model": {"pose": [.55, 0., .45, 1., 0., 0., 0.], "dims": [.8, 1., .3]},
             "surface_height_m": .6, "normal": [0., 0., 1.], "inlier_count": 100,
             "source": "initial_inventory_ransac_table"}
    side = deepcopy(table)
    side["model"]["pose"][1] = .75
    side["model"]["dims"][:2] = [.25, .25]
    return {"source_observation_id": "capture-42", "primary_table_id": "table",
            "tables": {"table": table, "side": side}}


def test_demo_mapping_is_validated_and_bound_to_scene_identity():
    original = wire()
    _, legacy_identity = decode_mapping_payload(original)
    demo, identity = decode_mapping_payload(original | {"demo_mapping": True, "demo_tables": table_evidence()})
    assert demo["scene_policy"] == "demo_inventory"
    assert identity["scene_token"] != legacy_identity["scene_token"]
    with pytest.raises(ValueError, match="demo_mapping"):
        decode_mapping_payload(original | {"demo_mapping": "yes"})
    with pytest.raises(ValueError, match="demo_initial_table_evidence_required"):
        decode_mapping_payload(original | {"demo_mapping": True})


def test_demo_discards_background_and_reuses_all_initial_table_cuboids():
    value, segmenter, mapper, table_inputs = builder(support=True)
    payload = request("selected") | {"demo_mapping": True, "scene_policy": "demo_inventory",
                                   "demo_tables": table_evidence()}
    def reject_crop(*args, **kwargs):
        raise AssertionError("main-table crop deletes listed side-table entities")
    segmenter.crop_semantic_observation = reject_crop
    candidate = value.build(payload)
    assert table_inputs == []
    assert not segmenter.separated
    assert [view.background_point_count for view in mapper.integrated] == [0, 0, 0]
    assert set(candidate.scene_model["cuboid"]) == {"semantic_table::table", "semantic_table::side"}
    assert value.diagnostics["semantic_only_active"] is True
    assert value.diagnostics["background_drop_policy"] == "discard_all_unlisted_and_tabletop_returns"
    frozen = deepcopy(dict(candidate.scene_model["cuboid"]))
    candidate = value.build(payload | {"observation_set_id": "new-observation"})
    assert value.diagnostics["table_cache"]["cache_hit"] is True
    assert dict(candidate.scene_model["cuboid"]) == frozen
    changed = deepcopy(payload)
    changed["demo_tables"]["tables"]["side"]["model"]["pose"][0] += .01
    with pytest.raises(ValueError, match="demo_table_inventory_changed_requires_reset"):
        value.build(changed)


def test_demo_rejects_raw_background_mapping():
    value, _, _, _ = builder()
    payload = request("observation") | {"demo_mapping": True, "mapping_mode": "background",
                                       "confirmed_instance_id": "", "sam3_generation": 0}
    for view in payload["views"].values():
        view["instance_masks"] = {}
    with pytest.raises(ValueError, match="demo_inventory_semantic_mapping_required"):
        value.build(payload)


def test_demo_partial_views_remain_all_three_without_background_fallback():
    decoded, _ = decode_mapping_payload(wire() | {"demo_mapping": True, "demo_tables": table_evidence()})
    decoded["views"]["1"]["depth_left_m"][:] = 0
    decoded["views"]["1"]["valid_left_mask"][:] = 0
    result = mapping_builder_payload(decoded)
    assert list(result["views"]) == ["1", "2", "3"]
    assert not np.any(result["views"]["1"]["depth_left_m"])


def test_labelled_height_completion_adds_separate_cuboid_not_measured_voxels():
    value, _, mapper, _ = builder()
    payload = request("selected") | {"demo_mapping": True, "scene_policy": "demo_inventory",
        "demo_tables": table_evidence(), "demo_completions": {"source": {
            "source": "measured_surface_and_user_height_prior", "height_prior_m": .10,
            "synthetic_points_are_measured": False,
            "completed_bounds_world_m": [[.5, .7, .6], [.56, .74, .7]]}}}
    candidate = value.build(payload)
    model = candidate.scene_model["cuboid"]["semantic_instance::source::height_completion"]
    np.testing.assert_allclose(model["dims"], [.06, .04, .10])
    assert sum(len(points) for points in mapper.instances["source"][0].values()) == 2


def test_selected_completion_is_disabled_together_with_its_voxel_layer():
    completion = {"source": "measured_surface_and_user_height_prior", "height_prior_m": .10,
                  "synthetic_points_are_measured": False,
                  "completed_bounds_world_m": [[.5, .7, .6], [.56, .74, .7]]}
    _, identity = decode_mapping_payload(wire() | {"demo_mapping": True, "demo_tables": table_evidence(),
                                                  "demo_completions": {"source": completion}})
    assert identity["disabled_obstacle_names"] == (
        "semantic_instance::source", "semantic_instance::source::height_completion")


def completed_black_observation():
    snapshot = observed_scene()
    snapshot.update(demo_mapping=True, demo_tables=table_evidence(), stage="unpacked")
    instance = snapshot["objects"][0]
    instance.update(color="black", current_appearance="unpacked", appearance_revision=1,
                    lifecycle_phase="unpacked_confirmed")
    instance["points_world_m"][:, 2] = .66
    complete_source_height(instance, support_z_m=.6, support_source="cached_measured_table_plane",
                           observation_id=snapshot["observation_id"], height_prior_m=.06)
    return snapshot


def build_demo_snapshot(snapshot):
    value, _, mapper, _ = builder()
    captured = {}

    def service(path, payload):
        assert path == "/perception/build"
        decoded, identity = decode_mapping_payload(payload)
        captured.update(payload=payload, identity=identity,
                        candidate=value.build(mapping_builder_payload(decoded)))
        return {"scene_token": identity["scene_token"]}

    Algorithms(curobo_request=service).build_scene(snapshot, "seen-red")
    return captured, mapper


def test_build_scene_accepts_current_confirmed_black_completion_without_changing_measurements():
    snapshot = completed_black_observation()
    original = deepcopy(snapshot["objects"][0])
    captured, mapper = build_demo_snapshot(snapshot)
    model = captured["candidate"].scene_model["cuboid"]["semantic_instance::seen-red::height_completion"]
    np.testing.assert_allclose(model["dims"], [.04, .04, .06], atol=1e-7)
    np.testing.assert_allclose(model["pose"][:3], [.5, .1, .63], atol=1e-7)
    assert captured["identity"]["disabled_obstacle_names"] == (
        "semantic_instance::seen-red", "semantic_instance::seen-red::height_completion")
    context = captured["payload"]["demo_completions"]["seen-red"]["instance_context"]
    assert context == {"instance_id": "seen-red", "observation_id": "capture-4", "color": "black",
                       "current_appearance": "unpacked", "appearance_revision": 1,
                       "lifecycle_phase": "unpacked_confirmed"}
    np.testing.assert_array_equal(snapshot["objects"][0]["points_world_m"], original["points_world_m"])
    np.testing.assert_array_equal(snapshot["objects"][0]["completed_geometry_points_world_m"],
                                  original["completed_geometry_points_world_m"])
    assert snapshot["objects"][0]["geometry_completion"] == original["geometry_completion"]
    # Only measured RGBD is integrated; the separately labelled completion is a cuboid.
    assert sum(len(points) for points in mapper.instances["seen-red"][0].values()) == 64 * 3


def test_height_completion_preserves_scene_workspace_bounds():
    captured, mapper = build_demo_snapshot(completed_black_observation())
    assert captured["candidate"].bounds is mapper.bounds


def test_build_scene_accepts_current_wrapped_pink_ten_cm_prior():
    snapshot = completed_black_observation()
    instance = snapshot['objects'][0]
    instance.update(color='pink', current_appearance='wrapped', appearance_revision=0,
                    lifecycle_phase='wrapped_source')
    instance['points_world_m'][:, 2] = .7
    complete_source_height(instance, support_z_m=.6, support_source='measured_bin_floor',
                           observation_id=snapshot['observation_id'], height_prior_m=.10)
    captured, _ = build_demo_snapshot(snapshot)
    model = captured['candidate'].scene_model['cuboid']['semantic_instance::seen-red::height_completion']
    assert model['dims'][2] == pytest.approx(.10)


@pytest.mark.parametrize("change", ["pink", "missing_context", "released", "old_observation",
                                   "zero_revision", "bool_revision", "wrong_height_source", "unpacked_ten_cm",
                                   "wrong_support_source", "unauthorized_height"])
def test_build_scene_rejects_wrongly_bound_black_height_completion(change):
    snapshot = completed_black_observation()
    instance = snapshot["objects"][0]
    if change == "pink":
        instance.update(color="pink", current_appearance="wrapped", appearance_revision=0)
    elif change == "missing_context":
        for key in ("current_appearance", "appearance_revision", "lifecycle_phase"):
            instance.pop(key)
    elif change == "released":
        instance["lifecycle_phase"] = "right_bin_released"
    elif change == "old_observation":
        instance["geometry_completion"]["observation_id"] = "old-capture"
    elif change == "zero_revision":
        instance["appearance_revision"] = 0
    elif change == "bool_revision":
        instance["appearance_revision"] = True
    elif change == "wrong_height_source":
        instance["geometry_completion"]["height_prior_source"] = "user_specified_100mm"
    elif change == "wrong_support_source":
        instance["geometry_completion"]["support_source"] = "mock_commanded_pose"
    elif change == "unauthorized_height":
        instance["geometry_completion"]["height_prior_m"] = .08
    else:
        instance["geometry_completion"].update(height_prior_m=.10, height_prior_source="user_specified_100mm")
    with pytest.raises(ValueError, match="demo_height_completion_invalid"):
        build_demo_snapshot(snapshot)
