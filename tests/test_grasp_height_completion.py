"""Labeled completion reaches inference without changing measured references."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_demo_planning import GraspClient, grasp_placement
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.demo_algorithms import DemoAlgorithms


def completed_scene():
    scene = observed_scene()
    target = scene["objects"][0]
    measured = target["points_world_m"]
    synthetic = measured[[0, 4, 20, 24]].copy()
    synthetic[:, 2] -= .1
    target["completed_geometry_points_world_m"] = np.vstack((measured, synthetic))
    target["geometry_completion"] = {
        "source": "measured_surface_and_user_height_prior",
        "observation_id": scene["observation_id"],
        "height_prior_m": .1, "height_prior_source": "user_specified_100mm",
        "support_z_m": .54, "support_source": "measured_bin_floor",
        "measured_point_count": len(measured), "synthetic_point_count": len(synthetic),
        "synthetic_points_are_measured": False,
    }
    return scene


@pytest.mark.parametrize("arm", ["A", "B"])
def test_demo_actual_input_uses_completion_but_frame_and_reference_stay_measured(arm):
    scene = completed_scene()
    target = scene["objects"][0]
    before = deepcopy(target)
    client = GraspClient()
    result = DemoAlgorithms(graspgen_client=client).generate_grasps(
        scene, "seen-red", arm=arm, placement_request=grasp_placement())
    origin = np.asarray(before["points_world_m"], dtype=float).mean(axis=0)
    expected = (before["completed_geometry_points_world_m"] - origin).astype(np.float32)
    assert len(client.calls) == 3
    actual_input = client.calls[0]["object_points_camera_m"]
    assert len(actual_input) > len(expected)
    for request in client.calls:
        np.testing.assert_array_equal(request["object_points_camera_m"], actual_input)
        np.testing.assert_array_equal(request["object_points_camera_m"][:len(expected)], expected)
    np.testing.assert_array_equal(np.array(result["T_world_cloud"])[:3, 3], origin)
    np.testing.assert_array_equal(result["fixed_slot_reach_filter"]["measured_reference_world_m"], origin)
    replay = result["candidate_replay"]
    assert replay["target_id"] == "seen-red"
    assert replay["graspgen_input_source"] == "height_completed"
    assert replay["completion_provenance"] == before["geometry_completion"]
    np.testing.assert_array_equal(replay["measured_points_world_m"], before["points_world_m"])
    np.testing.assert_array_equal(replay["completed_geometry_points_world_m"],
                                  before["completed_geometry_points_world_m"])
    np.testing.assert_array_equal(replay["graspgen_request"]["object_points_camera_m"], actual_input)
    assert result["cloud"]["point_count"] == len(actual_input)
    assert result["cloud"]["id"] == hashlib.sha256(actual_input.tobytes()).hexdigest()
    assert result["cloud"]["source"] == "registered_rgbd_surface_with_explicit_height_completion"
    assert result["measured_cloud_reference"] == Algorithms._cloud_reference(before)
    np.testing.assert_array_equal(target["points_world_m"], before["points_world_m"])
    np.testing.assert_array_equal(target["completed_geometry_points_world_m"],
                                  before["completed_geometry_points_world_m"])


@pytest.mark.parametrize("corruption", ["prefix", "count", "source", "measurement_claim", "stale"])
def test_invalid_completion_stops_before_inference(corruption):
    scene = completed_scene()
    target = scene["objects"][0]
    if corruption == "prefix":
        target["completed_geometry_points_world_m"][0, 0] += .01
    else:
        key, value = {"count": ("synthetic_point_count", 99), "source": ("source", "simulation"),
                      "measurement_claim": ("synthetic_points_are_measured", True),
                      "stale": ("observation_id", "previous-capture")}[corruption]
        target["geometry_completion"][key] = value
    client = GraspClient()
    with pytest.raises(AlgorithmError, match="grasp_height_completion_provenance_invalid"):
        DemoAlgorithms(graspgen_client=client).generate_grasps(
            scene, "seen-red", arm="A", placement_request=grasp_placement())
    assert not client.calls


def test_legacy_adapter_keeps_measured_input_and_demo_without_completion_labels_it():
    scene = completed_scene()
    client = GraspClient()
    legacy = Algorithms(graspgen_client=client).generate_grasps(scene, "seen-red", arm="A")
    assert client.calls[0]["object_points_camera_m"].shape == (25, 3)
    assert legacy["cloud"]["source"] == "rgbd_visible_surface"
    result = DemoAlgorithms(graspgen_client=GraspClient()).generate_grasps(
        observed_scene(), "seen-red", arm="A", placement_request=grasp_placement())
    assert result["candidate_replay"]["graspgen_input_source"] == "measured"
    assert "completed_geometry_points_world_m" not in result["candidate_replay"]


def test_failed_direction_filter_still_archives_actual_completed_input():
    class Upward(GraspClient):
        def generate(self, payload):
            self.calls.append(payload)
            return {"ok": True, "model": "upward", "candidates": [
                {"index": 0, "score": .9, "T_camera_grasp": np.eye(4).tolist()}]}

    client = Upward()
    with pytest.raises(AlgorithmError) as caught:
        DemoAlgorithms(graspgen_client=client).generate_grasps(
            completed_scene(), "seen-red", arm="A", placement_request=grasp_placement())
    assert caught.value.code == "grasp_no_downward_candidate"
    replay = caught.value.details["candidate_replay"]
    assert replay["graspgen_input_source"] == "height_completed"
    assert replay["graspgen_input_cloud"]["point_count"] == len(client.calls[0]["object_points_camera_m"])
    assert replay["graspgen_input_sampling"]["source_geometry_point_count"] == 29
    np.testing.assert_array_equal(replay["graspgen_request"]["object_points_camera_m"],
                                  client.calls[0]["object_points_camera_m"])


def test_frozen_top_surface_uses_completed_bottom_for_nominal_release_screen():
    folder = Path(__file__).parent / "fixtures"
    record = json.loads((folder / "luna_top_surface_grasp.json").read_text(encoding="utf-8"))
    record["planning_settings"]["attachment"]["sphere_budget"] = 24
    record["planning_settings"]["trajectory_collision_clearance_m"] = .01
    scene = observed_scene()
    target = scene["objects"][0]
    with np.load(folder / "luna_top_surface_grasp.npz") as fixture:
        target["points_world_m"] = fixture["measured_points_world_m"]
        target["completed_geometry_points_world_m"] = fixture["completed_geometry_points_world_m"]
    target["geometry_completion"] = record["geometry_completion"] | {"observation_id": scene["observation_id"]}
    placement = record["placement_request"] | {"object_id": "seen-red"}

    class FrozenClient:
        def generate(self, payload):
            return {"ok": True, "model": "frozen-actual-graspgen",
                    "candidates": [record["candidate"]]}

    result = DemoAlgorithms(graspgen_client=FrozenClient()).generate_grasps(
        scene, "seen-red", arm="A", placement_request=placement,
        planning_settings=record["planning_settings"])
    evidence = result["fixed_slot_reach_filter"]
    assert evidence["measured_cloud_min_world_z_m"] == pytest.approx(.5119968057)
    assert evidence["nominal_object_min_world_z_m"] == pytest.approx(.4119982123)
    assert evidence["nominal_object_geometry_source"] == "height_completed"
    assert evidence["nominal_height_source"] == "height_completed_minimum_and_bound_floor"
    # The historical 5 mm policy's top-only minimum incorrectly rejected this
    # actual candidate. With the current 10 mm floor gap, completion still
    # preserves its reachability. XY now aligns the calibrated finger front;
    # the original archived fixture is unchanged.
    assert evidence["candidate_evidence"][0]["nominal_release_wrist_distance_m"] == pytest.approx(
        .5818275736247825)
    assert evidence["candidate_evidence"][0]["reach_screening_passed"]
    assert len(result["candidates"]) == 3
