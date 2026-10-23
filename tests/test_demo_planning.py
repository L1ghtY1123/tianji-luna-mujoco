"""Demo behavior at the real planning adapter's external service boundary."""

import copy
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_grasp_fk_goal import fk_evidence
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.geometry import grasp_targets
from luna_mujoco.placement import release_endpoint_evidence


class GraspClient:
    def __init__(self):
        self.calls = []

    def generate(self, payload):
        self.calls.append(payload)
        poses = []
        for index, pitch in enumerate((0., 90., 180., 89.99999999)):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("y", pitch, degrees=True).as_matrix()
            poses.append({"index": index, "score": 1. - index * .1,
                          "T_camera_grasp": transform.tolist()})
        return {"ok": True, "model": "test-service", "candidates": poses}


def grasp_placement():
    return {"object_id": "seen-red", "slot_id": 1,
            "layout": {"floor_z_m": .6, "cells": [{"slot_id": 1, "center_world_xy_m": [.3, 0.]}]}}


@pytest.mark.parametrize("arm", ["A", "B"])
def test_downward_filter_preserves_indices_before_real_batched_ik(arm, monkeypatch):
    # Synthetic direction/transport fixture, independent of the frozen geometry tests.
    monkeypatch.setattr(DemoAlgorithms, "_gripper_quality", lambda *args: {
        "accepted": True, "reason": "accepted"})
    calls = []

    def request(path, payload):
        calls.append((path, payload))
        return {"scene_token": "map-1", "batch_size": len(payload["goal_matrices_world"]),
                "items": [{"candidate_index": index, "success": True,
                           "joints_rad": [0.] * 7,
                           "joint_names": fk_evidence(np.asarray(matrix), arm)["joint_names"],
                           "achieved_tool_pose": fk_evidence(np.asarray(matrix), arm)}
                          for index, matrix in enumerate(payload["goal_matrices_world"])]}

    observation = observed_scene()
    client = GraspClient()
    algorithms = DemoAlgorithms(curobo_request=request, graspgen_client=client)
    grasps = algorithms.generate_grasps(observation, "seen-red", arm=arm,
                                       placement_request=grasp_placement())
    assert len(client.calls) == 3
    assert grasps["requested_candidate_count"] == 256
    assert [candidate["index"] for candidate in grasps["candidates"]] == list(range(6, 9))
    evidence = grasps["downward_filter"]
    assert evidence["maximum_tilt_deg"] == 45.
    assert (evidence["input_count"], evidence["accepted_count"], evidence["rejected_count"]) == (12, 3, 9)
    assert len(grasps["candidate_replay"]["candidates_before_ik"]) == 12
    np.testing.assert_array_equal(grasps["candidate_replay"]["graspgen_request"]["object_points_camera_m"],
                                  client.calls[0]["object_points_camera_m"])
    scene = {"target_id": "seen-red", "generation": 1,
             "observation_id": "capture-4", "scene_token": "map-1"}
    result = algorithms.evaluate_arms(observation, "seen-red", grasps, scene, requested_arm=arm)
    assert len(calls) == 1 and calls[0][0] == "/tools/solve_ik_batch"
    assert len(calls[0][1]["goal_matrices_world"]) == 3
    assert [item["candidate_index"] for item in result[arm]["valid_candidates"]] == list(range(6, 9))
    world = np.asarray(grasps["T_world_cloud"]) @ np.asarray(
        [candidate["T_object_grasp"] for candidate in grasps["candidates"]])
    expected = grasp_targets(world, arm)
    expected[:, :3, 3] += .01 * expected[:, :3, 2]
    np.testing.assert_allclose(calls[0][1]["goal_matrices_world"], expected)


def test_no_downward_grasp_returns_distinct_failure_before_ik():
    class Upward:
        def generate(self, payload):
            return {"ok": True, "model": "upward", "candidates": [
                {"index": 0, "score": .9, "T_camera_grasp": np.eye(4).tolist()}]}

    algorithms = DemoAlgorithms(graspgen_client=Upward())
    with pytest.raises(AlgorithmError) as caught:
        algorithms.generate_grasps(observed_scene(), "seen-red", arm="A",
                                   placement_request=grasp_placement())
    assert caught.value.code == "grasp_no_downward_candidate"
    assert caught.value.details["accepted_count"] == 0


def test_background_bootstrap_is_rejected_by_inventory_only_demo_mapping():
    from luna_mujoco.perception_mapping import decode_mapping_payload

    observation = observed_scene()
    observation.update(objects=[], demo_background_mapping=True)
    requests = []
    algorithms = DemoAlgorithms(curobo_request=lambda path, body: (
        requests.append(body) or {"ok": True, "scene_token": "map-1"}))
    scene = algorithms.build_scene(observation, None, exclusion_state="observation", head_only=True)
    with pytest.raises(ValueError, match="demo_inventory_semantic_mapping_required"):
        decode_mapping_payload(requests[0])
    assert scene["target_id"] is None


def placement_inputs():
    observation = observed_scene()
    polygon = [[.5, -.3], [.7, -.3], [.7, -.1], [.5, -.1]]
    destination = {"id": "demo-destination", "kind": "marker",
        "points_world_m": [[.6, -.2, .6], [.59, -.2, .6], [.61, -.2, .6]],
        "placement_reference": {"position_world_m": [.6, -.2, .6], "floor_z_m": .6,
            "region_polygon_world_xy_m": polygon, "observation_id": "capture-4",
            "source": "measured_table_and_fixed_demo_region"}}
    observation["objects"].append(destination)
    grasp = np.eye(4)
    grasp[:3, :3] = np.diag([1., -1., -1.])
    grasp[:3, 3] = [.45, .1, .8]
    points = [[.03 + x, .02 + y, .15 + z]
              for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)]
    holding = {"arm": "A", "instance_id": "seen-red", "owner_id": "held-1",
        "verified": True, "attached": True, "points_tool_m": points,
        "grasp_support_reference": {"T_world_tcp_at_grasp": grasp.tolist(),
                                   "table_height_m": .6, "source": "measured_support"},
        "attachment_evidence": {"attached": True, "scene_token": "map-1",
            "attachment_spheres_tool_m": [[.03, .02, .15, .025]]}}
    scene = {"target_id": "seen-red", "generation": 1,
             "observation_id": "capture-4", "scene_token": "map-1"}
    placement = {"object_id": "seen-red", "region_id": "demo-region", "slot_id": 1,
        "target_count": 1, "layout_revision": 1, "layout": {"section_count": 1,
            "floor_z_m": .6, "region_polygon_world_xy_m": polygon,
            "cells": [{"slot_id": 1, "index": 0, "center_world_xy_m": [.6, -.2],
                       "polygon_world_xy_m": polygon}]}}
    return observation, scene, holding, placement


class PlaceService:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure
        self.resolved_height_increase = 0.

    def __call__(self, path, payload):
        self.calls.append((path, copy.deepcopy(payload)))
        if path == "/tools/resolve_release_height":
            from luna_mujoco.demo_placement import resolve_release_height

            _, _, holding, _ = placement_inputs()
            minimum = payload["goal_position_m"][2] + self.resolved_height_increase
            base_bottom = -.22 if payload["gripper_spheres_tool_m"][0][2] == .2 else -.175

            def world_cost(world):
                clearance = payload["trajectory_collision_clearance_m"]
                surface = minimum + base_bottom - clearance
                return np.maximum(0., clearance - (world[..., 2] - world[..., 3] - surface))

            result = resolve_release_height(holding["attachment_evidence"]["attachment_spheres_tool_m"],
                payload["gripper_spheres_tool_m"], SimpleNamespace(**payload), world_cost, scene_token="map-1")
            return {**result, "attached": True}
        if path == "/carrying/validate":
            return {"scene_token": "map-1", "collision_free": self.failure != "collision",
                "attached": "carrying_binding" in payload,
                "binding_sha256": hashlib.sha256(json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
        phase = {"positions_rad": [[0.] * 7, [.01] * 7],
            "velocities_rad_s": [[0.] * 7] * 2, "accelerations_rad_s2": [[0.] * 7] * 2,
            "joint_names": [f"Joint{i}_L" for i in range(1, 8)], "interpolation_dt_s": .02}
        if path == "/tools/plan_to_pose":
            return {"scene_token": "map-1", "attached": False, "trajectory": phase}
        if path == "/tools/plan_place":
            if self.failure == "timeout":
                raise AlgorithmError("timed out", code="algorithm_request_timeout")
            return {"scene_token": "stale" if self.failure == "stale" else "map-1",
                "phases": {"approach": phase, "place": phase},
                "collision_evidence": {"optimizer_attachment": self.failure != "attachment"}}
        pytest.fail("fixed placement must not call " + path)


def test_fixed_place_single_goal_targets_front_keeps_rotation_and_checks_retreat(monkeypatch):
    observation, scene, holding, placement = placement_inputs()
    service = PlaceService()
    algorithms = DemoAlgorithms(curobo_request=service)
    # A modeled fingertip sphere extends below the held object, so it determines Z.
    monkeypatch.setattr(algorithms, "gripper_release_spheres", lambda observation, arm: [[0., 0., .2, .02]],
                        raising=False)
    before = copy.deepcopy((observation, scene, holding, placement))
    result = algorithms.plan_place(observation, "seen-red", "A", scene, holding,
                                   "demo-destination", placement_request=placement)
    assert [path for path, _ in service.calls] == [
        "/tools/resolve_release_height", "/tools/plan_place", "/carrying/validate", "/carrying/validate",
        "/tools/plan_to_pose", "/carrying/validate"]
    tcp = np.asarray(result["goal"]["T_world_tcp"])
    np.testing.assert_allclose(tcp[:3, 3], [.6, -.2, .8301], atol=1e-12)
    np.testing.assert_array_equal(tcp[:3, :3], np.diag([1., -1., -1.]))
    np.testing.assert_allclose(result["goal"]["object_reference_world_m"][:2], [.6, -.2])
    assert result["goal"]["release_reference"]["height_limiting_geometry"] == "gripper"
    assert set(result["phases"]) == {"approach", "place", "retreat"}
    assert result["candidate_count"] == 1 and result["retreat_requires_replan"] is False
    assert result["selection_evidence"]["retreat_checked"] is True
    assert result["carrying_binding"]["owner_id"] == "held-1"
    endpoint = release_endpoint_evidence(result["goal"], holding, tcp)
    assert endpoint["at_release_goal"] is True
    assert endpoint["attachment_floor_gap_m"] == pytest.approx(.0551)
    assert endpoint["gripper_floor_gap_m"] == pytest.approx(.0101)
    moved = tcp.copy()
    moved[0, 3] += .004
    assert release_endpoint_evidence(result["goal"], holding, moved)["at_release_goal"] is True
    for actual, original in zip((observation, scene, holding, placement), before):
        assert str(actual) == str(original)


def test_fixed_place_uses_measured_height_and_endpoint_rejects_unbound_higher_goal(monkeypatch):
    observation, scene, holding, placement = placement_inputs()
    service = PlaceService()
    service.resolved_height_increase = .006
    algorithms = DemoAlgorithms(curobo_request=service)
    monkeypatch.setattr(algorithms, "gripper_release_spheres", lambda observation, arm: [[0., 0., .2, .02]])
    result = algorithms.plan_place(observation, "seen-red", "A", scene, holding,
                                   "demo-destination", placement_request=placement)
    tcp = np.asarray(result["goal"]["T_world_tcp"])
    np.testing.assert_allclose(tcp[:3, 3], [.6, -.2, .8361])
    assert service.calls[1][1]["goal_position_m"] == pytest.approx([.6, -.2, .8361])
    assert release_endpoint_evidence(result["goal"], holding, tcp)["at_release_goal"] is True
    tampered = copy.deepcopy(result["goal"])
    tampered["release_reference"]["height_resolution"]["scene_token"] = "another-scene"
    assert release_endpoint_evidence(tampered, holding, tcp)["at_release_goal"] is False


def test_fixed_place_allows_footprint_outside_region_with_front_inside(monkeypatch):
    observation, scene, holding, placement = placement_inputs()
    placement["layout"]["cells"][0]["polygon_world_xy_m"] = [
        [.59, -.21], [.61, -.21], [.61, -.19], [.59, -.19]]
    service = PlaceService()
    algorithms = DemoAlgorithms(curobo_request=service)
    monkeypatch.setattr(algorithms, "gripper_release_spheres", lambda observation, arm: [[0., 0., 0., .01]],
                        raising=False)
    result = algorithms.plan_place(observation, "seen-red", "A", scene, holding,
                                   "demo-destination", placement_request=placement)
    endpoint = release_endpoint_evidence(result["goal"], holding, result["goal"]["T_world_tcp"])
    assert endpoint['front_inside_region'] is True
    assert endpoint['footprint_inside_region'] is False
    assert endpoint['at_release_goal'] is True



@pytest.mark.parametrize("failure,code", [
    ("stale", "planning_stale_scene"), ("attachment", "attachment_not_applied"),
    ("collision", "execution_collision"), ("timeout", "algorithm_request_timeout")])
def test_fixed_place_propagates_failure_without_alternate_pose(monkeypatch, failure, code):
    observation, scene, holding, placement = placement_inputs()
    service = PlaceService(failure=failure)
    algorithms = DemoAlgorithms(curobo_request=service)
    monkeypatch.setattr(algorithms, "gripper_release_spheres", lambda observation, arm: [[0., 0., 0., .01]],
                        raising=False)
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_place(observation, "seen-red", "A", scene, holding,
                              "demo-destination", placement_request=placement)
    assert caught.value.code == code
    assert sum(path == "/tools/plan_place" for path, _ in service.calls) == 1


def test_fixed_place_cancellation_and_unverified_hold_do_not_issue_motion():
    observation, scene, holding, placement = placement_inputs()
    service = PlaceService()
    algorithms = DemoAlgorithms(curobo_request=service)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_place(observation, "seen-red", "A", scene, holding,
                              "demo-destination", placement_request=placement, cancelled=cancelled)
    assert caught.value.code == "planning_cancelled"
    holding["verified"] = False
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_place(observation, "seen-red", "A", scene, holding,
                              "demo-destination", placement_request=placement)
    assert caught.value.code == "place_requires_attachment"
    assert service.calls == []


def test_gripper_release_envelope_uses_own_model_and_current_driver():
    algorithms = DemoAlgorithms()
    observation = observed_scene()
    spheres = np.asarray(algorithms.gripper_release_spheres(observation, "A"))
    assert spheres.ndim == 2 and spheres.shape[1] == 4 and len(spheres) > 48
    assert np.isfinite(spheres).all() and np.all(spheres[:, 3] > 0)
    # Authored fingers extend 185 mm along the mount's positive Z.
    assert np.max(spheres[:, 2] + spheres[:, 3]) > .185
    assert np.max(spheres[:, 2] + spheres[:, 3]) < .3
