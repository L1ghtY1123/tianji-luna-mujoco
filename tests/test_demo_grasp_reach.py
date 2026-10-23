"""The fixed slot may require an unreachable wrist even when source IK succeeds."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.demo_algorithms import DemoAlgorithms


@pytest.fixture(autouse=True)
def isolate_reach_screen_from_direction_and_insertion_policy(monkeypatch):
    # These historical fixtures exercise only the reach calculation. The current
    # cone/insertion policy is independently tested in test_luna_grasp_policy.
    monkeypatch.setattr("luna_mujoco.demo_algorithms.prepare_grasp_candidates",
        lambda candidates, **kwargs: (deepcopy(candidates), deepcopy(candidates), {}))


def inputs(arm="A"):
    centroid = np.array([.341315727529533, .5766078970797198, .6581784948576461])
    poses = [
        [[-.9054858088493347, -.23453420400619507, -.3536793887615204, .4183666932559086],
         [-.42360374331474304, .4492391347885132, .7866029739379883, .43088128069712095],
         [-.025598634034395218, .8620777130126953, -.506129264831543, .76595397292266],
         [0., 0., 0., 1.]],
        [[.03875255584716797, -.5011089444160461, .8645161986351013, .1670559456920696],
         [-.9992465972900391, -.017586827278137207, .03459775820374489, .5714995793469089],
         [-.002133071655407548, -.8652055859565735, -.501413106918335, .7663617128003608],
         [0., 0., 0., 1.]],
    ]
    candidates = [{"index": index, "score": score, "source_batch": 1, "source_index": index,
                   "T_world_tcp": pose}
                  for index, score, pose in zip((80, 138), (.689030110836029, .651117742061615), poses)]
    snapshot = {"objects": [{"id": "target", "points_world_m": [
        (centroid - [.01, .002, .003]).tolist(), (centroid + [.01, .002, .003]).tolist()],
        # Deliberately disagree: the measured cloud mean must be used.
        "visible_centroid_world": [9., 9., 9.]}]}
    request = {"object_id": "target", "slot_id": 1, "layout_revision": 4,
               "layout": {"floor_z_m": .6, "cells": [{"slot_id": 1,
                                    "center_world_xy_m": [.6288367980980835, .08886111004014446]}]}}
    if arm == "B":
        # Reflect both world and tool Y; determinant remains +1 and tool Z remains down.
        mirror = np.diag([1., -1., 1.])
        for candidate in candidates:
            pose = np.array(candidate["T_world_tcp"])
            pose[:3, :3] = mirror @ pose[:3, :3] @ mirror
            pose[:3, 3] = mirror @ pose[:3, 3]
            candidate["T_world_tcp"] = pose.tolist()
        snapshot["objects"][0]["points_world_m"] = (
            np.array(snapshot["objects"][0]["points_world_m"]) @ mirror).tolist()
        request["layout"]["cells"][0]["center_world_xy_m"][1] *= -1
    return snapshot, candidates, request


@pytest.mark.parametrize("arm", ["A", "B"])
def test_known_impossible_grasp_rejected_before_ik_without_reordering(monkeypatch, arm):
    snapshot, candidates, request = inputs(arm)
    original = deepcopy(candidates)
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: {"candidates": deepcopy(candidates),
                                                "batches": [{"candidate_count": len(candidates)}]})
    result = DemoAlgorithms().generate_grasps(snapshot, "target", arm=arm, placement_request=request)
    assert result["candidates"] == [original[1]]
    evidence = result["fixed_slot_reach_filter"]
    assert (evidence["input_count"], evidence["accepted_count"], evidence["rejected_count"]) == (2, 1, 1)
    assert evidence["candidate_evidence"][0]["horizontal_wrist_distance_m"] > .8
    assert evidence["candidate_evidence"][1]["horizontal_wrist_distance_m"] < evidence["reach_screening_limit_m"]
    assert evidence["maximum_shoulder_wrist_distance_m"] == pytest.approx(.60208048, abs=1e-7)
    assert evidence["measurement_reserve_m"] == .005
    assert evidence["horizontal_screening_limit_m"] == pytest.approx(.59708048, abs=1e-7)
    assert evidence["slot_id"] == 1 and evidence["layout_revision"] == 4


@pytest.mark.parametrize("arm", ["A", "B"])
def test_five_mm_inward_reserve_rejects_near_boundary_candidate(monkeypatch, arm):
    from luna_mujoco.demo_grasp import arm_wrist_reach

    geometry = arm_wrist_reach(arm)
    rotation = Rotation.from_euler("x", np.pi).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    snapshot = {"objects": [{"id": "target", "points_world_m": [[0., 0., 0.]]}]}
    candidate = {"index": 91, "score": .7, "T_world_tcp": transform.tolist()}
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: {"candidates": [candidate], "batches": [{"candidate_count": 1}]})
    shoulder = np.asarray(geometry["shoulder_world_m"])
    offset = rotation @ np.asarray(geometry["tcp_to_wrist_tool_m"])
    for extra, accepted in ((-.0051, True), (-.0049, False)):
        distance = geometry["maximum_shoulder_wrist_distance_m"] + extra
        target = shoulder[:2] + [np.sqrt(distance ** 2 - .05 ** 2), 0.] - offset[:2]
        request = {"object_id": "target", "slot_id": 1,
                   "layout": {"floor_z_m": shoulder[2] - .05 - offset[2] - .01,
                              "cells": [{"slot_id": 1, "center_world_xy_m": target.tolist()}]}}
        if accepted:
            result = DemoAlgorithms().generate_grasps(snapshot, "target", arm=arm, placement_request=request)
            assert result["candidates"] == [candidate]
        else:
            with pytest.raises(AlgorithmError) as caught:
                DemoAlgorithms().generate_grasps(snapshot, "target", arm=arm, placement_request=request)
            assert caught.value.code == "grasp_no_fixed_slot_reachable_candidate"
            assert caught.value.details["fixed_slot_reach_filter"]["rejected_count"] == 1


@pytest.mark.parametrize("arm", ["A", "B"])
def test_eighth_run_horizontal_distance_is_outside_inward_reserve(monkeypatch, arm):
    from luna_mujoco.demo_grasp import arm_wrist_reach

    geometry = arm_wrist_reach(arm)
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1., -1., -1.])
    candidate = {"index": 91, "score": .7, "T_world_tcp": transform.tolist()}
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: {"candidates": [candidate], "batches": [{"candidate_count": 1}]})
    snapshot = {"objects": [{"id": "target", "points_world_m": [[0., 0., 0.]]}]}
    wrist_offset = transform[:3, :3] @ geometry["tcp_to_wrist_tool_m"]
    target = np.asarray(geometry["shoulder_world_m"])[:2] + [.601385, 0.] - wrist_offset[:2]
    request = {"object_id": "target", "slot_id": 1,
               "layout": {"floor_z_m": .6,
                          "cells": [{"slot_id": 1, "center_world_xy_m": target.tolist()}]}}
    with pytest.raises(AlgorithmError) as caught:
        DemoAlgorithms().generate_grasps(snapshot, "target", arm=arm, placement_request=request)
    evidence = caught.value.details["fixed_slot_reach_filter"]
    assert evidence["candidate_evidence"][0]["horizontal_wrist_distance_m"] == pytest.approx(.601385)
    assert evidence["horizontal_screening_limit_m"] < .598
    assert evidence["screening_scope"] == "nominal_release_and_approach_3d_with_inward_measurement_reserve"


def test_demo_grasps_require_bound_slot_before_external_inference(monkeypatch):
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: pytest.fail("must validate slot before GraspGen"))
    snapshot, _, request = inputs()
    for invalid in (None, {**request, "object_id": "other"}, {**request, "slot_id": 2}):
        with pytest.raises(ValueError, match="grasp_requires_bound_placement"):
            DemoAlgorithms().generate_grasps(snapshot, "target", arm="A", placement_request=invalid)


def test_reach_filter_keeps_original_order_and_scores_without_vertical_ranking(monkeypatch):
    snapshot, candidates, request = inputs()
    second = deepcopy(candidates[1])
    second.update(index=199, score=.999, source_index=17)
    candidates = [candidates[1], second]
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: {"candidates": deepcopy(candidates),
                                                "batches": [{"candidate_count": len(candidates)}]})
    result = DemoAlgorithms().generate_grasps(snapshot, "target", arm="A", placement_request=request)
    assert result["candidates"] == candidates


@pytest.mark.parametrize("arm", ["A", "B"])
def test_ninth_measured_nominal_release_rejected_despite_passing_horizontal_screen(monkeypatch, arm):
    snapshot, _, request = inputs(arm)
    centroid = np.array([.3413151213524463, .5766481536984354, .6580493018857076])
    minimum = .6149969696998596
    pose = np.array([
        [-.8038852214813232, .4952761232852936, .32934802770614624, .2763242018220546],
        [.5940588116645813, .6959211826324463, .40346962213516235, .497069767153254],
        [-.02937142550945282, .519995391368866, -.8536639213562012, .841832934998615],
        [0., 0., 0., 1.]])
    points = np.array([centroid, centroid])
    points[:, 2] = [minimum, 2 * centroid[2] - minimum]
    center = [.6288287801159992, .08882278563504792]
    if arm == "B":
        mirror = np.diag([1., -1., 1.])
        pose[:3, :3] = mirror @ pose[:3, :3] @ mirror
        pose[:3, 3] = mirror @ pose[:3, 3]
        points = points @ mirror
        center[1] *= -1
    snapshot["objects"][0]["points_world_m"] = points.tolist()
    request["layout"]["floor_z_m"] = .5999999642372131
    request["layout"]["cells"][0]["center_world_xy_m"] = center
    candidate = {"index": 163, "score": .6663368940353394, "T_world_tcp": pose.tolist()}
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: {"candidates": [candidate], "batches": [{"candidate_count": 1}]})
    with pytest.raises(AlgorithmError) as caught:
        DemoAlgorithms().generate_grasps(snapshot, "target", arm=arm, placement_request=request)
    evidence = caught.value.details["fixed_slot_reach_filter"]
    item = evidence["candidate_evidence"][0]
    assert item["horizontal_wrist_distance_m"] < evidence["reach_screening_limit_m"]
    assert item["nominal_release_wrist_distance_m"] > evidence["reach_screening_limit_m"]
    assert item["nominal_mount_world_m"][2] == pytest.approx(
        pose[2, 3] - minimum + .5999999642372131 + .01)


@pytest.mark.parametrize("floor", [None, float("nan"), float("inf"), "unmeasured"])
def test_missing_or_invalid_frozen_floor_fails_before_graspgen(monkeypatch, floor):
    snapshot, _, request = inputs()
    request["layout"].pop("floor_z_m")
    if floor is not None:
        request["layout"]["floor_z_m"] = floor
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: pytest.fail("floor must be validated before inference"))
    with pytest.raises(ValueError, match="grasp_requires_finite_bound_floor"):
        DemoAlgorithms().generate_grasps(snapshot, "target", arm="A", placement_request=request)


def test_nominal_z_and_tool_frame_approach_use_validated_current_settings(monkeypatch):
    snapshot, candidates, request = inputs()
    monkeypatch.setattr(Algorithms, "generate_grasps",
                        lambda *args, **kwargs: {"candidates": [candidates[1]], "batches": [{"candidate_count": 1}]})
    settings = {"trajectory_collision_clearance_m": .02, "place_approach_axis": "x",
                "place_approach_offset_m": .04, "place_approach_in_tool_frame": True}
    result = DemoAlgorithms().generate_grasps(
        snapshot, "target", arm="A", placement_request=request, planning_settings=settings)
    evidence = result["fixed_slot_reach_filter"]
    assert evidence["required_clearance_m"] == .02
    item = evidence["candidate_evidence"][0]
    rotation = np.array(candidates[1]["T_world_tcp"])[:3, :3]
    np.testing.assert_allclose(np.array(item["nominal_approach_wrist_world_m"])
                               - item["nominal_release_wrist_world_m"], rotation @ [.04, 0., 0.])
    with pytest.raises(ValueError):
        DemoAlgorithms().generate_grasps(snapshot, "target", arm="A", placement_request=request,
                                        planning_settings={"place_approach_offset_m": 0})


@pytest.mark.parametrize("arm,side", [("A", "L"), ("B", "R")])
def test_two_segment_bound_contains_authored_twist_and_elbow_fk(arm, side):
    import mujoco

    from luna_mujoco.demo_grasp import arm_wrist_reach
    from luna_mujoco.kinematics import robot_model

    model = robot_model()
    data = mujoco.MjData(model)
    twist, elbow = [model.joint(f"Joint{index}_{side}") for index in (3, 4)]
    shoulder, wrist = [model.body(f"Link{index}_{side}").id for index in (1, 7)]
    bound = arm_wrist_reach(arm)["maximum_shoulder_wrist_distance_m"]
    distances = []
    for twist_angle in np.linspace(*twist.range, 9):
        for elbow_angle in np.linspace(*elbow.range, 65):
            data.qpos[twist.qposadr[0]] = twist_angle
            data.qpos[elbow.qposadr[0]] = elbow_angle
            mujoco.mj_fwdPosition(model, data)
            distances.append(np.linalg.norm(data.xpos[wrist] - data.xpos[shoulder]))
    assert max(distances) <= bound
    assert bound - max(distances) < .0002
