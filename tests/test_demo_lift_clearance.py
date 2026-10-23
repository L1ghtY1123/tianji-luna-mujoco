"""Measured source bounds determine a bounded, shared lift before trajectory planning."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_planning_batch import batch_response

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.collision_mapping import collision_layer_bounds
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.demo_lift import source_clearance_lift
from luna_mujoco.geometry import TOOL_PROFILE
from luna_mujoco.planning_settings import PlanningSettings


def frozen():
    directory = Path(__file__).parent / "fixtures"
    values = json.loads((directory / "luna_lift_clearance.json").read_text(encoding="utf-8"))
    # Historical input geometry is unchanged; use the current bounded policy.
    values["settings"]["attachment"]["sphere_budget"] = 24
    values["settings"]["trajectory_collision_clearance_m"] = .01
    values["settings"] = PlanningSettings.model_validate(values["settings"]).model_dump()
    arrays = np.load(directory / "luna_lift_clearance.npz", allow_pickle=False)
    for obj in values["snapshot"]["objects"]:
        for key in ("points_world_m", "completed_geometry_points_world_m"):
            if key in obj:
                obj[key] = arrays[obj[key]["fixture_array"]]
    return values


def derive(value):
    return source_clearance_lift(value["snapshot"], value["target_id"], "A",
                                 value["candidates"], value["scene"], value["settings"])


def test_published_bounds_use_full_native_cells_and_cuboid_rotation_without_changing_map():
    identity = {"scene_token": "map", "observation_id": "rgbd", "generation": 4,
                "disabled_obstacle_names": ["semantic_instance::target"]}
    model = {"voxel": {"semantic_instance::target": {
                 "centers_base_m": [[999., 999., 999.]], "surface_voxel_size": .02,
                 "surface_cell_bounds_world_m": [[.09, .19, .29], [.21, .41, .61]],
                 "surface_cell_bounds_source": "all_native_surface_cells_before_preview_sampling",
                 "surface_cell_count": 2}},
             "cuboid": {"box": {"pose": [0., 0., 1., np.sqrt(.5), 0., 0., np.sqrt(.5)],
                               "dims": [.4, .2, .6]}}}
    original = copy.deepcopy(model)
    result = collision_layer_bounds([SimpleNamespace(name="semantic_instance::target", voxel_size=.02)],
                                    model, identity)
    assert result["scene_token"] == "map"
    assert result["layers"]["semantic_instance::target"]["enabled"] is False
    np.testing.assert_allclose(result["layers"]["semantic_instance::target"]["maximum_world_m"], [.21, .41, .61])
    np.testing.assert_allclose(result["layers"]["box"]["minimum_world_m"], [-.1, -.2, .7])
    assert model == original


def test_preview_without_full_cell_bounds_is_unavailable_for_lift_safety():
    grid = SimpleNamespace(name="semantic_instance::target", voxel_size=.02)
    result = collision_layer_bounds([grid],
        {"voxel": {grid.name: {"centers_base_m": [[.2, .3, .6]], "surface_voxel_size": .02}}},
        {"scene_token": "map", "observation_id": "rgbd", "generation": 1})
    assert result["layers"][grid.name]["status"] == "unavailable"
    assert "maximum_world_m" not in result["layers"][grid.name]


def test_actual_ten_centimeter_lift_is_below_neighbor_but_derived_lift_clears_source_bin():
    value = frozen()
    original = copy.deepcopy(value)
    settings, proof = derive(value)
    source = value["snapshot"]["scene_entities"][value["target_id"]]["initial_parent_id"]
    source_top = value["scene"]["collision_layer_bounds"]["layers"]["semantic_instance::" + source]["maximum_world_m"][2]
    assert source_top == pytest.approx(.6319981906563044)
    candidate = proof["candidates"][0]
    assert candidate["lowest_sphere_world_z_m"] + .1 < .53 + .005
    assert candidate["lowest_sphere_world_z_m"] + settings["lift_offset_m"] >= source_top + .01 - 1e-12
    assert .23 < settings["lift_offset_m"] < .25
    assert settings["lift_axis"] == "z" and settings["lift_in_tool_frame"] is False
    assert proof["source_container_id"] == source
    assert proof["candidates"][0]["candidate_index"] == value["candidates"][0]["candidate_index"]
    assert proof["candidates"][0]["sphere_count"] == 24
    assert proof["candidates"][0]["measurement_uncertainty_m"] == .005
    assert proof["candidates"][0]["margin_m"] == .001
    assert value["settings"] == original["settings"]
    assert value["scene"] == original["scene"]
    assert value["candidates"] == original["candidates"]


def test_larger_configured_lift_is_not_reduced_and_same_offset_covers_all_candidates():
    value = frozen()
    value["settings"]["lift_offset_m"] = .4
    second = copy.deepcopy(value["candidates"][0])
    second["candidate_index"] += 1
    second["score"] = .1
    second["T_world_tcp"][0][3] += .005
    value["candidates"].append(second)
    settings, proof = derive(value)
    assert settings["lift_offset_m"] == .4
    assert [c["candidate_index"] for c in proof["candidates"]] == [c["candidate_index"] for c in value["candidates"]]


def test_second_cycle_excludes_completed_and_outside_source_references_but_keeps_world_layers():
    value = frozen()
    target = value["target_id"]
    siblings = [o for o in value["snapshot"]["objects"] if o["kind"] == "object" and o["id"] != target]
    value["snapshot"]["completed_target_ids"] = [siblings[0]["id"]]
    siblings[1]["points_world_m"] = siblings[1]["points_world_m"] + [0., -1., 0.]
    original_scene = copy.deepcopy(value["scene"])
    _, proof = derive(value)
    assert proof["source_neighbor_ids"] == []
    assert proof["excluded_reference_ids"][siblings[0]["id"]] == "completed_target"
    assert proof["excluded_reference_ids"][siblings[1]["id"]] == "outside_measured_source_region"
    assert value["scene"] == original_scene
    assert all(value["scene"]["collision_layer_bounds"]["layers"]["semantic_instance::" + obj["id"]]["enabled"]
               for obj in siblings)


@pytest.mark.parametrize("damage", ["missing", "stale", "nonfinite", "disabled_source", "unknown_parent", "wrong_lift"])
def test_missing_or_inconsistent_geometry_stops_before_trajectory_call(damage):
    value = frozen()
    source = value["snapshot"]["scene_entities"][value["target_id"]]["initial_parent_id"]
    if damage == "missing":
        value["scene"].pop("collision_layer_bounds")
    elif damage == "stale":
        value["scene"]["collision_layer_bounds"]["scene_token"] = "old"
    elif damage == "nonfinite":
        value["scene"]["collision_layer_bounds"]["layers"]["semantic_instance::" + source]["maximum_world_m"][2] = float("nan")
    elif damage == "disabled_source":
        value["scene"]["collision_layer_bounds"]["layers"]["semantic_instance::" + source]["enabled"] = False
    elif damage == "unknown_parent":
        value["snapshot"]["scene_entities"][value["target_id"]]["initial_parent_id"] = None
    else:
        value["settings"]["lift_in_tool_frame"] = True
    with pytest.raises(ValueError):
        derive(value)


def test_batch_receives_derived_lift_and_records_proof_without_mutating_ik_identity():
    from test_grasp_fk_goal import fk_evidence

    value = frozen()
    snapshot, scene, settings = value["snapshot"], value["scene"], value["settings"]
    calls, events = [], []

    def request(path, body):
        calls.append((path, copy.deepcopy(body)))
        result = batch_response(body)
        for item, goal in zip(result["items"], body["goal_matrices_world"], strict=True):
            item["grasp_endpoint_fk"] = fk_evidence(np.asarray(goal),
                joints=item["phases"]["grasp"]["positions_rad"][-1][::-1], token=scene["scene_token"])
        return result

    algorithm = DemoAlgorithms(curobo_request=request)
    algorithm.validate_execution = lambda *args, **kwargs: {"collision_free": True}
    for candidate in value["candidates"]:
        fk = fk_evidence(np.asarray(candidate["T_world_tcp"]), token=scene["scene_token"])
        candidate.update(nominal_T_world_tcp=copy.deepcopy(candidate["T_world_tcp"]),
                         source_ik_fk=fk, ik={"achieved_tool_pose": fk,
                             "joints_rad": fk["joints_rad"], "joint_names": fk["joint_names"]})
    evaluation = {"A": {"arm": "A", "target_id": value["target_id"], "generation": snapshot["generation"],
        "scene_token": scene["scene_token"], "planning_settings": copy.deepcopy(settings),
        "tool_profile_revision": TOOL_PROFILE["revision"], "valid_candidates": value["candidates"],
        "start_joints_deg": np.rad2deg(snapshot["arms"]["A"]["joints_rad"]).tolist(),
        "other_joints_deg": np.rad2deg(snapshot["arms"]["B"]["joints_rad"]).tolist()}}
    original = copy.deepcopy(evaluation)
    plan = algorithm.plan_grasp(snapshot, value["target_id"], "A", evaluation, scene,
                                planning_settings=settings, progress=lambda *args: events.append(args))
    assert [path for path, _ in calls] == ["/tools/plan_grasp_batch"]
    sent = calls[0][1]
    assert sent["lift_offset_m"] > .23
    assert sent["planning_settings"]["lift_offset_m"] == sent["lift_offset_m"]
    assert sent["trajectory_collision_clearance_m"] == .01
    assert plan["planning_settings"] == settings
    assert plan["effective_grasp_settings"]["lift_offset_m"] == sent["lift_offset_m"]
    assert plan["lift_clearance"]["scene_token"] == scene["scene_token"]
    replay = next(e[2]["candidate_replay"] for e in events if e[0] == "planning_batch_replay")
    assert replay["lift_clearance"] == plan["lift_clearance"]
    assert evaluation == original
    scene.pop("collision_layer_bounds")
    calls.clear()
    with pytest.raises(AlgorithmError, match="lift"):
        algorithm.plan_grasp(snapshot, value["target_id"], "A", evaluation, scene, planning_settings=settings)
    assert not calls
