"""Measured world-height resolution must precede the only place trajectory solve."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_release_clearance import request

from luna_mujoco import demo_placement


def test_archived_completed_release_binding_requires_fresh_measured_geometry():
    from luna_mujoco.placement import visible_reference_tool_m
    fixture = json.loads((Path(__file__).parent / "fixtures/luna_loaded_release.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="synthetic_scene_geometry_unsupported"):
        visible_reference_tool_m(fixture["holding"])


def test_loaded_release_reserve_covers_archived_tracking_error_without_reusing_completed_geometry():
    from test_demo_planning import placement_inputs
    # The archive supplies only a measured servo error, never object geometry.
    fixture = json.loads((Path(__file__).parent / "fixtures/luna_loaded_release.json").read_text(encoding="utf-8"))
    error_z = fixture["actual_tcp_world"][2][3] - fixture["goal"]["T_world_tcp"][2][3]
    observation, _, holding, placement = placement_inputs()
    destination = next(obj for obj in observation['objects'] if obj['id'] == 'demo-destination')
    goal = demo_placement.fixed_release_goal(destination, holding, placement,
        holding['attachment_evidence']['attachment_spheres_tool_m'],
        [[0., 0., .1, .01]], clearance_m=.01, observation_id=observation['observation_id'])
    reference = goal['release_reference']
    assert reference['planning_reserve_m'] == .0001
    assert reference['required_clearance_m'] == .01
    assert reference['attachment_floor_gap_m'] == pytest.approx(.0101)
    from scipy.spatial.transform import Rotation

    from luna_mujoco.planning_contracts import PlanPlaceRequest
    transform = np.asarray(goal['T_world_tcp'])
    planning_request = PlanPlaceRequest.create(arm='A', start_joints_deg=[0.] * 7,
        other_joints_deg=[0.] * 7, goal_position_m=transform[:3, 3].tolist(),
        goal_quaternion_wxyz=Rotation.from_matrix(transform[:3, :3]).as_quat(scalar_first=True).tolist(),
        trajectory_collision_clearance_m=.01, freq_hz=50, max_attempts=32)
    proof = resolve(reference['attachment_spheres_tool_m'], reference['gripper_spheres_tool_m'],
        planning_request, lambda spheres: np.zeros(spheres.shape[:-1]))
    goal = demo_placement.apply_release_height(goal, proof, scene_token='scene-current')
    actual = np.asarray(goal['T_world_tcp']).copy()
    actual[2, 3] += error_z - reference['planning_reserve_m']
    assert not demo_placement.release_endpoint_evidence(goal, holding, actual)['at_release_goal']
    actual[2, 3] += reference['planning_reserve_m']
    verdict = demo_placement.release_endpoint_evidence(goal, holding, actual)
    assert verdict['at_release_goal']
    assert verdict['attachment_floor_gap_m'] >= .01
    assert verdict['required_clearance_m'] == .01


def resolve(attachment, gripper, goal, query):
    resolver = getattr(demo_placement, "resolve_release_height", None)
    assert callable(resolver), "demo release height must query the measured hard world checker"
    return resolver(attachment, gripper, goal, query, scene_token="scene-current")


def test_measured_surface_above_analytic_plane_sets_first_clear_height_with_fixed_xy_rotation():
    goal = request(goal_position_m=[.5, .2, .7])
    seen = []

    def query(world):
        seen.append(world.copy())
        # Measured ESDF boundary is 16.4 mm above the analytic support.
        return np.maximum(0., .01 - (world[..., 2] - world[..., 3] - .6464))

    proof = resolve([[0., 0., -.02, .025]], [[0., 0., .02, .01]], goal, query)
    assert proof["selected_goal_position_m"] == pytest.approx([.5, .2, .702])
    assert proof["height_increase_m"] == pytest.approx(.002)
    assert proof["last_conflicting_z_m"] == pytest.approx(.701)
    assert proof["resolution_m"] == .001
    assert len(seen) == 1
    assert seen[0].shape == (1, 21, 2, 4)
    assert all(item[0, 0, 0, 3] == pytest.approx(.025) for item in seen)
    assert proof["final_world_collision_costs"] == [0., 0.]
    assert goal.goal_position_m == (.5, .2, .7)


def test_gripper_envelope_is_included_in_measured_height_resolution():
    goal = request(goal_position_m=[.5, .2, .7])

    def query(world):
        return np.maximum(0., .01 - (world[..., 2] - world[..., 3] - .65))

    proof = resolve([[0., 0., .04, .01]], [[0., 0., -.03, .02]], goal, query)
    assert proof["selected_goal_position_m"][2] == pytest.approx(.710)
    assert proof["attachment_sphere_count"] == proof["gripper_sphere_count"] == 1
    assert proof["tested_heights"][0]["conflicting_sphere_indices"] == [1]


def test_no_feasible_height_stops_after_bounded_sweep_without_claiming_safe_pose():
    queries = []

    def query(world):
        queries.append(world)
        return np.ones(world.shape[:-1])

    with pytest.raises(ValueError, match="release_height_no_valid_goal") as caught:
        resolve([[0., 0., 0., .01]], [[0., 0., 0., .01]], request(), query)
    assert caught.value.code == "release_height_no_valid_goal"
    assert len(queries) == 1
    assert len(caught.value.details['tested_heights']) == 21
    assert caught.value.details["maximum_height_increase_m"] == .02


@pytest.mark.parametrize("cost", [np.nan, np.inf, -.01])
def test_invalid_world_query_is_not_treated_as_clearance(cost):
    with pytest.raises(ValueError, match="world_cost"):
        resolve([[0., 0., 0., .01]], [[0., 0., 0., .01]], request(),
                lambda world: np.full(world.shape[:-1], cost))


def test_release_height_route_validates_binding_and_keeps_query_separate_from_plan():
    from test_carrying import binding

    from luna_mujoco.geometry_service import run_operation
    from luna_mujoco.planning_settings import PlanningSettings

    settings = PlanningSettings()
    calls = []

    class Runtime:
        geometry_solver_profile = settings.solver_payload()

        def _scene_snapshot_unlocked(self):
            return {"scene_token": "scene-2", "observation_id": "rgbd-2",
                    "generation": 2, "target_id": "sam-source"}

        def configure_planning(self, value):
            assert value == settings

        def resolve_release_height_scoped(self, goal, carried, feedback, gripper):
            calls.append(goal)
            assert carried.owner_id == binding()["owner_id"]
            assert feedback == {"A": .4, "B": 0.}
            assert gripper == [[0., 0., .2, .01]]
            return {"scene_token": "scene-2", "selected_goal_position_m": [.5, .2, .72]}

    payload = {"arm": "A", "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
        "goal_position_m": [.5, .2, .7], "goal_quaternion_wxyz": [1., 0., 0., 0.],
        "scene_token": "scene-2", "planning_settings": settings.model_dump(),
        "carrying_binding": binding(), "gripper_joint_positions_rad": {"A": .4, "B": 0.},
        "gripper_spheres_tool_m": [[0., 0., .2, .01]]}
    result = run_operation(Runtime(), "/tools/resolve_release_height", payload)
    assert result["selected_goal_position_m"] == [.5, .2, .72]
    assert len(calls) == 1 and result["attached"] is True
    assert result["planner"] == "release_height_world_query"


def test_runtime_height_resolution_checks_scene_under_lock_without_planning():
    from contextlib import nullcontext

    from test_pre_release_place import runtime_fixture

    runtime, backend, _, calls = runtime_fixture()
    runtime.carrying_scope = lambda binding, feedback: nullcontext()
    binding = SimpleNamespace(scene_token="current", owner_id="owner")
    proof = {"scene_token": "current"}

    def query(goal, gripper):
        assert runtime.gpu_lock.locked()
        calls.append("height_query")
        return proof

    backend.resolve_release_height = query
    method = getattr(runtime, "resolve_release_height_scoped", None)
    assert callable(method), "runtime needs the scoped read-only release-height query"
    result = method(request(), binding, {"A": .4, "B": 0.}, [[0., 0., 0., .01]])
    assert result == proof
    assert calls == ["prepare", "refresh", "peer", "height_query", "unchanged"]
    assert not runtime.gpu_lock.locked()


def test_height_response_cannot_move_both_reported_initial_and_selected_xy():
    from test_demo_planning import placement_inputs

    observation, _, holding, placement = placement_inputs()
    goal = demo_placement.fixed_release_goal(observation["objects"][-1], holding, placement,
        holding["attachment_evidence"]["attachment_spheres_tool_m"], [[0., 0., .2, .02]],
        clearance_m=.01, observation_id="capture-4")
    matrix = np.asarray(goal["T_world_tcp"])
    planned = request(goal_position_m=matrix[:3, 3].tolist(), goal_quaternion_wxyz=[0., 1., 0., 0.])
    proof = resolve(holding["attachment_evidence"]["attachment_spheres_tool_m"],
        [[0., 0., .2, .02]], planned, lambda spheres: np.zeros(spheres.shape[:-1]))
    proof["initial_goal_position_m"][0] += .01
    proof["selected_goal_position_m"][0] += .01
    with pytest.raises(ValueError, match="release_height_invalid_response"):
        demo_placement.apply_release_height(goal, proof, scene_token="scene-current")
