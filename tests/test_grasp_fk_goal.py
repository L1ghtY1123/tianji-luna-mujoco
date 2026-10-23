"""IK's accepted configuration, rather than its nearby nominal goal, is planned."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_backend_builder import native_batch
from test_rgbd_algorithms import observed_scene

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.backend_builder import ValidatedBatchBackend
from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.geometry import TOOL_PROFILE
from luna_mujoco.planning_settings import PlanningSettings


@pytest.fixture(autouse=True)
def isolate_pose_binding_from_gripper_quality(monkeypatch):
    # These synthetic poses/clouds test FK binding and tolerance composition.
    # Full own-model quality and both frozen clouds are covered separately.
    monkeypatch.setattr(DemoAlgorithms, "_gripper_quality", lambda *args: {
        "accepted": True, "reason": "accepted"})


def pose(x=.5):
    result = np.diag([1., -1., -1., 1.])
    result[:3, 3] = [x, .1, .8]
    return result


def fk_evidence(matrix, arm="A", joints=None, token="map-1"):
    side = "L" if arm == "A" else "R"
    return {"source": "curobo_public_compute_kinematics", "frame": "world",
            "arm": arm, "tool_frame": f"TCP_Link_{side}", "scene_token": token,
            "joint_names": [f"Joint{i}_{side}" for i in range(1, 8)],
            "joints_rad": joints or [0.] * 7, "T_world_tcp": matrix.tolist()}


def inputs(arm="A", achieved=None):
    nominal = pose()
    achieved = pose(.503) if achieved is None else achieved
    snapshot = observed_scene()
    scene = {"target_id": "seen-red", "generation": 1,
             "observation_id": "capture-4", "scene_token": "map-1"}
    grasps = {"arm": arm, "tool_profile_revision": TOOL_PROFILE["revision"],
              "sampling_round": 2, "T_world_cloud": np.eye(4).tolist(),
              "candidates": [{"index": 981, "score": .8, "source_batch": 10,
                  "source_index": 81, "T_object_grasp": nominal.tolist(),
                  "T_world_tcp": nominal.tolist()}]}
    item = {"candidate_index": 0, "success": True, "joints_rad": [0.] * 7,
            "joint_names": fk_evidence(achieved, arm)["joint_names"],
            "position_error_m": .003, "rotation_error_rad": 0.,
            "achieved_tool_pose": fk_evidence(achieved, arm)}
    return snapshot, scene, grasps, item


@pytest.mark.parametrize("arm", ["A", "B"])
def test_accepted_source_fk_replaces_only_planning_goal_and_preserves_nominal(arm):
    snapshot, scene, grasps, item = inputs(arm)
    original = deepcopy(grasps)
    algorithms = DemoAlgorithms(curobo_request=lambda *args: {
        "scene_token": "map-1", "batch_size": 1, "items": [item]})
    result = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm=arm)[arm]
    candidate = result["valid_candidates"][0]
    np.testing.assert_allclose(candidate["T_world_tcp"], pose(.503))
    np.testing.assert_allclose(candidate["nominal_T_world_tcp"], pose())
    assert candidate["ik"] == item
    assert candidate["candidate_index"] == 981 and candidate["sampling_round"] == 2
    assert candidate["source_batch"] == 10 and candidate["source_index"] == 81
    assert candidate["fk_goal_validation"]["position_error_m"] == pytest.approx(.003)
    assert grasps == original  # Insertion is not applied for a second time.


@pytest.mark.parametrize("fault", ["missing", "token", "arm", "frame", "q"])
def test_missing_or_mismatched_fk_evidence_fails_instead_of_reusing_old_evaluation(fault):
    snapshot, scene, grasps, item = inputs()
    if fault == "missing":
        item.pop("achieved_tool_pose")
    else:
        key, value = {"token": ("scene_token", "old"), "arm": ("arm", "B"),
                      "frame": ("frame", "tool"), "q": ("joints_rad", [.1] * 7)}[fault]
        item["achieved_tool_pose"][key] = value
    algorithms = DemoAlgorithms(curobo_request=lambda *args: {
        "scene_token": "map-1", "batch_size": 1, "items": [item]})
    with pytest.raises(AlgorithmError, match="FK"):
        algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm="A")


@pytest.mark.parametrize("fault", ["position", "rotation", "toward", "tilt"])
def test_achieved_pose_failure_is_explicit_candidate_rejection(fault):
    achieved = pose()
    snapshot, scene, grasps, item = inputs(achieved=achieved)
    if fault == "position":
        achieved[0, 3] += .00501
    else:
        degrees = {"rotation": 3., "toward": 1., "tilt": -45.1}[fault]
        achieved[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix() @ achieved[:3, :3]
        if fault == "tilt":
            nominal = pose()
            nominal[:3, :3] = Rotation.from_euler("y", -44.9, degrees=True).as_matrix() @ nominal[:3, :3]
            grasps["candidates"][0]["T_world_tcp"] = nominal.tolist()
    item["achieved_tool_pose"] = fk_evidence(achieved)
    algorithms = DemoAlgorithms(curobo_request=lambda *args: {
        "scene_token": "map-1", "batch_size": 1, "items": [item]})
    result = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm="A")["A"]
    assert result["valid_candidates"] == []
    evidence = result["batches"][0]["candidate_filters"][0]
    assert evidence["accepted"] is False
    assert evidence["candidate_index"] == 981
    assert evidence["reason"] != "unknown"
    assert result["batches"][0]["success_count"] == 1
    assert result["batches"][0]["eligible_count"] == 0


def test_final_grasp_pose_cannot_stack_a_second_position_tolerance():
    snapshot, scene, grasps, item = inputs()
    algorithms = DemoAlgorithms(curobo_request=lambda *args: {
        "scene_token": "map-1", "batch_size": 1, "items": [item]})
    candidate = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene,
        requested_arm="A")["A"]["valid_candidates"][0]
    phases = {"grasp": {"positions_rad": [[0.] * 7, [.1] * 7]}}
    response = {"grasp_endpoint_fk": fk_evidence(pose(.506), joints=[.1] * 7)}
    with pytest.raises(AlgorithmError) as error:
        algorithms._grasp_plan_evidence(candidate, response, phases, snapshot,
            "seen-red", "A", scene, PlanningSettings().model_dump())
    assert error.value.code == "planning_grasp_endpoint_rejected"
    assert error.value.details["position_error_m"] == pytest.approx(.006)
    response["grasp_endpoint_fk"] = fk_evidence(pose(.504), joints=[.1] * 7)
    evidence = algorithms._grasp_plan_evidence(candidate, response, phases, snapshot,
        "seen-red", "A", scene, PlanningSettings().model_dump())
    assert evidence["final_grasp_validation"]["position_error_m"] == pytest.approx(.004)


def test_final_grasp_rotation_is_measured_from_original_nominal_not_source_fk():
    source = pose()
    source[:3, :3] = Rotation.from_euler("z", .04).as_matrix() @ source[:3, :3]
    snapshot, scene, grasps, item = inputs(achieved=source)
    algorithms = DemoAlgorithms(curobo_request=lambda *args: {
        "scene_token": "map-1", "batch_size": 1, "items": [item]})
    candidate = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene,
        requested_arm="A")["A"]["valid_candidates"][0]
    final = pose()
    final[:3, :3] = Rotation.from_euler("z", .08).as_matrix() @ final[:3, :3]
    phases = {"grasp": {"positions_rad": [[0.] * 7, [.1] * 7]}}
    with pytest.raises(AlgorithmError) as error:
        algorithms._grasp_plan_evidence(candidate,
            {"grasp_endpoint_fk": fk_evidence(final, joints=[.1] * 7)}, phases,
            snapshot, "seen-red", "A", scene, PlanningSettings().model_dump())
    assert error.value.details["rotation_error_rad"] == pytest.approx(.08)


@pytest.mark.parametrize("arm", ["A", "B"])
def test_achieved_pose_exact_45_degree_away_boundary_and_current_tolerance(arm):
    matrix = pose()
    matrix[:3, :3] = Rotation.from_euler("y", -45., degrees=True).as_matrix() @ matrix[:3, :3]
    snapshot, scene, grasps, item = inputs(arm, achieved=matrix)
    grasps["candidates"][0]["T_world_tcp"] = matrix.tolist()
    algorithms = DemoAlgorithms(curobo_request=lambda *args: {
        "scene_token": "map-1", "batch_size": 1, "items": [item]})
    result = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm=arm)[arm]
    assert len(result["valid_candidates"]) == 1
    item["achieved_tool_pose"]["T_world_tcp"][0][3] += .003
    result = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene,
        requested_arm=arm, planning_settings={"ik_position_threshold_m": .002})[arm]
    assert result["valid_candidates"] == []
    assert result["batches"][0]["candidate_filters"][0]["position_threshold_m"] == .002


def test_backend_fk_uses_selected_ik_solution_and_serializes_binding(monkeypatch):
    from tianji_curobo._curobo_impl import _NativeCuroboBackend
    from tianji_curobo.contracts import NativeIkBatchResult, NativeIkResult
    names = tuple(f"Joint{i}_L" for i in range(1, 8))
    native = NativeIkBatchResult((NativeIkResult(True, (.2,) * 7, names, .003, .01),
                                  NativeIkResult(False, None, names, 1., 1.)), "map-1", .1)
    monkeypatch.setattr(_NativeCuroboBackend, "solve_ik_batch_native", lambda self, request: native)
    backend = object.__new__(ValidatedBatchBackend)
    backend.arm, backend.tool_frame, backend.joint_names = "A", "TCP_Link_L", names
    checked = []
    backend._state = lambda degrees: np.deg2rad(degrees)
    class Tensor:
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return pose(.503)[None, None]
    def compute(state):
        checked.append(state)
        return SimpleNamespace(tool_poses=SimpleNamespace(get_link_pose=lambda name:
            SimpleNamespace(get_matrix=lambda: Tensor())))
    backend.motion_planner = SimpleNamespace(compute_kinematics=compute)
    result = backend.solve_ik_batch_native(SimpleNamespace(scene_token="map-1"))
    # Runtime replaces the token after its unchanged-scene check.
    payload = replace(result, scene_token="map-1").to_payload()
    assert len(checked) == 1
    np.testing.assert_allclose(checked[0], [.2] * 7)
    assert payload["items"][0]["achieved_tool_pose"] == fk_evidence(pose(.503), joints=[.2] * 7)
    assert payload["items"][1].get("achieved_tool_pose") is None


def test_backend_preserves_final_fk_through_trajectory_derivatives_and_hard_check(monkeypatch):
    from tianji_curobo._curobo_impl import _NativeCuroboBackend

    from luna_mujoco import backend_builder
    native = native_batch()
    monkeypatch.setattr(_NativeCuroboBackend, "plan_grasp_batch_native", lambda self, request: native)
    monkeypatch.setattr(backend_builder, "add_trajectory_derivatives", lambda result, raw: result)
    backend = object.__new__(ValidatedBatchBackend)
    backend.motion_planner = SimpleNamespace(last_grasp_result=None)
    backend.arm = 'A'
    backend.tool_fk_evidence = lambda joints, token: fk_evidence(pose(.504), joints=list(joints), token=token)
    backend.validate_trajectory_native = lambda request: None
    result = backend.plan_grasp_batch_native(SimpleNamespace(scene_token="scene"))
    assert result.items[0].grasp_endpoint_fk["joints_rad"] == [.1] * 7
    assert result.items[0].grasp_endpoint_fk["scene_token"] == "scene"


def test_original_runtime_and_service_preserve_fk_at_both_api_boundaries(monkeypatch):
    from tianji_curobo._curobo_impl import _NativeCuroboBackend
    from tianji_curobo.contracts import NativeIkBatchResult, NativeIkResult

    from luna_mujoco import backend_builder
    from luna_mujoco.geometry_service import GeometryRuntime, run_operation

    names = tuple(f"Joint{i}_L" for i in range(1, 8))
    ik = NativeIkResult(True, (.2,) * 7, names, .003, .01)
    monkeypatch.setattr(_NativeCuroboBackend, "solve_ik_batch_native", lambda self, request:
        NativeIkBatchResult((ik,) * len(request.goal_matrices_world), "scene", .1))
    def plan(self, request):
        native = native_batch()
        count = len(request.goal_matrices_world)
        return replace(native, items=tuple(replace(native.items[0], candidate_index=i)
                       for i in range(count)), batch_size=count, batch_capacity=count)
    monkeypatch.setattr(_NativeCuroboBackend, "plan_grasp_batch_native", plan)
    monkeypatch.setattr(backend_builder, "add_trajectory_derivatives", lambda result, raw: result)
    backend = object.__new__(ValidatedBatchBackend)
    backend.motion_planner = SimpleNamespace(last_grasp_result=None)
    backend.arm = 'A'
    backend.tool_fk_evidence = lambda joints, token: fk_evidence(pose(.504), joints=list(joints), token=token)
    backend.validate_trajectory_native = lambda request: None
    backend.update_peer_collision_model = lambda *args: None
    runtime = GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene"}
    runtime.geometry_solver_profile = PlanningSettings().solver_payload()
    runtime.gripper_feedback = {'A': 0., 'B': 0.}
    with runtime.carrying_scope(None, runtime.gripper_feedback):
        pass
    runtime.effective_config = {"A": {"ik": {}, "plan": {}}}
    monkeypatch.setattr(runtime, "_prepare_scene_cache_unlocked", dict)
    monkeypatch.setattr(runtime, "_refresh_scene_unlocked", lambda **kwargs: "scene")
    monkeypatch.setattr(runtime, "_require_scene_token_unchanged_unlocked", lambda token: None)
    monkeypatch.setattr(runtime, "_create_real_backend_unlocked", lambda *args, **kwargs: backend)
    monkeypatch.setattr(runtime, "_batch_backend_unlocked", lambda *args: backend)
    payload = {"arm": "A", "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
               "gripper_joint_positions_rad": {'A': 0., 'B': 0.},
               "goal_matrices_world": [pose().tolist()], "scene_token": "scene",
               "planning_settings": PlanningSettings().model_dump()}
    ik_payload = run_operation(runtime, "/tools/solve_ik_batch", payload)
    assert len(ik_payload["items"]) == 1
    assert ik_payload["items"][0]["achieved_tool_pose"]["joints_rad"] == [.2] * 7
    planned = run_operation(runtime, "/tools/plan_grasp_batch", payload)
    assert len(planned["items"]) == 1
    assert planned["items"][0]["grasp_endpoint_fk"]["joints_rad"] == [.1] * 7
    assert planned["items"][0]["grasp_endpoint_fk"]["scene_token"] == "scene"


def test_demo_planning_uses_one_fk_goal_for_lift_request_contact_and_return(monkeypatch):
    from test_planning_batch import batch_response

    from luna_mujoco import demo_algorithms

    snapshot, scene, grasps, ik = inputs()
    algorithms = DemoAlgorithms(curobo_request=lambda *args: {
        "scene_token": "map-1", "batch_size": 1, "items": [ik]})
    evaluations = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm="A")
    captured = {}
    def lift(snapshot, target, arm, candidates, scene, settings):
        captured["lift_goal"] = deepcopy(candidates[0]["T_world_tcp"])
        return settings, {"derived": True}
    monkeypatch.setattr(demo_algorithms, "source_clearance_lift", lift)
    def request(path, payload):
        captured["request"] = payload
        result = batch_response(payload)
        phase = result["items"][0]["phases"]["grasp"]
        result["items"][0]["grasp_endpoint_fk"] = fk_evidence(
            pose(.504), joints=phase["positions_rad"][-1][::-1])
        return result
    algorithms.request = request
    contacts = []
    algorithms.validate_execution = lambda *args, **kwargs: contacts.append(kwargs["contact_goal_pose_world"]) or {}
    result = algorithms.plan_grasp(snapshot, "seen-red", "A", evaluations, scene)
    for goal in [captured["lift_goal"], captured["request"]["goal_matrices_world"][0],
                 *contacts, result["T_world_tcp"]]:
        np.testing.assert_array_equal(goal, pose(.503))
    assert result["nominal_T_world_tcp"] == pose().tolist()
    assert result["source_index"] == 81 and result["sampling_round"] == 2
    assert result["final_grasp_validation"]["position_error_m"] == pytest.approx(.004)
    evaluations["A"]["valid_candidates"][0].pop("nominal_T_world_tcp")
    with pytest.raises(AlgorithmError, match="missing.*FK"):
        algorithms.plan_grasp(snapshot, "seen-red", "A", evaluations, scene)
