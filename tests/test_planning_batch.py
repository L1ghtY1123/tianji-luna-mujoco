import threading
from copy import deepcopy

import numpy as np
import pytest

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.planning_settings import PlanningSettings


def inputs(count=2):
    state = {"generation": 1, "arms": {arm: {"joints_rad": [0.] * 7, 'gripper_position_rad': 0.} for arm in ("A", "B")},
             "objects": [{"id": "red", "points_world_m": np.array([
                 [x, y, 0.] for x in np.linspace(-.02, .02, 5) for y in np.linspace(-.02, .02, 5)])}]}
    scene = {"generation": 1, "target_id": "red", "scene_token": "scene-1"}
    grasps = {"candidates": [{"index": index + 100, "score": index / max(count, 1),
                               "T_object_grasp": np.eye(4).tolist()} for index in range(count)]}
    return state, scene, grasps


def trajectory(arm="A", reverse=False):
    names = [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)]
    points = np.array([[0.] * 7, [.01 * i for i in range(1, 8)]])
    if reverse:
        names, points = names[::-1], points[:, ::-1]
    return {"joint_names": names, "positions_rad": points.tolist(), "interpolation_dt_s": .025,
            "velocities_rad_s": np.zeros_like(points).tolist(),
            "accelerations_rad_s2": np.zeros_like(points).tolist(),
            "command_positions_rad": None, "trajectory_point_scale": None}


def batch_response(body, *, successful=None):
    count = len(body["goal_matrices_world"])
    successful = set(range(count)) if successful is None else set(successful)
    result = {"ok": True, "scene_token": body["scene_token"], "batch_size": count,
            "batch_capacity": 16, "items": [
                {"candidate_index": i, "success": i in successful, "goalset_success": i in successful,
                 "approach_success": i in successful, "grasp_success": i in successful,
                 "lift_success": i in successful, "status": "SUCCESS" if i in successful else "IK_FAIL",
                 "phases": {phase: trajectory(body["arm"], reverse=True)
                            if i in successful else None
                            for phase in ("approach", "grasp", "lift")}} for i in range(count)]}
    start = np.deg2rad(body["start_joints_deg"])[::-1]
    step = np.arange(7, 0, -1) * .01
    for item in result["items"]:
        if item["success"]:
            for index, name in enumerate(("approach", "grasp", "lift")):
                phase = item["phases"][name]
                phase["positions_rad"] = (np.asarray(phase["positions_rad"]) + start + index * step).tolist()
    return result


def evaluated(count=2, *, scores=None, settings=None):
    state, scene, grasps = inputs(count)
    if scores is not None:
        for candidate, score in zip(grasps["candidates"], scores):
            candidate["score"] = score
    calls = []
    def ik(path, body):
        calls.append((path, deepcopy(body)))
        return {"ok": True, "scene_token": body["scene_token"], "batch_size": len(body["goal_matrices_world"]),
                "items": [{"candidate_index": i, "success": True}
                          for i in range(len(body["goal_matrices_world"]))]}
    algorithms = Algorithms(curobo_request=ik)
    # Selection tests replace the external collision boundary; dedicated tests
    # exercise validate_execution's real payload and response-proof checks.
    algorithms.validate_execution = lambda *args, **kwargs: {"collision_free": True}
    evaluations = algorithms.evaluate_arms(state, "red", grasps, scene, planning_settings=settings)
    return algorithms, state, scene, evaluations, calls


def test_ik_batches_preserve_global_indices_and_shared_settings():
    settings = PlanningSettings(trajectory_collision_clearance_m=.02, place_release_height_m=.025).model_dump()
    _, _, _, result, calls = evaluated(205, settings=settings)
    assert [len(body["goal_matrices_world"]) for _, body in calls] == [100, 100, 5] * 2
    for arm in ("A", "B"):
        assert [item["candidate_index"] for item in result[arm]["valid_candidates"]] == list(range(100, 305))
        assert result[arm]["planning_settings"] == settings
        assert result[arm]["arm"] == arm
        assert result[arm]["start_joints_deg"] == [0.] * 7
    assert all(body["planning_settings"] == settings for _, body in calls)
    assert all(body["trajectory_collision_clearance_m"] == .02 for _, body in calls)


def test_plan_stops_at_first_feasible_batch_and_uses_score_only_for_equal_travel():
    algorithms, state, scene, evaluations, ik_calls = evaluated(35)
    for index, candidate in enumerate(evaluations["A"]["valid_candidates"]):
        candidate["T_world_tcp"][0][3] = index / 100
    original = deepcopy(evaluations)
    calls, events = [], []
    def request(path, body):
        calls.append((path, deepcopy(body)))
        return batch_response(body)
    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene,
                                   progress=lambda *event: events.append(event))
    assert [path for path, _ in calls] == ["/tools/plan_grasp_batch"]
    assert [len(body["goal_matrices_world"]) for _, body in calls] == [16]
    assert [pose[0][3] for pose in calls[0][1]["goal_matrices_world"]] == [i / 100 for i in range(34, 18, -1)]
    np.testing.assert_equal(evaluations, original)
    assert result["candidate_index"] == 134
    assert result["score"] == pytest.approx(34 / 35)
    assert all(body["planning_settings"] == ik_calls[0][1]["planning_settings"] for _, body in calls)
    assert result["time_scale"] == 2
    assert result["raw_duration_s"] == pytest.approx(.075)
    assert result["command_duration_s"] == pytest.approx(.15)
    for index, phase in enumerate(result["phases"].values()):
        assert phase["raw_dt"] == .025
        assert phase["dt"] == .025
        expected = (np.array([0., .5, 1.])[:, None] + index) * np.arange(1, 8)[None, :] * .01
        np.testing.assert_allclose(phase["positions_rad"], expected)
        assert len(phase["positions_rad"]) == 3
    assert result["time_stretch_method"] == "quintic_hermite_point_insertion"
    assert [kind for kind, _, _ in events] == ["planning_batch_start", "planning_batch_replay", "planning_batch_complete"]
    assert events[-1][2]["candidate_indices"] == list(range(134, 118, -1))
    assert events[-1][2]["success_count"] == 16
    assert events[-1][2]["stopped_early"] is True
    assert events[-1][2]["unplanned_candidate_count"] == 19
    assert result["planned_candidate_count"] == 16
    assert result["unplanned_candidate_indices"] == list(range(118, 99, -1))
    assert result["selection"]["raw_complete_candidate_count"] == 16
    assert result["prior_failures"] == []


def test_batch_choice_ignores_higher_score_failed_candidates_and_keeps_ties_stable():
    algorithms, state, scene, evaluations, _ = evaluated(3, scores=[.8, .9, .8])
    evaluations["A"]["valid_candidates"][0]["candidate_index"] = 999
    algorithms.request = lambda path, body: batch_response(body, successful=[1, 2])
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 999
    assert result["prior_failures"][0]["candidate_index"] == 101
    assert result["unplanned_candidate_indices"] == []
    assert result["batches"][0]["stopped_early"] is False


def test_failed_batch_continues_and_preserves_selected_raw_derivatives():
    algorithms, state, scene, evaluations, _ = evaluated(35)
    batch_count = 0
    def request(path, body):
        nonlocal batch_count
        batch_count += 1
        response = batch_response(body, successful=[] if batch_count == 1 else None)
        if batch_count == 2:
            for phase in response["items"][0]["phases"].values():
                phase["velocities_rad_s"][0][-1] = .4
        return response
    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 118
    assert batch_count == 2
    assert result["planned_candidate_count"] == 32
    assert result["unplanned_candidate_indices"] == [102, 101, 100]
    assert [item["candidate_index"] for item in result["prior_failures"]] == list(range(134, 118, -1))
    assert result["batches"][0]["stopped_early"] is False
    # At u=1/2 for the default 2x stretch, q term=.005, v term=.0015625.
    assert result["phases"]["approach"]["positions_rad"][1][0] == pytest.approx(.0065625)


def test_plan_submits_only_ik_survivors_with_identical_frozen_joints_and_poses():
    state, scene, grasps = inputs(3)
    for i, candidate in enumerate(grasps["candidates"]):
        candidate["T_object_grasp"][0][3] = i / 100
    state["arms"]["A"]["joints_rad"] = [.1] * 7
    state["arms"]["B"]["joints_rad"] = [-.2] * 7
    ik_calls, plan_calls = [], []
    def request(path, body):
        if path.endswith("solve_ik_batch"):
            ik_calls.append(deepcopy(body))
            return {"scene_token": "scene-1", "batch_size": 3,
                    "items": [{"candidate_index": i, "success": i != 1} for i in range(3)]}
        plan_calls.append(deepcopy(body))
        return batch_response(body)
    algorithms = Algorithms(curobo_request=request)
    algorithms.validate_execution = lambda *args, **kwargs: {"collision_free": True}
    evaluations = algorithms.evaluate_arms(state, "red", grasps, scene)
    state["arms"]["A"]["joints_rad"][0] += .001  # harmless holding-controller drift
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert plan_calls[0]["goal_matrices_world"] == [ik_calls[0]["goal_matrices_world"][i] for i in (2, 0)]
    for field in ("start_joints_deg", "other_joints_deg", "planning_settings", "scene_token"):
        assert plan_calls[0][field] == ik_calls[0][field]
    assert result["candidate_index"] == 102
    assert result["joint_drift_threshold_deg"] == 1.
    assert result["joint_drift_deg"]["A"] == pytest.approx(np.rad2deg(.001))


def test_equal_score_candidates_keep_original_order_across_batch_boundary():
    algorithms, state, scene, evaluations, _ = evaluated(18, scores=[.8] * 18)
    candidates = evaluations["A"]["valid_candidates"]
    candidates[15]["candidate_index"] = 999
    calls = []
    def request(path, body):
        calls.append(body)
        return batch_response(body, successful=[15])
    algorithms.request = request
    result = algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert len(calls) == 1
    assert result["candidate_index"] == 999
    assert result["unplanned_candidate_indices"] == [116, 117]
    assert len(result["prior_failures"]) == 15


def test_all_failed_batches_report_every_failure_without_interpolating(monkeypatch):
    from luna_mujoco import algorithms as algorithms_module

    def reject_interpolation(*args):
        raise AssertionError("No executable trajectory exists")
    monkeypatch.setattr(algorithms_module, "build_execution_phases", reject_interpolation)
    algorithms, state, scene, evaluations, _ = evaluated(35)
    calls = []
    def request(path, body):
        calls.append(body)
        return batch_response(body, successful=[])
    algorithms.request = request
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(state, "red", "A", evaluations, scene)
    assert caught.value.code == "planning_no_valid_candidate"
    assert [len(body["goal_matrices_world"]) for body in calls] == [16, 16, 3]
    assert [item["candidate_index"] for item in caught.value.details["failures"]] == list(range(134, 99, -1))


@pytest.mark.parametrize("change", ["settings", "arm", "generation", "target", "scene", "joints"])
def test_plan_rejects_changed_ik_binding_before_submitting(change):
    algorithms, state, scene, evaluations, _ = evaluated()
    settings = PlanningSettings().model_dump()
    if change == "settings":
        settings["trajectory_collision_clearance_m"] = .02
        settings['place_release_height_m'] = .025
    elif change == "joints":
        state["arms"]["A"]["joints_rad"][0] = .1
    else:
        field = {"arm": "arm", "generation": "generation", "target": "target_id", "scene": "scene_token"}[change]
        evaluations["A"][field] = "changed"
    calls = []
    algorithms.request = lambda *args: calls.append(args)
    with pytest.raises(AlgorithmError):
        algorithms.plan_grasp(state, "red", "A", evaluations, scene, planning_settings=settings)
    assert calls == []


@pytest.mark.parametrize("malformation", ["partial", "duplicate", "missing", "scene", "dt", "nan", "shape"])
def test_invalid_batch_output_never_produces_an_executable_plan(malformation):
    algorithms, state, scene, evaluations, _ = evaluated()
    def request(path, body):
        result = batch_response(body)
        if malformation == "partial":
            result["items"][1]["lift_success"] = False
        elif malformation == "duplicate":
            result["items"][1]["candidate_index"] = 0
        elif malformation == "missing":
            result["items"].pop()
        elif malformation == "scene":
            result["scene_token"] = "other-scene"
        elif malformation == "dt":
            result["items"][1]["phases"]["lift"]["interpolation_dt_s"] = 0
        elif malformation == "nan":
            result["items"][1]["phases"]["lift"]["positions_rad"][0][0] = float("nan")
        else:
            result["items"][1]["phases"]["lift"]["positions_rad"] = [1, 2]
        return result
    algorithms.request = request
    with pytest.raises(AlgorithmError):
        algorithms.plan_grasp(state, "red", "A", evaluations, scene)


@pytest.mark.parametrize("field", ["velocities_rad_s", "accelerations_rad_s2"])
@pytest.mark.parametrize("malformation", ["missing", "nan", "shape", "count"])
def test_invalid_raw_derivatives_never_produce_an_executable_plan(field, malformation):
    algorithms, state, scene, evaluations, _ = evaluated()

    def request(path, body):
        result = batch_response(body)
        phase = result["items"][1]["phases"]["lift"]
        if malformation == "missing":
            del phase[field]
        elif malformation == "nan":
            phase[field][1][0] = float("nan")
        elif malformation == "shape":
            phase[field] = [1, 2]
        else:
            phase[field].pop()
        return result

    algorithms.request = request
    with pytest.raises(AlgorithmError):
        algorithms.plan_grasp(state, "red", "A", evaluations, scene)


def test_cancelled_after_first_batch_never_submits_second_or_publishes_completion():
    algorithms, state, scene, evaluations, _ = evaluated(20)
    cancelled = threading.Event()
    calls, events = [], []
    def request(path, body):
        calls.append(path)
        cancelled.set()
        return batch_response(body)
    algorithms.request = request
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(state, "red", "A", evaluations, scene, cancelled=cancelled,
                              progress=lambda *event: events.append(event))
    assert caught.value.code == "planning_cancelled"
    assert calls == ["/tools/plan_grasp_batch"]
    assert [kind for kind, _, _ in events] == ["planning_batch_start"]


def test_build_scene_uses_same_settings_for_esdf_padding():
    from test_algorithms import snapshot
    calls = []
    settings = PlanningSettings(trajectory_collision_clearance_m=.08, place_release_height_m=.09,
                                optimizer_collision_activation_distance_m=.3).model_dump()
    algorithms = Algorithms(curobo_request=lambda path, body: calls.append((path, body)) or {"scene_token": "s"})
    algorithms.build_scene(snapshot(), "red", planning_settings=settings)
    assert calls[0][0] == "/perception/build"
    assert calls[0][1]["planning_settings"] == settings


def test_ik_reset_discards_in_flight_batch_and_prevents_remaining_batches_and_peer_arm():
    state, scene, grasps = inputs(205)
    cancelled = threading.Event()
    calls, events = [], []
    def request(path, body):
        calls.append(path)
        cancelled.set()
        return {"scene_token": body["scene_token"], "batch_size": 100,
                "items": [{"candidate_index": i, "success": True} for i in range(100)]}
    algorithms = Algorithms(curobo_request=request)
    with pytest.raises(AlgorithmError) as caught:
        algorithms.evaluate_arms(state, "red", grasps, scene, cancelled=cancelled,
                                 progress=lambda *event: events.append(event))
    assert caught.value.code == "planning_cancelled"
    assert calls == ["/tools/solve_ik_batch"]
    assert [kind for kind, _, _ in events] == ["ik_batch_start"]
