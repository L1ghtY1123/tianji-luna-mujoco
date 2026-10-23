"""A complete deterministic planner exhaustion may spend the next frozen pool."""
import asyncio
from copy import deepcopy

import pytest
from test_demo_sampling import sampling_case
from test_planning_batch import batch_response, evaluated

from luna_mujoco.algorithms import AlgorithmError


def exhausted(evaluation):
    indices = [item["candidate_index"] for item in evaluation["valid_candidates"]]
    proof = {key: deepcopy(evaluation[key]) for key in (
        "arm", "target_id", "generation", "scene_token", "planning_settings",
        "tool_profile_revision", "sampling_round", "start_joints_deg", "other_joints_deg")}
    proof.update(outcome="determinate_planner_failure", candidate_indices=indices,
                 planned_candidate_count=len(indices), unplanned_candidate_indices=[])
    return AlgorithmError("pool failed", code="planning_no_valid_candidate", details={
        "arm": evaluation["arm"], "pool_exhaustion": proof,
        "failures": [{"candidate_index": index, "stage": "planner",
            "status": "Grasp planning completed.", "error": None,
            "phase_success": {"goalset": True, "approach": True, "grasp": True, "lift": False}}
            for index in indices]})


def plan_results(runner, original, outcomes):
    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.plan_grasp:
            outcome = outcomes.pop(0)
            if outcome == "exhausted":
                raise exhausted(args[3]["A"])
            if isinstance(outcome, BaseException):
                raise outcome
        return result
    runner.call_algorithm = call


async def test_zero_ik_then_failed_plan_uses_third_pool_with_one_deadline(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [False, True, True])
    plan_results(runner, original, ["exhausted", "success"])
    result = await runner.plan_grasp(snapshot, "source", "A", context)
    assert result["sampling_round"] == 3
    assert [call[0] for call in calls] == ["evaluate_arms", "generate_grasps", "evaluate_arms",
        "plan_grasp", "generate_grasps", "evaluate_arms", "plan_grasp"]
    assert [call[2]["timeout_s"] for call in calls] == [180., 175., 170., 165., 160., 155., 150.]
    evidence = result["sampling_evidence"]
    assert evidence["total_requested_candidate_count"] == 1768
    assert evidence["trajectory_attempt_count"] == 2
    assert [row["outcome"] for row in evidence["rounds"]] == [
        "no_eligible_source_ik", "determinate_planner_failure", "planned"]
    assert len(runner.state["planning_attempts"]) == 1
    assert runner.state["planning_attempts"][0]["status"] == "succeeded"


async def test_three_failed_planner_pools_stop_without_a_fourth(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True] * 3)
    plan_results(runner, original, ["exhausted"] * 3)
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert caught.value.code == "planning_no_valid_candidate"
    evidence = caught.value.details["sampling_evidence"]
    assert evidence["total_requested_candidate_count"] == 1768
    assert evidence["trajectory_attempt_count"] == 3
    assert [call[0] for call in calls].count("generate_grasps") == 2
    assert len(caught.value.details["failures"]) == 1


@pytest.mark.parametrize("fault", ["missing", "partial", "wrong_round", "wrong_scene", "wrong_settings",
    "duplicate", "unknown", "service", "hard_check", "command_validation", "endpoint_rejected",
    "wrong_count", "wrong_candidate", "all_true", "invalid_phase_order", "unknown_status"])
async def test_incomplete_or_nondeterminate_failure_never_resamples(tmp_path, monkeypatch, fault):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True])
    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.plan_grasp:
            error = exhausted(args[3]["A"])
            proof, failure = error.details["pool_exhaustion"], error.details["failures"][0]
            if fault == "missing":
                error.details.pop("pool_exhaustion")
            elif fault == "partial":
                proof["unplanned_candidate_indices"] = [17]
            elif fault == "wrong_round":
                proof["sampling_round"] = 9
            elif fault == "wrong_scene":
                proof["scene_token"] = "old"
            elif fault == "wrong_settings":
                proof["planning_settings"]["lift_offset_m"] += .01
            elif fault == "duplicate":
                error.details["failures"].append(deepcopy(failure))
            elif fault == "wrong_count":
                proof["planned_candidate_count"] = 0
            elif fault == "wrong_candidate":
                proof["candidate_indices"] = [18]
            elif fault == "all_true":
                failure["phase_success"]["lift"] = True
            elif fault == "invalid_phase_order":
                failure["phase_success"]["goalset"] = False
            elif fault == "unknown_status":
                failure["status"] = "unknown"
            elif fault == "unknown":
                failure["phase_success"]["lift"] = None
            elif fault == "service":
                failure["error"] = "service failed"
            elif fault == "hard_check":
                failure["status"] = "CUROBO_TRAJECTORY_COLLISION: clearance"
            else:
                failure["stage"] = fault
            raise error
        return result
    runner.call_algorithm = call
    with pytest.raises(AlgorithmError):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert [call[0] for call in calls] == ["evaluate_arms", "plan_grasp"]


@pytest.mark.parametrize("stop", ["timeout", "scene", "cancel"])
async def test_no_next_pool_after_stop_during_planning(tmp_path, monkeypatch, stop):
    runner, snapshot, context, calls, clock, original = sampling_case(tmp_path, monkeypatch, [True])
    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.plan_grasp:
            if stop == "timeout":
                clock[0] += 180.
            elif stop == "scene":
                context["scene"]["scene_token"] = "changed"
            else:
                raise asyncio.CancelledError()
            raise exhausted(args[3]["A"])
        return result
    runner.call_algorithm = call
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert [call[0] for call in calls] == ["evaluate_arms", "plan_grasp"]


def test_algorithm_reports_all_candidates_and_frozen_binding_on_exhaustion():
    algorithms, snapshot, scene, evaluations, _ = evaluated(35)
    evaluations["A"]["sampling_round"] = 2
    def request(path, body):
        response = batch_response(body, successful=[])
        for item in response["items"]:
            item["status"] = "Grasp planning completed."
        return response
    algorithms.request = request
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(snapshot, "red", "A", evaluations, scene)
    proof = caught.value.details["pool_exhaustion"]
    assert proof["outcome"] == "determinate_planner_failure"
    assert proof["sampling_round"] == 2
    assert proof["planned_candidate_count"] == 35
    assert proof["candidate_indices"] == list(range(134, 99, -1))
    assert proof["unplanned_candidate_indices"] == []
    assert proof["scene_token"] == "scene-1"
    assert all(row["stage"] == "planner" for row in caught.value.details["failures"])


async def test_final_empty_ik_pool_preserves_earlier_planner_failure(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True, False, False])
    plan_results(runner, original, ["exhausted"])
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert caught.value.code == "planning_no_valid_candidate"
    evidence = caught.value.details["sampling_evidence"]
    assert evidence["trajectory_attempt_count"] == 1
    assert evidence["total_requested_candidate_count"] == 1768
    assert evidence["rounds"][-1]["outcome"] == "no_eligible_source_ik"
    assert [call[0] for call in calls].count("plan_grasp") == 1


async def test_stale_evaluation_round_cannot_be_relabelled_before_planning(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True])
    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.evaluate_arms:
            result["A"]["sampling_round"] = 9
        return result
    runner.call_algorithm = call
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert caught.value.code == "planning_stale_evaluation"
    assert [call[0] for call in calls] == ["evaluate_arms"]


async def test_generated_pool_must_have_the_requested_round_before_ik(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [False, True])
    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.generate_grasps:
            result["sampling_round"] = 99
        return result
    runner.call_algorithm = call
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert caught.value.code == "planning_stale_evaluation"
    assert [call[0] for call in calls] == ["evaluate_arms", "generate_grasps"]


@pytest.mark.parametrize("failure", ["hard_check", "endpoint_rejected", "command_validation", "service"])
def test_algorithm_never_certifies_validation_or_service_failure_as_pool_exhaustion(failure):
    algorithms, snapshot, scene, evaluations, _ = evaluated(2)
    def request(path, body):
        result = batch_response(body, successful=[] if failure in {"hard_check", "service"} else None)
        for item in result["items"]:
            item["status"] = "Grasp planning completed."
            if failure == "hard_check":
                item["status"] = "CUROBO_TRAJECTORY_COLLISION: clearance"
            elif failure == "service":
                item["error"] = {"code": "unavailable"}
        return result
    algorithms.request = request
    def reject(*args, **kwargs):
        raise AlgorithmError("rejected", code="planning_grasp_endpoint_rejected"
                             if failure == "endpoint_rejected" else "execution_collision")
    if failure == "endpoint_rejected":
        algorithms._grasp_plan_evidence = reject
    elif failure == "command_validation":
        algorithms.validate_execution = reject
    with pytest.raises(AlgorithmError) as caught:
        algorithms.plan_grasp(snapshot, "red", "A", evaluations, scene)
    assert caught.value.details["pool_exhaustion"]["outcome"] == "not_retryable"
