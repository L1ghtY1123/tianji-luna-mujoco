"""Empty source IK and error boundaries within the shared frozen-pool budget."""
import asyncio
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from test_grasp_candidate_pool import GraspClient
from test_rgbd_algorithms import observed_scene
from test_runner import Sim

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.geometry import TOOL_PROFILE
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.runner import Runner


def sampling_case(tmp_path, monkeypatch, outcomes):
    clock = [100.]
    monkeypatch.setattr("luna_mujoco.runner.time.monotonic", lambda: clock[0])
    algorithms = SimpleNamespace(evaluate_arms=object(), generate_grasps=object(), plan_grasp=object())
    robot = Sim()
    runner = DemoRunner(robot, algorithms, None, tmp_path)
    runner.task_progress = SimpleNamespace(completed_ids=[], snapshot=dict)
    runner.state.update(task="grasp_place", arm="A", target_id="source", destination_id="demo-destination",
                        generation=1, planning_attempts=[])
    snapshot = {**robot.robot_snapshot(), "observation_id": "frozen-rgbd"}
    settings = PlanningSettings(observation={"A": [0.] * 7, "B": [0.] * 7}).model_dump()
    grasps = {"sampling_round": 1, "generation": 1, "observation_id": "frozen-rgbd",
              "arm": "A", "model": "fixed-model", "tool_profile_revision": TOOL_PROFILE["revision"],
              "cloud": {"id": "same-cloud"}, "T_world_cloud": np.eye(4).tolist(),
              "sampling_input_digest": "same-scene-and-object", "requested_candidate_count": 256,
              "candidates": [{"index": 17, "score": .7, "source_batch": 2, "source_index": 4}],
              "downward_filter": {"input_count": 256, "accepted_count": 1}}
    context = {"grasps": grasps, "scene": {"scene_token": "frozen-map", "observation_id": "frozen-rgbd"},
               "observation": snapshot, "observation_pose_capture_id": "frozen-rgbd",
               "planning_settings": settings, "grasp_placement_request": {"slot_id": 1}}
    calls = []

    async def call(function, *args, **kwargs):
        name = next(name for name, value in vars(algorithms).items() if value is function)
        calls.append((name, args, copy.deepcopy(kwargs)))
        clock[0] += 5.
        if name == "generate_grasps":
            sampling_round = kwargs["sampling_round"]
            count = {2: 512, 3: 1000}[sampling_round]
            return {**copy.deepcopy(grasps), "sampling_round": sampling_round,
                    "requested_candidate_count": count,
                    "downward_filter": {"input_count": count, "accepted_count": 1}}
        if name == "plan_grasp":
            return {"planned": True, "sampling_round": args[3]["A"]["sampling_round"]}
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        evaluation = {"arm": "A", "target_id": "source", "generation": 1,
                      "scene_token": "frozen-map", "planning_settings": settings,
                      "tool_profile_revision": TOOL_PROFILE["revision"],
                      "destination_id": "demo-destination", "tested_count": 1, "batches": [],
                      "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
                      "sampling_round": args[2]["sampling_round"],
                      "valid_candidates": [{"candidate_index": 17}] if outcome else []}
        return {"A": evaluation}

    runner.call_algorithm = call
    return runner, snapshot, context, calls, clock, call


async def test_zero_ik_draws_one_fresh_pool_under_the_original_deadline(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, _ = sampling_case(tmp_path, monkeypatch, [False, True])
    result = await runner.plan_grasp(snapshot, "source", "A", context)
    assert result["sampling_round"] == 2
    assert [call[0] for call in calls] == ["evaluate_arms", "generate_grasps", "evaluate_arms", "plan_grasp"]
    assert [call[2]["timeout_s"] for call in calls] == [180., 175., 170., 165.]
    assert calls[1][2]["report_progress"] is True
    assert calls[1][2]["placement_request"] == context["grasp_placement_request"]
    assert calls[1][2]["planning_settings"] == context["planning_settings"]
    assert calls[0][1][0] is calls[1][1][0] is calls[2][1][0]
    assert result["sampling_evidence"]["total_requested_candidate_count"] == 768
    assert [row["requested_candidate_count"] for row in result["sampling_evidence"]["rounds"]] == [256, 512]
    assert len(runner.state["planning_attempts"]) == 1


async def test_first_pool_success_never_resamples(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, _ = sampling_case(tmp_path, monkeypatch, [True])
    result = await runner.plan_grasp(snapshot, "source", "A", context)
    assert result["sampling_round"] == 1
    assert [call[0] for call in calls] == ["evaluate_arms", "plan_grasp"]


async def test_empty_transport_prefilter_spends_next_pool_without_fabricating_ik(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True])
    context['grasps']['candidates'] = []
    context['grasps']['fixed_slot_reach_filter'] = {'applied': True,
        'screening_scope': 'nominal_transport_wrist_reach', 'input_count': 1,
        'accepted_count': 0, 'rejected_count': 1,
        'candidate_evidence': [{'candidate_index': 17, 'reach_screening_passed': False}]}

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.generate_grasps:
            result['candidates'] = [{'index': 17, 'score': .7, 'source_batch': 2, 'source_index': 4}]
        return result

    runner.call_algorithm = call
    result = await runner.plan_grasp(snapshot, 'source', 'A', context)
    assert result['sampling_round'] == 2
    assert [c[0] for c in calls] == ['generate_grasps', 'evaluate_arms', 'plan_grasp']
    row = result['sampling_evidence']['rounds'][0]
    assert row['outcome'] == 'no_transport_reachable_candidate'
    assert row['tested_count'] == 0 and row['valid_count'] == 0


@pytest.mark.parametrize('invalid', [False, True])
async def test_empty_transport_pool_budget_and_incomplete_evidence(tmp_path, monkeypatch, invalid):
    runner, snapshot, context, calls, _, _ = sampling_case(tmp_path, monkeypatch, [])
    context['grasps']['candidates'] = []
    context['grasps']['fixed_slot_reach_filter'] = {'applied': True,
        'screening_scope': 'nominal_transport_wrist_reach', 'input_count': 1,
        'accepted_count': 0, 'rejected_count': 1,
        'candidate_evidence': [{'candidate_index': 17, 'reach_screening_passed': False}]}
    if invalid:
        context['grasps']['fixed_slot_reach_filter']['candidate_evidence'] = []
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, 'source', 'A', context)
    if invalid:
        assert not calls
    else:
        assert caught.value.code == 'grasp_no_transport_reachable_candidate'
        assert caught.value.details['total_requested_candidate_count'] == 1768
        assert [c[0] for c in calls] == ['generate_grasps', 'generate_grasps']
        assert caught.value.details['total_tested_count'] == 0


async def test_three_empty_pools_return_counts_without_trajectory_attempt(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, _ = sampling_case(tmp_path, monkeypatch, [False] * 3)
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert caught.value.code == "no_collision_free_source_ik"
    details = caught.value.details
    assert details["total_requested_candidate_count"] == 1768
    assert details["total_tested_count"] == 3
    assert details["trajectory_attempt_count"] == 0
    assert [row["sampling_round"] for row in details["rounds"]] == [1, 2, 3]
    assert [row["requested_candidate_count"] for row in details["rounds"]] == [256, 512, 1000]
    assert [call[0] for call in calls].count("generate_grasps") == 2
    assert all(call[0] != "plan_grasp" for call in calls)


@pytest.mark.parametrize("error", [
    AlgorithmError("service failed"), AlgorithmError("unknown", code="unknown"),
    AlgorithmError("stale", code="planning_stale_evaluation"),
    AlgorithmError("collision", code="execution_collision"), asyncio.CancelledError(),
])
async def test_ik_errors_and_cancellation_do_not_resample(tmp_path, monkeypatch, error):
    runner, snapshot, context, calls, _, _ = sampling_case(tmp_path, monkeypatch, [error])
    with pytest.raises(type(error)):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert [call[0] for call in calls] == ["evaluate_arms"]


@pytest.mark.parametrize("mutation", ["scene", "settings", "generation", "stale_empty", "unknown_empty"])
async def test_changed_or_unknown_inputs_cannot_authorize_another_pool(tmp_path, monkeypatch, mutation):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [False])

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is not runner.algorithms.evaluate_arms:
            return result
        if mutation == "scene":
            context["scene"]["scene_token"] = "different"
        elif mutation == "settings":
            context["planning_settings"]["approach_offset_m"] += .01
        elif mutation == "generation":
            runner.simulation.state["generation"] += 1
        elif mutation == "stale_empty":
            result["A"]["scene_token"] = "stale"
        else:
            result["A"].pop("valid_candidates")
        return result

    runner.call_algorithm = call
    with pytest.raises((AlgorithmError, ValueError)):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert len(calls) == 1


async def test_no_next_call_after_shared_deadline(tmp_path, monkeypatch):
    runner, snapshot, context, calls, clock, original = sampling_case(tmp_path, monkeypatch, [False])

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        clock[0] += 180.
        return result

    runner.call_algorithm = call
    with pytest.raises(RuntimeError, match="plan_grasp_timeout"):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert len(calls) == 1


@pytest.mark.parametrize("field,changed", [
    ("sampling_input_digest", "changed"), ("cloud", {"id": "different-cloud"}),
    ("arm", "B"), ("model", "different-model"), ("observation_id", "different-frame"),
    ("generation", 2), ("tool_profile_revision", "changed"),
    ("T_world_cloud", np.diag([1., 1., -1., 1.]).tolist()),
])
async def test_changed_generated_input_stops_before_second_ik(tmp_path, monkeypatch, field, changed):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [False])

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.generate_grasps:
            result[field] = changed
        return result

    runner.call_algorithm = call
    with pytest.raises(AlgorithmError, match="grasp_sampling_input_changed"):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert [call[0] for call in calls] == ["evaluate_arms", "generate_grasps"]


@pytest.mark.parametrize("count", [256, 1000, 512., True, None])
async def test_unscheduled_retry_budget_stops_before_second_ik(tmp_path, monkeypatch, count):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [False])

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.generate_grasps:
            result["requested_candidate_count"] = count
        return result

    runner.call_algorithm = call
    with pytest.raises(AlgorithmError, match="grasp_sampling_candidate_budget_changed"):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert [call[0] for call in calls] == ["evaluate_arms", "generate_grasps"]


@pytest.mark.parametrize("count", [512, 1000, 256., True, None])
async def test_unscheduled_initial_budget_stops_before_first_ik(tmp_path, monkeypatch, count):
    runner, snapshot, context, calls, _, _ = sampling_case(tmp_path, monkeypatch, [True])
    context["grasps"]["requested_candidate_count"] = count
    with pytest.raises(AlgorithmError, match="grasp_sampling_candidate_budget_changed"):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert not calls


async def test_trajectory_failure_does_not_draw_another_pool(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True])

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.plan_grasp:
            raise AlgorithmError("trajectory failed", code="planning_no_valid_candidate")
        return result

    runner.call_algorithm = call
    with pytest.raises(AlgorithmError, match="trajectory failed"):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert [call[0] for call in calls] == ["evaluate_arms", "plan_grasp"]


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_inflight_sampling_stops_remaining_calls_after_task_cancellation(tmp_path, stop):
    entered, release = threading.Event(), threading.Event()

    class BlockingClient(GraspClient):
        def generate(self, payload):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test did not release inference")
            return super().generate(payload)

    client = BlockingClient()
    algorithms = Algorithms(graspgen_client=client)
    runner = Runner(Sim(), algorithms, None, tmp_path)
    runner.state.update(status="running", current_step={"skill": "plan_grasp"})
    task = asyncio.create_task(runner.call_algorithm(algorithms.generate_grasps,
        observed_scene(), "seen-red", candidate_count=1000, report_progress=True,
        timeout_s=.1 if stop == "timeout" else 2.))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="plan_grasp_timeout"):
                await task
        assert runner.planning_cancelled.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(runner.planning_worker, return_exceptions=True)
    assert len(client.calls) == 1


def test_round_and_original_indices_survive_all_three_replay_stages(monkeypatch):
    from test_demo_planning import GraspClient as DemoClient
    from test_demo_planning import grasp_placement
    from test_grasp_fk_goal import fk_evidence
    from test_planning_batch import batch_response, evaluated

    from luna_mujoco.demo_algorithms import DemoAlgorithms

    # This metadata fixture deliberately has no physical gripper/target geometry.
    monkeypatch.setattr(DemoAlgorithms, "_gripper_quality", lambda *args: {
        "accepted": True, "reason": "accepted"})

    def request(path, payload):
        return {"scene_token": "map-1", "batch_size": len(payload["goal_matrices_world"]),
                "items": [{"candidate_index": index, "success": True,
                           "joints_rad": [0.] * 7,
                           "joint_names": fk_evidence(np.asarray(matrix))["joint_names"],
                           "achieved_tool_pose": fk_evidence(np.asarray(matrix))}
                          for index, matrix in enumerate(payload["goal_matrices_world"])]}

    algorithms = DemoAlgorithms(graspgen_client=DemoClient(), curobo_request=request)
    snapshot = observed_scene()
    grasps = algorithms.generate_grasps(snapshot, "seen-red", arm="A", sampling_round=2,
                                       placement_request=grasp_placement())
    assert grasps["candidate_replay"]["sampling_round"] == 2
    scene = {"target_id": "seen-red", "generation": 1,
             "observation_id": "capture-4", "scene_token": "map-1"}
    evaluation = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm="A")["A"]
    assert evaluation["sampling_round"] == 2
    assert evaluation["batches"][0]["candidate_replay"]["sampling_round"] == 2
    assert evaluation["batches"][0]["candidate_replay"]["candidate_indices"] == list(range(12, 18))
    for original, survivor in zip(grasps["candidates"], evaluation["valid_candidates"]):
        assert survivor["sampling_round"] == 2
        assert survivor["candidate_index"] == original["index"]
        assert survivor["source_index"] == original["source_index"]
        assert survivor["source_batch"] == original["source_batch"]
        assert survivor["score"] == original["score"]

    planner, snapshot, scene, evaluations, _ = evaluated(2)
    evaluations["A"]["sampling_round"] = 2
    planner.request = lambda path, payload: batch_response(payload)
    events = []
    result = planner.plan_grasp(snapshot, "red", "A", evaluations, scene,
        progress=lambda kind, title, data: events.append(data))
    assert result["sampling_round"] == 2
    replay = next(item["candidate_replay"] for item in events if "candidate_replay" in item)
    assert replay["sampling_round"] == 2
    assert replay["candidate_indices"] == [101, 100]
