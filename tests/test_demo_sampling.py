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
from luna_mujoco.geometry import TOOL_PROFILE
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.robot_skills.operations import RobotOperations
from luna_mujoco.runner import Runner


def sampling_case(tmp_path, monkeypatch, outcomes):
    clock = [100.]
    monkeypatch.setattr("luna_mujoco.runner.time.monotonic", lambda: clock[0])
    algorithms = SimpleNamespace(evaluate_arms=object(), generate_grasps=object(), plan_grasp=object())
    robot = Sim()
    runner = RobotOperations(robot, algorithms, None, tmp_path)
    runner.task_progress = SimpleNamespace(completed_ids=[], snapshot=dict)
    runner.state.update(task="grasp_place", arm="A", target_id="source", destination_id="demo-destination",
                        generation=1, planning_attempts=[])
    snapshot = {**robot.robot_snapshot(), "observation_id": "frozen-rgbd"}
    settings = PlanningSettings(observation={"A": [0.] * 7, "B": [0.] * 7}).model_dump()
    grasps = {"sampling_round": 1, "generation": 1, "observation_id": "frozen-rgbd",
              "arm": "A", "tcp_frame": "gripper_front_A", "model": "fixed-model", "tool_profile_revision": TOOL_PROFILE["revision"],
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
        evaluation = {"tcp_frame": "gripper_front_A", "arm": "A", "target_id": "source", "generation": 1,
                      "scene_token": "frozen-map", "planning_settings": settings,
                      "tool_profile_revision": TOOL_PROFILE["revision"],
                      "destination_id": "demo-destination", "tested_count": 1, "batches": [],
                      "start_joints_deg": [0.] * 7, "other_joints_deg": [0.] * 7,
                      "sampling_round": args[2]["sampling_round"],
                      "valid_candidates": [{"candidate_index": 17}] if outcome else []}
        return {"A": evaluation}

    runner.call_algorithm = call
    return runner, snapshot, context, calls, clock, call


async def test_zero_ik_draws_one_fresh_pool_without_wrapper_deadline(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, _ = sampling_case(tmp_path, monkeypatch, [False, True])
    result = await runner.plan_grasp(snapshot, "source", "A", context)
    assert result["sampling_round"] == 2
    assert [call[0] for call in calls] == ["evaluate_arms", "generate_grasps", "evaluate_arms", "plan_grasp"]
    assert all('timeout_s' not in call[2] for call in calls)
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


def rejected_direction_pool(grasps):
    grasps['candidates'] = []
    grasps['raw_candidate_count'] = 1
    grasps['candidate_replay'] = {'all_generated_candidates': [{'index': 17}]}
    grasps['downward_filter'] = {'axis': 'actual_tool_positive_z_in_world',
        'input_count': 1, 'accepted_count': 0, 'rejected_count': 1,
        'candidate_evidence': [{'candidate_index': 17, 'downward_dot': .02,
            'horizontal_away_from_robot_dot': -1., 'tilt_passed': False,
            'away_from_robot_passed': False}]}


async def test_empty_direction_pool_spends_remaining_budget_without_ik(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [True])
    rejected_direction_pool(context['grasps'])

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.generate_grasps:
            result['candidates'] = [{'index': 17}]
        return result

    runner.call_algorithm = call
    result = await runner.plan_grasp(snapshot, 'source', 'A', context)
    assert result['sampling_round'] == 2
    assert [c[0] for c in calls] == ['generate_grasps', 'evaluate_arms', 'plan_grasp']
    assert result['sampling_evidence']['rounds'][0]['outcome'] == 'no_downward_candidate'
    assert result['sampling_evidence']['rounds'][0]['tested_count'] == 0


@pytest.mark.parametrize('fault', [None, 'missing', 'count', 'positive', 'unknown', 'nonfinite',
                                 'truncated', 'wrong_id', 'duplicate', 'missing_raw'])
async def test_direction_exhaustion_requires_complete_proof_and_never_uses_fourth_pool(
        tmp_path, monkeypatch, fault):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [])
    rejected_direction_pool(context['grasps'])
    proof = context['grasps']['downward_filter']
    if fault == 'missing': proof['candidate_evidence'] = []
    elif fault == 'count': proof['rejected_count'] = 2
    elif fault == 'positive':
        proof['candidate_evidence'][0].update(tilt_passed=True, away_from_robot_passed=True)
    elif fault == 'unknown': proof['candidate_evidence'][0]['tilt_passed'] = None
    elif fault == 'nonfinite': proof['candidate_evidence'][0]['downward_dot'] = float('nan')
    elif fault == 'truncated':
        context['grasps']['raw_candidate_count'] = 2
        context['grasps']['candidate_replay']['all_generated_candidates'].append({'index': 18})
    elif fault == 'wrong_id': proof['candidate_evidence'][0]['candidate_index'] = 18
    elif fault == 'duplicate':
        context['grasps']['raw_candidate_count'] = 2
        context['grasps']['candidate_replay']['all_generated_candidates'].append({'index': 17})
    elif fault == 'missing_raw': context['grasps'].pop('raw_candidate_count')

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.generate_grasps:
            rejected_direction_pool(result)
        return result

    runner.call_algorithm = call
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, 'source', 'A', context)
    if fault:
        assert not calls
    else:
        assert caught.value.code == 'grasp_no_downward_candidate'
        assert caught.value.details['total_requested_candidate_count'] == 1768
        assert caught.value.details['total_tested_count'] == 0
        assert [c[0] for c in calls] == ['generate_grasps', 'generate_grasps']


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
    runner.state['facts']['grasps_ready'] = True
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert caught.value.code == "no_collision_free_source_ik"
    assert runner.state['facts']['grasps_ready'] is False
    details = caught.value.details
    assert details["total_requested_candidate_count"] == 1768
    assert details["total_tested_count"] == 3
    assert details["trajectory_attempt_count"] == 0
    assert [row["sampling_round"] for row in details["rounds"]] == [1, 2, 3]
    assert [row["requested_candidate_count"] for row in details["rounds"]] == [256, 512, 1000]
    assert [call[0] for call in calls].count("generate_grasps") == 2
    assert all(call[0] != "plan_grasp" for call in calls)


async def test_destination_exhaustion_preserves_successful_source_ik_evidence(tmp_path, monkeypatch):
    runner, snapshot, context, calls, _, original = sampling_case(tmp_path, monkeypatch, [False] * 3)

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if function is runner.algorithms.evaluate_arms:
            result['A']['destination_screening'] = {'applied': True, 'input_count': 1,
                'accepted_count': 0, 'rejected_count': 1, 'attached': False,
                'scope': 'fixed_gripper_at_finite_placement_endpoints',
                'trajectory_checked': False, 'whole_destination_infeasible': False,
                'candidate_evidence': [{'accepted': False, 'reason': 'no_sampled_endpoint_inside_current_region'}]}
        return result

    runner.call_algorithm = call
    with pytest.raises(AlgorithmError) as caught:
        await runner.plan_grasp(snapshot, 'source', 'A', context)
    assert caught.value.code == 'grasp_destination_samples_exhausted'
    assert caught.value.details['trajectory_attempt_count'] == 0
    assert caught.value.details['whole_destination_infeasible'] is False
    for row in caught.value.details['rounds']:
        assert row['outcome'] == 'no_sampled_destination_endpoint'
        assert row['destination_screening']['input_count'] == 1
        assert row['destination_screening']['rejection_reasons'] == {'no_sampled_endpoint_inside_current_region': 1}
    assert all(row[0] != 'plan_grasp' for row in calls)


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


async def test_slow_sampling_is_bounded_by_pool_count_not_wrapper_deadline(tmp_path, monkeypatch):
    runner, snapshot, context, calls, clock, original = sampling_case(tmp_path, monkeypatch, [False, False, False])

    async def call(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        clock[0] += 180.
        return result

    runner.call_algorithm = call
    with pytest.raises(AlgorithmError, match="no eligible source grasp target in the bounded candidate pools"):
        await runner.plan_grasp(snapshot, "source", "A", context)
    assert len(calls) == 5
    assert all('timeout_s' not in call[2] for call in calls)


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
async def test_inflight_sampling_stops_remaining_calls_after_task_cancellation(tmp_path, stop, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockingClient(GraspClient):
        def generate(self, payload):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test did not release inference")
            return super().generate(payload)

    client = BlockingClient()
    algorithms = Algorithms(graspgen_client=client)
    # This test starts its deadline at the service boundary; cold geometry-library
    # imports can otherwise consume the entire 100 ms before inference enters.
    monkeypatch.setattr(algorithms, '_graspgen_object_points',
                        lambda snapshot, target: np.asarray(target['points_world_m']))
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
                                       placement_request=grasp_placement(), planning_settings={'grasp_insertion_m': .01, 'grasp_maximum_tilt_deg': 45., 'grasp_require_away_from_robot': True})
    assert grasps["candidate_replay"]["sampling_round"] == 2
    scene = {"target_id": "seen-red", "generation": 1,
             "observation_id": "capture-4", "scene_token": "map-1"}
    evaluation = algorithms.evaluate_arms(snapshot, "seen-red", grasps, scene, requested_arm="A", planning_settings={'grasp_insertion_m': .01, 'grasp_maximum_tilt_deg': 45., 'grasp_require_away_from_robot': True})["A"]
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
