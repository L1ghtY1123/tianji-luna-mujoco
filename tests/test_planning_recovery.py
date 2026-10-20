import asyncio
import copy

import pytest
from test_runner import Algorithms, Decisions, Sim, wait

from jev_mujoco.algorithms import AlgorithmError
from jev_mujoco.planning_settings import PlanningSettings
from jev_mujoco.runner import Runner


class BothArms(Algorithms):
    def __init__(self, failed_arms=("A",), error_code="planning_no_valid_candidate", lift_offset_m=.1):
        self.failed_arms = failed_arms
        self.error_code = error_code
        self.lift_offset_m = lift_offset_m
        self.planned_arms = []

    def evaluate_arms(self, *args, **kwargs):
        return {arm: {"valid_candidates": [{"index": 1}]} for arm in ("A", "B")}

    def evaluate_reachability(self, *args, **kwargs):
        return {arm: {'reachable': True, 'feasibility_scope': 'offline_workspace'} for arm in ('A', 'B')}

    def plan_grasp(self, snapshot, target, arm, evaluations, scene, *, planning_settings=None, **kwargs):
        self.planned_arms.append(arm)
        assert planning_settings["lift_in_tool_frame"] is True
        assert planning_settings["lift_offset_m"] == self.lift_offset_m
        if arm in self.failed_arms:
            raise AlgorithmError("lift planning failed", code=self.error_code,
                                 details={"arm": arm, "failures": [{"candidate_index": 1,
                                          "phase_success": {"approach": True, "grasp": True, "lift": False}}]})
        return super().plan_grasp(snapshot, target, arm, evaluations, scene,
                                  planning_settings=planning_settings, **kwargs)


class Selecting(Decisions):
    def __init__(self, arms=("A", "B"), stop_after_failure=False):
        self.arms = iter(arms)
        self.stop_after_failure = stop_after_failure
        self.states = []

    async def choose_skill(self, state, registry):
        self.states.append(copy.deepcopy(state))
        if self.stop_after_failure and state.get("planning_attempts"):
            return "stop"
        return await super().choose_skill(state, registry)

    async def choose_arm(self, state, evaluations):
        return next(self.arms)

    async def choose_recovery(self, state):
        return ('retry_plan' if state['failure']['code'] == 'planning_no_valid_candidate'
                else 'stop')


def make_runner(tmp_path, algorithms, decisions, *, lift_offset_m=.1):
    sim = Sim()
    sim.state["table"] = {"position": [.55, 0., .56], "size": [.7, 1., .08]}
    settings = PlanningSettings(lift_in_tool_frame=True, lift_offset_m=lift_offset_m)
    return Runner(sim, algorithms, lambda record: decisions, tmp_path, planning_settings=settings)


@pytest.mark.parametrize("lift_offset_m", [.1, -.1])
async def test_failed_plan_returns_evidence_to_jev_and_retries_only_bound_arm(tmp_path, lift_offset_m):
    class FirstFailure(BothArms):
        def plan_grasp(self, *args, **kwargs):
            if self.planned_arms:
                self.failed_arms = ()
            return super().plan_grasp(*args, **kwargs)
    choices = Selecting(arms=('A',))
    algorithms = FirstFailure(lift_offset_m=lift_offset_m)
    runner = make_runner(tmp_path, algorithms, choices, lift_offset_m=lift_offset_m)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "succeeded"
    assert runner.state["arm"] == "A"
    assert algorithms.planned_arms == ["A", "A"]
    completed = [e["title"] for e in runner.events if e["kind"] == "skill_complete"]
    for name in ("observe_scene", "build_scene", "generate_grasps", "evaluate_reachability"):
        assert completed.count(name) == 1
    assert completed.count("plan_grasp") == 1
    assert completed.count("choose_arm") == 1
    failed = next(s for s in choices.states if s.get('failure'))
    assert failed['execution_arm'] == 'A'  # Failure itself never chooses a replacement.
    recovered = next(s for s in choices.states if s.get("planning_attempts") and not s.get('failure'))
    assert recovered["execution_arm"] == 'A'
    assert recovered["facts"]["arm_selected"] is True
    assert recovered["facts"]["plan_ready"] is False
    assert recovered["facts"]["arms_evaluated"] is True
    assert recovered["planning_attempts"][0]["arm"] == "A"
    assert recovered["planning_attempts"][0]["code"] == "planning_no_valid_candidate"
    assert recovered["planning_attempts"][0]["details"]["failures"][0]["phase_success"]["lift"] is False
    assert "table" not in recovered["simulation"]
    assert recovered["observation"]["table"]["position"] == [.55, 0., -.04]
    assert recovered["observation"]["table"]["surface_height_m"] == 0.
    assert recovered["planning_settings"]["lift_in_tool_frame"] is True
    assert recovered["planning_settings"]["lift_offset_m"] == lift_offset_m
    assert set(recovered["arm_evaluations"]) == {"A", "B"}
    assert [a["status"] for a in runner.state["planning_attempts"]] == ["failed", "succeeded"]
    assert runner.planning_settings.lift_in_tool_frame is True
    assert any(e["kind"] == "planning_attempt_failed" for e in runner.events)
    assert any(e['title'] == 'recover_task' for e in runner.events if e['kind'] == 'skill_complete')
    # Later updates must not change the actual state already presented to JEV.
    assert len(recovered["planning_attempts"]) == 1


async def test_jev_can_stop_after_failed_planning_without_automatic_arm_switch(tmp_path):
    algorithms = BothArms()
    runner = make_runner(tmp_path, algorithms, Selecting(stop_after_failure=True))
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["error"] == "planning_no_valid_candidate"
    assert algorithms.planned_arms == ["A"]
    assert not any(e["title"] == "execute_grasp" for e in runner.events)


async def test_explicit_same_arm_plan_retry_stops_at_task_attempt_budget(tmp_path):
    algorithms = BothArms()
    runner = make_runner(tmp_path, algorithms, Selecting(arms=("A", "A")))
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["error"] == "skill_selection_budget_exhausted"
    assert algorithms.planned_arms == ['A', 'A', 'A']
    assert runner.task_progress.failure['attempt_count'] == 3
    errors = [event['data'].get('code') for event in runner.events if event['kind'] == 'skill_error']
    assert any(code.startswith('recovery_retry_budget_exhausted') for code in errors)


async def test_stage_arm_change_is_refused_before_second_gpu_call(tmp_path):
    algorithms = BothArms(failed_arms=("A", "B"))
    runner = make_runner(tmp_path, algorithms, Selecting(stop_after_failure=True))
    await runner.start("抓取黑色物体")
    await wait(runner)
    with pytest.raises(ValueError, match='stage_arm_is_fixed'):
        await runner.execute('choose_arm', {'observation': {}, 'evaluations': {'B': {'reachable': True}}}, {'arm': 'B'})
    assert algorithms.planned_arms == ['A']
    assert len(runner.state["planning_attempts"]) == 1
    assert not any(e["title"] == "execute_grasp" for e in runner.events)


@pytest.mark.parametrize("code", ["CUROBO_BACKEND_FAILED", "planning_stale_evaluation"])
async def test_service_and_stale_input_errors_are_terminal(tmp_path, code):
    algorithms = BothArms(error_code=code)
    runner = make_runner(tmp_path, algorithms, Selecting())
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "failed"
    assert runner.state["failed_step"]["skill"] == "plan_grasp"
    assert algorithms.planned_arms == ["A"]
    assert not any(e["kind"] == "planning_attempt_failed" for e in runner.events)


async def test_unreachable_arm_selection_is_validated_after_model_answer(tmp_path):
    runner = make_runner(tmp_path, Algorithms(), Selecting(arms=("B",)))
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["error"] == "no_reachable_arm"
    assert runner.state["failed_step"]["skill"] == "choose_arm"
    assert not any(e["title"] == "plan_grasp" for e in runner.events)


async def test_reset_while_jev_reconsiders_failed_plan_cannot_launch_other_arm(tmp_path):
    entered = asyncio.Event()

    class WaitingChoice(Selecting):
        async def choose_skill(self, state, registry):
            if state.get("planning_attempts"):
                entered.set()
                await asyncio.Event().wait()
            return await super().choose_skill(state, registry)

    algorithms = BothArms()
    runner = make_runner(tmp_path, algorithms, WaitingChoice())
    await runner.start("抓取黑色物体")
    try:
        await asyncio.wait_for(entered.wait(), 2)
        runner.simulation.state["generation"] += 1
        await runner.cancel()
        runner.clear()
        assert runner.state == runner.idle_state()
        assert algorithms.planned_arms == ["A"]
        assert runner.events == []
    finally:
        await runner.cancel()
