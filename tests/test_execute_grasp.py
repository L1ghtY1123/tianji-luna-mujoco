import asyncio
import copy
import threading

import pytest
from test_runner import Algorithms, Decisions, Sim, wait

from jev_mujoco.runner import Runner
from jev_mujoco.skills import Registry

MOTOR_STAGES = ["open_gripper", "approach", "grasp", "close_gripper", "lift"]
ALL_STAGES = ["open_gripper", "approach", "dwell_approach", "grasp", "close_gripper",
              "dwell_grasp", "capture_reference", "lift", "dwell_lift"]


class DistinctPlan(Algorithms):
    def plan_grasp(self, *args, planning_settings=None, progress=None, cancelled=None):
        plan = super().plan_grasp(*args, planning_settings=planning_settings,
                                  progress=progress, cancelled=cancelled)
        result = plan | {"candidate_index": 7, "phases": {
            phase: {"positions_rad": [[start] * 7, [end] * 7], "dt": .1}
            for phase, start, end in (("approach", 0., .5), ("grasp", .5, 1.), ("lift", 1., 2.))}}
        result['phases']['approach']['positions_rad'][0] = list(args[0]['arms'][args[2]]['joints_rad'])
        return result

    def plan_carried_lift(self, *args, **kwargs):
        raise AssertionError("grasp must execute the original lift without attachment")


class RecordingSim(Sim):
    def __init__(self, failure=None, blocked=None):
        super().__init__()
        self.calls = []
        self.executed_commands = {}
        self.failure = failure
        self.blocked = blocked
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def record(self, stage, arm, generation):
        assert arm == "A"
        assert generation == 1
        self.calls.append(stage)
        if stage == self.blocked:
            self.entered.set()
            try:
                if not self.release.wait(5):
                    raise RuntimeError("test_stage_timeout")
                if generation != self.state["generation"]:
                    raise ValueError("stale_generation")
            finally:
                self.finished.set()
        if stage == self.failure:
            raise RuntimeError("failed_" + stage)

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        stage = {.5: "approach", 1.: "grasp", 2.: "lift"}[positions_rad[-1][0]]
        self.record(stage, arm, generation)
        self.executed_commands[stage] = copy.deepcopy(positions_rad)
        self.state["arms"][arm]["joints_rad"] = list(positions_rad[-1])
        if stage == "lift":
            self.state["objects"][0]["T_world_object"][2][3] = .1
            self.state["arms"][arm]["tcp"][2][3] = .1
        return {"completed": True}

    def set_gripper(self, arm, closed, generation):
        self.record("close_gripper" if closed else "open_gripper", arm, generation)
        if closed:
            # Closure feedback is sampled without another RGBD observation.
            self.state["arms"][arm]["tcp"][2][3] = 0.
        return super().set_gripper(arm, closed, generation)


def test_only_complete_execution_is_selectable_after_planning():
    registry = Registry()
    retired = {"execute_approach", "close_gripper", "execute_lift"}
    assert retired.isdisjoint(registry.contracts)
    assert retired.isdisjoint(registry.documents)
    facts = dict.fromkeys(("observed", "target_selected", "scene_ready", "grasps_ready",
                           "arms_evaluated", "arm_selected", "plan_ready"), True)
    observation_skills = {'observe', 'observe_scene', 'observe_geometry', 'observe_result', 'build_scene', 'inspect_visual_result'}
    assert set(registry.eligible(facts)) - observation_skills == {"execute_grasp"}
    facts.update(dict.fromkeys(registry.contracts["execute_grasp"].provides, True))
    assert set(registry.eligible(facts)) - observation_skills == {"verify_grasp"}


async def test_one_skill_runs_all_stages_and_verifies_without_attachment_or_postgrasp_capture(tmp_path):
    simulation = RecordingSim()
    choices = []

    class RecordingDecisions(Decisions):
        async def choose_skill(self, state, registry):
            skill = await super().choose_skill(state, registry)
            choices.append((skill, list(simulation.calls)))
            return skill

    runner = Runner(simulation, DistinctPlan(), lambda record: RecordingDecisions(), tmp_path)
    await runner.start("抓取左侧黑色方块")
    await wait(runner)
    assert runner.state["status"] == "succeeded"
    assert simulation.calls == MOTOR_STAGES
    assert len(choices) == 9
    assert choices[-2:] == [("execute_grasp", []), ("verify_grasp", MOTOR_STAGES)]
    complete = [event for event in runner.events if event["kind"] == "execution_stage_complete"
                and event['data']['skill'] == 'execute_grasp']
    assert [event["data"]["stage"] for event in complete] == ALL_STAGES
    assert all(event["data"]["skill"] == "execute_grasp" for event in complete)
    reference = next(event["data"]["result"] for event in complete if event["data"]["stage"] == "capture_reference")
    assert reference["source"] == "cached_rgbd_and_closure_feedback"
    assert reference["observation_id"].startswith("rgbd-1-")
    assert reference["tcp_before_lift"][2][3] == 0.
    assert runner.state["facts"]["lifted"]
    assert runner.holding_state["verified"] is True
    assert len(runner.holding_state["points_tool_m"]) >= 16
    assert runner.holding_state["attached"] is False
    assert "estimated_bounds_tool_m" not in runner.holding_state
    assert simulation.frame == 2


@pytest.mark.parametrize("failed_stage", MOTOR_STAGES)
async def test_stage_failure_stops_execution_and_never_enables_verification(tmp_path, failed_stage):
    simulation = RecordingSim(failure=failed_stage)
    runner = Runner(simulation, DistinctPlan(), lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert simulation.calls == MOTOR_STAGES[:MOTOR_STAGES.index(failed_stage) + 1]
    assert runner.state["status"] == "failed"
    assert runner.state["error"] == "failed_" + failed_stage
    assert not runner.state["facts"].get("lifted")
    assert not any(event["title"] == "verify_grasp" for event in runner.events)
    errors = [event for event in runner.events if event["kind"] == "execution_stage_error"]
    assert len(errors) == 1
    assert errors[0]["data"]["stage"] == failed_stage


@pytest.mark.parametrize("blocked_stage", MOTOR_STAGES)
async def test_reset_cancellation_cannot_continue_late_stage_or_restore_old_trace(tmp_path, blocked_stage):
    simulation = RecordingSim(blocked=blocked_stage)
    runner = Runner(simulation, DistinctPlan(), lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    try:
        assert await asyncio.to_thread(simulation.entered.wait, 3)
        # Same ordering as /api/reset: invalidate physics before cancelling the runner.
        simulation.state["generation"] += 1
        await runner.cancel()
        runner.clear()
    finally:
        simulation.release.set()
    assert await asyncio.to_thread(simulation.finished.wait, 3)
    await asyncio.sleep(0)
    assert simulation.calls == MOTOR_STAGES[:MOTOR_STAGES.index(blocked_stage) + 1]
    assert runner.state == runner.idle_state()
    assert runner.events == []


async def test_generation_change_after_motor_return_prevents_next_stage(tmp_path):
    class ResetAfterOpening(RecordingSim):
        def set_gripper(self, arm, closed, generation):
            result = super().set_gripper(arm, closed, generation)
            self.state["generation"] += 1
            return result

    simulation = ResetAfterOpening()
    runner = Runner(simulation, DistinctPlan(), lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert simulation.calls == ["open_gripper"]
    assert runner.state["error"] == "simulation_generation_changed"
    assert not runner.state["facts"].get("lifted")


async def test_dwell_settings_are_not_scaled_and_reference_follows_closure_dwell(tmp_path):
    from jev_mujoco.planning_settings import PlanningSettings

    class DwellSim(RecordingSim):
        def dwell(self, duration_s, generation):
            self.calls.append(("dwell", duration_s))
            return super().dwell(duration_s, generation)

    sim = DwellSim()
    settings = PlanningSettings(time_scale=4, dwell_approach_s=.3, dwell_grasp_s=.6, dwell_lift_s=.9)
    runner = Runner(sim, DistinctPlan(), lambda record: Decisions(), tmp_path, planning_settings=settings)
    await runner.start("抓取黑色方块")
    # Even direct replacement of the idle configuration cannot change this run's snapshot.
    runner.planning_settings = PlanningSettings(time_scale=1, dwell_grasp_s=10)
    await wait(runner)
    assert runner.state["status"] == "succeeded"
    assert sim.calls == ["open_gripper", "approach", ("dwell", .3), "grasp", "close_gripper",
                         ("dwell", .6), "lift", ("dwell", .9)]


async def test_initial_opening_uses_the_real_detached_phase_contract(tmp_path):
    from jev_mujoco.carrying import validate_phase_request

    class ContractCheckedPlan(DistinctPlan):
        def __init__(self):
            self.checked_phases = []

        def validate_execution(self, observation, arm, scene, phase, command, binding=None, *,
                               planning_settings=None, gripper_end_position_rad=None,
                               contact_goal_pose_world=None):
            payload = {"phase": phase, "gripper_joint_positions_rad": {
                name: feedback["gripper_position_rad"] for name, feedback in observation["arms"].items()},
                "gripper_start_position_rad": observation["arms"][arm]["gripper_position_rad"],
                "gripper_end_position_rad": gripper_end_position_rad,
                "contact_goal_pose_world": contact_goal_pose_world}
            if binding is not None:
                payload["carrying_binding"] = binding
            validate_phase_request(payload)
            self.checked_phases.append(phase)
            return {"collision_free": True}

    algorithms = ContractCheckedPlan()
    runner = Runner(RecordingSim(), algorithms, lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "succeeded", runner.state["error"]
    assert algorithms.checked_phases == ["approach_grasp", "grasp"]


@pytest.mark.parametrize("failure", ["post_grasp_observation", "attachment"])
async def test_unavailable_postlift_services_do_not_block_verified_hold_or_allow_second_grasp(tmp_path, failure):
    class FailingAfterLift(DistinctPlan):
        def observe(self, capture, *, instruction="", checkpoint=None, source_hint=None):
            if failure == "post_grasp_observation" and capture["views"]["1"]["rgb"][0, 0, 0] == 100:
                raise RuntimeError("rgbd_post_grasp_unavailable")
            return super().observe(capture, instruction=instruction, checkpoint=checkpoint)

        def prepare_carrying(self, *args, **kwargs):
            raise RuntimeError("attachment_unavailable")

    class CaptureFailure(RecordingSim):
        def capture_rgbd(self):
            if failure == "post_grasp_observation" and self.frame >= 2:
                raise RuntimeError("rgbd_post_grasp_unavailable")
            return super().capture_rgbd()
    simulation = CaptureFailure()
    runner = Runner(simulation, FailingAfterLift(), lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert simulation.calls == MOTOR_STAGES
    assert runner.holding_state is not None
    assert runner.holding_state["verified"] is True
    assert runner.holding_state["attached"] is False
    assert runner.state["facts"].get("grasp_completed")
    await runner.start("再次抓取黑色物体")
    await wait(runner)
    assert runner.state["error"] == "robot_already_holding_object"
    assert simulation.calls == MOTOR_STAGES


async def test_unused_postgrasp_visual_does_not_cancel_position_confirmation(tmp_path):
    class ObjectStayedBehind(DistinctPlan):
        def observe(self, capture, *, instruction="", checkpoint=None, source_hint=None):
            observation = super().observe(capture, instruction=instruction)
            if capture["views"]["1"]["rgb"][0, 0, 0] == 100:
                obj = observation["objects"][0]
                obj["position"][2] = .04
                obj["visible_centroid_world"][2] = .04
                obj["points_world_m"][:, 2] -= .08
            return observation

    runner = Runner(RecordingSim(), ObjectStayedBehind(), lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "succeeded"
    assert runner.state["verification"]["success"] is True
    assert runner.state["facts"]["verified"] is True
    assert runner.state["facts"].get("grasp_completed")
    assert runner.state["facts"].get("grasp_held")
    assert runner.holding_state["verified"] is True


async def test_small_live_joint_connector_is_checked_and_executed_identically(tmp_path):
    class ProofRecordingPlan(DistinctPlan):
        def __init__(self):
            self.checked_commands = {}

        def validate_execution(self, observation, arm, scene, phase, command, binding=None, **kwargs):
            self.checked_commands[phase] = copy.deepcopy(command["positions_rad"])
            return {"collision_free": True}

    simulation = RecordingSim()
    class DriftAfterPlan(Decisions):
        async def choose_skill(self, state, registry):
            skill = await super().choose_skill(state, registry)
            if skill == 'execute_grasp':
                simulation.state['arms']['A']['joints_rad'] = [
                    value + .005 for value in simulation.state['arms']['A']['joints_rad']]
            return skill
    algorithms = ProofRecordingPlan()
    runner = Runner(simulation, algorithms, lambda record: DriftAfterPlan(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "succeeded", runner.state["error"]
    checked = algorithms.checked_commands["approach_grasp"]
    assert checked[0] == pytest.approx([value + .005 for value in checked[1]])
    assert checked == simulation.executed_commands["approach"]
