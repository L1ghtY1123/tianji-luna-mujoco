"""Current Responses runner rejects stale service results before motor commands.

The service doubles model an uninterruptible remote request. Scheduling, deadlines,
generation checks, cancellation, contracts, and the program interpreter stay real.
"""
import asyncio
import threading

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.demo_runner import DemoRunner


class RecordedRobot(DemoRobot):
    def __init__(self):
        super().__init__()
        self.motor_commands = []

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        self.motor_commands.append(("trajectory", arm))
        return super().execute_trajectory(arm, positions_rad, dt, generation)

    def set_gripper(self, arm, closed, generation):
        self.motor_commands.append(("gripper", closed))
        return super().set_gripper(arm, closed, generation)


class WaitingVision(DemoVision):
    def __init__(self, blocked_phase):
        super().__init__(count=1)
        self.blocked_phase = blocked_phase
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = None

    def wait_for_service(self, progress, cancelled):
        self.cancelled = cancelled
        if progress:
            progress("planning_batch_start", "Waiting for service", {
                "arm": "A", "batch_index": 1, "batch_count": 2, "candidate_indices": [1]})
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("test_service_not_released")
        if progress:
            progress("planning_batch_complete", "Late result", {
                "arm": "A", "batch_index": 2, "batch_count": 2, "success_count": 1})

    def evaluate_arms(self, *args, progress=None, cancelled=None, **kwargs):
        if self.blocked_phase == "ik":
            self.wait_for_service(progress, cancelled)
        return super().evaluate_arms(*args, **kwargs)

    def plan_grasp(self, *args, progress=None, cancelled=None, **kwargs):
        if self.blocked_phase == "plan":
            self.wait_for_service(progress, cancelled)
        return super().plan_grasp(*args, **kwargs)


async def finish_service(runner, algorithms):
    """Always release the worker thread, including when an assertion fails."""
    algorithms.release.set()
    if runner.worker is not None:
        await asyncio.gather(runner.worker, return_exceptions=True)
    if runner.planning_worker is not None:
        await asyncio.gather(runner.planning_worker, return_exceptions=True)
    await asyncio.sleep(0)


@pytest.mark.parametrize("phase", ["ik", "plan"])
async def test_reset_cancels_current_demo_request_and_discards_late_result(tmp_path, phase):
    robot, algorithms = RecordedRobot(), WaitingVision(phase)
    runner = DemoRunner(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把一个粉色物体放到指定区域")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        assert runner.state["current_step"]["skill"] == "plan_grasp"
        robot.generation += 1
        await runner.cancel()
        runner.clear()
        assert algorithms.cancelled.is_set()
        assert runner.algorithm_pending
        with pytest.raises(ValueError, match="algorithm_request_pending"):
            await runner.start("把一个粉色物体放到指定区域")
    finally:
        await finish_service(runner, algorithms)
    assert not runner.algorithm_pending
    assert runner.state == runner.idle_state()
    assert runner.events == []
    assert robot.motor_commands == []


@pytest.mark.parametrize("phase", ["ik", "plan"])
async def test_pending_demo_plan_has_no_extra_wall_deadline_and_can_be_cancelled(tmp_path, phase):
    robot, algorithms = RecordedRobot(), WaitingVision(phase)
    runner = DemoRunner(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把一个粉色物体放到指定区域")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        await asyncio.sleep(.2)
        assert runner.state["status"] == "running"
        assert "timeout_s" not in runner.state["current_step"]
        assert not runner.state["facts"].get("plan_ready")
        await runner.cancel()
        assert algorithms.cancelled.is_set()
    finally:
        await finish_service(runner, algorithms)
    assert robot.motor_commands == []
    assert not runner.algorithm_pending


@pytest.mark.parametrize("phase", ["ik", "plan"])
async def test_generation_change_rejects_service_success_before_demo_execution(tmp_path, phase):
    robot, algorithms = RecordedRobot(), WaitingVision(phase)
    runner = DemoRunner(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把一个粉色物体放到指定区域")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        robot.generation += 1
    finally:
        await finish_service(runner, algorithms)
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"] == "simulation_generation_changed"
    assert runner.task_progress.completed_ids == []
    assert robot.motor_commands == []


async def test_demo_planning_progress_is_visible_while_request_is_pending(tmp_path):
    robot, algorithms = RecordedRobot(), WaitingVision("plan")
    runner = DemoRunner(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把一个粉色物体放到指定区域")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        await asyncio.sleep(0)
        step = runner.state["current_step"]
        assert step["skill"] == "plan_grasp"
        assert step["batch_index"] == 1 and step["batch_count"] == 2
        assert any(event["kind"] == "planning_batch_start" for event in runner.events)
        assert not runner.state["facts"].get("plan_ready")
        await runner.cancel()
    finally:
        await finish_service(runner, algorithms)
    assert robot.motor_commands == []


async def test_demo_service_failure_preserves_error_and_never_starts_motor(tmp_path):
    class BrokenVision(DemoVision):
        def plan_grasp(self, *args, **kwargs):
            raise AlgorithmError("planner unavailable", code="CUROBO_BACKEND_FAILED",
                                 details={"arm": "A"})

    robot = RecordedRobot()
    runner = DemoRunner(robot, BrokenVision(count=1), None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把一个粉色物体放到指定区域")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert runner.state["failed_step"]["skill"] == "plan_grasp"
    assert runner.state["error_code"] == "CUROBO_BACKEND_FAILED"
    failure = next(event for event in runner.events if event["kind"] == "skill_error")
    assert failure["data"]["details"] == {"arm": "A"}
    assert robot.motor_commands == []
    assert runner.task_progress.completed_ids == []


class WaitingMotorRobot(RecordedRobot):
    def __init__(self, blocked_stage):
        super().__init__()
        self.blocked_stage = blocked_stage
        self.dispatched_stages = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def before_command(self, stage, generation):
        self.dispatched_stages.append(stage)
        if stage == self.blocked_stage:
            self.entered.set()
            try:
                if not self.release.wait(5):
                    raise RuntimeError("test_motor_not_released")
            finally:
                self.finished.set()
        return generation == self.generation

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        stage = {0: "approach", 1: "grasp", 2: "lift"}[int(positions_rad[-1][0])]
        if not self.before_command(stage, generation):
            # Model a motor response that arrives after physics was reset.
            return {"completed": True}
        return super().execute_trajectory(arm, positions_rad, dt, generation)

    def set_gripper(self, arm, closed, generation):
        if not self.before_command("close_gripper" if closed else "open_gripper", generation):
            return {"completed": True, "closed_command": closed}
        return super().set_gripper(arm, closed, generation)


@pytest.mark.parametrize("stage", ["open_gripper", "approach", "grasp", "close_gripper", "lift"])
async def test_reset_during_demo_motor_command_cannot_dispatch_the_next_stage(tmp_path, stage):
    robot = WaitingMotorRobot(stage)
    runner = DemoRunner(robot, DemoVision(count=1), None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把一个粉色物体放到指定区域")
    try:
        assert await asyncio.to_thread(robot.entered.wait, 2)
        assert runner.state["current_step"]["skill"] == "execute_grasp"
        dispatched = list(robot.dispatched_stages)
        robot.generation += 1
        await runner.cancel()
        runner.clear()
    finally:
        robot.release.set()
        assert await asyncio.to_thread(robot.finished.wait, 2)
        await asyncio.sleep(0)
    assert robot.dispatched_stages == dispatched
    assert runner.state == runner.idle_state()
    assert runner.events == []
