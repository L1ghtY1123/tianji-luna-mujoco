import asyncio
import copy
import threading
import time

import pytest
from test_runner import Algorithms, Decisions, Sim, wait

from jev_mujoco.algorithms import AlgorithmError
from jev_mujoco.runner import Runner


class WaitingPlanner(Algorithms):
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = None

    def plan_grasp(self, *args, planning_settings=None, progress=None, cancelled=None):
        self.cancelled = cancelled
        if progress:
            progress("planning_batch_start", "规划 batch 1/2", {
                "arm": "A", "batch_index": 1, "batch_count": 2, "candidate_indices": [7, 8]})
            progress("planning_batch_complete", "batch 1/2 完成", {
                "arm": "A", "batch_index": 1, "batch_count": 2, "candidate_indices": [7, 8],
                "error": "cuRobo grasp planning failed", "code": "CUROBO_GRASP_FAILED",
                "duration_s": .01})
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError("test_planner_not_released")
        # Deliberately model a server that cannot be interrupted and returns late.
        if progress:
            progress("planning_batch_start", "迟到事件", {"batch_index": 2, "batch_count": 2})
        return super().plan_grasp(*args)


@pytest.mark.parametrize("timeout_s", [None, 180])
async def test_service_timeout_is_not_relabelled_as_runner_deadline(tmp_path, timeout_s):
    runner = Runner(Sim(), Algorithms(), lambda record: Decisions(), tmp_path)

    def service_timeout():
        raise TimeoutError("service read timed out")

    with pytest.raises(TimeoutError, match="service read timed out"):
        await runner.call_algorithm(service_timeout, timeout_s=timeout_s)


async def test_reset_during_ik_keeps_pending_and_signals_cancellation(tmp_path):
    class WaitingIK(WaitingPlanner):
        def evaluate_arms(self, *args, planning_settings=None, progress=None, cancelled=None, **kwargs):
            self.cancelled = cancelled
            self.entered.set()
            self.release.wait(3)
            return Algorithms.evaluate_arms(self, *args)

    algorithms = WaitingIK()
    runner = Runner(Sim(), algorithms, lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        assert runner.state["current_step"]["skill"] == "plan_grasp"
        runner.simulation.state["generation"] += 1
        await runner.cancel()
        runner.clear()
        assert algorithms.cancelled.is_set()
        assert runner.algorithm_pending
        with pytest.raises(ValueError, match="algorithm_request_pending"):
            await runner.start("抓取")
    finally:
        algorithms.release.set()
        await asyncio.gather(runner.planning_worker, return_exceptions=True)
    assert not runner.algorithm_pending
    assert not runner.events


async def test_planning_progress_is_visible_before_the_service_returns(tmp_path):
    algorithms = WaitingPlanner()
    runner = Runner(Sim(), algorithms, lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        await asyncio.sleep(0)
        step = runner.state.get("current_step")
        assert step and step["skill"] == "plan_grasp"
        assert step["batch_index"] == 1
        assert step['stage'] == 'planning_batch'
        assert step["last_error"] == "cuRobo grasp planning failed"
        assert step["timeout_s"] > 0
        assert any(event["kind"] == "planning_batch_complete" for event in runner.events)
        assert not runner.state["facts"].get("plan_ready")
    finally:
        algorithms.release.set()
        await wait(runner)


async def test_planning_timeout_is_terminal_and_late_result_cannot_move_robot(tmp_path):
    algorithms = WaitingPlanner()
    runner = Runner(Sim(), algorithms, lambda record: Decisions(), tmp_path)
    runner.plan_timeout_s = .05
    await runner.start("抓取黑色物体")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        await asyncio.sleep(.12)
        assert runner.state["status"] == "failed"
        assert "plan_grasp_timeout" in runner.state["error"]
        assert runner.state["failed_step"]["skill"] == "plan_grasp"
        # asyncio may dispatch within one clock tick of the deadline. On
        # Windows/Python 3.12 this tick is 15.625ms, unlike Linux's fine clock.
        clock_tick = max(time.get_clock_info("monotonic").resolution, time.get_clock_info("time").resolution)
        assert runner.state["failed_step"]["duration_s"] >= .05 - clock_tick
        assert algorithms.cancelled.is_set()
        assert not runner.active
        assert runner.algorithm_pending
        before = copy.deepcopy(runner.events)
        duration = runner.state["metrics"]["run_duration_s"]
        with pytest.raises(ValueError, match="algorithm_request_pending"):
            await runner.start("抓取黑色物体")
    finally:
        algorithms.release.set()
        await wait(runner)
    if hasattr(runner, "planning_worker") and runner.planning_worker:
        await asyncio.gather(runner.planning_worker, return_exceptions=True)
    await asyncio.sleep(0)
    assert runner.events == before
    assert runner.state["metrics"]["run_duration_s"] == duration
    assert not runner.algorithm_pending
    assert not runner.state["facts"].get("plan_ready")
    assert not any(event["title"] == "execute_grasp" for event in runner.events)


async def test_reset_marks_planner_cancelled_and_discards_late_events(tmp_path):
    algorithms = WaitingPlanner()
    runner = Runner(Sim(), algorithms, lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    try:
        assert await asyncio.to_thread(algorithms.entered.wait, 2)
        runner.simulation.state["generation"] += 1
        await runner.cancel()
        runner.clear()
        assert algorithms.cancelled is not None and algorithms.cancelled.is_set()
        assert runner.algorithm_pending
        with pytest.raises(ValueError, match="algorithm_request_pending"):
            await runner.start("抓取黑色物体")
    finally:
        algorithms.release.set()
    if hasattr(runner, "planning_worker") and runner.planning_worker:
        await asyncio.gather(runner.planning_worker, return_exceptions=True)
    await asyncio.sleep(0)
    assert runner.state == runner.idle_state()
    assert runner.events == []
    assert not runner.algorithm_pending


async def test_unwritable_run_log_fails_visibly_instead_of_staying_running(tmp_path):
    log_directory = tmp_path / "not_a_directory"
    log_directory.write_text("occupied", encoding="utf-8")
    runner = Runner(Sim(), Algorithms(), lambda record: Decisions(), log_directory)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "failed"
    assert "run_log_write_failed" in runner.state["error"]
    assert not any(event["kind"] == "skill_start" for event in runner.events)
    assert runner.events[-1]["kind"] == "run_end"


async def test_service_error_records_failed_step_and_never_selects_execution(tmp_path):
    class BrokenPlanner(Algorithms):
        def plan_grasp(self, *args, **kwargs):
            raise AlgorithmError("cuRobo service unavailable", code="CUROBO_BACKEND_FAILED",
                                 details={"arm": "A"})

    runner = Runner(Sim(), BrokenPlanner(), lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "failed"
    assert runner.state.get("failed_step", {}).get("skill") == "plan_grasp"
    assert runner.state.get("current_step") is None
    assert not runner.state["facts"].get("plan_ready")
    assert any(event["kind"] == "skill_error" for event in runner.events)
    failure = next(event for event in runner.events if event["kind"] == "skill_error")
    assert failure["data"]["code"] == "CUROBO_BACKEND_FAILED"
    assert failure["data"]["details"] == {"arm": "A"}
    assert runner.state["error_code"] == "CUROBO_BACKEND_FAILED"
    assert not any(event["title"] == "execute_grasp" for event in runner.events)
