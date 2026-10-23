import threading

import pytest
from test_runner import Algorithms, Decisions, Sim

from luna_mujoco.runner import Runner


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
