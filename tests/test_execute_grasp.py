import copy
import threading

from test_runner import Algorithms, Sim

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
