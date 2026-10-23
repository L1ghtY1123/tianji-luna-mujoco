import copy

from test_runner import Algorithms, Decisions, Sim

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.runner import Runner


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
