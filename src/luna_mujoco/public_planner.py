"""Composition only: official planners own all internal algorithm calls."""
import time


class PublicPlanner:
    def __init__(self, sdk, settings, *, batch=False):
        self.sdk = sdk
        self.planning_settings = settings
        self.batch = batch
        self.last_grasp_result = None
        self.last_pose_result = None
        self.last_grasp_wall_s = None

    def __getattr__(self, name):
        # Compatibility with the existing transport wrapper. No methods or
        # attributes are installed on the SDK instance; internal self calls
        # always resolve to the original NVIDIA class.
        return getattr(self.sdk, name)

    def plan_pose(self, goal_tool_poses, current_state, *args, **kwargs):
        kwargs.setdefault('max_attempts', self.planning_settings.plan_max_attempts)
        if self.batch:
            kwargs.setdefault('finetune_attempts', self.planning_settings.finetune_attempts)
            kwargs.setdefault('finetune_dt_scale', self.planning_settings.finetune_dt_scale)
        self.last_pose_result = self.sdk.plan_pose(goal_tool_poses, current_state, *args, **kwargs)
        return self.last_pose_result

    def plan_grasp(self, *args, **kwargs):
        self.last_grasp_result = None
        started = time.perf_counter()
        # In particular, do not clear dt, clone phase buffers, skip the goalset
        # stage, or override parameters of plan_pose inside plan_grasp.
        self.last_grasp_result = self.sdk.plan_grasp(*args, **kwargs)
        self.last_grasp_wall_s = time.perf_counter() - started
        return self.last_grasp_result
