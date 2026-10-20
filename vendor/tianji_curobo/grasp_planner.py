"""Project boundary for the pinned cuRobo compound planner's data lifetimes."""

from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.motion_planner import MotionPlanner


# In a35a708, IKSolverCfg documents current_state.dt as a velocity-IK
# horizon. plan_grasp passes a trajectory sample's dt into the next global
# pose solve. Remove only that sampling interval; preserve the motion state.
# GoalToolPose.clone() keeps solver goal updates out of the caller's poses.
# TrajOptSolverResult.clone() owns the output buffers before the next phase
# reuses them. Inherited plan_grasp and all solver parameters stay unchanged.
class GraspMotionPlanner(MotionPlanner):
    def plan_pose(self, goal_tool_poses, current_state, *args, **kwargs):
        start = current_state.clone()
        start.dt = None
        result = super().plan_pose(goal_tool_poses.clone(), start, *args, **kwargs)
        return result.clone() if result is not None else None


class GraspBatchMotionPlanner(BatchMotionPlanner):
    def plan_pose(self, goal_tool_poses, current_state, *args, **kwargs):
        start = current_state.clone()
        start.dt = None
        result = super().plan_pose(goal_tool_poses.clone(), start, *args, **kwargs)
        return result.clone() if result is not None else None
