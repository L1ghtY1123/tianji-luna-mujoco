"""Project dual path retaining the copied interpolation and hard validation."""
import numpy as np
from tianji_curobo._curobo_impl import (
    _hermite_command_positions,
    _NativeDualCuroboBackend,
    _planner_interpolation_dt,
    _planning_time,
    _success_matrix,
    _trajectory_values,
)
from tianji_curobo.contracts import CuroboToolError
from tianji_curobo.dual_contracts import JOINT_NAMES

from .planning_contracts import DualValidationRequest


class ValidatedDualBackend(_NativeDualCuroboBackend):
    def warmup(self):
        planner = self.motion_planner
        planner.warmup(enable_graph=True, num_warmup_iterations=1)
        current = planner.default_joint_state.clone().unsqueeze(0)
        goal = current.clone()
        goal.position[..., 0] += .2
        planner.plan_cspace(goal, current, max_attempts=1, enable_graph_attempt=0)
        planner.reset_seed()

    def plan_dual_native(self, request):
        """Use the project contract for the internal post-interpolation check.

        Mirrors the pinned wrapper method because it constructs its validation
        request internally. Solver settings and actual hard checker remain the
        same 10 mm profile; no compatibility clearance is sent to the solver.
        """
        current = self._state(self._ordered_dual_joints(request.start_joints_deg))
        if request.goal_joints_deg is not None:
            goal = self._state(self._ordered_dual_joints(request.goal_joints_deg))
            result = self.motion_planner.plan_cspace(
                goal, current, max_attempts=request.max_attempts, enable_graph_attempt=1)
        else:
            result = self.motion_planner.plan_pose(
                self._dual_goal(request.goal_transforms_world), current,
                max_attempts=request.max_attempts, enable_graph_attempt=1)
        if result is None or not bool(np.all(_success_matrix(result))):
            raise CuroboToolError("cuRobo found no collision-free dual-arm trajectory",
                                 stage="trajopt", code="CUROBO_TRAJOPT_FAILED", status=422)
        plan = result.get_interpolated_plan()
        positions, names = _trajectory_values(plan, self.joint_names)
        command = np.asarray(_hermite_command_positions(plan, names, positions, request.trajectory_point_scale))
        start = dict(zip(self.joint_names, self._ordered_dual_joints(request.start_joints_deg)))
        if np.max(np.abs(np.rad2deg(positions[0]) - [start[name] for name in names])) > .001:
            raise CuroboToolError("dual trajectory start does not match request", stage="validation",
                                 code="CUROBO_INVALID_TRAJECTORY", status=500)
        if request.goal_joints_deg is not None:
            target = dict(zip(self.joint_names, self._ordered_dual_joints(request.goal_joints_deg)))
            if np.max(np.abs(np.rad2deg(positions[-1]) - [target[name] for name in names])) > .001:
                raise CuroboToolError("dual trajectory end does not match request", stage="validation",
                                     code="CUROBO_INVALID_TRAJECTORY", status=500)
        raw_paths, command_paths = {}, {}
        for arm in ("A", "B"):
            columns = [names.index(name) for name in JOINT_NAMES[arm]]
            raw_paths[arm] = np.rad2deg(positions[:, columns]).tolist()
            command_paths[arm] = np.rad2deg(command[:, columns]).tolist()
        checked = self.validate_dual_native(DualValidationRequest.from_mapping({
            "points_deg": command_paths,
            "trajectory_collision_clearance_m": request.trajectory_collision_clearance_m,
            "optimizer_collision_activation_distance_m": request.optimizer_collision_activation_distance_m,
        }))
        return {"ok": True, "planner": "curobo_dual", "trajectories": command_paths,
                "raw_trajectories": raw_paths, "joint_names": {a: list(JOINT_NAMES[a]) for a in ("A", "B")},
                "sample_dt_s": 1. / request.freq_hz,
                "raw_sample_dt_s": _planner_interpolation_dt(self.motion_planner),
                "trajectory_point_scale": request.trajectory_point_scale,
                "time_parameterization": "quintic_hermite",
                "planning_time_s": _planning_time(result), "checked_point_count": checked,
                "point_count": len(command)}
