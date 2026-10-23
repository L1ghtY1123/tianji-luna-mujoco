"""Project-owned construction of the pinned cuRobo public solver APIs.

The NVIDIA package and copied Tianji wrapper stay read-only. Both standalone
IK and the grasp planner receive the same IK configuration before construction.
"""

import hashlib
import logging
import sys
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from traceback import clear_frames
from types import SimpleNamespace

import numpy as np

from .carrying import (
    carrying_geometry,
    carrying_spheres,
    configure_gripper_feedback,
)
from .place_batch import PLACE_BATCH_CAPACITY, place_batch_items_from_result
from .release_clearance import release_clearance_evidence
from .resident_robot import ResidentRobotGeometry, reserve_attachment_links
from .sdk_scene import sdk_scene_model
from .trajectory_postprocessing import (
    add_pose_derivatives,
    add_trajectory_derivatives,
    pre_release_place_from_result,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
from tianji_curobo._curobo_impl import (
    _create_world_clearance_checker,
    _graph_planner_rollout_config,
    _NativeCuroboBackend,
    _planner_interpolation_dt,
    _success_flag_is_true,
)
from tianji_curobo.contracts import (
    CuroboToolError,
    NativeGraspBatchItem,
    NativeIkBatchResult,
    PlanGraspBatchRequest,
    PlanPlaceRequest,
)
from tianji_curobo.curobo_robot_config import (
    _exclude_collision_links,
    collision_links_for_arm,
    create_frozen_peer_arm_robot_config,
)


def create_solver_configs(settings, *, robot, scene_model, collision_cache, batch_size,
                          ik_config_type, motion_config_type=None, graph_rollout=None):
    """Construct typed configs; never swap a live solver or its CUDA buffers."""
    common = {
        "robot": robot, "scene_model": sdk_scene_model(scene_model),
        "collision_cache": collision_cache,
        "self_collision_check": settings.self_collision_check,
        "position_tolerance": settings.ik_position_threshold_m,
        "orientation_tolerance": settings.ik_rotation_threshold_rad,
        "random_seed": settings.random_seed, "use_cuda_graph": True,
        "max_batch_size": batch_size, "multi_env": False, "max_goalset": 1,
    }
    ik = ik_config_type.create(
        **common, optimizer_configs=["ik/lbfgs_ik.yml"],
        num_seeds=settings.ik_num_seeds,
        seed_solver_num_seeds=settings.ik_seed_solver_num_seeds,
        success_requires_convergence=True,
        optimizer_collision_activation_distance=settings.trajectory_collision_clearance_m,
    )
    if motion_config_type is None:
        return ik, None
    # The public API accepts the already parsed RobotCfg. Sharing it within
    # this backend also keeps IK, TrajOpt and graph collision spheres in sync.
    common['robot'] = ik.robot_config
    motion = motion_config_type.create(
        **common, ik_optimizer_configs=["ik/lbfgs_ik.yml"],
        num_ik_seeds=settings.ik_num_seeds, num_trajopt_seeds=settings.trajopt_num_seeds,
        optimizer_collision_activation_distance=settings.optimizer_collision_activation_distance_m,
        graph_planner_rollout=graph_rollout, interpolation_dt=1.0 / settings.freq_hz,
    )
    # MotionPlannerCfg.create otherwise gives its embedded IK the wider TrajOpt
    # cost distance and cannot forward the LM seed count. This is a typed config
    # assignment before any planner/solver is instantiated, matching the SDK API.
    motion.ik_solver_config = ik
    return ik, motion


def validate_grasp_batch(result, validate_trajectory):
    """Keep only three-phase candidates passing the existing dense hard check."""
    checked = []
    for item in result.items:
        if not item.success:
            checked.append(item)
            continue
        phases = (item.approach, item.grasp, item.lift)
        try:
            # The Tianji checker samples every segment at <= 0.25 degrees,
            # including boundaries, and checks graph feasibility plus hard
            # world clearance for both active and stationary arm spheres.
            points = [point for phase in phases for point in phase.positions_rad]
            validate_trajectory(SimpleNamespace(points_deg=np.rad2deg(points).tolist()))
        except CuroboToolError as error:
            if error.code != "CUROBO_TRAJECTORY_COLLISION":
                raise
            checked.append(replace(
                item, success=False, approach_success=False, grasp_success=False,
                lift_success=False, approach=None, grasp=None, lift=None,
                grasp_end_joints_rad=None,
                status=f"{error.code}: {error}"[:256],
            ))
        else:
            checked.append(item)
    return replace(result, items=tuple(checked))


@dataclass(frozen=True)
class IkBatchWithToolPoses(NativeIkBatchResult):
    achieved_tool_poses: tuple = ()

    def to_payload(self):
        payload = super().to_payload()
        for item, pose in zip(payload["items"], self.achieved_tool_poses, strict=True):
            item["achieved_tool_pose"] = deepcopy(pose)
        return payload


@dataclass(frozen=True)
class GraspItemWithToolPose(NativeGraspBatchItem):
    grasp_endpoint_fk: dict | None = None


class ValidatedBatchBackend(_NativeCuroboBackend):
    def warmup(self):
        if not self.ik_only:
            return self.motion_planner.warmup(enable_graph=True,
                num_warmup_iterations=self._warmup_iterations)
        from curobo.types import JointState

        planner = self.motion_planner
        current = planner.default_joint_state
        positions = current.position.reshape(1, -1).repeat(planner.config.max_batch_size, 1)
        state = JointState.from_position(positions, joint_names=list(self.joint_names))
        goal = planner.compute_kinematics(state).tool_poses.as_goal()
        planner.solve_pose(goal, return_seeds=planner.config.num_seeds)
        planner.reset_seed()

    def tool_fk_evidence(self, joints_rad, scene_token):
        # Both public InverseKinematics and MotionPlanner expose this same FK
        # API. Evaluate the selected configuration, not the optimizer's goal.
        state = self._state(np.rad2deg(joints_rad).tolist())
        kinematics = self.motion_planner.compute_kinematics(state)
        pose = kinematics.tool_poses.get_link_pose(self.tool_frame)
        matrix = pose.get_matrix().detach().cpu().numpy().reshape(-1, 4, 4)
        if matrix.shape != (1, 4, 4) or not np.isfinite(matrix).all():
            raise CuroboToolError("invalid achieved tool FK", code="CUROBO_INVALID_RESULT",
                                  stage="validation", status=500)
        return {"source": "curobo_public_compute_kinematics", "frame": "world",
                "arm": self.arm, "tool_frame": self.tool_frame, "scene_token": scene_token,
                "joint_names": list(self.joint_names), "joints_rad": list(joints_rad),
                "T_world_tcp": matrix[0].tolist()}

    def solve_ik_batch_native(self, request):
        started = time.perf_counter()
        result = super().solve_ik_batch_native(request)
        solved = time.perf_counter()
        poses = tuple(self.tool_fk_evidence(item.joints_rad, request.scene_token)
                      if item.reachable else None for item in result.items)
        logging.getLogger(__name__).info("IK timing arm=%s count=%d solve_and_hard_check_s=%.4f per_candidate_fk_s=%.4f",
            self.arm, len(result.items), solved - started, time.perf_counter() - solved)
        return IkBatchWithToolPoses(result.items, result.scene_token, result.solve_time_s, poses)

    def attach_measured_carrying(self, binding, joints_deg):
        # IK-only configurations already contain these same fixed link spheres.
        # Only public motion planners expose AttachmentManager.
        spheres = self._torch.as_tensor(carrying_spheres(binding), device=self._device, dtype=self._dtype)
        link = f"jev_carried_object_{binding.arm}"
        if not self.ik_only:
            self.motion_planner.attachment_manager.update(spheres, self._state(joints_deg), link_name=link)
        self.carrying_link = link
        self.carrying_spheres_tool = spheres.detach().cpu().numpy().copy()
        self.carrying_scene_token = binding.scene_token

    def plan_place_batch_native(self, request):
        from scipy.spatial.transform import Rotation

        if not isinstance(request, PlanGraspBatchRequest) or len(request.goal_matrices_world) != PLACE_BATCH_CAPACITY:
            raise ValueError(f"place batch requires exactly {PLACE_BATCH_CAPACITY} validated goal matrices")
        if (getattr(self, "carrying_link", None) != f"jev_carried_object_{request.arm}"
                or getattr(self, "carrying_scene_token", None) != request.scene_token
                or getattr(self, "carrying_spheres_tool", None) is None):
            raise CuroboToolError("place batch requires the current uploaded attachment",
                code="CUROBO_RELEASE_GEOMETRY_STALE", stage="validation", status=422)
        matrices = np.asarray(request.goal_matrices_world, dtype=float)
        quaternions = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        def query_world_cost(world):
            tensor = self._torch.as_tensor(world, device=self._device, dtype=self._dtype)
            return self._motion_clearance_checker.get_collision_distance(tensor).detach().cpu().numpy().copy()
        clearance = []
        for matrix, quaternion in zip(matrices, quaternions, strict=True):
            endpoint = SimpleNamespace(arm=request.arm, goal_position_m=matrix[:3, 3],
                goal_quaternion_wxyz=quaternion,
                trajectory_collision_clearance_m=request.trajectory_collision_clearance_m)
            clearance.append(release_clearance_evidence(self.carrying_spheres_tool, endpoint,
                query_world_cost, scene_token=request.scene_token))
        current = self._state(request.start_joints_deg)
        state = self._joint_state_type.from_position(current.position.repeat(PLACE_BATCH_CAPACITY, 1),
            joint_names=list(self.joint_names))
        target = self._pose_type(
            position=self._torch.as_tensor(matrices[:, :3, 3], device=self._device, dtype=self._dtype),
            quaternion=self._torch.as_tensor(quaternions, device=self._device, dtype=self._dtype))
        goal = self._goal_tool_pose_type.from_poses({self.tool_frame: target},
            ordered_tool_frames=[self.tool_frame], num_goalset=1)
        result = None
        self.motion_planner.last_grasp_result = None
        try:
            try:
                # The application has already filtered place goals with held-object
                # IK. The public planner still runs its own IK before TrajOpt;
                # its internal preflight cannot be skipped through this API.
                result = self.motion_planner.plan_grasp(goal, state,
                    grasp_approach_axis=request.approach_axis,
                    grasp_approach_offset=request.approach_offset_m,
                    grasp_approach_in_tool_frame=request.approach_in_tool_frame,
                    plan_approach_to_grasp=True, plan_grasp_to_lift=False, disable_collision_links=[])
            finally:
                try:
                    self.motion_planner.update_tool_pose_criteria({self.tool_frame: self._tool_pose_criteria_type()})
                except Exception as error:
                    raise CuroboToolError("cuRobo batch criteria restoration failed",
                        code="CUROBO_GRASP_RESTORE_FAILED", stage="validation", status=500) from error
            try:
                items = place_batch_items_from_result(result, self.joint_names,
                    _planner_interpolation_dt(self.motion_planner))
            except (ValueError, TypeError, AttributeError) as error:
                clear_frames(error.__traceback__)
                raise CuroboToolError(str(error), code="CUROBO_INVALID_TRAJECTORY",
                    stage="validation", status=500) from error
            for item, evidence in zip(items, clearance, strict=True):
                item["release_clearance"] = evidence
                if evidence["conflict"]:
                    item.update(success=False, place_success=False, phases={}, status="release_clearance_conflict")
                    continue
                if not item["success"]:
                    continue
                # The SDK has already checked its raw path. The application
                # checks the final, time-scaled command after interpolation.
            return {"items": items, "batch_size": PLACE_BATCH_CAPACITY, "batch_capacity": PLACE_BATCH_CAPACITY,
                "pose_call_timings": list(getattr(self.motion_planner, "pose_call_timings", []))}
        finally:
            result = None
            self.motion_planner.last_grasp_result = None

    def destroy(self):
        try:
            link = getattr(self, "carrying_link", None) or getattr(self, "configured_carrying_link", None)
            if link is not None and self.motion_planner is not None and not self.ik_only:
                self.motion_planner.attachment_manager.detach(link_name=link)
                self.motion_planner.disable_link_collision([link])
        finally:
            self.carrying_link = self.carrying_spheres_tool = self.carrying_scene_token = None
            super().destroy()

    def plan_grasp_batch_native(self, request):
        started = time.perf_counter()
        self.motion_planner.last_grasp_result = None
        try:
            result = super().plan_grasp_batch_native(request)
            result = add_trajectory_derivatives(result, self.motion_planner.last_grasp_result)
        finally:
            self.motion_planner.last_grasp_result = None
        native_done = time.perf_counter()
        items = []
        for item in result.items:
            evidence = (self.tool_fk_evidence(item.grasp.positions_rad[-1], request.scene_token)
                        if item.grasp_success and item.grasp is not None else None)
            items.append(GraspItemWithToolPose(
                **{field.name: getattr(item, field.name) for field in fields(NativeGraspBatchItem)},
                grasp_endpoint_fk=evidence))
        fk_done = time.perf_counter()
        logging.getLogger(__name__).info("grasp timing arm=%s count=%d native_s=%.4f fk_s=%.4f",
            self.arm, len(items), native_done - started, fk_done - native_done)
        return replace(result, items=tuple(items))


def support_departure_criteria(criteria_type, request, non_terminal_scale):
    criteria = criteria_type.linear_motion(axis=request.axis,
        non_terminal_scale=non_terminal_scale * 100000,
        project_distance_to_goal=request.reference_frame == "tool")
    return criteria


@contextmanager
def linear_retreat_criteria(backend, axis, in_tool_frame):
    """Official tool-pose criterion for the open-hand lift; restore on failure."""
    criteria_type = backend._tool_pose_criteria_type
    try:
        criteria = criteria_type.linear_motion(axis=axis, project_distance_to_goal=in_tool_frame)
        backend.motion_planner.update_tool_pose_criteria({backend.tool_frame: criteria})
        yield
    finally:
        backend.motion_planner.update_tool_pose_criteria({backend.tool_frame: criteria_type()})


class ValidatedSingleBackend(_NativeCuroboBackend):
    def _state(self, joints_deg):
        state = super()._state(joints_deg)
        terminal = getattr(self, 'release_start_state', None)
        if terminal is not None:
            # Public JointState input, not solver internals. Official plan_grasp
            # retains dt here; dropping it disables the SDK's velocity-aware IK.
            for name in ('velocity', 'acceleration'):
                setattr(state, name, self._torch.as_tensor([terminal[name]], device=self._device, dtype=self._dtype))
            state.dt = self._torch.as_tensor([terminal['dt']], device=self._device, dtype=self._dtype)
        return state

    def warmup(self):
        return self.motion_planner.warmup(enable_graph=True,
            num_warmup_iterations=self._warmup_iterations)

    def resolve_release_height(self, request, gripper_spheres):
        from .demo_placement import resolve_release_height

        if (getattr(self, "carrying_link", None) != f"jev_carried_object_{request.arm}"
                or not getattr(self, "carrying_scene_token", None)
                or getattr(self, "carrying_spheres_tool", None) is None):
            raise CuroboToolError("release height requires the current uploaded attachment",
                code="CUROBO_RELEASE_GEOMETRY_STALE", stage="validation", status=422)

        def query_world_cost(world):
            tensor = self._torch.as_tensor(world, device=self._device, dtype=self._dtype)
            return self._motion_clearance_checker.get_collision_distance(tensor).detach().cpu().numpy().copy()

        return resolve_release_height(self.carrying_spheres_tool, gripper_spheres, request, query_world_cost,
                                      scene_token=self.carrying_scene_token)

    def _plan_pre_release_native(self, request):
        """Use the public two-phase mode without constructing a vendor place plan."""
        if not isinstance(request, PlanPlaceRequest):
            raise CuroboToolError("request must be a PlanPlaceRequest", code="CUROBO_INVALID_REQUEST",
                stage="input", status=400)
        state, goal = self._state_and_goal(request.start_joints_deg,
            request.goal_position_m, request.goal_quaternion_wxyz)
        result = None
        try:
            try:
                result = self.motion_planner.plan_grasp(goal, state,
                    grasp_approach_axis=request.approach_axis,
                    grasp_approach_offset=request.approach_offset_m,
                    grasp_approach_in_tool_frame=request.approach_in_tool_frame,
                    plan_approach_to_grasp=True, plan_grasp_to_lift=False,
                    disable_collision_links=[])
            except CuroboToolError:
                raise
            except Exception as error:
                raise CuroboToolError("cuRobo pre-release trajectory optimization failed",
                    code="CUROBO_GRASP_FAILED", stage="trajopt", status=422,
                    details={"arm": self.arm}) from error
            finally:
                try:
                    self.motion_planner.update_tool_pose_criteria({self.tool_frame: self._tool_pose_criteria_type()})
                except Exception as error:
                    raise CuroboToolError("cuRobo grasp planner criteria restoration failed",
                        code="CUROBO_GRASP_RESTORE_FAILED", stage="validation", status=500,
                        details={"arm": self.arm}) from error
            for sdk_name, phase_name in (("approach", "approach"), ("grasp", "place")):
                if not _success_flag_is_true(getattr(result, f"{sdk_name}_success", None)):
                    status = str(getattr(result, "status", "") or "")
                    raise CuroboToolError(f"cuRobo {sdk_name} planning failed", code="CUROBO_GRASP_FAILED",
                        stage="trajopt", status=422, details={"arm": self.arm, "phase": sdk_name,
                            "compound_phase": phase_name, "status": status})
            try:
                return pre_release_place_from_result(result, self.joint_names,
                    _planner_interpolation_dt(self.motion_planner))
            except ValueError as error:
                # Failed derivative conversion frames otherwise retain the
                # borrowed SDK result while this validation error is reported.
                clear_frames(error.__traceback__)
                raise CuroboToolError(str(error), code="CUROBO_INVALID_TRAJECTORY",
                    stage="validation", status=500, details={"arm": self.arm}) from error
        finally:
            # Exception tracebacks must not retain the SDK's reused GPU buffers.
            result = None
            self.motion_planner.last_grasp_result = None
            self.motion_planner.last_pose_result = None

    def _plan_axis_linear_candidate(self, request, current_state, goal, non_terminal_scale):
        if not getattr(self, "support_departure_planning", False):
            return super()._plan_axis_linear_candidate(request, current_state, goal, non_terminal_scale)
        try:
            criteria = support_departure_criteria(self._tool_pose_criteria_type, request, non_terminal_scale)
            self.motion_planner.update_tool_pose_criteria({self.tool_frame: criteria})
            return self._plan_goal_native(request, current_state, goal)
        finally:
            self.motion_planner.update_tool_pose_criteria({self.tool_frame: self._tool_pose_criteria_type()})

    def attach_measured_carrying(self, binding, joints_deg):
        # Configured spheres already reach IK, graph and checker, whose models
        # are built separately. Public AttachmentManager updates TrajOpt's same
        # measured link without touching any private SDK state.
        spheres = self._torch.as_tensor(carrying_spheres(binding), device=self._device, dtype=self._dtype)
        state = self._state(joints_deg)
        self.motion_planner.attachment_manager.update(spheres, state, link_name=f"jev_carried_object_{binding.arm}")
        self.carrying_link = f"jev_carried_object_{binding.arm}"
        self.carrying_spheres_tool = spheres.detach().cpu().numpy().copy()
        self.carrying_scene_token = binding.scene_token

    def plan_place_native(self, request):
        self.motion_planner.last_grasp_result = None
        try:
            if (getattr(self, "carrying_link", None) != f"jev_carried_object_{request.arm}"
                    or not getattr(self, "carrying_scene_token", None)
                    or getattr(self, "carrying_spheres_tool", None) is None):
                raise CuroboToolError("release endpoint requires the current uploaded attachment",
                    code="CUROBO_RELEASE_GEOMETRY_STALE", stage="validation", status=422)

            def query_world_cost(world):
                tensor = self._torch.as_tensor(world, device=self._device, dtype=self._dtype)
                return self._motion_clearance_checker.get_collision_distance(tensor).detach().cpu().numpy().copy()

            # Typed PlanPlaceRequest has no scene token. The runtime validates the
            # binding and refreshes this upload from it under its scene/GPU lock.
            evidence = release_clearance_evidence(self.carrying_spheres_tool, request, query_world_cost,
                                                  scene_token=self.carrying_scene_token)
            if evidence["conflict"]:
                raise CuroboToolError("release_clearance_conflict: attached surface spheres violate hard world clearance at the requested release goal",
                    code="release_clearance_conflict", stage="validation", status=422, details=evidence)
            retained = self._plan_pre_release_native(request)
            return retained
        finally:
            self.motion_planner.last_grasp_result = None
            self.motion_planner.last_pose_result = None

    def plan_to_pose_native(self, request):
        self.motion_planner.last_pose_result = None
        try:
            try:
                departure = getattr(self, 'release_departure', None)
                scope = linear_retreat_criteria(self, *departure) if departure else nullcontext()
                # The native runtime owns the GPU lock while calling here.
                with scope:
                    result = super().plan_to_pose_native(request)
            except CuroboToolError as error:
                # The pinned wrapper uses one code for an SDK exception and a
                # valid unsuccessful result. Only the latter permits another yaw.
                if (error.code == "CUROBO_TRAJOPT_FAILED"
                        and str(error) == "cuRobo trajectory optimization found no valid trajectory"):
                    raise CuroboToolError(str(error), code="CUROBO_NO_VALID_TRAJECTORY",
                        stage=error.stage, status=error.status, retryable=error.retryable,
                        details=error.details) from error
                raise
            retained = add_pose_derivatives(result, self.motion_planner.last_pose_result)
            return retained
        finally:
            self.motion_planner.last_pose_result = None

    def plan_axis_linear_native(self, request):
        self.motion_planner.last_pose_result = None
        try:
            result = super().plan_axis_linear_native(request)
            return add_pose_derivatives(result, self.motion_planner.last_pose_result)
        finally:
            self.motion_planner.last_pose_result = None

    def destroy(self):
        try:
            link = getattr(self, "carrying_link", None) or getattr(self, "configured_carrying_link", None)
            if link is not None and self.motion_planner is not None:
                self.motion_planner.attachment_manager.detach(link_name=link)
                self.motion_planner.disable_link_collision([link])
        finally:
            self.carrying_link = None
            self.carrying_spheres_tool = None
            self.carrying_scene_token = None
            super().destroy()


def effective_solver_config(planner, settings, *, ik_only, robot_config_path):
    """Read values from instantiated solver configs, not the HTTP payload."""
    ik = planner.config if ik_only else planner.ik_solver.config
    values = {
        "ik_num_seeds": ik.num_seeds,
        "ik_seed_solver_num_seeds": ik.seed_solver_num_seeds,
        "ik_position_threshold_m": ik.position_tolerance,
        "ik_rotation_threshold_rad": ik.orientation_tolerance,
        "random_seed": ik.random_seed, "self_collision_check": ik.self_collision_check,
        "ik_collision_activation_distance_m": ik.optimizer_collision_activation_distance,
        "success_requires_convergence": ik.success_requires_convergence,
        "use_cuda_graph": ik.use_cuda_graph, "max_goalset": ik.max_goalset,
        "multi_env": ik.multi_env, "max_batch_size": ik.max_batch_size,
        "ik_optimizer_configs": ["ik/lbfgs_ik.yml"],
        "trajectory_collision_clearance_m": settings.trajectory_collision_clearance_m,
        "world_clearance_geometry": "configured_robot_spheres",
        "robot_config_sha256": hashlib.sha256(robot_config_path.read_bytes()).hexdigest(),
    }
    if not ik_only:
        trajectory = planner.trajopt_solver.config
        values.update(
            trajopt_num_seeds=trajectory.num_seeds,
            optimizer_collision_activation_distance_m=trajectory.optimizer_collision_activation_distance,
            trajopt_position_threshold_m=trajectory.position_tolerance,
            trajopt_rotation_threshold_rad=trajectory.orientation_tolerance,
            trajopt_self_collision_check=trajectory.self_collision_check,
            interpolation_dt_s=trajectory.interpolation_dt,
            plan_max_attempts=planner.planning_settings.plan_max_attempts,
            finetune_attempts=planner.planning_settings.finetune_attempts,
            finetune_dt_scale=planner.planning_settings.finetune_dt_scale,
            trajectory_hard_check=True,
            planner_implementation='unmodified_public_sdk_composition',
            grasp_internal_pose_settings='official_defaults_not_overridden',
            grasp_internal_max_attempts=1 if planner.batch else 5,
            grasp_advanced_attempt_controls_applied=False,
        )
    return values


def maximum_world_sphere_radius(runtime):
    """Use the original spheres and their authored per-link padding."""
    import yaml

    combined = yaml.safe_load(runtime._robot_config_path.read_text(encoding="utf-8"))
    kinematics = combined["robot_cfg"]["kinematics"]
    buffers = kinematics.get("collision_sphere_buffer", 0.0)
    largest = 0.0
    for link, spheres in kinematics["collision_spheres"].items():
        padding = buffers.get(link, 0.0) if isinstance(buffers, dict) else buffers
        for sphere in spheres:
            largest = max(largest, float(sphere["radius"]) + float(padding))
    return largest


def create_support_frame_robot_config(robot_config, frame):
    result = deepcopy(robot_config)
    kin = result["robot_cfg"]["kinematics"]
    name = frame["link_name"]
    kin.setdefault("extra_links", {})[name] = {"fixed_transform": frame["fixed_transform"],
        "joint_name": f"jev_support_axis_joint_{frame['arm']}", "joint_type": "FIXED",
        "link_name": name, "parent_link_name": frame["parent_link_name"]}
    kin["tool_frames"] = [name]
    return result


def build_geometry_backend(runtime, arm, *, batch_size, ik_only, batch_planner_type=None,
                           single_planner_type=None):
    """Assemble the A/B application adapter using only public NVIDIA classes."""
    # Isolated SDK comparisons may inject the public planner constructor.
    # Production keeps the configured adapter and its advanced settings.
    import curobo
    import torch
    import yaml
    from curobo.batch_motion_planner import BatchMotionPlanner
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Cuboid
    from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

    from .public_planner import PublicPlanner

    if arm not in ("A", "B"):
        raise ValueError("geometry service requires an A/B solver")
    settings = runtime.planning_settings
    combined = yaml.safe_load(runtime._robot_config_path.read_text(encoding="utf-8"))
    peer_arm = "B" if arm == "A" else "A"
    robot = create_frozen_peer_arm_robot_config(
        combined, arm, urdf_path=runtime._urdf_path, asset_root_path=runtime._asset_root_path,
    )
    robot = _exclude_collision_links(robot, runtime._excluded_collision_links)
    feedback = getattr(runtime, "gripper_feedback", None)
    base_robot = deepcopy(robot)
    robot = reserve_attachment_links(configure_gripper_feedback(robot, {"A": 0., "B": 0.}, runtime._urdf_path))
    carrying = getattr(runtime, "carrying_binding", None)
    support_frame = getattr(runtime, "support_frame", None)
    if support_frame is not None:
        if support_frame["arm"] != arm or batch_size is not None or ik_only or carrying is not None:
            raise ValueError("support frame is scoped to one detached prefix planner")
        robot = create_support_frame_robot_config(robot, support_frame)
        base_robot = create_support_frame_robot_config(base_robot, support_frame)
    carrying_link = None
    if carrying is not None:
        if carrying.arm != arm:
            raise ValueError("carrying must match the planning arm")
        carrying_link = "jev_carried_object_" + arm
    configured_links = set(robot["robot_cfg"]["kinematics"]["collision_link_names"])
    active_links = [name for name in collision_links_for_arm(combined, arm, include_shared=True)
                    if name in configured_links]
    active_links.append('jev_carried_object_' + arm)
    peer_links = [name for name in collision_links_for_arm(combined, peer_arm) if name in configured_links]
    scene = {name: value for name, value in (runtime._scene_model or {}).items() if name != "voxel"}
    ik_config, motion_config = create_solver_configs(
        settings, robot=robot, scene_model=scene or {"cuboid": {}},
        collision_cache=runtime._collision_cache, batch_size=1 if batch_size is None else batch_size,
        ik_config_type=InverseKinematicsCfg,
        motion_config_type=None if ik_only else MotionPlannerCfg,
        graph_rollout=None if ik_only else _graph_planner_rollout_config(),
    )
    if ik_only:
        planner = InverseKinematics(ik_config)
    else:
        if batch_size is None:
            sdk = (single_planner_type or MotionPlanner)(motion_config)
        else:
            sdk = (batch_planner_type or BatchMotionPlanner)(motion_config)
        planner = PublicPlanner(sdk, settings, batch=batch_size is not None)
    # The hard checker shares the optimizer's measured gripper configuration
    # and original spheres, including the frozen peer.
    checker = _create_world_clearance_checker(
        checker_config_type=RobotCollisionCheckerCfg, checker_type=RobotCollisionChecker,
        robot_config=ik_config.robot_config, motion_planner=planner,
        activation_distance_m=settings.trajectory_collision_clearance_m,
    )
    reference = Kinematics(KinematicsCfg.from_data_dict(
        robot, tool_frames=peer_links, device_cfg=planner.device_cfg,
    ))
    zero = torch.zeros((1, len(reference.joint_names)), device=planner.device_cfg.device,
                       dtype=planner.device_cfg.dtype)
    poses = reference.get_link_poses(zero, peer_links)
    matrices = Pose(position=poses.position.reshape(-1, 3),
                    quaternion=poses.quaternion.reshape(-1, 4)).get_matrix()
    peer_inverses = {name: torch.linalg.inv(matrices[index].to(dtype=torch.float64)).detach().clone()
                     for index, name in enumerate(peer_links)}
    # Both IK and batch planning need only FK to place the stationary arm's
    # spheres. Constructing another MotionPlanner for this would allocate
    # unnecessary solvers and introduce a different planning configuration.
    peer_config = create_frozen_peer_arm_robot_config(
        combined, peer_arm, urdf_path=runtime._urdf_path, asset_root_path=runtime._asset_root_path,
    )
    peer_config = _exclude_collision_links(peer_config, runtime._excluded_collision_links)
    base_peer_config = deepcopy(peer_config)
    peer_config = configure_gripper_feedback(peer_config, {"A": 0., "B": 0.}, runtime._urdf_path)
    peer_kinematics = Kinematics(KinematicsCfg.from_data_dict(peer_config, device_cfg=planner.device_cfg))
    backend_type = ValidatedSingleBackend if batch_size is None else ValidatedBatchBackend
    backend = backend_type(
        arm=arm, motion_planner=planner, peer_kinematics=peer_kinematics, torch_module=torch,
        joint_state_type=JointState, pose_type=Pose, obstacle_cuboid_type=Cuboid,
        goal_tool_pose_type=GoalToolPose, tool_pose_criteria_type=ToolPoseCriteria,
        motion_clearance_checker=checker, active_collision_links=active_links,
        peer_collision_links=peer_links, peer_baseline_inverse_by_link=peer_inverses,
        warmup_iterations=1, warmup_joint_space=False,
    )
    backend.configured_carrying_link = carrying_link
    backend.ik_only = ik_only
    backend.resident_geometry = ResidentRobotGeometry(base_robot, runtime._urdf_path,
        peer_robot=base_peer_config, peer_links=peer_links)
    runtime._curobo_source = str(Path(curobo.__file__).resolve().parents[1])
    runtime._curobo_version = str(getattr(curobo, "__version__", "unknown"))
    runtime._gpu_available = bool(torch.cuda.is_available())
    effective = effective_solver_config(planner, settings, ik_only=ik_only,
                                        robot_config_path=runtime._robot_config_path)
    attachment_geometry = None
    if carrying is not None:
        attachment_geometry = carrying_geometry(carrying)
        attachment_geometry.pop("spheres")
    effective.update(optimizer_attachment=carrying_link is not None,
                     attachment_geometry=attachment_geometry,
                     carried_link=carrying_link,
                     geometry_coverage="measured_surface_voxels_with_uncertainty" if carrying_link else None,
                     full_object_coverage=False,
                     carrying_owner_id=None if carrying is None else carrying.owner_id,
                     gripper_configuration_source="measured_feedback" if feedback is not None else "configured_open_gripper")
    if batch_size is None and not ik_only:
        effective.update(finetune_attempts=None, finetune_dt_scale=None,
            finetune_settings_applied=False,
            api_limitations=["Pinned single MotionPlanner.plan_pose does not expose batch finetuning controls"])
    runtime.effective_config.setdefault(arm, {})["ik" if ik_only else "plan"] = effective
    backend.effective_config = effective
    return backend


def build_dual_geometry_backend(runtime):
    """One official MotionPlanner and checker for both TCPs and fourteen joints."""
    import torch
    import yaml
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.inverse_kinematics import InverseKinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria
    from tianji_curobo.curobo_robot_config import create_dual_arm_robot_config

    from .dual_planning import ValidatedDualBackend

    combined = yaml.safe_load(runtime._robot_config_path.read_text(encoding="utf-8"))
    robot = create_dual_arm_robot_config(combined, urdf_path=runtime._urdf_path,
                                         asset_root_path=runtime._asset_root_path)
    base_robot = deepcopy(robot)
    robot = reserve_attachment_links(configure_gripper_feedback(robot, {"A": 0., "B": 0.}, runtime._urdf_path))
    scene = {k: v for k, v in (runtime._scene_model or {}).items() if k != "voxel"}
    ik_config, config = create_solver_configs(runtime.planning_settings, robot=robot,
        scene_model=scene or {"cuboid": {}}, collision_cache=runtime._collision_cache,
        batch_size=1, ik_config_type=InverseKinematicsCfg, motion_config_type=MotionPlannerCfg,
        graph_rollout=_graph_planner_rollout_config())
    planner = MotionPlanner(config)
    checker = _create_world_clearance_checker(
        checker_config_type=RobotCollisionCheckerCfg, checker_type=RobotCollisionChecker,
        robot_config=ik_config.robot_config, motion_planner=planner,
        activation_distance_m=runtime.planning_settings.trajectory_collision_clearance_m)
    backend = ValidatedDualBackend(arm="AB", motion_planner=planner, torch_module=torch,
        joint_state_type=JointState, pose_type=Pose, goal_tool_pose_type=GoalToolPose,
        tool_pose_criteria_type=ToolPoseCriteria, motion_clearance_checker=checker,
        active_collision_links=list(robot["robot_cfg"]["kinematics"]["collision_link_names"]),
        peer_collision_links=(), warmup_joint_space=False)
    backend.ik_only = False
    backend.resident_geometry = ResidentRobotGeometry(base_robot, runtime._urdf_path)
    return backend

