"""GPU service for registered RGBD maps and the copied planning wrapper."""

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .inventory_map import (
    apply_inventory_update,
    certify_workspace,
    freeze_inventory_scene,
    reconcile_inventory,
)
from .motion_coverage import MotionCoverage
from .planning_batches import TRAJECTORY_BATCH_CAPACITY, pad_trajectory_batch
from .planning_settings import PlanningSettings, load_settings
from .planning_timing import collect_timings, measured, timed_stage
from .sdk_scene import sdk_scene_model, sdk_voxel_grid
from .trajectory_postprocessing import PreReleasePlacePlan

ROOT = Path(__file__).resolve().parents[2]
from tianji_curobo._curobo_impl import CuroboRuntime, _busy_error, _validated_timeout
from tianji_curobo.contracts import (
    CuroboToolError,
    PlanAxisLinearRequest,
    PlanToPoseRequest,
)
from tianji_curobo.contracts import (
    PlanPlaceRequest as VendorPlanPlaceRequest,
)
from tianji_curobo.curobo_scene import collision_cache_for_scene

from .backend_builder import build_dual_geometry_backend, build_geometry_backend
from .carrying import (
    CarryingBinding,
    canonical_gripper_feedback,
    check_command_path,
    support_departure_frame,
    support_gap_m,
    support_prefix_distance,
    validate_phase_request,
    validate_support_goal_projection,
)
from .perception_mapping import (
    ObservedFreeSpaceGuard,
    PerceptionMapper,
    create_feedback_segmenter,
    decode_mapping_payload,
    filter_mapping_views,
    mapping_filter_settings,
)
from .planning_contracts import PlanGraspBatchRequest, PlanPlaceRequest, SolveIkBatchRequest

SDK_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"
LOGGER = logging.getLogger(__name__)
IK_BATCH_CAPACITY = 100
INITIAL_VOXEL_LAYER_CAPACITY = 16


def request_size_limit(path):
    # Bounded sphere batches contain up to 262144 four-number records. Keep
    # the existing limit for all other endpoints; decoded geometry is checked too.
    return (32 if path == '/tools/query_world_spheres' else 8) * 1024 * 1024


def geometry_collision_cache(scene_model):
    """Small collision arrays; grow before upload without dropping geometry.

    cuRobo launches sphere-obstacle work across allocated obstacle slots, so
    reserving hundreds of empty mesh/cuboid slots slows every optimizer step.
    Keep common empty-world minima and round actual growth to capacity buckets.
    """
    cache = collision_cache_for_scene(scene_model)
    model = scene_model or {}
    # Keep a small four-mesh working set resident: a third CAD instance must
    # not destroy every warmed solver partway through manipulation.
    for kind, minimum in (('mesh', 4), ('cuboid', 5)):
        count = len(model.get(kind, {}))
        cache[kind] = max(minimum, 1 << (max(1, count) - 1).bit_length())
    count = len(model.get('voxel', {}))
    # Retained surfaces from multiple captures can exceed the object count.
    # Reserve their slots during startup, so a fifth measured layer does not
    # destroy every warmed IK and motion planner in the middle of a task.
    cache['voxel']['layers'] = max(INITIAL_VOXEL_LAYER_CAPACITY,
                                   1 << (max(1, count) - 1).bit_length())
    return cache


def operation_error_response(error):
    """Keep resource exhaustion distinct from a valid planner rejection."""
    cause = error
    while cause is not None:
        if type(cause).__name__ == 'OutOfMemoryError' and type(cause).__module__.startswith('torch'):
            return 503, {'ok': False, 'error': {
                'code': 'CUROBO_CUDA_OUT_OF_MEMORY',
                'message': 'GPU memory exhausted during cuRobo computation; no planning result was obtained',
                'details': {'resource': 'cuda_memory', 'planning_result': 'unavailable',
                            'allocator_message': str(cause)[:2000]}}}
        cause = cause.__cause__
    return getattr(error, 'status', 422), {'ok': False, 'error': {
        'code': getattr(error, 'code', type(error).__name__),
        'message': str(error), 'details': getattr(error, 'details', None)}}


def cuda_memory_snapshot():
    """Report this process's allocator and device-wide headroom in bytes."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'free_bytes': free, 'total_bytes': total}


def measured_target_layers(runtime, scene):
    target = scene.get('target_id')
    if not target or scene.get('exclusion_state') != 'selected':
        raise ValueError('open grasp requires a selected measured target')
    prefix = 'semantic_instance::' + target
    layers = [(kind, name) for kind in ('cuboid', 'mesh', 'voxel')
              for name in (runtime.geometry_model or {}).get(kind, {})
              if name == prefix or name.startswith(prefix + '::')]
    if not layers:
        raise ValueError('open grasp target collision layers missing')
    return layers


@contextmanager
def measured_target_collision_scope(runtime, planner, phase):
    """Retain the measured target during open approach, restoring selection.

    The caller holds the runtime GPU/scene lock and has refreshed this planner
    from geometry_scene, so disabled_obstacle_names records its original flags.
    Closing contact and carrying retain their separate phase policy.
    """
    proof = {'policy': 'existing_phase_selection', 'target_layers': []}
    if phase not in {'approach_grasp', 'grasp', 'grasp_closure'}:
        yield proof
        return
    scene = runtime.geometry_scene
    layers = measured_target_layers(runtime, scene)
    disabled = set(scene.get('disabled_obstacle_names', ()))
    changed = []
    try:
        for kind, name in layers:
            if name in disabled:
                data = getattr(planner.scene_collision_checker.data,
                               {'cuboid': 'cuboids', 'mesh': 'meshes', 'voxel': 'voxels'}[kind])
                changed.append((data, name))
                data.set_enabled(name, enabled=True, env_idx=0)
        yield {'policy': 'measured_target_retained_during_open_approach',
               'target_layers': [name for _, name in layers], 'target_contact_exemptions': []}
    finally:
        for data, name in reversed(changed):
            data.set_enabled(name, enabled=False, env_idx=0)


class GeometryRuntime(CuroboRuntime):
    def __init__(self, **kwargs):
        kwargs.setdefault("robot_config_path", ROOT / "resources/robot/marvin_gui_dual.yml")
        kwargs.setdefault("urdf_path", ROOT / "assets/marvin_urdf/marvin_with_grippers.urdf")
        kwargs.setdefault("asset_root_path", ROOT / "assets")
        super().__init__(scene="none", warmup=False, **kwargs)
        self.geometry_scene = None
        self.geometry_grids = None
        self.geometry_model = None
        self._reachability_solvers = {}
        self.planning_settings = PlanningSettings()
        self.geometry_solver_profile = None
        self.effective_config = {}
        self._maximum_sphere_radius = None
        self.perception_mapper = None
        self.perception_gripper_feedback = None
        self.observed_free_guard = None
        self.motion_coverage = None
        self._inventory_archive = None
        self.carrying_binding = None
        self.gripper_feedback = None
        self.support_frame = None
        self.logical_carrying_binding = None
        self._carrying_configuration_key = None
        self._command_checkers = {}
        self._planner_configurations = {}
        self._planner_mode = "open"
        self._carrying_scope_lock = threading.RLock()
        self._planning_scope = threading.local()
        self.warmup_status = {"ready": False, "profiles": []}

    def warm_start(self, settings):
        """Prepare resident open-hand solvers before accepting task requests."""
        from .reachability import PositionReachability

        self.configure_planning(settings)
        self.warmup_status = {"ready": False, "profiles": [], "profile_memory": []}
        started = time.perf_counter()
        with self.carrying_scope(None, {"A": 0., "B": 0.}), self.gpu_lock:
            self.warmup_status['initial_memory'] = cuda_memory_snapshot()
            self._collision_cache = geometry_collision_cache(None)
            self._scene_model = {"cuboid": {}}
            for arm in ('A', 'B'):
                # Larger trajectory batches remain available on demand. Their
                # resident CUDA graphs otherwise displace first-capture masking.
                for capacity, ik_only in ((100, True), (2, False), (None, False)):
                    backend = self._create_real_backend_unlocked(arm, batch_size=capacity, ik_only=ik_only)
                    try:
                        backend.warmup()
                    except Exception:
                        backend.destroy()
                        raise
                    clearance = settings.trajectory_collision_clearance_m
                    if ik_only:
                        self._ik_batch_backends[(arm, clearance, capacity)] = backend
                    elif capacity is None:
                        self._backends[arm] = backend
                    else:
                        self._batch_backends[(arm, capacity, clearance,
                                              settings.optimizer_collision_activation_distance_m)] = backend
                    self._record_warmup_profile([arm, capacity, ik_only])
                    LOGGER.info('Startup warmed arm=%s capacity=%s ik_only=%s', arm, capacity, ik_only)
                # Compound grasp/place has its own cache in the pinned adapter.
                # Warming the ordinary pose instance does not populate it.
                backend = self._compound_backend_unlocked(arm,
                    settings.trajectory_collision_clearance_m,
                    settings.optimizer_collision_activation_distance_m)
                backend.warmup()
                self._record_warmup_profile([arm, 'compound', False])
                LOGGER.info('Startup warmed arm=%s compound grasp/place', arm)
            backend = self._dual_backend_unlocked()
            backend.warmup()
            self._record_warmup_profile(['AB', None, False])
            for arm in ('A', 'B'):
                solver = PositionReachability(self._robot_config_path, self._urdf_path,
                                              self._asset_root_path, arm)
                solver.warmup()
                self._reachability_solvers[arm] = solver
                self._record_warmup_profile([arm, 'position_reachability', True])
        self.warmup_status.update(ready=True, wall_s=time.perf_counter() - started)

    def _record_warmup_profile(self, profile):
        memory = {'profile': profile, **cuda_memory_snapshot()}
        self.warmup_status['profiles'].append(profile)
        self.warmup_status['profile_memory'].append(memory)
        LOGGER.info('Startup profile memory %s', json.dumps(memory))

    def configure_planning(self, settings):
        """CUDA graphs cannot outlive the solver configuration they captured."""
        if settings.solver_payload() == self.planning_settings.solver_payload():
            self.planning_settings = settings
            return
        with self.gpu_lock:
            LOGGER.info('Invalidating solver caches: solver settings changed old=%s new=%s',
                        self.planning_settings.solver_payload(), settings.solver_payload())
            self._clear_planners_unlocked()
            self.planning_settings = settings

    def _clear_planners_unlocked(self):
        """Invalidate all references before releasing configuration-bound buffers."""
        caches = (self._backends, self._dual_backends, self._compound_backends,
                  self._batch_backends, self._ik_batch_backends)
        backends = {id(backend): backend for cache in caches for backend in cache.values()}
        for _, saved in self._planner_configurations.values():
            for name in self._planner_cache_fields[:5]:
                backends.update({id(backend): backend for backend in saved[name].values()})
        self._planner_configurations.clear()
        for cache in caches:
            cache.clear()
        self._scene_snapshot_key_by_arm.clear()
        self._peer_collision_snapshot_by_arm.clear()
        self.effective_config.clear()
        self._command_checkers.clear()
        errors = []
        for backend in backends.values():
            try:
                backend.destroy()
            except Exception as error:
                LOGGER.exception("cuRobo backend cleanup failed")
                errors.append(error)
        if errors:
            raise CuroboToolError("cuRobo backend cleanup failed", stage="cleanup",
                                  code="CUROBO_BACKEND_CLEANUP_FAILED") from errors[0]

    _planner_cache_fields = ("_backends", "_dual_backends", "_compound_backends", "_batch_backends",
                            "_ik_batch_backends", "_scene_snapshot_key_by_arm",
                            "_peer_collision_snapshot_by_arm", "effective_config", "_command_checkers")

    def _select_planner_configuration_unlocked(self, key, mode):
        if key == self._carrying_configuration_key:
            return
        self._planner_configurations[self._planner_mode] = (
            self._carrying_configuration_key, {name: getattr(self, name) for name in self._planner_cache_fields})
        saved_key, saved = self._planner_configurations.get(mode, (None, {}))
        if saved_key != key:
            LOGGER.info('Selecting robot geometry mode=%s previous_key=%s new_key=%s', mode, saved_key, key)
            backends = {id(backend): backend for name in self._planner_cache_fields[:5]
                        for backend in saved.get(name, {}).values()}
            for backend in backends.values():
                backend.destroy()
            saved = {name: OrderedDict() if name in self._planner_cache_fields[2:5] else {}
                     for name in self._planner_cache_fields}
        for name in self._planner_cache_fields:
            setattr(self, name, saved[name])
        self._planner_configurations[mode] = (key, saved)
        self._planner_mode, self._carrying_configuration_key = mode, key

    @measured('backend_construction_and_warmup')
    def _create_real_backend_unlocked(self, arm, batch_size=None, compound_planner=False,
                                      trajectory_collision_clearance_m=.01,
                                      optimizer_collision_activation_distance_m=.03,
                                      ik_collision_activation_distance_m=None, ik_only=False):
        started = time.perf_counter()
        LOGGER.info("cuRobo backend construction: arm=%s capacity=%s ik_only=%s",
                    arm, batch_size, ik_only)
        if arm == "AB":
            backend = build_dual_geometry_backend(self)
        else:
            backend = build_geometry_backend(self, arm, batch_size=batch_size, ik_only=ik_only,
                                             compound_planner=compound_planner)
        if self.warmup_status['ready']:
            # Only an explicit topology/capacity/settings change can reach this
            # path after startup. Attachment and feedback retain the solvers.
            try:
                backend.warmup()
            except Exception:
                backend.destroy()
                raise
        LOGGER.info("cuRobo backend construction complete: arm=%s capacity=%s ik_only=%s wall_s=%.3f",
                    arm, batch_size, ik_only, time.perf_counter() - started)
        return backend

    @contextmanager
    def carrying_scope(self, binding, feedback=None, support_frame=None):
        # Cache compiled geometry, not task IDs or entire measured point clouds.
        # Binding freshness is checked at the request boundary and on upload.
        feedback = canonical_gripper_feedback(feedback or {"A": 0., "B": 0.})
        # Measured finger/attachment data update fixed buffers. Only a different
        # tool-frame definition needs a separate planner topology.
        key = json.dumps({'support_frame': support_frame}, sort_keys=True, allow_nan=False)
        mode = 'support' if support_frame is not None else 'open'
        with self._carrying_scope_lock:
            with self.gpu_lock:
                previous = (self.carrying_binding, self.gripper_feedback, self.support_frame)
                self._select_planner_configuration_unlocked(key, mode)
                self.carrying_binding, self.gripper_feedback, self.support_frame = binding, feedback, support_frame
                if binding is not None:
                    for configuration in self.effective_config.get(binding.arm, {}).values():
                        if 'carrying_owner_id' in configuration:
                            configuration['carrying_owner_id'] = binding.owner_id
            try:
                yield
            finally:
                with self.gpu_lock:
                    self.carrying_binding, self.gripper_feedback, self.support_frame = previous

    def prepare_carrying(self, request, binding, feedback, next_motion_kind='ik'):
        if binding.state != "held":
            raise ValueError("persistent attachment requires held evidence")
        if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
            raise ValueError("prepare requires measured A and B gripper feedback")
        if next_motion_kind not in ('ik', 'place', 'pose'):
            raise ValueError('invalid next attachment motion kind')
        started = time.perf_counter()
        with self.carrying_scope(binding, feedback), self.gpu_lock:
            self._prepare_scene_cache_unlocked()
            # Prepare the solver used by the next motion directly. Compatibility
            # callers that still screen release poses can explicitly prepare IK.
            clearance = self.planning_settings.trajectory_collision_clearance_m
            key = (request.arm, clearance, IK_BATCH_CAPACITY)
            if next_motion_kind == 'place':
                backend = self._compound_backend_unlocked(request.arm, clearance,
                    self.planning_settings.optimizer_collision_activation_distance_m)
            elif next_motion_kind == 'pose':
                backend = self._backend_unlocked(request.arm)
            else:
                backend = self._ik_batch_backends.get(key)
            if backend is None:
                # Keep the frozen wrapper's IK allocation and eviction contract.
                while self._ik_batch_backends and (
                    sum(item[2] for item in self._ik_batch_backends) + IK_BATCH_CAPACITY > 200
                    or sum(item[0] == request.arm for item in self._ik_batch_backends) >= 2
                ):
                    _, old_backend = self._ik_batch_backends.popitem(last=False)
                    old_backend.destroy()
                backend = self._create_real_backend_unlocked(request.arm,
                    batch_size=IK_BATCH_CAPACITY, ik_only=True,
                    trajectory_collision_clearance_m=clearance)
                self._ik_batch_backends[key] = backend
            if next_motion_kind == 'ik':
                self._ik_batch_backends.move_to_end(key)
            ready_at = time.perf_counter()
            self._refresh_scene_unlocked(arm=request.arm, backend=backend)
            self._refresh_peer_collision_unlocked(arm=request.arm, backend=backend,
                active_joints_deg=request.start_joints_deg, other_joints_deg=request.other_joints_deg)
            self._require_scene_token_unchanged_unlocked(binding.scene_token)
            evidence = dict(self.effective_config[request.arm]['ik' if next_motion_kind == 'ik' else 'plan'])
            attachment_spheres = backend.carrying_spheres_tool.tolist()
            self.logical_carrying_binding = binding
        return {"ok": True, "attached": True, "carrying_owner_id": binding.owner_id,
                "scene_token": binding.scene_token, "collision_evidence": evidence,
                "attachment_geometry": evidence.get("attachment_geometry"),
                "attachment_spheres_tool_m": attachment_spheres,
                "next_motion_kind": next_motion_kind,
                "prepare_timings_ms": {'configuration_and_warmup': (ready_at - started) * 1000,
                    'scene_and_attachment_upload': (time.perf_counter() - ready_at) * 1000},
                "prepare_wall_ms": (time.perf_counter() - started) * 1000}

    def detach_carrying(self):
        with self.gpu_lock:
            self.logical_carrying_binding = None
            self.carrying_binding = None
            self.gripper_feedback = None
            self.support_frame = None
            for cache in (self._backends, self._dual_backends, self._compound_backends,
                          self._batch_backends, self._ik_batch_backends):
                for backend in cache.values():
                    if hasattr(backend, 'resident_geometry') and getattr(backend, 'carrying_link', None):
                        backend.resident_geometry.detach(backend)
        return {"ok": True, "attached": False}

    def plan_observation_scoped(self, request, feedback):
        if self.geometry_scene.get("exclusion_state") != "observation":
            raise ValueError("observation motion requires all measured targets retained")
        with self.carrying_scope(None, feedback):
            native = self.plan_to_pose_native(request, lock_timeout_s=600)
            evidence = dict(self.effective_config[request.arm]["plan"])
        return native, evidence

    def clear_carrying(self):
        self.detach_carrying()
        with self.gpu_lock:
            self.carrying_binding = None
            self.gripper_feedback = None
            self.support_frame = None
            # Keep resident solver allocations, but require fresh world/peer evidence.
            self._scene_snapshot_key_by_arm.clear()
            self._peer_collision_snapshot_by_arm.clear()
            self._command_checkers.clear()
            self.geometry_scene = None
            self.geometry_model = None
            self.geometry_grids = None
            self.geometry_solver_profile = None
            self.observed_free_guard = None
            self.motion_coverage = None
            self._inventory_archive = None
            self.pointcloud_preview = None
            if self.perception_mapper is not None:
                self.perception_mapper.reset()
                self.perception_mapper = None
        return {"ok": True, "attached": False}

    def build_perception(self, payload):
        started = time.perf_counter()
        settings = PlanningSettings.model_validate(payload.get("planning_settings", {}))
        decoded, identity = decode_mapping_payload(payload)
        decoded_at = time.perf_counter()
        filtering = filter_mapping_views(decoded, mapping_filter_settings(payload))
        filtered_at = time.perf_counter()
        preview = filtering.pop("preview")
        self.configure_planning(settings)
        with self.gpu_lock:
            old = self.geometry_scene
            previous = ((self.geometry_scene, self.geometry_grids, self.geometry_model)
                        if self.geometry_scene is not None and self.geometry_grids is not None
                        and self.geometry_model is not None else self._inventory_archive)
            self._inventory_archive = freeze_inventory_scene(previous)
            if old is None or any(old.get(field) != identity.get(field) for field in ("generation", "target_id")):
                self.logical_carrying_binding = None
            self.carrying_binding = None
            self.gripper_feedback = None
            # The pinned public world loader copies new ESDF values into the
            # planner's allocated buffers. Scene identity is not robot geometry:
            # retain compatible CUDA graphs, but invalidate every mode's scene
            # evidence. Configuration changes and capacity growth still destroy
            # solvers in carrying_scope/_prepare_scene_cache_unlocked.
            for name in ("_scene_snapshot_key_by_arm", "_peer_collision_snapshot_by_arm", "_command_checkers"):
                getattr(self, name).clear()
                for _, saved in self._planner_configurations.values():
                    saved[name].clear()
            # Invalidate publication before reusing Mapper buffers. A failed
            # fresh capture may never leave the old map executable.
            self.geometry_scene = None
            self.geometry_grids = None
            self.geometry_model = None
            self.observed_free_guard = None
            self.motion_coverage = None
            self.pointcloud_preview = None
            feedback = identity["gripper_joint_positions_rad"]
            segmenter_started = time.perf_counter()
            hybrid = 'geometry_objects' in decoded
            if self.perception_mapper is not None:
                previous_hybrid = self.perception_mapper.__class__.__name__ == 'HybridGeometryMapper'
                if previous_hybrid != hybrid:
                    raise ValueError('geometry_policy_changed_requires_reset')
            if self.perception_mapper is None:
                self.perception_gripper_feedback = None
                segmenter = create_feedback_segmenter(robot_config_path=self._robot_config_path,
                    urdf_path=self._urdf_path, asset_root_path=self._asset_root_path, feedback=feedback)
                if hybrid:
                    from .hybrid_mapping import HybridGeometryMapper

                    self.perception_mapper = HybridGeometryMapper(segmenter.sdk_boundary)
                else:
                    self.perception_mapper = PerceptionMapper(segmenter.sdk_boundary)
            self.perception_mapper.segmenter_boundary.update_feedback(feedback)
            self.perception_gripper_feedback = dict(feedback)
            segmenter_ready = time.perf_counter()
            candidate, grids, model = self.perception_mapper.build(decoded)
            identity, grids, model = reconcile_inventory(self._inventory_archive, identity, grids, model,
                                                         **decoded.get('mapping_lifecycle', {}))
            from .pointcloud_preview import reconcile_obb_preview

            previous_obbs = (self._inventory_archive[0].get('mapped_object_obbs', [])
                             if self._inventory_archive else [])
            identity['mapped_object_obbs'] = reconcile_obb_preview(
                previous_obbs, payload.get('object_obb_observations', []), identity)
            mapper_done = time.perf_counter()
            diagnostics = dict(self.perception_mapper.diagnostics)
            identity['mapping_diagnostics'] = diagnostics
            identity['semantic_only_active'] = diagnostics.get('semantic_only_active', False)
            if identity["exclusion_state"] == "held" and candidate.selected_instance_id == "":
                # A fresh held/background capture can have no static source
                # layer. Keep the carrying identity, but disable only a layer
                # actually emitted by Mapper; all measured obstacles stay on.
                source_name = f"semantic_instance::{identity['target_id']}"
                if not any(source_name in model.get(kind, {}) for kind in ("voxel", "cuboid", "mesh")):
                    identity["disabled_obstacle_names"] = ()
            self.observed_free_guard = ObservedFreeSpaceGuard(decoded["views"],
                self.perception_mapper.capture_segmenter.filtered_depths, identity["scene_token"],
                registered_source_limits=self.perception_mapper.capture_segmenter.source_limits)
            self.motion_coverage = MotionCoverage(decoded.get('coverage_views') or decoded['views'],
                                                   identity['scene_token'])
            identity['observed_free_motion_gate'] = False
            identity['sensor_coverage_role'] = 'diagnostic_for_model_mapping_strategy'
            self.observed_free_guard.capture_spheres = self.perception_mapper.segmenter_boundary.robot_spheres(
                next(iter(decoded["views"].values()))["robot_joints_at_capture"])
            self.motion_coverage.capture_robot_spheres = self.observed_free_guard.capture_spheres.copy()
            certify_workspace(identity, self.motion_coverage)
            captured_joints = next(iter(decoded["views"].values()))["robot_joints_at_capture"]
            self.observed_free_guard.capture_joint_inputs = {
                f"Joint{index}_{'L' if arm == 'A' else 'R'}": float(np.deg2rad(value))
                for arm, values in captured_joints.items() for index, value in enumerate(values, 1)} | {
                "gripL_finger_joint": feedback["A"], "gripR_finger_joint": feedback["B"]}
            self.observed_free_guard.selected_surface_world_m = identity["selected_surface_world_m"]
            self.observed_free_guard.selected_instance_id = identity["target_id"]
            self.observed_free_guard.observation_id = identity["observation_id"]
            self.observed_free_guard.generation = identity["generation"]
            if not decoded.get('demo_mapping'):
                self.observed_free_guard.support_plane = getattr(candidate.table, "support_plane", None)
                self.observed_free_guard.set_measured_table_task_scope(candidate.table.model, decoded["views"],
                    self.perception_mapper.capture_segmenter.filtered_depths)
            self.geometry_scene = identity | {"support_plane": getattr(candidate.table, "support_plane", None)}
            self.geometry_grids = grids
            self.geometry_model = model
            self._inventory_archive = (self.geometry_scene, grids, model)
            self.geometry_solver_profile = settings.solver_payload()
            guard_done = time.perf_counter()
            published_preview = {**preview, **{key: identity[key] for key in
                ("scene_token", "observation_id", "generation")}, "checkpoint": payload.get("checkpoint")}
            published_preview['object_obbs'] = {
                'observed': deepcopy(payload.get('object_obb_observations', [])),
                'mapped': deepcopy(identity['mapped_object_obbs'])}
            from .collision_mapping import collision_layer_bounds, collision_scene_preview

            if decoded.get("demo_mapping"):
                identity["collision_layer_bounds"] = collision_layer_bounds(grids, model, identity)
                self.geometry_scene["collision_layer_bounds"] = identity["collision_layer_bounds"]
            try:
                published_preview["collision_scene"] = collision_scene_preview(candidate, grids, model, identity)
            except (AttributeError, ValueError, TypeError, KeyError) as error:
                # A display export must not invalidate a successfully built map.
                # Publish explicit unavailability rather than relabel raw points.
                published_preview["collision_scene"] = {"status": "unavailable", "reason": str(error),
                    **{key: identity[key] for key in ('scene_token', 'observation_id', 'generation')}}
            self.pointcloud_preview = published_preview
        from .collision_archive import freeze_collision_snapshot

        try:
            collision_snapshot = freeze_collision_snapshot(published_preview['collision_scene'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            collision_snapshot = {'status': 'unavailable', 'reason': str(error),
                **{key: identity[key] for key in ('scene_token', 'observation_id', 'generation')}}
        background_only = candidate.selected_instance_id == ""
        return {"ok": True, **identity, "layer_count": len(grids), "voxel_size_m": None if hybrid else .02,
                'collision_snapshot': collision_snapshot,
                "measured_table_task_scope": self.observed_free_guard.measured_table_task_scope,
                "background_includes_selected": background_only,
                "mapping_camera_ids": list(candidate.camera_metrics),
                "source_restoration": dict(self.perception_mapper.capture_segmenter.source_restoration),
                "implicit_sensor_filter": False, "implicit_pointcloud_filter": False,
                "excluded_object_ids": [identity["target_id"]] if identity["disabled_obstacle_names"] else [],
                "collision_objects": [name for name in identity['inventory_map']['entities']
                    if identity["exclusion_state"] in ("released", "observation") or name != identity["target_id"]],
                "table_collision_enabled": True, "table": dict(candidate.table.model),
                "instance_metrics": {key: dict(value) for key, value in candidate.instance_metrics.items()},
                "camera_metrics": {key: dict(value) for key, value in candidate.camera_metrics.items()},
                "timings_ms": dict(candidate.timings_ms),
                "service_timings_ms": {
                    "decode": (decoded_at - started) * 1000,
                    "depth_preparation": (filtered_at - decoded_at) * 1000,
                    "configure_lock_and_invalidate": (segmenter_started - filtered_at) * 1000,
                    "segmenter_configuration": (segmenter_ready - segmenter_started) * 1000,
                    "mapper_wall": (mapper_done - segmenter_ready) * 1000,
                    "free_space_guard": (guard_done - mapper_done) * 1000,
                    "preview_and_publication": (time.perf_counter() - guard_done) * 1000,
                    "total": (time.perf_counter() - started) * 1000},
                "segmenter_timings_ms": dict(getattr(self.perception_mapper.capture_segmenter, "timings_ms", {})),
                "pointcloud_filter": filtering, "solver_profile": self.geometry_solver_profile}

    def plan_place_scoped(self, request, binding, feedback=None):
        with self.carrying_scope(binding, feedback):
            native = self.plan_pre_release_place_native(request, lock_timeout_s=600)
            evidence = dict(self.effective_config[request.arm]["plan"])
        return native, evidence

    def resolve_release_height_scoped(self, request, binding, feedback, gripper_spheres, *,
                                     search=None, ik_payload=None):
        """Query tool geometry, then public held IK outside the non-reentrant GPU lock."""
        with self.carrying_scope(binding, feedback):
            with self.gpu_lock:
                snapshot = self._prepare_scene_cache_unlocked()
                backend = self._compound_backend_unlocked(request.arm,
                    request.trajectory_collision_clearance_m, request.optimizer_collision_activation_distance_m)
                token = self._refresh_scene_unlocked(snapshot=snapshot, arm=request.arm, backend=backend)
                if token != binding.scene_token:
                    self._raise_scene_changed(binding.scene_token, token)
                self._refresh_peer_collision_unlocked(arm=request.arm, backend=backend,
                    active_joints_deg=request.start_joints_deg, other_joints_deg=request.other_joints_deg)
                began = time.monotonic()
                result = (backend.resolve_release_height(request, gripper_spheres) if search is None else
                          backend.resolve_release_height(request, gripper_spheres, search=search))
                world_query_s = time.monotonic() - began
                self._require_scene_token_unchanged_unlocked(token)
            if search is not None:
                from .object_obb import held_geometry_points
                from .release_height_search import select_held_height

                def solve(poses):
                    query = {key: value for key, value in ik_payload.items()
                             if key not in ('goal_position_m', 'goal_quaternion_wxyz',
                                            'height_search', 'gripper_spheres_tool_m')}
                    query['goal_matrices_world'] = poses
                    return run_operation(self, '/tools/solve_ik_batch', query)

                result['world_query_s'] = world_query_s
                result = select_held_height(result, search, held_geometry_points(asdict(binding)), solve)
                with self.gpu_lock:
                    self._require_scene_token_unchanged_unlocked(token)
            return result

    def query_world_spheres(self, payload, settings):
        """Query proposed fixed-tool endpoints without attaching or planning motion."""
        from .world_sphere_query import prepare_query, query_result

        with self._carrying_scope_lock:
            with self.gpu_lock:
                if self.logical_carrying_binding is not None or self.carrying_binding is not None:
                    raise ValueError('pre-grasp world screening requires an empty hand')
                spheres = prepare_query(payload, self._scene_snapshot_unlocked())
                if settings.solver_payload() != self.geometry_solver_profile:
                    raise ValueError('scene planning settings changed; rebuild the geometry scene')
            self.configure_planning(settings)
            feedback = payload['gripper_joint_positions_rad']
            with self.carrying_scope(None, feedback), self.gpu_lock:
                snapshot = self._prepare_scene_cache_unlocked()
                prepare_query(payload, snapshot)
                clearance = settings.trajectory_collision_clearance_m
                backend = self._compound_backend_unlocked(payload['arm'], clearance,
                    settings.optimizer_collision_activation_distance_m)
                token = self._refresh_scene_unlocked(snapshot=snapshot, arm=payload['arm'], backend=backend)
                if token != payload['scene_token']:
                    self._raise_scene_changed(payload['scene_token'], token)
                self._refresh_peer_collision_unlocked(arm=payload['arm'], backend=backend,
                    active_joints_deg=payload['start_joints_deg'], other_joints_deg=payload['other_joints_deg'])
                costs = backend.query_world_sphere_costs(spheres[None])
                if costs.shape != (1, *spheres.shape[:2]):
                    raise ValueError('invalid world sphere collision cost shape')
                self._require_scene_token_unchanged_unlocked(token)
                return query_result(spheres, costs[0], token, clearance)


    def plan_pre_release_place_native(self, request, lock_timeout_s=600):
        """Own two-phase contract with the same scene/peer and failure boundary."""
        if not isinstance(request, VendorPlanPlaceRequest):
            raise CuroboToolError("request must be a PlanPlaceRequest", code="CUROBO_INVALID_REQUEST",
                stage="input", status=400)
        timeout = _validated_timeout(lock_timeout_s)
        deadline = time.monotonic() + timeout
        if not self.gpu_lock.acquire(timeout=timeout):
            raise _busy_error()
        try:
            snapshot = self._prepare_scene_cache_unlocked()
            backend = self._compound_backend_unlocked(request.arm,
                request.trajectory_collision_clearance_m, request.optimizer_collision_activation_distance_m)
            try:
                if timeout > 0 and time.monotonic() >= deadline:
                    raise _busy_error()
                token = self._refresh_scene_unlocked(snapshot=snapshot, arm=request.arm,
                    backend=backend, cache_snapshot=False)
                self._refresh_peer_collision_unlocked(arm=request.arm, backend=backend,
                    active_joints_deg=request.start_joints_deg, other_joints_deg=request.other_joints_deg,
                    cache_snapshot=False)
                if timeout > 0 and time.monotonic() >= deadline:
                    raise _busy_error()
                result = backend.plan_place_native(request)
                if not isinstance(result, PreReleasePlacePlan):
                    raise TypeError("backend must return a real two-phase pre-release plan")
                self._require_scene_token_unchanged_unlocked(token)
                return PreReleasePlacePlan(approach=replace(result.approach, scene_token=token),
                    place=replace(result.place, scene_token=token))
            except Exception as error:
                if isinstance(error, CuroboToolError) and (
                        error.code in {'CUROBO_SCENE_CHANGED', 'release_clearance_conflict'}
                        or error.code == 'CUROBO_GRASP_FAILED' and error.details.get('phase') in {'approach', 'grasp'}):
                    raise
                self._discard_compound_backend_unlocked(arm=request.arm,
                    trajectory_collision_clearance_m=request.trajectory_collision_clearance_m,
                    optimizer_collision_activation_distance_m=request.optimizer_collision_activation_distance_m,
                    backend=backend)
                if isinstance(error, CuroboToolError):
                    raise
                raise CuroboToolError("cuRobo backend pre-release planning failed", stage="sdk",
                    code="CUROBO_BACKEND_FAILED", status=500, details={"arm": request.arm}) from error
        finally:
            self.gpu_lock.release()

    def plan_retreat_scoped(self, request, feedback=None, *, departure_axis=None, departure_in_tool_frame=False,
                           departure_start_state=None):
        if self.geometry_scene.get("exclusion_state") not in {"selected", "held", "released"}:
            raise ValueError("retreat requires a manipulation scene")
        with self.carrying_scope(None, feedback):
            if departure_axis is None:
                native = self.plan_to_pose_native(request, lock_timeout_s=600)
            else:
                with self.gpu_lock:
                    self._prepare_scene_cache_unlocked()
                    backend = self._backend_unlocked(request.arm)
                    backend.release_departure = (departure_axis, departure_in_tool_frame)
                    backend.release_start_state = departure_start_state
                try:
                    native = self.plan_to_pose_native(request, lock_timeout_s=600)
                finally:
                    with self.gpu_lock:
                        backend.release_departure = None
                        backend.release_start_state = None
            evidence = dict(self.effective_config[request.arm]["plan"])
        return native, evidence

    def plan_carry_pose_scoped(self, request, binding, feedback, support_contact_policy=None):
        if self.geometry_scene.get("exclusion_state") not in ("selected", "held"):
            raise ValueError("carried pose requires a selected or held object map")
        prefix = None
        end_gap = None
        prefix_proof = None
        frame = None
        goal_projection = None
        plane = self.geometry_scene.get("support_plane")
        if support_contact_policy is not None:
            if support_contact_policy != "observed_plane_normal_departure":
                raise ValueError("unsupported support contact policy")
            if plane is not None:
                from scipy.spatial.transform import Rotation

                with self.carrying_scope(None, feedback), self.gpu_lock:
                    self._prepare_scene_cache_unlocked()
                    backend = self._backend_unlocked(request.arm)
                    position, quaternion = backend._forward_tool_pose(backend._state(request.start_joints_deg))
                    matrix = np.eye(4)
                    matrix[:3, :3] = Rotation.from_quat([*quaternion[1:], quaternion[0]]).as_matrix()
                    matrix[:3, 3] = position
                distance = support_prefix_distance(binding, plane, matrix, self.planning_settings.optimizer_collision_activation_distance_m)
                if distance:
                    try:
                        goal_projection = validate_support_goal_projection(position, request.goal_position_m, plane["normal_xyz"], distance)
                    except ValueError as error:
                        raise CuroboToolError(str(error), stage="validation", code="CUROBO_SUPPORT_DEPARTURE_GOAL_INVALID", status=422) from error
                    frame = support_departure_frame(request.arm, matrix, plane["normal_xyz"])
                    with self.carrying_scope(None, feedback, support_frame=frame):
                        with self.gpu_lock:
                            self._prepare_scene_cache_unlocked()
                            backend = self._backend_unlocked(request.arm)
                            backend.support_departure_planning = True
                        axis_request = PlanAxisLinearRequest.create(arm=request.arm,
                            start_joints_deg=request.start_joints_deg, other_joints_deg=request.other_joints_deg,
                            reference_frame="tool", axis="z", distance_m=distance,
                            freq_hz=request.freq_hz, max_attempts=request.max_attempts)
                        prefix = self.plan_axis_linear_native(axis_request, lock_timeout_s=600)
                        with self.gpu_lock:
                            end_position, end_quaternion = backend._forward_tool_pose(backend._state(np.rad2deg(prefix.positions_rad[-1])))
                            matrix[:3, :3] = Rotation.from_quat([*end_quaternion[1:], end_quaternion[0]]).as_matrix()
                            matrix[:3, 3] = end_position
                            matrix = matrix @ np.linalg.inv(frame["T_tool_frame"])
                        end_gap = support_gap_m(binding, plane, matrix)
                        required_gap = max(self.planning_settings.trajectory_collision_clearance_m,
                                           self.planning_settings.optimizer_collision_activation_distance_m)
                        if end_gap <= required_gap:
                            raise CuroboToolError("support prefix endpoint does not clear optimizer activation", stage="validation",
                                code="CUROBO_SUPPORT_DEPARTURE_FAILED", status=422,
                                details={"end_support_gap_m": end_gap, "required_gap_m": required_gap})
                if prefix is not None:
                    prefix_proof = self.validate_execution({"arm": request.arm, "phase": "lift",
                        "positions_rad": prefix.positions_rad, "joint_names": prefix.joint_names,
                        "other_joints_deg": request.other_joints_deg, "scene_token": self.geometry_scene["scene_token"],
                        "gripper_joint_positions_rad": feedback, "carrying_binding": asdict(binding),
                        "support_contact_policy": support_contact_policy}, binding)
                    if not prefix_proof["collision_free"]:
                        raise CuroboToolError("observed support departure prefix is unsafe", stage="validation",
                            code="CUROBO_SUPPORT_DEPARTURE_FAILED", status=422, details=prefix_proof)
                    request = replace(request, start_joints_deg=tuple(np.rad2deg(prefix.positions_rad[-1])))
        with self.carrying_scope(binding, feedback):
            native = self.plan_to_pose_native(request, lock_timeout_s=600)
            evidence = dict(self.effective_config[request.arm]["plan"])
        if prefix is not None:
            if not np.allclose(prefix.positions_rad[-1], native.positions_rad[0], atol=1e-5, rtol=0):
                raise ValueError("support departure and attached continuation do not join")
            native = replace(native, positions_rad=prefix.positions_rad + native.positions_rad[1:],
                velocities_rad_s=prefix.velocities_rad_s + native.velocities_rad_s[1:],
                accelerations_rad_s2=prefix.accelerations_rad_s2 + native.accelerations_rad_s2[1:],
                planning_time_s=prefix.planning_time_s + native.planning_time_s)
            evidence["support_departure_prefix"] = {"planner": "public_curobo_normal_frame_axis_linear_robot",
                "optimizer_attachment": False, "carried_body_guarded": True,
                "support_frame": frame, "user_goal_normal_projection_m": goal_projection,
                "end_support_gap_m": end_gap,
                "required_optimizer_clearance_m": self.planning_settings.optimizer_collision_activation_distance_m,
                "native_point_count": len(prefix.positions_rad), "collision_evidence": prefix_proof}
        return native, evidence

    @contextmanager
    def empty_hand_target_scope(self):
        """Include a selected measured object in both dual-arm motion queries.

        Selection is bookkeeping, not contact permission for an empty hand.
        The operation snapshot/cache key changes; the inventory does not.
        """
        with self._carrying_scope_lock:
            previous = getattr(self._planning_scope, 'retain_empty_hand_target', False)
            self._planning_scope.retain_empty_hand_target = True
            try:
                yield
            finally:
                self._planning_scope.retain_empty_hand_target = previous

    def dual_operation_native(self, request, *, validate=False, lock_timeout_s=120.0):
        with self.empty_hand_target_scope():
            return super().dual_operation_native(request, validate=validate, lock_timeout_s=lock_timeout_s)

    def validate_dual_execution(self, request, feedback):
        """Reuse the world, checking both command chains with exact current fingers."""
        with self.empty_hand_target_scope(), self.gpu_lock:
            self._prepare_scene_cache_unlocked()
            backend = next(iter(self._dual_backends.values()), None)
            if backend is None:
                backend = self._dual_backend_unlocked(request)
            if backend is not None:
                token = self._scene_snapshot_unlocked()["scene_token"]
                if request.expected_scene_token is not None and request.expected_scene_token != token:
                    self._raise_scene_changed(request.expected_scene_token, token)
                self._refresh_scene_unlocked(arm="AB", backend=backend)
                payload = {"arm": "AB", "phase": "observation", "scene_token": token,
                    "positions_rad": np.deg2rad(np.concatenate(
                        (request.points_deg["A"], request.points_deg["B"]), axis=1)).tolist(),
                    "joint_names": [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)],
                    "gripper_joint_positions_rad": feedback}
                result = check_command_path(self, backend.motion_planner, payload, None)
                self._require_scene_token_unchanged_unlocked(token)
                if not result['collision_free']:
                    coverage_missing = result.get('failure_reason') == 'observation_coverage_insufficient'
                    raise CuroboToolError(
                        "dual motion requires additional registered depth coverage" if coverage_missing else
                        "dual observation command intersects measured obstacles or the robot",
                        code="observation_coverage_insufficient" if coverage_missing else "CUROBO_TRAJECTORY_COLLISION",
                        stage="validation", status=422, details=result)
                return result

    def validate_execution(self, payload, binding):
        phase = validate_phase_request(payload)
        if phase == "observation" and self.geometry_scene.get("exclusion_state") != "observation":
            raise ValueError("observation validation requires all measured targets retained")
        if phase == "retreat" and self.geometry_scene.get("exclusion_state") not in {"selected", "held", "released"}:
            raise ValueError("retreat requires a manipulation scene")
        # Checks consume the exact command samples after project Hermite
        # insertion, then densely sample every connecting command segment.
        feedback = None if payload.get("gripper_geometry_mode") == "fixed_open" else payload.get("gripper_joint_positions_rad")
        with self._carrying_scope_lock:
            with self.gpu_lock:
                self._prepare_scene_cache_unlocked()
                # Borrow only world buffers. check_command_path independently
                # uses exact command samples, measured fingers and attachment.
                backend = None
                for cache in (self._backends, self._batch_backends, self._compound_backends):
                    for key, candidate in cache.items():
                        if key == payload["arm"] or isinstance(key, tuple) and key[0] == payload["arm"]:
                            backend = candidate
                            break
                    if backend is not None:
                        break
                if backend is not None:
                    self._refresh_scene_unlocked(arm=payload["arm"], backend=backend, update_robot_geometry=False)
                    with measured_target_collision_scope(self, backend.motion_planner, phase) as target_scope:
                        result = check_command_path(self, backend.motion_planner, payload, binding)
                    result['target_collision_scope'] = target_scope
                    self._require_scene_token_unchanged_unlocked(payload["scene_token"])
                    return result
            # With no world to borrow, retain the existing model-specific path.
            # carrying_scope acquires the non-reentrant GPU lock itself.
            with self.carrying_scope(binding, feedback), self.gpu_lock:
                self._prepare_scene_cache_unlocked()
                backend = self._compound_backend_unlocked(payload["arm"],
                    self.planning_settings.trajectory_collision_clearance_m,
                    self.planning_settings.optimizer_collision_activation_distance_m)
                self._refresh_scene_unlocked(arm=payload["arm"], backend=backend)
                with measured_target_collision_scope(self, backend.motion_planner, phase) as target_scope:
                    result = check_command_path(self, backend.motion_planner, payload, binding)
                result['target_collision_scope'] = target_scope
                self._require_scene_token_unchanged_unlocked(payload["scene_token"])
                return result

    def update_perception(self, payload):
        from .scene_update import prepare_scene_update, updated_preview

        with self.gpu_lock:
            identity, grids, model = prepare_scene_update(
                self.geometry_scene, self.geometry_grids, self.geometry_model, payload)
            for name in ('_scene_snapshot_key_by_arm', '_peer_collision_snapshot_by_arm', '_command_checkers'):
                getattr(self, name).clear()
                for _, saved in self._planner_configurations.values():
                    saved[name].clear()
            self.geometry_scene, self.geometry_grids, self.geometry_model = identity, grids, model
            apply_inventory_update(identity, payload, model)
            preview = updated_preview(getattr(self, 'pointcloud_preview', None), identity, payload)
            identity['mapped_object_obbs'] = deepcopy((preview or {}).get('object_obbs', {}).get('mapped', []))
            self._inventory_archive = (identity, grids, model)
            self.pointcloud_preview = preview
            self.carrying_binding = self.logical_carrying_binding = None
            if self.observed_free_guard is not None:
                # Rays remain attached to their original capture; only collision version changes.
                self.observed_free_guard.scene_token = identity['scene_token']
                self.observed_free_guard.selected_instance_id = identity['target_id']
                self.observed_free_guard.selected_surface_world_m = identity.get('selected_surface_world_m', [])
            if self.motion_coverage is not None:
                if payload.get('operation') == 'select':
                    self.motion_coverage.scene_token = identity['scene_token']
                else:
                    self.motion_coverage.invalidate('scene_geometry_changed_requires_registered_rgbd')
            names = [name.removeprefix('semantic_instance::') for name in model.get('voxel', {})
                     if name.startswith('semantic_instance::')]
            return {'ok': True, **identity, 'layer_count': len(grids),
                    'excluded_object_ids': [identity['target_id']] if identity['exclusion_state'] == 'selected' else [],
                    'collision_objects': [name for name in names if 'semantic_instance::' + name not in identity['disabled_obstacle_names']],
                    'collision_evidence': 'native_voxel_pose_and_tensor_version_checked_before_each_planning_query'}

    def build_geometry(self, payload):
        raise ValueError('simulation_geometry_mapping_unsupported:use_registered_rgbd')

    def plan_grasp_batch_native(self, request, lock_timeout_s=120.0):
        # Keep selection identity unchanged. Only this operation's uploaded
        # snapshot retains the target; its cache key records the enabled set.
        with self._carrying_scope_lock:
            previous = getattr(self._planning_scope, 'retain_grasp_target', False)
            self._planning_scope.retain_grasp_target = True
            try:
                return super().plan_grasp_batch_native(request, lock_timeout_s=lock_timeout_s)
            finally:
                self._planning_scope.retain_grasp_target = previous

    def _scene_snapshot_unlocked(self, *, allow_unstable_object=False):
        if self.geometry_scene is None:
            raise ValueError("build a registered RGBD scene before planning")
        snapshot = self.geometry_scene.copy()
        if (getattr(self._planning_scope, 'retain_grasp_target', False)
                or (getattr(self._planning_scope, 'retain_empty_hand_target', False)
                    and snapshot.get('exclusion_state') == 'selected')):
            names = {name for _, name in measured_target_layers(self, snapshot)}
            snapshot['disabled_obstacle_names'] = tuple(
                name for name in snapshot.get('disabled_obstacle_names', ()) if name not in names)
        return snapshot

    def _prepare_scene_cache_unlocked(self):
        from .inventory_map import require_native_inventory_geometry

        require_native_inventory_geometry(self.geometry_model or {})
        snapshot = self._scene_snapshot_unlocked()
        cache = geometry_collision_cache(self.geometry_model)
        # The upload list is authoritative even if a mapper omits background
        # metadata. Never reserve fewer layers than will actually be uploaded.
        grid_count = len(self.geometry_grids or ())
        cache['voxel']['layers'] = max(cache['voxel']['layers'],
            1 << (max(1, grid_count) - 1).bit_length())
        previous = self._collision_cache
        if previous is not None:
            for kind in ("mesh", "cuboid"):
                cache[kind] = max(cache[kind], previous[kind])
            cache["voxel"]["layers"] = max(cache["voxel"]["layers"], previous["voxel"]["layers"])
            if np.prod(previous["voxel"]["dims"]) >= np.prod(cache["voxel"]["dims"]):
                cache["voxel"]["dims"] = list(previous["voxel"]["dims"])
        if cache != previous:
            LOGGER.info('Invalidating solver caches: collision capacity changed old=%s new=%s', previous, cache)
            self._clear_planners_unlocked()
            self._collision_cache = cache
        self._scene_model = self.geometry_model
        return snapshot

    def _refresh_scene_unlocked(self, *, arm, backend, snapshot=None, update_robot_geometry=True, **kwargs):
        from .inventory_map import require_native_inventory_geometry

        require_native_inventory_geometry(self.geometry_model or {})
        if update_robot_geometry and hasattr(backend, 'resident_geometry'):
            with timed_stage('robot_geometry_update'):
                changed = backend.resident_geometry.update(backend, self.gripper_feedback, self.carrying_binding)
            if changed:
                self._peer_collision_snapshot_by_arm.pop(arm, None)
        # A warmed arm owns several batch shapes. Report the backend actually
        # selected for this call, rather than the last one constructed.
        effective = getattr(backend, 'effective_config', None)
        if effective is not None:
            effective = dict(effective)
            if self._collision_cache is not None:
                effective['collision_cache_capacity'] = {
                    'mesh': self._collision_cache['mesh'],
                    'cuboid': self._collision_cache['cuboid'],
                    'voxel_layers': self._collision_cache['voxel']['layers'],
                }
            if self.carrying_binding is not None and 'carrying_owner_id' in effective:
                effective['carrying_owner_id'] = self.carrying_binding.owner_id
            self.effective_config.setdefault(arm, {})['ik' if backend.ik_only else 'plan'] = effective
        snapshot = snapshot or self._scene_snapshot_unlocked()
        disabled = snapshot.get("disabled_obstacle_names", (f"semantic_instance::{snapshot['target_id']}",))
        key = (snapshot["scene_token"], tuple(sorted(disabled)))
        slot = ("uploaded_scene", id(backend))
        cached = self._scene_snapshot_key_by_arm.get(slot)
        restore_failed = getattr(getattr(backend, 'motion_planner', None), 'scene_restore_failed', False)
        if cached is not None and cached[0] is backend and cached[1] == key and not restore_failed:
            return snapshot["scene_token"]
        with timed_stage('scene_upload'):
            backend.update_voxel_scene(
                [sdk_voxel_grid(grid) for grid in self.geometry_grids],
                sdk_scene_model(self.geometry_model, include_voxel_metadata=True),
                disabled_obstacle_names=disabled)
        from .public_planner import PublicPlanner

        planner = getattr(backend, 'motion_planner', None)
        if isinstance(planner, PublicPlanner):
            from curobo.scene import Scene

            from .planning_meshes import optimizer_scene

            reduced, mesh_evidence = optimizer_scene(self.geometry_model) if not self.geometry_grids else (None, [])
            renamed = {row['obstacle']: row['optimizer_obstacle'] for row in mesh_evidence}
            original_disabled = {
                kind: [name for name in disabled if name in self.geometry_model.get(kind, {})]
                for kind in ('mesh', 'cuboid')}
            reduced_disabled = {
                kind: [renamed.get(name, name) for name in names]
                for kind, names in original_disabled.items()}
            planner.set_planning_scene(
                Scene.create(sdk_scene_model(self.geometry_model)) if mesh_evidence else None,
                Scene.create(sdk_scene_model(reduced)) if mesh_evidence else None,
                original_disabled, reduced_disabled)
            backend.effective_config['optimizer_meshes'] = mesh_evidence
            self.effective_config[arm]['plan']['optimizer_meshes'] = mesh_evidence
        self._scene_snapshot_key_by_arm[slot] = (backend, key)
        return snapshot["scene_token"]

    def _refresh_peer_collision_unlocked(self, *, arm, backend, active_joints_deg,
                                         other_joints_deg, cache_snapshot=True):
        # Reuse only an exact successful upload to this backend. A changed
        # resident model or temporary override invalidates it before any work.
        # At most one retained upload per arm. Switching backends forces a
        # refresh and does not retain discarded GPU planners in this cache.
        slot = arm
        geometry_key = getattr(getattr(backend, 'resident_geometry', None), 'key', None)
        key = (tuple(active_joints_deg), tuple(other_joints_deg), geometry_key)
        cached = self._peer_collision_snapshot_by_arm.get(slot)
        reuse = (cache_snapshot and geometry_key is not None and cached is not None
                 and cached[0] is backend and cached[1] == key)
        if not reuse:
            self._peer_collision_snapshot_by_arm.pop(slot, None)
            with timed_stage('peer_collision_update'):
                backend.update_peer_collision_model(None, active_joints_deg, other_joints_deg)
            if cache_snapshot and geometry_key is not None:
                self._peer_collision_snapshot_by_arm[slot] = (backend, key)
        # Attachment update depends on the active held pose; never skip it
        # merely because the stationary arm has not moved.
        if self.carrying_binding is not None:
            with timed_stage('attachment_upload'):
                backend.attach_measured_carrying(self.carrying_binding, active_joints_deg)
        return not reuse


def run_operation(runtime, path, payload):
    """Validate settings once, then dispatch to the existing typed GPU boundary."""
    if path == '/tools/evaluate_reachability':
        from .reachability import PositionReachability, validate_query

        arm, positions = validate_query(payload)
        with runtime.gpu_lock:
            if arm not in runtime._reachability_solvers:
                runtime._reachability_solvers[arm] = PositionReachability(
                    runtime._robot_config_path, runtime._urdf_path, runtime._asset_root_path, arm)
            items = runtime._reachability_solvers[arm].solve(positions.tolist())
        return {'ok': True, 'arm': arm, 'items': items}
    if path == "/geometry/build":
        return runtime.build_geometry(payload)
    if path == "/perception/build":
        return runtime.build_perception(payload)
    if path == "/perception/update":
        return runtime.update_perception(payload)
    if path == "/carrying/clear":
        return runtime.clear_carrying()
    if path == "/carrying/detach":
        return runtime.detach_carrying()
    if path == '/tools/query_world_spheres':
        settings = PlanningSettings.model_validate(payload['planning_settings'])
        return runtime.query_world_spheres(payload, settings)
    if path == "/tools/plan_to_pose" and payload.get("carrying_binding") is not None:
        raise ValueError("detach carrying before fresh retreat planning")
    if path in ("/tools/plan_dual_observation", "/tools/validate_dual_observation"):
        from .planning_contracts import DualPlanRequest, DualValidationRequest

        settings = PlanningSettings.model_validate(payload["planning_settings"])
        scene = runtime._scene_snapshot_unlocked()
        if payload.get("scene_token") != scene["scene_token"]:
            raise ValueError("request scene token is stale")
        returning = payload.get('return_after_place') is True
        if returning and getattr(runtime, 'logical_carrying_binding', None) is not None:
            raise ValueError('post_place_observation_requires_detached_object')
        if runtime.geometry_scene.get("exclusion_state") != "observation" and not returning:
            raise ValueError("dual observation requires a target-inclusive map")
        if settings.solver_payload() != runtime.geometry_solver_profile:
            raise ValueError("scene planning settings changed; rebuild the geometry scene")
        feedback = payload.get("gripper_joint_positions_rad")
        if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
            raise ValueError("dual observation requires measured gripper positions")
        values = {**payload, "trajectory_collision_clearance_m": settings.trajectory_collision_clearance_m,
                  "optimizer_collision_activation_distance_m": settings.optimizer_collision_activation_distance_m,
                  "trajectory_point_scale": settings.time_scale, "freq_hz": settings.freq_hz,
                  "max_attempts": settings.plan_max_attempts}
        validate = path.endswith("/validate_dual_observation")
        if not validate and payload.get("goal_joints_deg") is None:
            # Tianji's dual wrapper takes matrix translations in mm; project RGBD/FK uses m.
            goals = {}
            for arm in ("A", "B"):
                goal = np.asarray(payload["goal_transforms_world"][arm], dtype=float).copy()
                goal[:3, 3] *= 1000.
                goals[arm] = goal.tolist()
            values["goal_transforms_world"] = goals
        request = (DualValidationRequest if validate else DualPlanRequest).from_mapping(values)
        runtime.configure_planning(settings)
        if validate:
            result = runtime.validate_dual_execution(request, feedback)
            return {**result, "attached": False}
        with runtime.carrying_scope(None, feedback):
            result = runtime.dual_operation_native(request, validate=validate, lock_timeout_s=600.)
        return {**result, "attached": False}
    if path not in ("/tools/solve_ik_batch", "/tools/plan_grasp_batch", "/tools/plan_place", "/tools/resolve_release_height", "/tools/plan_carried_pose", "/tools/plan_to_pose", "/tools/plan_observation", "/carrying/prepare", "/carrying/validate"):
        raise ValueError("unknown endpoint")
    settings = PlanningSettings.model_validate(payload["planning_settings"])
    scene = runtime._scene_snapshot_unlocked()
    if payload.get("scene_token") != scene["scene_token"]:
        raise ValueError("request scene token is stale")
    if settings.solver_payload() != runtime.geometry_solver_profile:
        raise ValueError("scene planning settings changed; rebuild the geometry scene")
    if path in ("/tools/plan_place", "/tools/resolve_release_height", "/tools/plan_carried_pose", "/tools/plan_to_pose", "/tools/plan_observation", "/carrying/prepare", "/carrying/validate"):
        arm = payload.get("arm")
        carrying = payload.get("carrying_binding")
        binding = None if carrying is None else CarryingBinding.from_mapping(carrying, scene=scene, arm=arm)
        if path == "/carrying/prepare" and binding is None:
            raise ValueError("prepare requires measured carrying geometry")
        if path == "/carrying/validate":
            validate_phase_request(payload)
            runtime.configure_planning(settings)
            return runtime.validate_execution(payload, binding)
        feedback = payload.get('gripper_joint_positions_rad')
        if not isinstance(feedback, dict) or set(feedback) != {'A', 'B'}:
            raise ValueError('pose and place planning requires measured A and B gripper feedback')
        if path in ("/tools/plan_place", "/tools/resolve_release_height") and binding is None:
            raise ValueError("plan_place requires measured or scoped preflight carrying geometry")
        if path == "/tools/plan_carried_pose":
            if scene.get("exclusion_state") not in ("selected", "held"):
                raise ValueError("carried pose requires a selected or held object map")
            if binding is None:
                raise ValueError("carried pose requires a measured carrying binding")
            feedback = payload.get("gripper_joint_positions_rad")
            if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
                raise ValueError("carried pose requires measured A and B gripper driver feedback")
        values = {"freq_hz": settings.freq_hz, "max_attempts": settings.plan_max_attempts}
        if path in ("/tools/plan_place", "/tools/resolve_release_height"):
            values.update(trajectory_collision_clearance_m=settings.trajectory_collision_clearance_m,
                optimizer_collision_activation_distance_m=settings.optimizer_collision_activation_distance_m,
                approach_axis=settings.place_approach_axis, approach_offset_m=settings.place_approach_offset_m,
                approach_in_tool_frame=settings.place_approach_in_tool_frame,
                lift_axis=settings.place_retreat_axis, lift_offset_m=settings.place_retreat_offset_m,
                lift_in_tool_frame=settings.place_retreat_in_tool_frame)
        for name, value in values.items():
            if name in payload and payload[name] != value:
                raise ValueError(f"{name} conflicts with planning_settings")
        request_type = PlanPlaceRequest if path in ("/tools/plan_place", "/tools/resolve_release_height") else PlanToPoseRequest
        request = request_type.from_mapping(payload | values)
        if request.trajectory_point_scale is not None:
            raise ValueError("project execution retains raw derivatives before Hermite insertion")
        runtime.configure_planning(settings)
        feedback = payload.get("gripper_joint_positions_rad")
        if path == "/tools/resolve_release_height":
            if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
                raise ValueError("release height requires measured A and B gripper feedback")
            search = payload.get('height_search')
            if search is None:
                result = runtime.resolve_release_height_scoped(
                    request, binding, feedback, payload.get("gripper_spheres_tool_m"))
            else:
                from .object_obb import held_geometry_points
                from .release_height_search import validate_search

                validate_search(search, held_geometry_points(asdict(binding)), payload)
                result = runtime.resolve_release_height_scoped(request, binding, feedback,
                    payload.get('gripper_spheres_tool_m'), search=search, ik_payload=payload)
            return result | {"ok": True, "planner": ("release_height_world_query" if search is None else
                             "release_height_world_and_held_ik"), "attached": True,
                             "planning_settings_revision": settings.revision()}
        if path == "/carrying/prepare":
            if 'next_motion_kind' in payload:
                return runtime.prepare_carrying(request, binding, feedback, payload['next_motion_kind'])
            return runtime.prepare_carrying(request, binding, feedback)
        if path == "/tools/plan_observation":
            native, evidence = runtime.plan_observation_scoped(request, feedback)
            return {"ok": True, "planner": "curobo_observation_pose", "scene_token": native.scene_token,
                    "trajectory": asdict(native), "collision_evidence": evidence,
                    "attached": False, "planning_settings_revision": settings.revision()}
        if path == "/tools/plan_carried_pose":
            policy = payload.get("support_contact_policy")
            if policy is None:
                native, evidence = runtime.plan_carry_pose_scoped(request, binding, feedback)
            else:
                native, evidence = runtime.plan_carry_pose_scoped(request, binding, feedback, support_contact_policy=policy)
            return {"ok": True, "planner": "curobo_carried_pose", "scene_token": native.scene_token,
                    "trajectory": asdict(native), "collision_evidence": evidence,
                    "attached": True, "planning_settings_revision": settings.revision()}
        if path == "/tools/plan_place":
            native, evidence = runtime.plan_place_scoped(request, binding, feedback)
            return {"ok": True, "planner": "curobo_place", "scene_token": native.scene_token,
                    "phases": {name: asdict(getattr(native, name)) for name in ("approach", "place")},
                    "carrying_binding": asdict(binding), "collision_evidence": evidence,
                    "retreat_requires_replan": True,
                    "planning_settings_revision": settings.revision()}
        departure_axis = payload.get('departure_axis')
        if departure_axis is not None:
            if (departure_axis != settings.place_retreat_axis
                    or payload.get('departure_in_tool_frame') is not settings.place_retreat_in_tool_frame):
                raise ValueError('retreat direction conflicts with planning_settings')
            terminal = payload.get('departure_start_state')
            if terminal is not None and (
                        not isinstance(terminal, dict) or set(terminal) != {'position', 'velocity', 'acceleration', 'dt'}
                        or type(terminal['dt']) not in (int, float) or not np.isfinite(terminal['dt']) or terminal['dt'] <= 0
                        or any(np.asarray(terminal[name]).shape != (7,) or not np.isfinite(terminal[name]).all()
                               for name in ('position', 'velocity', 'acceleration'))):
                raise ValueError('invalid native release JointState')
            native, evidence = runtime.plan_retreat_scoped(request, feedback, departure_axis=departure_axis,
                departure_in_tool_frame=settings.place_retreat_in_tool_frame, departure_start_state=terminal)
            evidence = {**evidence, 'departure_criterion': 'official_ToolPoseCriteria.linear_motion',
                'axis': departure_axis, 'in_tool_frame': settings.place_retreat_in_tool_frame,
                'native_release_state_preserved': terminal is not None,
                'native_release_dt_s': None if terminal is None else terminal['dt']}
        else:
            native, evidence = runtime.plan_retreat_scoped(request, feedback)
        return {"ok": True, "planner": "curobo_pose", "scene_token": native.scene_token,
                "trajectory": asdict(native), "collision_evidence": evidence,
                "attached": False, "planning_settings_revision": settings.revision()}
    values = {"trajectory_collision_clearance_m": settings.trajectory_collision_clearance_m}
    ik_only = path == "/tools/solve_ik_batch"
    if not ik_only:
        fields = ("freq_hz", "optimizer_collision_activation_distance_m", "approach_axis",
                  "approach_offset_m", "approach_in_tool_frame", "lift_axis", "lift_offset_m",
                  "lift_in_tool_frame")
        snapshot = settings.model_dump()
        values.update({name: snapshot[name] for name in fields})
    for name, value in values.items():
        if name in payload and payload[name] != value:
            raise ValueError(f"{name} conflicts with planning_settings")
    request_payload = payload | values
    request_type = SolveIkBatchRequest if ik_only else PlanGraspBatchRequest
    request = request_type.from_mapping(request_payload)
    runtime.configure_planning(settings)
    if ik_only:
        candidate_count = len(request.goal_matrices_world)
        if candidate_count > IK_BATCH_CAPACITY:
            raise ValueError('IK requests must contain at most 100 candidates; split larger sets')
        request = replace(request, goal_matrices_world=(request.goal_matrices_world
                          + (request.goal_matrices_world[-1],) * (IK_BATCH_CAPACITY - candidate_count)))
        carrying = payload.get("carrying_binding")
        if carrying is None:
            feedback = payload.get('gripper_joint_positions_rad')
            if not isinstance(feedback, dict) or set(feedback) != {'A', 'B'}:
                raise ValueError('grasp IK requires measured A and B gripper feedback')
            with runtime.carrying_scope(None, {**feedback, request.arm: 0.}):
                result = runtime.solve_ik_batch_native(request, lock_timeout_s=600).to_payload()
        else:
            feedback = payload.get("gripper_joint_positions_rad")
            if not isinstance(feedback, dict) or set(feedback) != {"A", "B"}:
                raise ValueError("carried IK requires measured A and B gripper feedback")
            binding = CarryingBinding.from_mapping(carrying, scene=scene, arm=request.arm)
            with runtime.carrying_scope(binding, feedback):
                result = runtime.solve_ik_batch_native(request, lock_timeout_s=600).to_payload()
                evidence = dict(runtime.effective_config[request.arm]["ik"])
            if evidence.get("optimizer_attachment") is not True:
                raise ValueError("carried IK did not attach the measured object")
            result.update(attached=True, collision_evidence=evidence)
        result['items'] = result['items'][:candidate_count]
        result['batch_size'] = candidate_count
        result['batch_capacity'] = IK_BATCH_CAPACITY
    else:
        candidate_count = len(request.goal_matrices_world)
        # Solve the first ranked pair before spending work on later candidates.
        # Keep fixed shapes for CUDA graphs; never truncate a larger request.
        capacity = next((size for size in (2, 4, 8) if candidate_count <= size), TRAJECTORY_BATCH_CAPACITY)
        if candidate_count < capacity:
            request = replace(request, goal_matrices_world=pad_trajectory_batch(request.goal_matrices_world, capacity))
        started = time.perf_counter()
        feedback = payload.get('gripper_joint_positions_rad')
        if not isinstance(feedback, dict) or set(feedback) != {'A', 'B'}:
            raise ValueError('grasp planning requires measured A and B gripper feedback')
        with runtime.carrying_scope(None, {**feedback, request.arm: 0.}):
            native = runtime.plan_grasp_batch_native(request, lock_timeout_s=600)
        LOGGER.info("cuRobo grasp batch complete: arm=%s candidates=%d capacity=%d wall_s=%.3f",
                    request.arm, candidate_count, native.batch_capacity, time.perf_counter() - started)
        items = []
        for item in native.items[:candidate_count]:
            value = asdict(item)
            value["phases"] = {name: value.pop(name) for name in ("approach", "grasp", "lift")}
            items.append(value)
        result = {"ok": True, "planner": "curobo_grasp_batch", "items": items,
                  "batch_size": candidate_count, "batch_capacity": native.batch_capacity,
                  "planning_time_s": native.planning_time_s, "scene_token": native.scene_token,
                  "target_collision_policy": "measured_target_retained_during_open_grasp_planning"}
    result["effective_config"] = runtime.effective_config[request.arm]["ik" if ik_only else "plan"]
    result["planning_settings_revision"] = settings.revision()
    return result


def sdk_identity():
    import curobo
    path = Path(curobo.__file__).resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            result = subprocess.run(["git", "-C", str(parent), "rev-parse", "HEAD"],
                                    text=True, capture_output=True, check=True)
            return result.stdout.strip(), str(path)
    # Wheels built from a local checkout can omit Git metadata. Verify every
    # packaged Python source byte against the pinned official Git tree instead.
    manifest_path = Path(os.environ.get("TIANJI_ALGORITHMS_ROOT", ROOT.parent / "tianji-algorithms")) / "manifests/curobo-sdk-files.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        mismatches = []
        for relative, expected in manifest["files"].items():
            source = path.parent / relative
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                mismatches.append(relative)
        if not mismatches:
            return manifest["commit"], str(path)
        raise RuntimeError(f"installed SDK differs from official source: {mismatches[:12]}")
    # Installed immutable image package may carry its source marker instead.
    for marker in (Path("/opt/curobo-src/CUROBO_SOURCE_SHA"), Path("/opt/curobo/CUROBO_SOURCE_SHA")):
        if marker.is_file():
            return marker.read_text().strip(), str(path)
    return "unavailable", str(path)


def main():
    import torch
    from tianji_curobo.curobo_service import materialize_urdf
    logging.basicConfig(level=logging.INFO)
    source, module_path = sdk_identity()
    if source != SDK_COMMIT:
        raise RuntimeError(f"cuRobo source identity mismatch: {source}; expected {SDK_COMMIT}")
    urdf_path = Path(os.environ["LUNA_ROBOT_URDF"])
    urdf = materialize_urdf(urdf_path,
        marvin_meshes=Path(os.environ["LUNA_MARVIN_MESHES"]),
        rag_meshes=Path(os.environ["LUNA_RAG_MESHES"]))
    runtime = GeometryRuntime(robot_config_path=ROOT / "resources/robot/marvin_gui_dual.yml",
                              urdf_path=urdf, asset_root_path=urdf.parent)
    runtime.warm_start(load_settings(ROOT / 'data/planning-settings.json'))
    operation_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self):
            if self.path == "/perception/preview":
                preview = getattr(runtime, "pointcloud_preview", None)
                self.send_json(404 if preview is None else 200,
                               {"ok": False, "error": "no_saved_observation"} if preview is None else preview)
                return
            if self.path != "/health":
                self.send_json(404, {"ok": False, "error": "unknown endpoint"})
                return
            self.send_json(200, {"ok": True, "ready": runtime.warmup_status['ready'] and torch.cuda.is_available(),
                "warmup": runtime.warmup_status,
                "gpu_available": torch.cuda.is_available(), "curobo_source": source,
                "module_path": module_path, "scene_token": (runtime.geometry_scene or {}).get("scene_token"),
                "robot_config_sha256": hashlib.sha256((ROOT / "resources/robot/marvin_gui_dual.yml").read_bytes()).hexdigest()})

        def do_POST(self):
            timings = {}
            started = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < request_size_limit(self.path):
                    self.close_connection = True  # The unread body cannot become another pooled request.
                    raise ValueError("invalid request size")
                with collect_timings() as timings:
                    with timed_stage('request_read_decode'):
                        payload = json.loads(self.rfile.read(length))
                    with timed_stage('operation_queue'):
                        operation_lock.acquire()
                    try:
                        with timed_stage('operation'):
                            result = run_operation(runtime, self.path, payload)
                    finally:
                        operation_lock.release()
                self.send_json(200, result, timings=timings)
            except Exception as error:
                self.close_connection = True  # Parsing errors may leave unread bytes on this connection.
                LOGGER.exception("GPU operation failed")
                status, response = operation_error_response(error)
                self.send_json(status, response, timings=timings)
            finally:
                LOGGER.info('request timing path=%s wall_s=%.6f stages=%s', self.path,
                            time.perf_counter() - started, json.dumps(timings))

        def send_json(self, status, value, *, timings=None):
            if timings is not None:
                value = {**value, 'request_timings': {'scope': 'inclusive_wall_seconds', 'stages': timings}}
            started = time.perf_counter()
            data = json.dumps(value, allow_nan=False).encode()
            encode_s = time.perf_counter() - started
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header('Server-Timing', f'encode;dur={encode_s * 1000:.3f}')
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("0.0.0.0", int(os.getenv("LUNA_CUROBO_PORT", "8310"))), Handler)
    print("Luna geometry cuRobo service ready", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

