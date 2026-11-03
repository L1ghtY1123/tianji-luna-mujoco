"""Synchronous boundaries to the independent GraspGen and cuRobo services."""

import hashlib
import json
import logging
import os
import threading
import time
from copy import deepcopy
from pathlib import Path

import httpx
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import TCP_FRAMES, TOOL_PROFILE, grasp_targets, rigid_matrix
from .placement import observed_object
from .planning_batches import trajectory_batches
from .planning_settings import PlanningSettings
from .trajectory_postprocessing import build_execution_phases



class AlgorithmError(RuntimeError):
    def __init__(self, message, *, code="algorithm_error", details=None):
        super().__init__(message)
        self.code = code
        self.details = details


def determinate_grasp_failure(failure):
    """Accept completed solver failure or a finite, proven contact shortfall."""
    if (isinstance(failure, dict) and failure.get('stage') == 'command_validation'
            and failure.get('code') == 'execution_collision'):
        proof = failure.get('details') or {}
        index, count = proof.get('first_invalid_index'), proof.get('command_point_count')
        checked, digest = proof.get('checked_point_count'), proof.get('binding_sha256')
        return (proof.get('collision_free') is False and proof.get('motion_collision_checks_passed') is False
            and proof.get('failure_reason') == 'collision_or_joint_bounds'
            and proof.get('phase') in {'approach_grasp', 'grasp', 'grasp_closure', 'grasp_lift'}
            and proof.get('collision_scope') == 'measured_obstacles_and_both_robot_arms'
            and type(index) is int and type(count) is int and 0 <= index < count
            and type(checked) is int and checked > 0 and bool(proof.get('scene_token'))
            and isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest))
    if (isinstance(failure, dict) and failure.get('stage') == 'command_validation'
            and failure.get('code') == 'planning_bounded_contact_unreachable'):
        from .grasp_contact_prediction import is_proven_contact_shortfall

        return is_proven_contact_shortfall(failure.get('details'))
    if (not isinstance(failure, dict) or failure.get("stage") != "planner"
            or failure.get("error") is not None or failure.get("status") != "Grasp planning completed."):
        return False
    phases = failure.get("phase_success", {})
    if not isinstance(phases, dict):
        return False
    flags = [phases.get(name) for name in ("goalset", "approach", "grasp", "lift")]
    return (all(type(value) is bool for value in flags) and not all(flags)
            and not any(flags[index] and not flags[index - 1] for index in range(1, 4)))


class Algorithms:
    grasp_trajectory_batch_capacity = 16
    def __init__(self, *, curobo_request=None, graspgen_client=None, perception=None, workspace_lookup=None,
                 cad_grasp_library=None):
        from tianji_graspgen.client import GraspGenClient

        from .perception import Perception
        from .workspace import WorkspaceAtlas
        self.curobo_url = os.getenv("LUNA_CUROBO_URL", "http://127.0.0.1:8310")
        self.graspgen = graspgen_client or GraspGenClient(
            os.getenv("LUNA_GRASPGEN_URL", "http://127.0.0.1:8323"))
        self.request = curobo_request or self._request
        self.perception = perception if perception is not None else Perception()
        self.workspace_lookup = workspace_lookup or WorkspaceAtlas().query
        self.cad_grasp_library = cad_grasp_library
        self._client = None
        self._client_lock = threading.Lock()

    def close(self):
        """Release the transport after the application drains algorithm workers."""
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _request(self, path, body=None):
        started = time.perf_counter()
        try:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=600)
                client = self._client
            response = (client.get(self.curobo_url + path) if body is None
                        else client.post(self.curobo_url + path, json=body))
            received = time.perf_counter()
        except httpx.TimeoutException as error:
            raise AlgorithmError(f"cuRobo {path}: request timed out ({type(error).__name__})",
                                 code="algorithm_request_timeout") from error
        except httpx.HTTPError as error:
            raise AlgorithmError(f"cuRobo {path}: service unavailable ({error})",
                                 code="algorithm_service_unavailable") from error
        try:
            result = response.json()
        except ValueError as error:
            raise AlgorithmError(f"cuRobo {path}: response is not valid JSON",
                                 code="algorithm_invalid_response") from error
        if not isinstance(result, dict):
            raise AlgorithmError(f"cuRobo {path}: expected a JSON object",
                                 code="algorithm_invalid_response")
        result['transport_timings'] = {
            'request_encode_http_roundtrip_s': received - started,
            'response_decode_s': time.perf_counter() - received,
            'response_bytes': len(response.content),
            'server_timing': response.headers.get('Server-Timing'),
        }
        logging.getLogger(__name__).info('cuRobo request path=%s transport=%s service=%s',
            path, result['transport_timings'], result.get('request_timings'))
        if response.is_error or result.get("ok") is False:
            error = result.get("error")
            if isinstance(error, dict):
                raise AlgorithmError(f"cuRobo {path}: {error.get('message', 'service error')}",
                                     code=error.get("code") or "algorithm_error", details=error.get("details"))
            raise AlgorithmError(f"cuRobo {path}: {error or f'HTTP {response.status_code}'}")
        return result

    def health(self):
        result = {}
        try:
            result["curobo"] = self.request("/health")
        except (AlgorithmError, OSError, ValueError) as error:
            result["curobo"] = {"ready": False, "error": str(error)}
        try:
            value = self.graspgen.health()
            result["graspgen"] = {**value, "ready": value.get("model_ready") is True}
        except (OSError, ValueError) as error:
            result["graspgen"] = {"ready": False, "error": str(error)}
        return result

    @staticmethod
    def _target(snapshot, target_id):
        for obj in snapshot["objects"]:
            if obj["id"] == target_id:
                return obj
        raise AlgorithmError(f"target {target_id!r} is absent from the observed scene")

    @staticmethod
    def _binding(snapshot, target_id, scene):
        if scene.get("generation") != snapshot["generation"]:
            raise AlgorithmError("scene generation is stale")
        if scene.get("target_id") != target_id:
            raise AlgorithmError("scene target differs from the requested target")
        if snapshot.get("observation_id") != scene.get("observation_id"):
            raise AlgorithmError("scene observation is stale", code="planning_stale_observation")

    def observe(self, capture, instruction="", *, checkpoint=None, source_hint=None):
        return self.perception.observe(capture, instruction=instruction, checkpoint=checkpoint,
                                       source_hint=source_hint)

    @staticmethod
    def _cloud_reference(target):
        points = np.asarray(target["points_world_m"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16 or not np.isfinite(points).all():
            raise AlgorithmError("target has insufficient measured depth", code="perception_depth_insufficient")
        return {"id": hashlib.sha256(points.tobytes()).hexdigest(), "frame": f"cloud:{target['id']}",
                "point_count": len(points), "source": "rgbd_visible_surface"}

    @staticmethod
    def _cloud_frame(target):
        """A numerical point-cloud origin, not an estimated physical object pose."""
        points = np.asarray(target["points_world_m"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16 or not np.isfinite(points).all():
            raise AlgorithmError("target has insufficient measured depth", code="perception_depth_insufficient")
        frame = np.eye(4)
        frame[:3, 3] = points.mean(axis=0)
        return frame

    def build_scene(self, snapshot, target_id, *, planning_settings=None, exclusion_state="selected",
                    head_only=False):
        from .cameras import letterbox_view
        from .inventory_map import mapping_strategy
        from .semantic_transport import encode_project_semantic_request

        if any(obj.get('completed_geometry_points_world_m') is not None
               or obj.get('geometry_completion', {}).get('source') == 'measured_surface_and_user_height_prior'
               for obj in snapshot.get('objects', [])):
            raise AlgorithmError('synthetic_scene_geometry_unsupported:reobserve_required')
        if snapshot.get("source") != "rgbd":
            raise AlgorithmError("scene requires registered RGBD observation")
        # Observation maps include measured instances from the requested views.
        if exclusion_state == "observation":
            target_id = None
            head_only = head_only or tuple(snapshot['views']) == ('3',)
        demo_background = snapshot.get("demo_background_mapping") is True
        if demo_background:
            if exclusion_state != "observation":
                raise AlgorithmError("demo background bootstrap requires an observation map")
            head_only = False
        held_surface = snapshot.get("held_surface_world_m") if exclusion_state == "held" else None
        no_target = exclusion_state == 'observation' and not target_id
        target = self._target(snapshot, target_id) if held_surface is None and not no_target else None
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        entities = snapshot.get('scene_entities') or snapshot['objects']
        if isinstance(entities, dict):
            entities = list(entities.values())
        inventory = {entity['id']: entity for entity in entities}
        lifecycle = dict(snapshot.get('mapping_lifecycle', {}))
        strategy = mapping_strategy(**{key: lifecycle[key] for key in
            ('requested_scope', 'mapping_entity_ids', 'strategy_source', 'scope_reason') if key in lifecycle})
        retained_required = set()
        if exclusion_state in {'observation', 'selected'}:
            unresolved = snapshot.get('mapping_unresolved', [])
            if strategy:
                retained_rows = [row for row in unresolved
                    if row.get('status') == 'unseen' and not row.get('candidates')
                    and row.get('previous_id') in inventory
                    and row['previous_id'] not in lifecycle.get('retired_entity_ids', [])]
                retained_required = {row['previous_id'] for row in retained_rows}
                unresolved = [row for row in unresolved if row not in retained_rows]
            if unresolved:
                raise AlgorithmError('semantic_scene_identity_unresolved', code='semantic_scene_identity_unresolved',
                    details={'unresolved': unresolved})
            task_ids = set(snapshot.get('task_member_ids', [])) | {target_id}
            for query in snapshot.get('declared_environment', []):
                observed = any(obj['id'] not in task_ids and obj.get('kind') not in {'bin', 'marker', 'table'}
                               and obj.get('color') == query['color'] for obj in snapshot['objects'])
                historical = set()
                if strategy and query.get('status') == 'observed':
                    for identifier in query.get('observed_ids', []):
                        entity = inventory.get(identifier, {})
                        position = np.asarray(entity.get('position', []), dtype=float)
                        if (identifier not in task_ids and identifier not in lifecycle.get('retired_entity_ids', [])
                                and entity.get('kind') not in {'bin', 'marker', 'table'}
                                and entity.get('color') == query['color'] and entity.get('observation_id')
                                and entity.get('position_source') != 'simulation_truth'
                                and position.shape == (3,) and np.isfinite(position).all()):
                            historical.add(identifier)
                if not observed and not historical:
                    raise AlgorithmError('declared_obstacle_not_observed', code='declared_obstacle_not_observed',
                        details={'query': query, 'reason': 'no_distinct_non_task_object'})
                retained_required.update(historical)
        views = {}
        coverage_views = {}
        camera_capture_metadata = {}
        markers = {obj['id'] for obj in snapshot['objects'] if obj.get('kind') == 'marker'}
        for camera_id in (("3",) if head_only else ("1", "2", "3")):
            captured_view = snapshot["views"][camera_id]
            view = letterbox_view(captured_view)
            coverage_view = deepcopy(captured_view)
            coverage_view['depth_m'] = captured_view.get('coverage_depth_m', captured_view['depth_m'])
            coverage_view['valid_depth'] = captured_view.get('coverage_valid_depth', captured_view['valid_depth'])
            coverage_view = letterbox_view(coverage_view)
            captured_depth = np.asarray(captured_view["depth_m"])
            captured_valid = np.asarray(captured_view["valid_depth"], dtype=bool) & np.isfinite(captured_depth) & (captured_depth > 0)
            camera_capture_metadata[camera_id] = {
                "image_shape_hw": list(captured_depth.shape),
                "valid_depth_count": int(captured_valid.sum()),
                "invalid_depth_count": int(captured_valid.size - captured_valid.sum()),
                "depth_sha256": hashlib.sha256(np.ascontiguousarray(captured_depth).tobytes()).hexdigest(),
                "calibration_id": captured_view.get("calibration_id"),
                "capture_time": snapshot.get("time"),
                "intrinsics": dict(captured_view["intrinsics"]),
                "letterbox_offset_rc": [(view["depth_m"].shape[0] - captured_depth.shape[0]) // 2,
                                        (view["depth_m"].shape[1] - captured_depth.shape[1]) // 2],
            }
            if held_surface is not None:
                from scipy.spatial import cKDTree

                from .cameras import deproject_view

                # Remove only current measured returns near the rigid held reference.
                # This is robot/held geometry filtering, not a new semantic observation.
                view = deepcopy(view)
                valid = np.asarray(view["valid_depth"], dtype=bool) & np.isfinite(view["depth_m"]) & (view["depth_m"] > 0)
                matches = cKDTree(np.asarray(held_surface)).query(deproject_view(view))[0] <= .01
                rows, columns = np.where(valid)
                view["valid_depth"][rows[matches], columns[matches]] = False
                view["depth_m"][rows[matches], columns[matches]] = 0.
                view["instance_masks"] = {}
            k = view["intrinsics"]
            matrix = [[k["fx"], 0., k["cx"]], [0., k["fy"], k["cy"]], [0., 0., 1.]]
            views[camera_id] = {"camera_id": camera_id, "stereo_sequence": snapshot["generation"],
                                "depth_left_m": np.asarray(view["depth_m"], dtype=np.float32),
                                "valid_left_mask": np.asarray(view["valid_depth"], dtype=np.uint8),
                                "left_intrinsics": matrix, "rgb_intrinsics": matrix,
                                "T_base_left_camera": view["T_world_camera"],
                                "robot_joints_at_capture": view["robot_joints_deg"],
                                "rgb_size": [k["width"], k["height"]], "left_to_rgb": np.eye(4).tolist(),
                                "instance_masks": {key: mask for key, mask in view['instance_masks'].items()
                                                   if key not in markers}}
            coverage_views[camera_id] = {**views[camera_id],
                'depth_left_m': np.asarray(coverage_view['depth_m'], dtype=np.float32),
                'valid_left_mask': np.asarray(coverage_view['valid_depth'], dtype=np.uint8)}
        semantic = {"observation_set_id": snapshot["observation_id"],
                    "sam3_generation": snapshot["generation"], "confirmed_instance_id": target_id or '',
                    "selection_generation": snapshot["generation"], "views": views}
        if held_surface is not None or demo_background:
            semantic.update(mapping_mode="background", confirmed_instance_id="", sam3_generation=0)
            for view in (*views.values(), *coverage_views.values()):
                view['instance_masks'] = {}
        payload = encode_project_semantic_request(semantic, exclusion_state=exclusion_state)
        payload['coverage_rgbd'] = encode_project_semantic_request(
            {**semantic, 'views': coverage_views}, exclusion_state=exclusion_state)
        lifecycle.setdefault('required_entity_ids', [obj['id'] for obj in entities
            if obj.get('kind') not in {'marker', 'table'} and not obj.get('retired', False)])
        lifecycle['required_entity_ids'] = sorted(set(lifecycle['required_entity_ids']) | retained_required)
        payload['mapping_lifecycle'] = lifecycle
        from .pointcloud_preview import observed_obb_preview

        payload['object_obb_observations'] = observed_obb_preview(snapshot)
        if os.getenv('LUNA_GEOMETRY_MODE') == 'cad_obb':
            from .object_obb import observed_object_obb

            visible_ids = set().union(*(set(view['instance_masks']) for view in views.values()))
            payload['geometry_objects'] = []
            for obj in snapshot['objects']:
                if obj.get('kind') in {'table', 'marker'} or obj['id'] not in visible_ids:
                    continue
                geometry = ({'cad': deepcopy(obj['cad'])} if obj.get('cad') is not None
                            else {'obb': observed_object_obb(obj)})
                payload['geometry_objects'].append({'id': obj['id'], 'kind': obj['kind'],
                    'observation_id': snapshot['observation_id'], 'generation': snapshot['generation'], **geometry})
        if snapshot.get("demo_mapping") is True or demo_background:
            payload["demo_mapping"] = True
            payload["demo_tables"] = snapshot.get("demo_tables")
        if held_surface is not None:
            payload["held_instance_id"] = target_id
        result = self.request("/perception/build", {**payload, "planning_settings": settings,
                            "instance_floor_heights_m": {obj['id']: float(obj['placement_reference']['floor_z_m'])
                                for obj in snapshot['objects'] if obj.get('kind') == 'bin'
                                and 'floor_z_m' in obj.get('placement_reference', {})},
                            "camera_capture_metadata": camera_capture_metadata,
                            "checkpoint": snapshot.get("checkpoint"),
                            "full_observation_complete": snapshot.get('observation_pose_verified') is True,
                            "mapping_max_sphere_radius_m": (0. if target is None else
                                float(np.linalg.norm(np.ptp(np.asarray(target['points_world_m']), axis=0))
                                      + target.get('uncertainty_m', .005) + settings['attachment']['margin_m'])),
                            "exclusion_state": exclusion_state, "gripper_joint_positions_rad": {
                                arm: float(state["gripper_position_rad"])
                                for arm, state in snapshot["arms"].items()},
                            "selected_surface_world_m": [] if target is None else np.asarray(target["points_world_m"]).tolist()})
        return {**result, "generation": snapshot["generation"], "target_id": target_id,
                "observation_id": snapshot["observation_id"], "exclusion_state": exclusion_state}

    @staticmethod
    def _graspgen_object_points(snapshot, target):
        """Sample the observed OBB; keep the existing measured cloud frame."""
        from .object_obb import obb_surface_points, observed_object_obb
        from .placement import measured_object_geometry

        measured_object_geometry(target)
        return obb_surface_points(observed_object_obb(target))

    def generate_grasps(self, snapshot, target_id, *, arm=None, candidate_count=200,
                        progress=None, cancelled=None):
        from .cameras import deproject_view

        self._check_cancelled(cancelled)
        if type(candidate_count) is not int or not 1 <= candidate_count <= 1000:
            raise ValueError("candidate_count must be an integer from 1 to 1000")
        if arm not in (None, 'A', 'B'):
            raise ValueError('invalid_requested_arm')
        if snapshot.get("source") != "rgbd":
            raise AlgorithmError("GraspGen requires an observed RGBD cloud")
        target = self._target(snapshot, target_id)
        if target.get('cad') is not None:
            return self._generate_cad_grasps(snapshot, target, arm)
        reference = self._cloud_reference(target)
        cloud_frame = self._cloud_frame(target)
        object_points = (self._graspgen_object_points(snapshot, target) - cloud_frame[:3, 3]).astype(np.float32)
        groups = [deproject_view(view) for view in snapshot["views"].values()]
        world = np.concatenate(groups)
        if not len(world):
            raise AlgorithmError("scene RGBD has no valid depth")
        world = world[::max(1, len(world) // 50000)]
        inverse = np.linalg.inv(cloud_frame)
        scene_points = (world @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.float32)
        request = {"scene_points_camera_m": scene_points,
                   "object_points_camera_m": object_points, "gripper_config_id": "robotiq_2f_140"}
        candidates, batches = [], []
        model = None
        # Keep the service's top-100 inference protocol. The fixed budget sets
        # the number of samples before any downstream filtering or IK.
        for batch in range(1, (candidate_count + 99) // 100 + 1):
            self._check_cancelled(cancelled)
            started = time.monotonic()
            result = self.graspgen.generate(request)
            self._check_cancelled(cancelled)
            if result.get("ok") is False:
                raise AlgorithmError(f"GraspGen: {result['error']}")
            if not result.get("candidates"):
                raise AlgorithmError("GraspGen returned no candidates")
            if model is not None and result["model"] != model:
                raise AlgorithmError("GraspGen model changed between candidate batches")
            model = result["model"]
            candidates.extend({"source_batch": batch, "source_index": c["index"], "score": c["score"],
                               "T_object_grasp": rigid_matrix(c["T_camera_grasp"]).tolist()}
                              for c in result["candidates"])
            batches.append({"batch_index": batch, "candidate_count": len(result["candidates"]),
                            "duration_s": time.monotonic() - started})
            if progress is not None:
                progress("graspgen_batch_complete", "GraspGen 候选采样完成", batches[-1])
        candidates.sort(key=lambda candidate: (-candidate["score"], candidate["source_batch"],
                                               candidate["source_index"]))
        candidates = [{"index": index, **candidate} for index, candidate in enumerate(candidates)]
        if arm is not None:
            world = cloud_frame @ np.asarray([candidate['T_object_grasp'] for candidate in candidates])
            for candidate, tcp in zip(candidates, grasp_targets(world, arm)):
                candidate['T_world_tcp'] = tcp.tolist()
        all_generated = deepcopy(candidates)
        candidates = candidates[:candidate_count]
        from .object_obb import observed_object_obb

        return {"candidates": candidates, "all_generated_candidates": all_generated,
                "object_obb": observed_object_obb(target),
                "graspgen_input_source": "observed_obb_mesh_surface_samples",
                "model": model, "frame": f"cloud:{target_id}",
                'arm': arm, 'tool_profile_revision': TOOL_PROFILE['revision'],
                'tcp_frame': TCP_FRAMES.get(arm),
                "T_world_cloud": cloud_frame.tolist(),
                "graspgen_input": request,
                "requested_candidate_count": candidate_count, "candidate_count": len(candidates), "batches": batches,
                "generation": snapshot["generation"], "observation_id": snapshot["observation_id"],
                "cloud": reference}

    def _generate_cad_grasps(self, snapshot, target, arm):
        from .cad_grasps import CadGraspError, CadGraspLibrary

        cad = target['cad']
        if (cad.get('instance_id') != target['id'] or any(cad.get(key) != snapshot.get(key)
                for key in ('observation_id', 'generation'))):
            raise AlgorithmError('cad_pose_binding_stale')
        if cad.get('pose_validation', {}).get('accepted') is not True:
            raise AlgorithmError('cad_pose_not_validated')
        library = self.cad_grasp_library or CadGraspLibrary(
            os.getenv('LUNA_CAD_ROOT', str(Path(__file__).resolve().parents[3] / 'CAD')))
        try:
            source = {'source_asset_id': cad['source_asset_id']} if cad.get('source_asset_id') else {}
            result = library.generate(asset_id=cad['asset_id'], source_sha256=cad['source_sha256'],
                arm=arm, T_world_source=cad['T_world_source'], **source)
        except CadGraspError as error:
            raise AlgorithmError(str(error), code=error.code) from error
        return {**result, 'all_generated_candidates': deepcopy(result['candidates']),
            'requested_candidate_count': result['candidate_count'], 'model': cad['asset_id'],
            'frame': 'original_CAD', 'arm': arm, 'tcp_frame': TCP_FRAMES[arm],
            'tool_profile_revision': TOOL_PROFILE['revision'],
            'generation': snapshot['generation'], 'observation_id': snapshot['observation_id'],
            'cloud': self._cloud_reference(target), 'sampling_input_digest': result['library_sha256'],
            'batches': [{'batch_index': 1, 'candidate_count': result['candidate_count'],
                        'source': 'cad_offline_library'}]}

    @staticmethod
    def _batch_items(response, count, scene_token, operation):
        if response.get("scene_token") != scene_token:
            raise AlgorithmError(f"{operation} returned a different scene token")
        items = response.get("items")
        if type(response.get("batch_size")) is not int or response["batch_size"] != count:
            raise AlgorithmError(f"{operation} returned an invalid batch size")
        if not isinstance(items, list) or len(items) != count:
            raise AlgorithmError(f"{operation} returned an invalid number of results")
        for index, item in enumerate(items):
            if (not isinstance(item, dict) or type(item.get("candidate_index")) is not int
                    or item["candidate_index"] != index or type(item.get("success")) is not bool):
                raise AlgorithmError(f"{operation} returned invalid candidate indices or success flags")
        return items

    @staticmethod
    def _check_cancelled(cancelled):
        if cancelled is not None and cancelled.is_set():
            raise AlgorithmError("algorithm operation was cancelled", code="planning_cancelled")

    @staticmethod
    def _graspgen_replay_inputs(replay):
        # Preserve submitted arrays and their frame/provenance; never rebuild
        # missing historical inference input from a measured target cloud.
        fields = ("graspgen_request", "T_world_cloud", "observation_id",
                  "graspgen_input_source", "graspgen_input_sampling",
                  "completion_provenance", "graspgen_input_cloud", "object_obb",
                  "measured_cloud_reference", "completed_geometry_points_world_m")
        return deepcopy({key: replay[key] for key in fields if key in replay})

    def evaluate_reachability(self, snapshot, target_id, destination_id=None, *, requested_arm=None):
        from .task_progress import measured_position

        if requested_arm not in (None, 'A', 'B'):
            raise ValueError('invalid_requested_arm')
        target = self._target(snapshot, target_id)
        destination = self._target(snapshot, destination_id) if destination_id else None
        positions = []
        for obj in ([target, destination] if destination is not None else [target]):
            position = measured_position(obj)
            if position is None:
                raise ValueError('reachability_requires_measured_center:' + obj['id'])
            positions.append(position.tolist())
        result = {}
        for arm in ((requested_arm,) if requested_arm else ('A', 'B')):
            items = self.workspace_lookup(arm, positions)
            if (not isinstance(items, list) or len(items) != len(positions)
                    or any(not isinstance(item, dict) or type(item.get('reachable')) is not bool
                    or item.get('position_world_m') != position for item, position in zip(items, positions))):
                raise AlgorithmError('invalid workspace lookup response')
            source_ik = items[0]
            destination_ik = items[1] if destination else None
            result[arm] = {'arm': arm, 'target_id': target_id, 'destination_id': destination_id,
                'source': source_ik, 'destination': destination_ik,
                'reachable': source_ik['reachable'] and (destination_ik is None or destination_ik['reachable']),
                'feasibility_scope': 'offline_position_workspace', 'collision_checked': False,
                'complete_path_status': 'not_planned'}
        return result

    def evaluate_arms(self, snapshot, target_id, grasps, scene, *, planning_settings=None,
                      progress=None, cancelled=None, destination_id=None, requested_arm=None):
        if requested_arm not in (None, 'A', 'B'):
            raise ValueError('invalid_requested_arm')
        bound_arm = grasps.get('arm')
        if bound_arm is not None:
            if requested_arm not in (None, bound_arm):
                raise AlgorithmError('grasp_candidate_arm_mismatch')
            if grasps.get('tool_profile_revision') != TOOL_PROFILE['revision']:
                raise AlgorithmError('grasp_candidate_tool_profile_changed')
            if grasps.get('tcp_frame') != TCP_FRAMES.get(bound_arm):
                raise AlgorithmError('grasp_candidate_tcp_frame_mismatch')
            requested_arm = bound_arm
        self._check_cancelled(cancelled)
        self._binding(snapshot, target_id, scene)
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        sampling = ({"sampling_round": grasps["sampling_round"]} if "sampling_round" in grasps else {})
        replay_inputs = self._graspgen_replay_inputs(grasps.get("candidate_replay", {}))
        target = self._target(snapshot, target_id)
        candidates = grasps["candidates"]
        if not candidates:
            raise AlgorithmError("cannot evaluate an empty grasp candidate set")
        indices = [candidate["index"] for candidate in candidates]
        if (any(type(index) is not int for index in indices) or len(set(indices)) != len(indices)
                or any(not np.isfinite(candidate["score"]) for candidate in candidates)):
            raise AlgorithmError("GraspGen returned invalid candidate indices or scores")
        cloud_frame = rigid_matrix(grasps.get("T_world_cloud", self._cloud_frame(target)))
        world = None if bound_arm is not None else cloud_frame @ np.asarray([c["T_object_grasp"] for c in candidates])
        result = {}
        for arm in ((requested_arm,) if requested_arm else ('A', 'B')):
            peer = "B" if arm == "A" else "A"
            poses = (np.asarray([rigid_matrix(candidate['T_world_tcp']) for candidate in candidates])
                     if bound_arm is not None else grasp_targets(world, arm))
            request = {"arm": arm, "start_joints_deg": np.rad2deg(snapshot["arms"][arm]["joints_rad"]).tolist(),
                       "other_joints_deg": np.rad2deg(snapshot["arms"][peer]["joints_rad"]).tolist(),
                       "gripper_joint_positions_rad": {a: snapshot['arms'][a]['gripper_position_rad'] for a in ('A', 'B')},
                       "scene_token": scene["scene_token"], "planning_settings": settings,
                       "trajectory_collision_clearance_m": settings["trajectory_collision_clearance_m"]}
            valid, batches = [], []
            for start in range(0, len(candidates), 100):
                self._check_cancelled(cancelled)
                goals = poses[start:start + 100].tolist()
                batch_data = {**sampling, "arm": arm, "batch_index": start // 100 + 1,
                              "batch_count": (len(candidates) + 99) // 100,
                              "candidate_indices": indices[start:start + 100],
                              "duration_s": 0., "success_count": 0}
                if progress is not None:
                    progress("ik_batch_start", "cuRobo batched IK 开始", batch_data)
                self._check_cancelled(cancelled)
                started = time.monotonic()
                try:
                    ik = self.request("/tools/solve_ik_batch", {**request, "goal_matrices_world": goals})
                except AlgorithmError:
                    self._check_cancelled(cancelled)
                    raise
                self._check_cancelled(cancelled)
                items = self._batch_items(ik, len(goals), scene["scene_token"], "IK")
                candidate_filters = []
                for local_index, item in enumerate(items):
                    index = start + local_index
                    if item["success"]:
                        candidate = {**sampling, **{key: candidates[index][key]
                                      for key in ("source_batch", "source_index", "source", "grasp_id", "pair_id") if key in candidates[index]},
                                      "candidate_index": candidates[index]["index"],
                                      "score": candidates[index]["score"],
                                      "T_world_tcp": poses[index].tolist(), "ik": item}
                        candidate, evidence = self._grasp_ik_candidate(
                            candidate, snapshot, target_id, arm, scene, settings)
                        if evidence is not None:
                            candidate_filters.append(evidence)
                        if candidate is not None:
                            valid.append(candidate)
                completed = {**batch_data, "duration_s": time.monotonic() - started,
                             "success_count": sum(item["success"] for item in items),
                              "eligible_count": sum(item["success"] for item in items) - sum(
                                  not row["accepted"] for row in candidate_filters),
                              "candidate_filters": candidate_filters,
                              "effective_config": ik.get("effective_config"),
                              "candidate_replay": {**replay_inputs, **sampling, "stage": "after_ik", "target_id": target_id,
                                  "candidate_indices": indices[start:start + 100],
                                  "request": {**request, "goal_matrices_world": goals},
                                  "response": deepcopy(ik),
                                  "candidate_filters": deepcopy(candidate_filters),
                                  "measured_points_world_m": np.asarray(target["points_world_m"]).copy(),
                                  "T_world_cloud": cloud_frame.tolist()}}
                batches.append(completed)
                if progress is not None:
                    progress("ik_batch_complete", "cuRobo batched IK 完成", completed)
                self._check_cancelled(cancelled)
            valid, destination_screening = self._grasp_destination_candidates(
                valid, snapshot, target_id, arm, scene, settings, destination_id)
            self._check_cancelled(cancelled)
            result[arm] = {**sampling, **({"graspgen_replay_inputs": replay_inputs} if replay_inputs else {}),
                           "valid_candidates": valid, "tested_count": len(candidates),
                           "arm": arm, "target_id": target_id,
                           "start_joints_deg": request["start_joints_deg"],
                           "other_joints_deg": request["other_joints_deg"],
                           "planning_settings": deepcopy(settings), "batches": batches,
                           "orientation_deg": TOOL_PROFILE[arm]["orientation_deg"],
                           "scene_token": scene["scene_token"], "generation": snapshot["generation"],
                           "tool_profile_revision": TOOL_PROFILE["revision"], "tcp_frame": TCP_FRAMES[arm]}
            if destination_id is not None:
                observed_object(snapshot, destination_id)
                # Fixed tool geometry can reject a bounded set of destination
                # endpoints before closure; carried motion still needs actual feedback.
                screened = isinstance(destination_screening, dict) and destination_screening.get('applied') is True
                scope = 'grasp_ik_and_fixed_tool_destination_screen' if screened else 'grasp_ik_only'
                result[arm].update(destination_id=destination_id, feasibility_scope=scope,
                                   place_planning_stage="after_grasp_attachment")
                if destination_screening is not None:
                    result[arm]['destination_screening'] = destination_screening
        return result

    @staticmethod
    def _plan_phases(item, arm):
        flags = ("goalset_success", "approach_success", "grasp_success", "lift_success")
        if any(type(item.get(flag)) is not bool for flag in flags):
            raise AlgorithmError("planner returned invalid phase success flags")
        if not item["success"]:
            return None
        if not all(item[flag] for flag in flags):
            raise AlgorithmError("planner marked an incomplete three-phase trajectory as successful")
        names = [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)]
        phases = {}
        try:
            for name in ("approach", "grasp", "lift"):
                phase = item["phases"][name]
                joint_names = phase["joint_names"]
                if len(joint_names) != len(set(joint_names)):
                    raise ValueError("duplicate joint names")
                order = [joint_names.index(joint) for joint in names]
                raw = np.asarray(phase["positions_rad"], dtype=float)
                if raw.ndim != 2 or raw.shape[1] != len(joint_names) or raw.shape[0] < 2:
                    raise ValueError("invalid trajectory shape")
                velocity = np.asarray(phase["velocities_rad_s"], dtype=float)
                acceleration = np.asarray(phase["accelerations_rad_s2"], dtype=float)
                if velocity.shape != raw.shape or acceleration.shape != raw.shape:
                    raise ValueError("invalid trajectory derivative shape")
                raw_dt = float(phase["interpolation_dt_s"])
                if (not all(np.isfinite(values).all() for values in (raw, velocity, acceleration))
                        or not np.isfinite(raw_dt) or raw_dt <= 0):
                    raise ValueError("invalid trajectory samples or dt")
                phases[name] = {"positions_rad": raw[:, order].tolist(),
                                "velocities_rad_s": velocity[:, order].tolist(),
                                "accelerations_rad_s2": acceleration[:, order].tolist(),
                                "interpolation_dt_s": raw_dt}
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise AlgorithmError(f"planner returned invalid trajectory samples: {error}") from error
        return phases

    @staticmethod
    def _native_phases(values, arm):
        joint_names = [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)]
        result = {}
        for name, phase in values.items():
            try:
                names = phase["joint_names"]
                if len(names) != len(set(names)):
                    raise ValueError("duplicate joint names")
                order = [names.index(joint) for joint in joint_names]
                arrays = [np.asarray(phase[key], dtype=float) for key in
                          ("positions_rad", "velocities_rad_s", "accelerations_rad_s2")]
                if (arrays[0].ndim != 2 or arrays[0].shape[1] != len(names) or len(arrays[0]) < 2
                        or any(values.shape != arrays[0].shape or not np.isfinite(values).all()
                               for values in arrays)):
                    raise ValueError("invalid trajectory samples or derivatives")
                dt = float(phase["interpolation_dt_s"])
                if not np.isfinite(dt) or dt <= 0:
                    raise ValueError("invalid sample period")
                result[name] = {key: array[:, order].tolist() for key, array in zip(
                    ("positions_rad", "velocities_rad_s", "accelerations_rad_s2"), arrays)}
                result[name]["interpolation_dt_s"] = dt
                if phase.get('terminal_state') is not None:
                    result[name]['terminal_state'] = deepcopy(phase['terminal_state'])
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise AlgorithmError("invalid native phase " + name + ": " + str(error),
                                     code="algorithm_invalid_response") from error
        return result

    @staticmethod
    def _motion_request(observation, arm, scene, goal, settings):
        transform = rigid_matrix(goal)
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        peer = "B" if arm == "A" else "A"
        return {"arm": arm,
                "start_joints_deg": np.rad2deg(observation["arms"][arm]["joints_rad"]).tolist(),
                "other_joints_deg": np.rad2deg(observation["arms"][peer]["joints_rad"]).tolist(),
                "goal_position_m": transform[:3, 3].tolist(),
                "goal_quaternion_wxyz": [float(quaternion[3]), *quaternion[:3].tolist()],
                "freq_hz": settings["freq_hz"], "max_attempts": settings["plan_max_attempts"],
                "scene_token": scene["scene_token"], "planning_settings": settings,
                "gripper_joint_positions_rad": {
                    name: float(feedback["gripper_position_rad"]) for name, feedback in observation["arms"].items()}}

    def validate_execution(self, observation, arm, scene, phase, command, binding=None, *,
                           planning_settings=None, gripper_end_position_rad=None,
                           contact_goal_pose_world=None, previous_check=None, closure_driver_limit_rad=None):
        self._binding(observation, scene["target_id"], scene)
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        peer = "B" if arm == "A" else "A"
        payload = {"arm": arm, "phase": phase, "scene_token": scene["scene_token"],
                   "positions_rad": command["positions_rad"],
                   "current_joints_rad": observation["arms"][arm]["joints_rad"],
                   "joint_names": [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)],
                   "other_joints_deg": np.rad2deg(observation["arms"][peer]["joints_rad"]).tolist(),
                   "planning_settings": settings, "gripper_joint_positions_rad": {
                       name: float(value["gripper_position_rad"]) for name, value in observation["arms"].items()}}
        payload["gripper_geometry_mode"] = (
            "fixed_open" if binding is None and phase in {"approach_grasp", "grasp", "grasp_closure", "close", "open_grasp"}
            else "actual")
        if binding is not None:
            payload["carrying_binding"] = binding
        if closure_driver_limit_rad is not None:
            payload['closure_driver_limit_rad'] = closure_driver_limit_rad
        if contact_goal_pose_world is not None:
            payload["contact_goal_pose_world"] = rigid_matrix(contact_goal_pose_world).tolist()
        if phase in ("close", "open", "open_grasp"):
            payload["gripper_start_position_rad"] = observation["arms"][arm]["gripper_position_rad"]
            payload["gripper_end_position_rad"] = gripper_end_position_rad
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()
        # Reuse only a proof for the identical final command and geometry.
        # Any changed path, peer, fingers, attachment or scene sends a fresh query.
        if (previous_check is not None and previous_check.get('binding_sha256') == expected
                and previous_check.get('scene_token') == scene['scene_token']
                and previous_check.get('collision_free') is True
                and previous_check.get('attached') is (binding is not None)):
            return {**deepcopy(previous_check), 'reused': True}
        result = self.request("/carrying/validate", payload)
        if result.get("scene_token") != scene["scene_token"] or result.get("binding_sha256") != expected:
            raise AlgorithmError("collision proof differs from submitted command", code="collision_proof_mismatch")
        if result.get("collision_free") is not True:
            if result.get("failure_reason") == "observation_coverage_insufficient":
                raise AlgorithmError("command lacks depth evidence inside the observed task region",
                                     code="execution_observation_coverage_insufficient", details=result)
            raise AlgorithmError("command collides", code="execution_collision", details=result)
        if result.get("attached") is not (binding is not None):
            raise AlgorithmError("collision proof has a different attachment state", code="collision_proof_mismatch")
        return result

    @staticmethod
    def _held_binding(observation, target_id, arm, scene, holding):
        from .carrying import held_collision_binding

        return held_collision_binding(holding, scene, arm, target_id)


    def _complete_place_candidate(self, observation, arm, scene, binding, raw, settings, *, cancelled=None):
        phases = build_execution_phases(raw, settings["time_scale"])
        checks = {}
        checking = deepcopy(observation)
        for name in ("approach", "place"):
            self._check_cancelled(cancelled)
            checks[name] = self.validate_execution(checking, arm, scene, name, phases[name], binding,
                                                   planning_settings=settings)
            checking["arms"][arm]["joints_rad"] = phases[name]["positions_rad"][-1]
        # Plan the third stage with an open, detached model. This is hypothetical;
        # the runtime's logical attachment remains held until execution opens it.
        from .kinematics import robot_tcp

        checking["arms"][arm].update(gripper_position_rad=0., gripper_closed_fraction=0.)
        release_tcp = np.asarray(robot_tcp(checking["arms"], arm), dtype=float)
        checking["arms"][arm]["tcp"] = release_tcp.tolist()
        retreat_goal = release_tcp.copy()
        offset = np.eye(3)["xyz".index(settings["place_retreat_axis"])] * settings["place_retreat_offset_m"]
        if settings["place_retreat_in_tool_frame"]:
            offset = release_tcp[:3, :3] @ offset
        retreat_goal[:3, 3] += offset
        self._check_cancelled(cancelled)
        retreat = self.plan_retreat(checking, arm, scene, retreat_goal.tolist(), planning_settings=settings,
                                    start_state=raw['place'].get('terminal_state'))
        self._check_cancelled(cancelled)
        phases["retreat"], checks["retreat"] = retreat["command"], retreat["check"]
        return phases, checks

    def plan_retreat(self, observation, arm, scene, goal, *, planning_settings=None, start_state=None):
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        payload = self._motion_request(observation, arm, scene, goal, settings)
        payload.update(departure_axis=settings['place_retreat_axis'],
                       departure_in_tool_frame=settings['place_retreat_in_tool_frame'])
        if start_state is not None:
            payload['departure_start_state'] = deepcopy(start_state)
        response = self.request("/tools/plan_to_pose", payload)
        if response.get("scene_token") != scene["scene_token"] or response.get("attached") is not False:
            raise AlgorithmError("retreat requires a detached current scene")
        raw = self._native_phases({"retreat": response["trajectory"]}, arm)
        command = build_execution_phases(raw, settings["time_scale"])["retreat"]
        check = self.validate_execution(observation, arm, scene, "retreat", command, planning_settings=settings)
        return {"command": command, "check": check}

    def clear_carrying(self):
        return self.request("/carrying/clear", {})

    def prepare_carrying(self, observation, arm, scene, holding, *, planning_settings=None,
                         placement_request=None):
        self._binding(observation, holding["instance_id"], scene)
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        binding = self._held_binding(observation, holding["instance_id"], arm, scene, holding)
        payload = self._motion_request(observation, arm, scene, observation["arms"][arm]["tcp"], settings)
        if placement_request:
            if placement_request.get('transport_region'):
                payload['next_motion_kind'] = 'pose'
            elif placement_request.get('region'):
                payload['next_motion_kind'] = 'place'
        result = self.request("/carrying/prepare", {**payload, "carrying_binding": binding})
        if result.get("attached") is not True or result.get("scene_token") != scene["scene_token"]:
            raise AlgorithmError("attachment preparation failed", code="attachment_not_applied")
        return result

    def detach_carrying(self):
        return self.request("/carrying/detach", {})

    def plan_dual_observation_motion(self, observation, scene, goals, *, planning_settings=None):
        self._binding(observation, scene["target_id"], scene)
        if scene.get("exclusion_state") != "observation" and not scene.get('return_after_place'):
            raise AlgorithmError("observation motion requires target-inclusive scene")
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        payload = {"scene_token": scene["scene_token"], "planning_settings": settings,
                   'return_after_place': bool(scene.get('return_after_place')),
                   "start_joints_deg": {a: np.rad2deg(observation["arms"][a]["joints_rad"]).tolist() for a in ("A", "B")},
                   "goal_joints_deg": {a: np.rad2deg(goals[a]).tolist() for a in ("A", "B")},
                   "gripper_joint_positions_rad": {a: observation["arms"][a]["gripper_position_rad"] for a in ("A", "B")}}
        result = self.request("/tools/plan_dual_observation", payload)
        if result.get("scene_token") != scene["scene_token"] or result.get("attached") is not False:
            raise AlgorithmError("dual observation returned stale or attached trajectory")
        return {"paths": {a: np.deg2rad(result["trajectories"][a]).tolist() for a in ("A", "B")},
                "dt": result["sample_dt_s"], "scene_token": scene["scene_token"]}

    def validate_dual_observation(self, observation, scene, paths, *, planning_settings=None):
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        result = self.request("/tools/validate_dual_observation", {
            'return_after_place': bool(scene.get('return_after_place')),
            "scene_token": scene["scene_token"], "planning_settings": settings,
            "expected_scene_token": scene["scene_token"],
            "points_deg": {a: np.rad2deg(paths[a]).tolist() for a in ("A", "B")},
            "gripper_joint_positions_rad": {a: observation["arms"][a]["gripper_position_rad"] for a in ("A", "B")}})
        if result.get("scene_token") != scene["scene_token"] or result.get("collision_free") is not True:
            raise AlgorithmError("dual observation command failed collision checking")
        return result

    def _grasp_ik_candidate(self, candidate, snapshot, target_id, arm, scene, settings):
        return candidate, None

    def _grasp_destination_candidates(self, valid, snapshot, target_id, arm, scene, settings, destination_id):
        return valid, None

    def _grasp_plan_evidence(self, candidate, item, phases, snapshot, target_id, arm, scene, settings):
        return {}

    def _grasp_lift_settings(self, snapshot, target_id, arm, candidates, scene, settings):
        return settings, None

    def plan_grasp(self, snapshot, target_id, arm, evaluations, scene, *, planning_settings=None,
                   progress=None, cancelled=None, destination_id=None):
        def check_cancelled():
            self._check_cancelled(cancelled)

        check_cancelled()
        self._binding(snapshot, target_id, scene)
        if arm not in {"A", "B"}:
            raise AlgorithmError("execution arm must be A or B")
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        evaluation = evaluations[arm]
        sampling = {key: deepcopy(evaluation[key]) for key in ("sampling_round", "sampling_evidence")
                    if key in evaluation}
        replay_inputs = self._graspgen_replay_inputs(evaluation.get("graspgen_replay_inputs", {}))
        expected = {"arm": arm, "target_id": target_id, "generation": snapshot["generation"],
                    "scene_token": scene["scene_token"], "planning_settings": settings,
                    "tool_profile_revision": TOOL_PROFILE["revision"], "tcp_frame": TCP_FRAMES[arm]}
        for field, value in expected.items():
            if evaluation.get(field) != value:
                raise AlgorithmError(f"IK evaluation {field} differs from the requested plan",
                                     code="planning_stale_evaluation")
        if destination_id is not None and evaluation.get("destination_id") != destination_id:
            raise AlgorithmError("paired IK destination differs from place target", code="planning_stale_evaluation")
        peer = "B" if arm == "A" else "A"
        joint_drift = {}
        for field, checked_arm in (("start_joints_deg", arm), ("other_joints_deg", peer)):
            frozen = np.asarray(evaluation.get(field), dtype=float)
            current = np.rad2deg(snapshot["arms"][checked_arm]["joints_rad"])
            if frozen.shape != (7,) or not np.isfinite(frozen).all():
                raise AlgorithmError("IK evaluation has invalid frozen start joints")
            joint_drift[checked_arm] = float(np.max(np.abs(current - frozen)))
            if not np.isfinite(joint_drift[checked_arm]) or joint_drift[checked_arm] > 1.:
                raise AlgorithmError(f"arm {checked_arm} moved more than 1 degree since batched IK",
                                     code="planning_start_changed", details={"joint_drift_deg": joint_drift})
        failures, batches = [], []
        candidates = evaluation["valid_candidates"]
        indices = [candidate["candidate_index"] for candidate in candidates]
        if (len(indices) != len(set(indices)) or any(type(index) is not int for index in indices)
                or any(not np.isfinite(candidate["score"]) for candidate in candidates)):
            raise AlgorithmError("IK evaluation has invalid candidate indices or scores")
        # Preserve planner batching. Rank its successful returned paths by travel.
        candidates = sorted(candidates, key=lambda candidate: -candidate["score"])
        indices = [candidate["candidate_index"] for candidate in candidates]
        effective_settings, lift_clearance = self._grasp_lift_settings(
            snapshot, target_id, arm, candidates, scene, settings) if candidates else (settings, None)
        lift_metadata = ({} if lift_clearance is None else
                         {"effective_grasp_settings": effective_settings, "lift_clearance": lift_clearance})
        planner_fields = ("freq_hz", "approach_axis", "approach_offset_m", "approach_in_tool_frame",
                          "lift_axis", "lift_offset_m", "lift_in_tool_frame",
                          "trajectory_collision_clearance_m", "optimizer_collision_activation_distance_m")
        request_base = {"arm": arm, "start_joints_deg": evaluation["start_joints_deg"],
                        "other_joints_deg": evaluation["other_joints_deg"], "scene_token": scene["scene_token"],
                        "gripper_joint_positions_rad": {a: snapshot['arms'][a]['gripper_position_rad'] for a in ('A', 'B')},
                        "planning_settings": effective_settings,
                        **{key: effective_settings[key] for key in planner_fields}}
        planned_count = 0
        measured_start = np.asarray(snapshot["arms"][arm]["joints_rad"], dtype=float)
        chunks = trajectory_batches(candidates, self.grasp_trajectory_batch_capacity)
        for batch_index, chunk in enumerate(chunks, 1):
            check_cancelled()
            request = {**request_base, "goal_matrices_world":
                       [rigid_matrix(candidate["T_world_tcp"]).tolist() for candidate in chunk]}
            batch_data = {**sampling, "arm": arm, "planning_stage": "grasp", "batch_index": batch_index,
                          "batch_count": len(chunks),
                          "candidate_indices": [candidate["candidate_index"] for candidate in chunk],
                          "duration_s": 0., "success_count": 0}
            if progress is not None:
                progress("planning_batch_start", "cuRobo plan_grasp_batch 开始", batch_data)
            check_cancelled()
            started = time.monotonic()
            try:
                response = self.request("/tools/plan_grasp_batch", request)
            except AlgorithmError:
                check_cancelled()
                raise
            check_cancelled()
            if progress is not None:
                progress("planning_batch_replay", "保存plan_grasp_batch完整候选与轨迹", {
                    **batch_data, "candidate_replay": {**replay_inputs, **sampling,
                        "stage": "after_plan_grasp_batch", "target_id": target_id,
                        **lift_metadata,
                        "candidate_indices": batch_data["candidate_indices"],
                        "planning_candidates": deepcopy(chunk),
                        "request": deepcopy(request), "response": deepcopy(response),
                        "measured_points_world_m": np.asarray(
                            self._target(snapshot, target_id)["points_world_m"]).copy()}})
            items = self._batch_items(response, len(chunk), scene["scene_token"], "plan_grasp_batch")
            batch_failures, feasible = [], []
            for candidate, item in zip(chunk, items):
                check_cancelled()
                phases = self._plan_phases(item, arm)
                if phases is None:
                    batch_failures.append({"candidate_index": candidate["candidate_index"],
                                           "stage": "planner",
                                           "score": candidate["score"], "status": item.get("status"),
                                           "error": item.get("error"),
                                           "phase_success": {phase: item[f"{phase}_success"]
                                                             for phase in ("goalset", "approach", "grasp", "lift")}})
                    continue
                try:
                    goal_evidence = self._grasp_plan_evidence(
                        candidate, item, phases, snapshot, target_id, arm, scene, settings)
                except AlgorithmError as error:
                    if error.code != "planning_grasp_endpoint_rejected":
                        raise
                    batch_failures.append({"candidate_index": candidate["candidate_index"],
                        "score": candidate["score"], "code": error.code,
                        "stage": "grasp_endpoint_validation", "details": error.details})
                    continue
                # Score the returned samples, including the measured start and
                # phase connections. Interpolate only the selected candidate.
                positions = np.concatenate([measured_start[None, :], *[
                    np.asarray(phases[name]["positions_rad"], dtype=float)
                    for name in ("approach", "grasp", "lift")]])
                travel = float(np.abs(np.diff(positions, axis=0)).sum())
                feasible.append((travel, candidate, phases, goal_evidence))
            best = None
            for travel, candidate, phases, goal_evidence in sorted(feasible, key=lambda item: item[0]):
                check_cancelled()
                try:
                    phases = build_execution_phases(phases, settings["time_scale"])
                except (TypeError, ValueError) as error:
                    raise AlgorithmError(f"cannot prepare Hermite execution trajectory: {error}") from error
                checking = deepcopy(snapshot)
                checks = {}
                try:
                    for name in ("approach", "grasp", "lift"):
                        check_cancelled()
                        current = np.asarray(checking["arms"][arm]["joints_rad"], dtype=float)
                        command = phases[name]
                        points = np.asarray(command["positions_rad"], dtype=float)
                        drift = float(np.max(np.abs(current - points[0])))
                        if drift > np.deg2rad(1.):
                            raise AlgorithmError("grasp phase start differs by more than 1 degree",
                                code="planning_phase_start_changed",
                                details={"phase": name, "maximum_drift_deg": float(np.rad2deg(drift))})
                        if name != "lift":
                            if drift > 1e-8:
                                command = {**command, "positions_rad": np.vstack((current, points)).tolist()}
                            checks[name] = self.validate_execution(
                                checking, arm, scene, "approach_grasp" if name == "approach" else "grasp",
                                command, planning_settings=settings,
                                contact_goal_pose_world=candidate["T_world_tcp"])
                        checking["arms"][arm]["joints_rad"] = points[-1].tolist()
                        if name == 'grasp':
                            close_command = {'positions_rad': [points[-1].tolist()], 'dt': command['dt']}
                            try:
                                checks['closure'] = self.validate_execution(checking, arm, scene, 'grasp_closure',
                                    close_command, planning_settings=settings)
                            except AlgorithmError as error:
                                prefix = (error.details or {}).get('grasp_closure_preflight') or {}
                                limit = prefix.get('safe_driver_limit_rad')
                                samples = prefix.get('driver_samples_rad', [])
                                if error.code != 'execution_collision' or limit is None or len(samples) < 2:
                                    raise
                                # Keep two feedback tolerances before the last proven
                                # position. Revalidate the exact bounded command.
                                reserve = 2 * settings['holding_position_tolerance_fraction'] * (samples[-1] - samples[0])
                                limit -= reserve
                                if limit <= samples[0] + reserve:
                                    raise
                                checks['closure'] = self.validate_execution(checking, arm, scene, 'grasp_closure',
                                    close_command, planning_settings=settings, closure_driver_limit_rad=limit)
                                checks['closure']['full_range_rejection'] = {
                                    'first_world_collision': error.details.get('first_world_collision'),
                                    'safe_driver_limit_rad': prefix['safe_driver_limit_rad'],
                                    'command_reserve_rad': reserve}
                                from .grasp_contact_prediction import bounded_contact_prediction

                                contact = bounded_contact_prediction(
                                    self._target(snapshot, target_id)['points_world_m'], candidate['T_world_tcp'],
                                    arm, limit, settings['holding_position_tolerance_fraction'])
                                checks['closure']['bounded_contact_prediction'] = contact
                                if not contact['opposing_contact_predicted']:
                                    raise AlgorithmError('bounded closure cannot reach observed opposing pad contacts',
                                        code='planning_bounded_contact_unreachable', details=contact)
                except AlgorithmError as error:
                    check_cancelled()
                    if error.code not in {"planning_phase_start_changed", "execution_collision",
                                          "planning_bounded_contact_unreachable",
                                          "execution_observation_coverage_insufficient"}:
                        raise
                    batch_failures.append({"candidate_index": candidate["candidate_index"],
                        "score": candidate["score"], "joint_travel_rad": travel,
                        "stage": "command_validation", "code": error.code,
                        "error": str(error), "details": error.details})
                    continue
                check_cancelled()
                best = {"candidate_index": candidate["candidate_index"], "score": candidate["score"],
                        "T_world_tcp": candidate["T_world_tcp"], "phases": phases,
                        "joint_travel_rad": travel, "command_checks": checks, **goal_evidence}
                if destination_id is not None:
                    best["place_preflight"] = {"destination_id": destination_id,
                        "scope": "deferred_until_attachment", "formal_place_requires_attachment": True}
                break
            failures.extend(batch_failures)
            planned_count += len(chunk)
            remaining_count = len(candidates) - planned_count
            completed = {**batch_data, "duration_s": time.monotonic() - started,
                         "success_count": len(feasible), "failures": batch_failures,
                         "stopped_early": best is not None and remaining_count > 0,
                         "unplanned_candidate_count": remaining_count,
                         "effective_config": response.get("effective_config")}
            batches.append(completed)
            if progress is not None:
                progress("planning_batch_complete", "cuRobo plan_grasp_batch 完成", completed)
            check_cancelled()
            if best is not None:
                return {**best, **lift_metadata, **sampling, "arm": arm, "scene_token": scene["scene_token"], "target_id": target_id,
                        "excluded_object_ids": [target_id], "prior_failures": failures, "batches": batches,
                        "planned_candidate_count": planned_count,
                        "unplanned_candidate_indices": indices[planned_count:],
                        "selection": {"rule": "minimum_total_joint_travel",
                            "metric": "sum_absolute_planner_joint_increments_rad",
                            "scope": "first_feasible_batch",
                            "all_ik_candidates_evaluated": remaining_count == 0,
                            "raw_complete_candidate_count": len(feasible),
                            "command_validated_candidate_count": 1,
                            "lift_command_collision_checked": False,
                            "tie_break": "higher_graspgen_score_then_original_order",
                            "start_joints_rad": measured_start.tolist(),
                            "joint_names": [f"Joint{i}_{'L' if arm == 'A' else 'R'}" for i in range(1, 8)]},
                        "planning_settings": settings, "time_scale": settings["time_scale"],
                        "raw_duration_s": sum(phase["raw_duration_s"] for phase in best["phases"].values()),
                        "command_duration_s": sum(phase["command_duration_s"] for phase in best["phases"].values()),
                        "time_stretch_method": "quintic_hermite_point_insertion",
                        "start_joints_deg": evaluation["start_joints_deg"],
                        "other_joints_deg": evaluation["other_joints_deg"], "joint_drift_deg": joint_drift,
                        "joint_drift_threshold_deg": 1.}
        exhaustion = {**expected, "sampling_round": evaluation.get("sampling_round"),
            "start_joints_deg": deepcopy(evaluation["start_joints_deg"]),
            "other_joints_deg": deepcopy(evaluation["other_joints_deg"]),
            "candidate_indices": indices, "planned_candidate_count": planned_count,
            "unplanned_candidate_indices": indices[planned_count:],
            "outcome": "determinate_planner_failure" if candidates and len(failures) == len(candidates)
                and all(determinate_grasp_failure(row) for row in failures) else "not_retryable"}
        raise AlgorithmError(f"no valid three-phase grasp plan for arm {arm}: {failures}",
            code="planning_no_valid_candidate", details={"arm": arm, "failures": failures,
                "batches": batches, "pool_exhaustion": exhaustion})
