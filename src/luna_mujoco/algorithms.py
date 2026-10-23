"""Synchronous boundaries to the independent GraspGen and cuRobo services."""

import hashlib
import json
import os
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import httpx
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import TOOL_PROFILE, grasp_targets, rigid_matrix
from .placement import observed_object, release_candidate_groups
from .planning_batches import TRAJECTORY_BATCH_CAPACITY, pad_trajectory_batch, trajectory_batches
from .planning_settings import PlanningSettings
from .trajectory_postprocessing import build_execution_phases

VENDOR = Path(__file__).resolve().parents[2] / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


class AlgorithmError(RuntimeError):
    def __init__(self, message, *, code="algorithm_error", details=None):
        super().__init__(message)
        self.code = code
        self.details = details


def determinate_grasp_failure(failure):
    """Only the pinned SDK's completed result authorizes another candidate pool."""
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
    def __init__(self, *, curobo_request=None, graspgen_client=None, perception=None, workspace_lookup=None):
        from tianji_graspgen.client import GraspGenClient

        from .perception import Perception
        from .workspace import WorkspaceAtlas
        self.curobo_url = os.getenv("LUNA_CUROBO_URL", "http://127.0.0.1:8310")
        self.graspgen = graspgen_client or GraspGenClient(
            os.getenv("LUNA_GRASPGEN_URL", "http://127.0.0.1:8323"))
        self.request = curobo_request or self._request
        self.perception = perception if perception is not None else Perception()
        self.workspace_lookup = workspace_lookup or WorkspaceAtlas().query
        self._client = None
        self._client_lock = threading.Lock()

    def close(self):
        """Release the transport after the application drains algorithm workers."""
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _request(self, path, body=None):
        try:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=600)
                client = self._client
            response = (client.get(self.curobo_url + path) if body is None
                        else client.post(self.curobo_url + path, json=body))
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
        from .semantic_transport import encode_project_semantic_request

        if snapshot.get("source") != "rgbd":
            raise AlgorithmError("scene requires registered RGBD observation")
        # Observation motion includes every measured instance. Only the initial
        # scene is head-only; subsequent full-obstacle updates use all cameras.
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
        if exclusion_state in {'observation', 'selected'}:
            unresolved = snapshot.get('mapping_unresolved', [])
            if unresolved:
                raise AlgorithmError('semantic_scene_identity_unresolved', code='semantic_scene_identity_unresolved',
                    details={'unresolved': unresolved})
            task_ids = set(snapshot.get('task_member_ids', [])) | {target_id}
            for query in snapshot.get('declared_environment', []):
                if not any(obj['id'] not in task_ids and obj.get('kind') not in {'bin', 'marker'}
                           and obj.get('color') == query['color'] for obj in snapshot['objects']):
                    raise AlgorithmError('declared_obstacle_not_observed', code='declared_obstacle_not_observed',
                        details={'query': query, 'reason': 'no_distinct_non_task_object'})
        views = {}
        camera_capture_metadata = {}
        markers = {obj['id'] for obj in snapshot['objects'] if obj.get('kind') == 'marker'}
        for camera_id in (("3",) if head_only else ("1", "2", "3")):
            captured_view = snapshot["views"][camera_id]
            view = letterbox_view(captured_view)
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
        semantic = {"observation_set_id": snapshot["observation_id"],
                    "sam3_generation": snapshot["generation"], "confirmed_instance_id": target_id or '',
                    "selection_generation": snapshot["generation"], "views": views}
        if held_surface is not None or demo_background:
            semantic.update(mapping_mode="background", confirmed_instance_id="", sam3_generation=0)
            for view in views.values():
                view['instance_masks'] = {}
        payload = encode_project_semantic_request(semantic, exclusion_state=exclusion_state)
        if snapshot.get("demo_mapping") is True or demo_background:
            payload["demo_mapping"] = True
            payload["demo_tables"] = snapshot.get("demo_tables")
            task = snapshot.get('demo_task', {})
            bin_ids = {task.get('source_bin_id'), task.get('right_bin_id')}
            from .cad_assets import cad_asset
            bin_asset = cad_asset('blue_box')
            payload['demo_bin_priors'] = {obj['id']: {
                'kind': 'bin', 'color': 'blue', 'instance_id': obj['id'],
                'observation_id': snapshot['observation_id'], 'generation': snapshot['generation'],
                'outer_dimensions_m': bin_asset['size_m'], 'source': 'user_supplied_cad_shape',
                'asset_id': 'blue_box', 'cad_sha256': bin_asset['source']['sha256']}
                for obj in snapshot['objects'] if obj['id'] in bin_ids
                and obj.get('kind') == 'bin' and obj.get('color') == 'blue'}
            payload["demo_completions"] = {}
            for obj in snapshot['objects']:
                if obj.get('geometry_completion', {}).get('source') != 'measured_surface_and_user_height_prior':
                    continue
                completion = deepcopy(obj['geometry_completion'])
                # Bind the prior to the current inventory, not a copied old completion.
                completion.pop('instance_context', None)
                lifecycle_keys = ('current_appearance', 'appearance_revision', 'lifecycle_phase')
                if any(key in obj for key in lifecycle_keys):
                    completion['instance_context'] = {
                        'instance_id': obj['id'], 'observation_id': snapshot['observation_id'],
                        'color': obj.get('color'), **{key: obj.get(key) for key in lifecycle_keys}}
                payload['demo_completions'][obj['id']] = completion
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
        """Object geometry for inference; the cloud origin stays measurement-based."""
        return np.asarray(target["points_world_m"])

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
        return {"candidates": candidates, "all_generated_candidates": all_generated,
                "model": model, "frame": f"cloud:{target_id}",
                'arm': arm, 'tool_profile_revision': TOOL_PROFILE['revision'],
                "T_world_cloud": cloud_frame.tolist(),
                "graspgen_input": request,
                "requested_candidate_count": candidate_count, "candidate_count": len(candidates), "batches": batches,
                "generation": snapshot["generation"], "observation_id": snapshot["observation_id"],
                "cloud": reference}

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
                  "completion_provenance", "graspgen_input_cloud",
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
        world = cloud_frame @ np.asarray([c["T_object_grasp"] for c in candidates])
        result = {}
        for arm in ((requested_arm,) if requested_arm else ('A', 'B')):
            peer = "B" if arm == "A" else "A"
            poses = (np.asarray([rigid_matrix(candidate['T_world_tcp']) for candidate in candidates])
                     if bound_arm is not None else grasp_targets(world, arm))
            request = {"arm": arm, "start_joints_deg": np.rad2deg(snapshot["arms"][arm]["joints_rad"]).tolist(),
                       "other_joints_deg": np.rad2deg(snapshot["arms"][peer]["joints_rad"]).tolist(),
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
                                      for key in ("source_batch", "source_index") if key in candidates[index]},
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
            result[arm] = {**sampling, **({"graspgen_replay_inputs": replay_inputs} if replay_inputs else {}),
                           "valid_candidates": valid, "tested_count": len(candidates),
                           "arm": arm, "target_id": target_id,
                           "start_joints_deg": request["start_joints_deg"],
                           "other_joints_deg": request["other_joints_deg"],
                           "planning_settings": deepcopy(settings), "batches": batches,
                           "orientation_deg": TOOL_PROFILE[arm]["orientation_deg"],
                           "scene_token": scene["scene_token"], "generation": snapshot["generation"],
                           "tool_profile_revision": TOOL_PROFILE["revision"]}
            if destination_id is not None:
                observed_object(snapshot, destination_id)
                # Actual closure and carried geometry exist only after grasp.
                # An open-hand destination failure cannot reject grasp IK.
                result[arm].update(destination_id=destination_id, feasibility_scope="grasp_ik_only",
                                   place_planning_stage="after_grasp_attachment")
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
                           contact_goal_pose_world=None):
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
            "fixed_open" if binding is None and phase in {"approach_grasp", "grasp", "close", "open_grasp"}
            else "actual")
        if binding is not None:
            payload["carrying_binding"] = binding
        if contact_goal_pose_world is not None:
            payload["contact_goal_pose_world"] = rigid_matrix(contact_goal_pose_world).tolist()
        if phase in ("close", "open", "open_grasp"):
            payload["gripper_start_position_rad"] = observation["arms"][arm]["gripper_position_rad"]
            payload["gripper_end_position_rad"] = gripper_end_position_rad
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()
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
        binding = {"arm": arm, "owner_id": holding["owner_id"], "instance_id": target_id,
                   "observation_id": scene["observation_id"], "generation": scene["generation"],
                   "scene_token": scene["scene_token"], "frame": "tool", "state": "held",
                   "points_tool_m": deepcopy(holding["points_tool_m"]),
                   "uncertainty_m": holding.get("uncertainty_m", .005),
                   "measurement_provenance": deepcopy(holding.get("measurement_provenance"))}
        # Geometry belongs to the existing held session, including across standalone place tasks.
        for name in ("minimum_partition_size_m", "voxel_size_m", "margin_m", "sphere_budget"):
            if name in holding:
                binding[name] = holding[name]
        return binding

    def plan_place(self, observation, target_id, arm, scene, holding, destination_id, *,
                   planning_settings=None, placement_request=None, progress=None, cancelled=None):
        self._check_cancelled(cancelled)
        self._binding(observation, target_id, scene)
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        destination = observed_object(observation, destination_id)
        if not holding.get("verified") or not holding.get("attached"):
            raise AlgorithmError("place requires an already attached held object", code="place_requires_attachment")
        prepared = holding.get("attachment_evidence", {})
        if "attachment_spheres_tool_m" not in prepared:
            prepared = self.prepare_carrying(observation, arm, scene, holding, planning_settings=settings)
            self._check_cancelled(cancelled)
        groups = release_candidate_groups({}, destination, holding,
            spheres_tool_m=prepared.get("attachment_spheres_tool_m"),
            clearance_m=settings["trajectory_collision_clearance_m"],
            map_resolution_m=scene.get("voxel_size_m", .02),
            observation={**observation, 'semantic_only_active': scene.get('semantic_only_active', False)},
            placement_request=placement_request, progress=progress)
        binding = self._held_binding(observation, target_id, arm, scene, holding)
        failures = []
        goals = []
        batches = self._place_candidate_batches(groups, goals, failures, observation, arm, scene,
                                                binding, settings, progress=progress, cancelled=cancelled)
        for batch_index, batch_count, selected_goals in batches:
            payload = self._motion_request(observation, arm, scene, selected_goals[0]["T_world_tcp"], settings)
            payload.pop("goal_position_m")
            payload.pop("goal_quaternion_wxyz")
            payload.update(carrying_binding=binding, approach_axis=settings["place_approach_axis"],
                           approach_offset_m=settings["place_approach_offset_m"],
                           approach_in_tool_frame=settings["place_approach_in_tool_frame"],
                           lift_axis=settings["place_retreat_axis"], lift_offset_m=settings["place_retreat_offset_m"],
                           lift_in_tool_frame=settings["place_retreat_in_tool_frame"],
                           trajectory_collision_clearance_m=settings["trajectory_collision_clearance_m"],
                           optimizer_collision_activation_distance_m=settings["optimizer_collision_activation_distance_m"])
            padded = pad_trajectory_batch(selected_goals)
            payload['goal_matrices_world'] = [goal['T_world_tcp'] for goal in padded]
            batch = {"arm": arm, "planning_stage": "place", "batch_index": batch_index, "batch_count": batch_count,
                     "batch_scope": "generated_positions",
                     "candidate_indices": [goal["candidate_index"] for goal in selected_goals], "frame": "world",
                     "release_reference": goals[0]["release_reference"]}
            if progress is not None:
                progress("planning_batch_start", "cuRobo 放置偏航 batch 开始", batch)
            started = time.monotonic()
            response = self.request("/tools/plan_place_batch", payload)
            self._check_cancelled(cancelled)
            if response.get("scene_token") != scene["scene_token"]:
                raise AlgorithmError("place plan returned a stale scene token")
            if response.get("collision_evidence", {}).get("optimizer_attachment") is not True:
                raise AlgorithmError("place plan did not optimize the carried object", code="attachment_not_applied")
            items = response.get("items")
            if (not isinstance(items, list) or len(items) != TRAJECTORY_BATCH_CAPACITY
                    or any(not isinstance(item, dict) or item.get("index") != index or type(item.get("success")) is not bool
                           for index, item in enumerate(items))):
                raise AlgorithmError("invalid place batch candidate correspondence", code="algorithm_invalid_response")
            if progress is not None:
                progress("planning_batch_complete", "cuRobo 放置偏航 batch 完成", {
                    **batch, "duration_s": time.monotonic() - started,
                    "service_timings_s": response.get("service_timings_s"),
                    "pose_call_timings": response.get("pose_call_timings"),
                    "tested_count": len(selected_goals),
                    "success_count": sum(item["success"] for item in items[:len(selected_goals)])})
            feasible = []
            for goal, item in zip(selected_goals, items):
                if not item["success"]:
                    failures.append({"candidate_index": goal["candidate_index"],
                        "yaw_delta_world_rad": goal["yaw_delta_world_rad"],
                        "status": item.get("status"), "error": item.get("error"),
                        "release_clearance": item.get("release_clearance")})
                    continue
                if item.get("approach_success") is not True or item.get("place_success") is not True:
                    raise AlgorithmError("incomplete successful place candidate", code="algorithm_invalid_response")
                raw = self._native_phases({name: item["phases"][name] for name in ("approach", "place")}, arm)
                travel = sum(float(np.linalg.norm(np.diff(phase["positions_rad"], axis=0), axis=1).sum())
                             for phase in raw.values())
                yaw_change = abs((goal["yaw_delta_world_rad"] + np.pi) % (2 * np.pi) - np.pi)
                feasible.append((yaw_change, travel, goal["candidate_index"], raw))
            for yaw_change, travel, index, raw in sorted(feasible, key=lambda item: item[:3]):
                self._check_cancelled(cancelled)
                try:
                    phases, checks = self._complete_place_candidate(
                        observation, arm, scene, binding, raw, settings, cancelled=cancelled)
                except AlgorithmError as error:
                    if error.code not in {"execution_collision", "CUROBO_TRAJECTORY_COLLISION",
                                          "CUROBO_PLANNING_FAILED", "CUROBO_NO_VALID_TRAJECTORY"}:
                        raise
                    failures.append({"candidate_index": index, "yaw_delta_world_rad": goals[index]["yaw_delta_world_rad"],
                                     "code": error.code, "error": str(error), "details": error.details})
                    continue
                selection = {'candidate_index': index,
                    'position_index': goals[index]['release_reference'].get('region_position_index', 0),
                    'yaw_delta_deg': float(np.rad2deg(goals[index]['yaw_delta_world_rad'])),
                    'absolute_yaw_change_deg': float(np.rad2deg(yaw_change)), 'joint_travel_rad': travel,
                    'successful_candidates_in_batch': len(feasible),
                    'rule': 'first_complete_plan_in_first_feasible_batch_ranked_by_yaw_then_joint_travel_then_index',
                    'retreat_checked': True, 'later_batches_evaluated': False}
                if progress is not None:
                    progress('placement_candidate_selected', '选定完整放置及撤离方案，停止后续候选', selection)
                return {"arm": arm, "target_id": target_id, "destination_id": destination_id,
                    "observation_id": observation["observation_id"], "generation": observation["generation"],
                    "scene_token": scene["scene_token"], "phases": phases, "goal": goals[index],
                    "carrying_binding": binding, "collision_evidence": response["collision_evidence"],
                    "holding_reference": deepcopy(holding["points_tool_m"]),
                    "command_checks": checks, "retreat_requires_replan": False,
                    "planning_settings": settings, "time_scale": settings["time_scale"],
                    "candidate_count": len(goals), "candidate_failures": failures,
                    "selection_evidence": selection, "placement_request": deepcopy(placement_request)}
        raise AlgorithmError("no world-yaw candidate has a complete collision-free place and retreat",
            code="place_no_valid_candidate", details={"candidate_failures": failures,
                "planning_summary": {"generated_goal_count": len(goals),
                    "ik_rejected_count": sum(item.get('code') == 'place_ik_unreachable' for item in failures),
                    "trajectory_rejected_count": sum(item.get('code') != 'place_ik_unreachable' for item in failures)},
                "release_reference": goals[0]["release_reference"]})

    def _place_candidate_batches(self, groups, generated, failures, observation, arm, scene,
                                 binding, settings, *, progress=None, cancelled=None):
        batch_index = 0
        for group in groups:
            self._check_cancelled(cancelled)
            generated.extend(group)
            payload = self._motion_request(observation, arm, scene, group[0]["T_world_tcp"], settings)
            payload.pop("goal_position_m")
            payload.pop("goal_quaternion_wxyz")
            payload.update(carrying_binding=binding,
                           trajectory_collision_clearance_m=settings["trajectory_collision_clearance_m"],
                           goal_matrices_world=[goal["T_world_tcp"] for goal in group])
            batch = {"arm": arm, "planning_stage": "place", "batch_index": batch_index + 1,
                     "batch_scope": "one_release_position", "candidate_indices":
                     [goal["candidate_index"] for goal in group]}
            if progress is not None:
                progress("ik_batch_start", "cuRobo 放置 batched IK 开始", batch)
            started = time.monotonic()
            response = self.request("/tools/solve_ik_batch", payload)
            self._check_cancelled(cancelled)
            items = self._batch_items(response, len(group), scene["scene_token"], "place IK")
            if (response.get("attached") is not True
                    or response.get("collision_evidence", {}).get("optimizer_attachment") is not True):
                raise AlgorithmError("place IK did not include the carried object", code="attachment_not_applied")
            survivors = []
            for goal, item in zip(group, items):
                if item["success"]:
                    survivors.append(goal)
                else:
                    failures.append({"candidate_index": goal["candidate_index"],
                                     "yaw_delta_world_rad": goal["yaw_delta_world_rad"],
                                     "code": "place_ik_unreachable", "status": "collision_aware_ik_rejected",
                                     "failure_scope": "collision_aware_ik_not_proof_of_kinematic_unreachability",
                                     "position_error_m": item.get("position_error_m"),
                                     "rotation_error_rad": item.get("rotation_error_rad")})
            if progress is not None:
                progress("ik_batch_complete", "cuRobo 放置 batched IK 完成", {
                    **batch, "duration_s": time.monotonic() - started,
                    "success_count": len(survivors), "tested_count": len(group),
                    "items": items,
                    "effective_config": response.get("effective_config")})
            chunks = trajectory_batches(survivors)
            batch_count = batch_index + len(chunks)
            for chunk in chunks:
                batch_index += 1
                yield batch_index, batch_count, chunk

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
        retreat = self.plan_retreat(checking, arm, scene, retreat_goal.tolist(), planning_settings=settings)
        self._check_cancelled(cancelled)
        phases["retreat"], checks["retreat"] = retreat["command"], retreat["check"]
        return phases, checks

    def plan_retreat(self, observation, arm, scene, goal, *, planning_settings=None):
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        payload = self._motion_request(observation, arm, scene, goal, settings)
        payload.update(departure_axis=settings['place_retreat_axis'],
                       departure_in_tool_frame=settings['place_retreat_in_tool_frame'])
        response = self.request("/tools/plan_to_pose", payload)
        if response.get("scene_token") != scene["scene_token"] or response.get("attached") is not False:
            raise AlgorithmError("retreat requires a detached current scene")
        raw = self._native_phases({"retreat": response["trajectory"]}, arm)
        command = build_execution_phases(raw, settings["time_scale"])["retreat"]
        check = self.validate_execution(observation, arm, scene, "retreat", command, planning_settings=settings)
        return {"command": command, "check": check}

    def clear_carrying(self):
        return self.request("/carrying/clear", {})

    def prepare_carrying(self, observation, arm, scene, holding, *, planning_settings=None):
        self._binding(observation, holding["instance_id"], scene)
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        binding = self._held_binding(observation, holding["instance_id"], arm, scene, holding)
        payload = self._motion_request(observation, arm, scene, observation["arms"][arm]["tcp"], settings)
        if observation.get('stage') in ('wrapped', 'unpacked'):
            payload['next_motion_kind'] = 'pose' if observation['stage'] == 'unpacked' else 'place'
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
                    "tool_profile_revision": TOOL_PROFILE["revision"]}
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
                except AlgorithmError as error:
                    check_cancelled()
                    if error.code not in {"planning_phase_start_changed", "execution_collision",
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
