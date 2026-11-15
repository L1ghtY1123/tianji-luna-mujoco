"""Demo policies composed over the existing service and execution boundaries."""

import hashlib
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .algorithms import AlgorithmError, Algorithms
from .demo_grasp import (
    arm_wrist_reach,
    prepare_grasp_candidates,
)
from .demo_gripper_quality import predict_gripper_compatibility
from .demo_placement import (
    fixed_release_goal,
)
from .demo_sampling import grasp_candidate_count
from .geometry import TCP_FRAMES, front_tcp_offset, rigid_matrix
from .kinematics import robot_model
from .placement import measured_object_geometry, observed_object
from .planning_settings import PlanningSettings
from .robot_skills.placement_selection import (
    check_goal_workspace,
)
from .scene_update import SCENE_INVALIDATION_SOURCES


@lru_cache(maxsize=8)
def registered_grasp_geometry(asset_id, mesh_key, source_sha256, mesh_sha256):
    from .cad_perception import CadCatalog

    catalog = CadCatalog.from_file()
    geometry = catalog.geometry(mesh_key)
    if any(geometry[key] != value for key, value in (
            ('asset_id', catalog.record(asset_id)['asset_id']), ('source_sha256', source_sha256), ('mesh_sha256', mesh_sha256))):
        raise ValueError('cad_grasp_geometry_binding_changed')
    # Preserve authored surface recesses for contact screening; a convex hull
    # would incorrectly fill openings between the gripping pads.
    return np.unique(geometry['vertices_m'], axis=0)


def verified_unpack_product(snapshot, target_id):
    """Use verified capability lineage, never a scene-wide task stage, for this geometry policy."""
    entity = snapshot.get('object_memory_snapshot', {}).get('entities', {}).get(target_id, {})
    evidence = entity.get('creation_evidence', {})
    source_id = entity.get('input_entity_id')
    return (entity.get('id') == target_id and entity.get('processing_skill') == 'vla_unpack'
        and entity.get('identity_source') == 'confirmed_current_product_creation'
        and bool(source_id) and source_id != target_id and bool(entity.get('process_id'))
        and evidence.get('source') == 'confirmed_unpack_process_output'
        and evidence.get('processing_skill') == 'vla_unpack'
        and evidence.get('output_entity_id') == target_id
        and evidence.get('input_entity_id') == source_id
        and evidence.get('process_id') == entity['process_id']
        and type(evidence.get('generation')) is int and evidence['generation'] == snapshot.get('generation')
        and all(evidence.get(key) for key in ('action_id', 'request_hash', 'observation_id')))


class DemoAlgorithms(Algorithms):
    def evaluate_reachability(self, snapshot, target_id, destination_id=None, *, requested_arm=None,
                              placement_request=None, planning_settings=None, scene=None, release_mode=None):
        from .transfer_reachability import evaluate_transfer_workspace

        if destination_id is None:
            raise ValueError('single_arm_transfer_requires_destination')
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        return evaluate_transfer_workspace(self.workspace_lookup, snapshot, target_id, destination_id,
            requested_arm=requested_arm, settings=settings, scene=scene, release_mode=release_mode)

    def verify_release(self, capture, binding, reference, instruction=""):
        return self.perception.verify_release(capture, binding, reference, instruction)

    @staticmethod
    def _bound_tool_fk(evidence, joints_rad, joint_names, arm, scene):
        side = "L" if arm == "A" else "R"
        names = [f"Joint{i}_{side}" for i in range(1, 8)]
        try:
            expected = {"source": "curobo_public_compute_kinematics", "frame": "world",
                        "arm": arm, "tool_frame": TCP_FRAMES[arm],
                        "scene_token": scene["scene_token"], "joint_names": names}
            if not isinstance(evidence, dict) or any(evidence.get(k) != v for k, v in expected.items()):
                raise ValueError("FK binding differs from the current arm, frame or scene")
            q = np.asarray(joints_rad, dtype=float)
            fk_q = np.asarray(evidence["joints_rad"], dtype=float)
            if (list(joint_names) != names or q.shape != (7,) or fk_q.shape != (7,)
                    or not np.isfinite(q).all() or not np.array_equal(q, fk_q)):
                raise ValueError("FK configuration differs from the accepted joints")
            return rigid_matrix(evidence["T_world_tcp"])
        except (KeyError, TypeError, ValueError) as error:
            raise AlgorithmError(f"invalid achieved tool FK: {error}",
                                 code="planning_fk_evidence_invalid") from error

    @staticmethod
    def _grasp_pose_validation(nominal, achieved, snapshot, target_id, arm, settings):
        from .demo_grasp import grasp_direction_evidence

        target = np.asarray(observed_object(snapshot, target_id)["points_world_m"], dtype=float)
        away = (target.mean(axis=0) - np.asarray(arm_wrist_reach(arm)["robot_base_world_m"]))[:2]
        length = np.linalg.norm(away)
        if not np.isfinite(away).all() or (length <= 1e-9 and settings['grasp_require_away_from_robot']):
            raise AlgorithmError("FK direction has no measured robot-to-target reference",
                                 code="planning_fk_evidence_invalid")
        away = away / length if length > 1e-9 else np.zeros(2)
        position = float(np.linalg.norm(achieved[:3, 3] - nominal[:3, 3]))
        rotation = float(np.arccos(np.clip((np.trace(nominal[:3, :3].T @ achieved[:3, :3]) - 1) / 2, -1, 1)))
        downward = float(-achieved[2, 2])
        away_dot = float(achieved[:2, 2] @ away)
        reasons = []
        if position > settings["ik_position_threshold_m"] + 1e-9:
            reasons.append("nominal_position_tolerance_exceeded")
        if rotation > settings["ik_rotation_threshold_rad"] + 1e-9:
            reasons.append("nominal_rotation_tolerance_exceeded")
        if downward < np.cos(np.deg2rad(settings['grasp_maximum_tilt_deg'])) - 1e-9:
            reasons.append("achieved_tool_tilt_exceeded")
        if settings['grasp_require_away_from_robot'] and away_dot < -1e-9:
            reasons.append("achieved_tool_points_toward_robot")
        sector = grasp_direction_evidence(achieved[:3, 2], settings['grasp_maximum_lateral_angle_deg'],
            applied=observed_object(snapshot, target_id).get('cad') is None)
        if sector['applied'] and not sector['forward_downward_passed']:
            reasons.append('achieved_tool_outside_forward_down_sector')
        if sector['applied'] and not sector['lateral_passed']:
            reasons.append('achieved_tool_lateral_angle_exceeded')
        return {"accepted": not reasons, "reason": "accepted" if not reasons else ";".join(reasons),
                'sector': sector,
                "position_error_m": position, "rotation_error_rad": rotation,
                "position_threshold_m": settings["ik_position_threshold_m"],
                "rotation_threshold_rad": settings["ik_rotation_threshold_rad"],
                "downward_dot": downward, "horizontal_away_from_robot_dot": away_dot,
                "maximum_tilt_deg": settings['grasp_maximum_tilt_deg'],
                "require_away_from_robot": settings['grasp_require_away_from_robot'],
                "reference": "original_graspgen_tool_pose_with_arm_jaw_symmetry"}


    def _gripper_quality(self, snapshot, target_id, achieved, arm, settings):
        target = observed_object(snapshot, target_id)
        geometry = self._completed_object_geometry(snapshot, target)
        try:
            evidence = predict_gripper_compatibility(geometry, achieved, arm=arm,
                uncertainty_m=target["uncertainty_m"], margin_m=settings["attachment"]["margin_m"])
        except (ValueError, KeyError, TypeError, OSError, ET.ParseError, yaml.YAMLError) as error:
            raise AlgorithmError(f"invalid predicted gripper geometry: {error}",
                                 code="grasp_gripper_prediction_invalid") from error
        return {**evidence, "observation_id": snapshot["observation_id"],
                "target_id": target_id, "measured_point_count": len(target["points_world_m"]),
                "completion_provenance": deepcopy(target.get("geometry_completion"))}


    def _grasp_plan_evidence(self, candidate, item, phases, snapshot, target_id, arm, scene, settings):
        endpoint = item.get("grasp_endpoint_fk")
        side = "L" if arm == "A" else "R"
        achieved = self._bound_tool_fk(endpoint, phases["grasp"]["positions_rad"][-1],
            [f"Joint{i}_{side}" for i in range(1, 8)], arm, scene)
        validation = self._grasp_pose_validation(rigid_matrix(candidate["nominal_T_world_tcp"]),
            achieved, snapshot, target_id, arm, settings)
        if not validation["accepted"]:
            raise AlgorithmError("planned grasp endpoint violates the original nominal goal bounds",
                                 code="planning_grasp_endpoint_rejected", details=validation)
        if verified_unpack_product(snapshot, target_id):
            validation["gripper_prediction_applied"] = False
            validation["gripper_prediction_reason"] = "verified_processing_product_uses_actual_attachment_checks"
        else:
            prediction = self._gripper_quality(snapshot, target_id, achieved, arm, settings)
            validation["predicted_gripper_compatibility"] = prediction
            if not prediction["accepted"]:
                validation.update(accepted=False, reason=prediction["reason"])
                raise AlgorithmError("planned grasp endpoint violates predicted gripper compatibility",
                                     code="planning_grasp_endpoint_rejected", details=validation)
        return {key: deepcopy(candidate[key]) for key in (
            "nominal_T_world_tcp", "source_ik_fk", "fk_goal_validation", "ik",
            "source_batch", "source_index") if key in candidate} | {
                "grasp_endpoint_fk": deepcopy(endpoint), "final_grasp_validation": validation}

    @staticmethod
    def _completed_object_geometry(snapshot, target):
        """Use registered CAD surface vertices or the original unknown surface."""
        cad = target.get('cad')
        if cad is not None:
            if (cad.get('instance_id') != target['id'] or any(cad.get(key) != snapshot.get(key)
                    for key in ('observation_id', 'generation'))):
                raise ValueError('cad_pose_binding_stale')
            if cad.get('pose_validation', {}).get('accepted') is not True:
                raise ValueError('cad_pose_not_validated')
            vertices = registered_grasp_geometry(cad['asset_id'], cad['mesh_key'],
                cad['source_sha256'], cad['mesh_sha256'])
            transform = rigid_matrix(cad['T_world_object'])
            return vertices @ transform[:3, :3].T + transform[:3, 3]
        return measured_object_geometry(target)

    @staticmethod
    def _graspgen_object_points(snapshot, target):
        return Algorithms._graspgen_object_points(snapshot, target)

    def generate_grasps(self, snapshot, target_id, *, arm=None, placement_request=None,
                        planning_settings=None, sampling_round=1, progress=None, cancelled=None):
        self._check_cancelled(cancelled)
        candidate_count = grasp_candidate_count(sampling_round)
        if arm not in ("A", "B"):
            raise ValueError("demo_grasps_require_selected_arm")
        settings = PlanningSettings.model_validate(planning_settings or {})
        target = observed_object(snapshot, target_id)
        input_source = 'registered_cad' if target.get('cad') is not None else 'measured'
        reach_context = {'robot_base_world_m': arm_wrist_reach(arm)['robot_base_world_m'],
                         'measured_reference_world_m': np.asarray(target['points_world_m']).mean(axis=0)}
        result = super().generate_grasps(
            snapshot, target_id, arm=arm, candidate_count=candidate_count,
            progress=progress, cancelled=cancelled)
        result["sampling_round"] = sampling_round
        offline = result.get('source') == 'cad_offline_library'
        if offline:
            candidate_count = result['requested_candidate_count']
        input_source = result.get('graspgen_input_source', result.get('source'))
        result["raw_candidate_count"] = sum(batch["candidate_count"] for batch in result["batches"])
        graspgen_request = result.pop("graspgen_input", None)
        if graspgen_request is not None:
            digest = hashlib.sha256()
            for key in ("scene_points_camera_m", "object_points_camera_m"):
                values = graspgen_request[key]
                digest.update(str((key, values.shape, values.dtype)).encode())
                digest.update(values.tobytes())
            digest.update(graspgen_request["gripper_config_id"].encode())
            result["sampling_input_digest"] = digest.hexdigest()
            # Identity and frame origin retain the measured cloud separately.
            # The inference digest/count describe the exact submitted array.
            result["measured_cloud_reference"] = result["cloud"]
            input_points = graspgen_request["object_points_camera_m"]
            result["cloud"] = {
                "id": hashlib.sha256(input_points.tobytes()).hexdigest(),
                "frame": f"cloud:{target_id}", "hash_frame": f"cloud:{target_id}",
                "point_count": len(input_points),
                "source": input_source,
                "measured_point_count": len(target["points_world_m"]),
                "synthetic_point_count": 0}
        all_generated = result.pop('all_generated_candidates', result['candidates'])
        candidates, accepted, evidence = prepare_grasp_candidates(all_generated,
            planning_settings=settings.model_dump(),
            robot_base_world_m=reach_context["robot_base_world_m"],
            target_reference_world_m=reach_context["measured_reference_world_m"],
            reference_tcp=snapshot.get('arms', {}).get(arm, {}).get('tcp'))
        replay = {"stage": "before_ik", "target_id": target_id, "sampling_round": sampling_round,
            "all_generated_candidates": all_generated, "filter_all_generated": True,
            "requested_candidate_count": candidate_count, "raw_candidate_count": result["raw_candidate_count"],
            "measured_points_world_m": np.asarray(target["points_world_m"]).copy(),
            "T_world_cloud": result.get("T_world_cloud"), "candidates_before_ik": candidates,
            "observation_id": snapshot.get("observation_id"), "arm": arm,
            "graspgen_request": graspgen_request, "graspgen_input_source": input_source,
            "cad_library": ({key: deepcopy(result[key]) for key in
                ('library_id', 'library_sha256', 'asset_id', 'source_sha256', 'T_world_source')}
                if offline else None),
            "graspgen_input_cloud": deepcopy(result.get("cloud")),
            "measured_cloud_reference": deepcopy(result.get("measured_cloud_reference")),
            "direction_filter": evidence}
        if not accepted:
            # The planner owns the shared bounded pool budget. A complete
            # direction rejection is eligible for its next frozen pool too.
            return {**result, 'candidates': [], 'candidate_count': 0,
                    'downward_filter': evidence, 'candidate_replay': replay}
        replay['goalset_candidate_indices'] = [candidate['index'] for candidate in accepted]
        return {**result, 'candidates': accepted, 'candidate_count': len(accepted),
                'downward_filter': evidence, 'candidate_replay': replay}

    def publish_released_object(self, scene, target_id, grasp_tcp, release_tcp, after_capture_id, *,
                                support_floor_z_m=None):
        transform = rigid_matrix(release_tcp) @ np.linalg.inv(rigid_matrix(grasp_tcp))
        payload = {"scene_token": scene["scene_token"], "generation": scene["generation"],
            "target_id": target_id, "operation": "transfer", "exclusion_state": "released",
            "source": "release_feedback_rigid_prediction", "after_capture_id": after_capture_id,
            "T_new_world_from_old_world": transform.tolist()}
        if support_floor_z_m is not None:
            payload["support_floor_z_m"] = float(support_floor_z_m)
        result = self.request("/perception/update", payload)
        return {**scene, **result}

    def publish_container_occupancy(self, scene, receipt):
        result = self.request('/perception/update', {
            'scene_token': scene['scene_token'], 'generation': scene['generation'],
            'target_id': receipt['target_id'], 'container_id': receipt['destination_id'],
            'operation': 'container_entry', 'exclusion_state': 'released',
            'source': receipt['source'], 'receipt_id': receipt['receipt_id'],
            'binding_id': receipt['binding_id'], 'after_capture_id': receipt['observation_id']})
        return {**scene, **result}

    def measure_local_object(self, capture, work_area, appearance):
        from .local_object_obstacle import measure_object_obstacle

        cameras = ['3', '2', '1']
        attempts = []
        for camera_id in cameras:
            try:
                measured = measure_object_obstacle(capture, self.perception.detector, work_area,
                                                    appearance, camera_id=camera_id)
                return measured
            except ValueError as error:
                if str(error) != 'local_object_depth_missing':
                    raise
                attempts.append(getattr(error, 'details', {'camera_id': camera_id}))
                if camera_id == cameras[-1]:
                    error.details = {'camera_attempts': attempts}
                    raise

    def measured_object_workspace(self, measurement, arm):
        points = np.asarray(measurement['points_world_m'], float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
            raise ValueError('local_object_depth_missing')
        center = ((points.min(axis=0) + points.max(axis=0)) / 2).tolist()
        result = self.workspace_lookup(arm, [center])[0]
        return {**result, 'arm': arm, 'observation_id': measurement['observation_id'],
            'source': 'current_rgbd_object_center_and_offline_workspace',
            'scope': 'position_workspace_only_grasp_pose_and_path_not_yet_planned'}

    def invalidate_scene_entity(self, scene, target_id, action_id, source="mock_vla"):
        """Record geometry changed by an action; never invent replacement occupancy."""
        if (not isinstance(target_id, str) or not target_id
                or not isinstance(action_id, str) or not action_id
                or source not in SCENE_INVALIDATION_SOURCES):
            raise ValueError("invalid scene invalidation provenance")
        result = self.request("/perception/update", {
            "scene_token": scene["scene_token"], "generation": scene["generation"],
            "target_id": target_id, "operation": "invalidate", "exclusion_state": "released",
            "action_id": action_id, "source": source})
        if not isinstance(result.get("scene_token"), str) or not result["scene_token"]:
            raise AlgorithmError("scene invalidation did not return a scene token")
        return {**scene, **result}

    def plan_dual_vla_motion(self, observation, scene, goals, *, planning_settings=None, cancelled=None):
        """Use the dual joint planner and validate its already interpolated command."""
        self._check_cancelled(cancelled)
        self._binding(observation, scene["target_id"], scene)
        if scene.get("exclusion_state") != "observation" and not scene.get("return_after_place"):
            raise AlgorithmError("VLA motion requires a retained target-inclusive scene")
        if not isinstance(goals, dict) or set(goals) != {"A", "B"}:
            raise ValueError("VLA motion requires both seven-joint goals in degrees")
        goals = {arm: np.asarray(goals[arm], dtype=float) for arm in ("A", "B")}
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in goals.values()):
            raise ValueError("VLA goals must contain seven finite angles per arm")
        goals = {arm: q.tolist() for arm, q in goals.items()}
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        result = self.request("/tools/plan_dual_observation", {
            "scene_token": scene["scene_token"], "planning_settings": settings,
            "return_after_place": bool(scene.get("return_after_place")),
            "start_joints_deg": {a: np.rad2deg(observation["arms"][a]["joints_rad"]).tolist() for a in ("A", "B")},
            "goal_joints_deg": goals,
            "gripper_joint_positions_rad": {a: observation["arms"][a]["gripper_position_rad"] for a in ("A", "B")}})
        self._check_cancelled(cancelled)
        if result.get("scene_token") != scene["scene_token"] or result.get("attached") is not False:
            raise AlgorithmError("dual VLA plan returned stale or attached geometry")
        try:
            paths = {a: np.deg2rad(result["trajectories"][a]) for a in ("A", "B")}
            dt = float(result["sample_dt_s"])
            if (not np.isfinite(dt) or dt <= 0 or paths["A"].shape != paths["B"].shape
                    or any(q.ndim != 2 or q.shape[1] != 7 or len(q) < 2 or not np.isfinite(q).all()
                           for q in paths.values())):
                raise ValueError("invalid paired trajectories")
        except (KeyError, TypeError, ValueError) as error:
            raise AlgorithmError("dual VLA plan returned invalid trajectories") from error
        if any(np.max(np.abs(np.rad2deg(paths[a][-1]) - goals[a])) > 1. for a in ("A", "B")):
            raise AlgorithmError("VLA trajectory does not reach the requested joint posture")
        paths = {a: q.tolist() for a, q in paths.items()}
        check = self.validate_dual_observation(observation, scene, paths, planning_settings=settings)
        self._check_cancelled(cancelled)
        return {"paths": paths, "dt": dt, "scene_token": scene["scene_token"], "command_check": check,
                "goal_joints_deg": goals, "planning_settings": settings}

    @staticmethod
    def gripper_release_spheres(observation, arm):
        """Current closed and fully open gripper endpoint spheres, in front frame."""
        from .carrying import configure_gripper_feedback

        root = Path(__file__).resolve().parents[2]
        urdf_path = root / "assets/marvin_urdf/marvin_with_grippers.urdf"
        config = yaml.safe_load((root / "resources/robot/marvin_gui_dual.yml").read_text(encoding="utf-8"))
        feedback = {name: float(value["gripper_position_rad"]) for name, value in observation["arms"].items()}
        config = configure_gripper_feedback(config, feedback, urdf_path)
        kin = config["robot_cfg"]["kinematics"]
        prefix = "gripL_" if arm == "A" else "gripR_"
        tool = "TCP_Link_L" if arm == "A" else "TCP_Link_R"
        driver = prefix + "finger_joint"
        closed = kin["lock_joints"][driver]
        angles = (closed, 0.)
        mimics = [(joint.get("name"), joint.find("mimic")) for joint in ET.parse(urdf_path).getroot()
                  .findall("joint") if joint.find("mimic") is not None]
        model = robot_model()
        data = mujoco.MjData(model)
        spheres = []
        for angle in angles:
            joints = dict(kin["lock_joints"])
            joints[driver] = float(angle)
            for name, mimic in mimics:
                joints[name] = (joints[mimic.get("joint")] * float(mimic.get("multiplier", 1))
                                + float(mimic.get("offset", 0)))
            for name, value in joints.items():
                data.qpos[int(model.joint(name).qposadr[0])] = value
            mujoco.mj_fwdPosition(model, data)
            tool_id = model.body(tool).id
            tool_rotation = data.xmat[tool_id].reshape(3, 3)
            for link in kin["collision_link_names"]:
                if not link.startswith(prefix) and link != tool:
                    continue
                body = model.body(link).id
                rotation = data.xmat[body].reshape(3, 3)
                buffer = kin["collision_sphere_buffer"]
                if isinstance(buffer, dict):
                    buffer = buffer.get(link, 0.)
                for sphere in kin["collision_spheres"][link]:
                    center_world = data.xpos[body] + rotation @ np.asarray(sphere["center"])
                    center_tool = tool_rotation.T @ (center_world - data.xpos[tool_id]) - front_tcp_offset(arm)
                    radius = float(sphere["radius"]) + buffer
                    spheres.append([*center_tool.tolist(), radius])
        return spheres


    def prepare_fixed_placement(self, observation, target_id, arm, scene, holding, destination_id, *,
                          placement_request, placement_zone, planning_settings=None, progress=None, cancelled=None):
        from .robot_skills.placement_resolution import digest, resolution_input

        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        if placement_zone['destination_id'] != destination_id:
            raise ValueError('placement_destination_changed')
        resolved = self._resolve_place_goal(observation, target_id, arm, scene, holding, destination_id,
            planning_settings=settings, placement_request=placement_request, progress=progress, cancelled=cancelled)
        inputs = resolution_input(observation, target_id, arm, scene, holding, destination_id, placement_request, settings)
        candidate = {'candidate_id': 'placement-0', 'request': deepcopy(placement_request), **resolved}
        result = {'status': 'success', 'candidates': [candidate], 'candidate_failures': [],
            'selection_evidence': {'source': 'operator_configured_square', 'candidate_count': 1,
                'xy_search_performed': False, 'height_search_performed': False},
            'semantic_selection': {'status': 'success', 'scope': 'configured_xy', 'candidate_ids': ['placement-0']},
            'placement_zone': deepcopy(placement_zone), 'trajectory_checked': False,
            'robot_feedback': deepcopy(inputs['robot_feedback']),
            'robot_feedback_time': observation.get('feedback_time'), 'input_digest': digest(inputs)}
        result['resolution_id'] = digest(result)
        return result

    def _plan_resolved_placement(self, observation, target_id, arm, scene, holding, destination_id, *,
                                planning_settings, placement_request, resolution, progress, cancelled):
        from .robot_skills.placement_resolution import (
            RETRYABLE_PLACEMENT_ERRORS,
            resolution_input,
            validate_resolution,
        )

        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        # Cached endpoint evidence is historical when servo feedback changes.
        # Only fixed poses and immutable scene/holding/semantic bindings survive;
        # each candidate still needs a new current-feedback path and all phase checks.
        feedback_check = validate_resolution(resolution, resolution_input(observation, target_id, arm, scene, holding,
            destination_id, placement_request, settings), require_same_feedback=False)
        if resolution['placement_zone'].get('semantic_scope'):
            selection = resolution.get('semantic_selection', {})
            accepted = set(selection.get('candidate_ids', []))
            if selection.get('status') != 'success' or any(
                    candidate['candidate_id'] not in accepted for candidate in resolution['candidates']):
                raise ValueError('placement_semantic_selection_required')
        failures = []
        for index, candidate in enumerate(resolution['candidates']):
            self._check_cancelled(cancelled)
            try:
                result = self._plan_place_at_reference(observation, target_id, arm, scene, holding, destination_id,
                    placement_request=candidate['request'], planning_settings=settings,
                    progress=progress, cancelled=cancelled, resolved=candidate)
            except AlgorithmError as error:
                if error.code not in RETRYABLE_PLACEMENT_ERRORS:
                    raise
                failures.append({'candidate_id': candidate['candidate_id'], 'code': error.code, 'details': error.details})
                continue
            feedback_check.update(authorization='current_mode_specific_trajectory_checks',
                reference_endpoint_proof_current=bool(feedback_check['consistent']))
            result.update(placement_request=deepcopy(placement_request),
                resolution_feedback_check=feedback_check,
                placement_zone=deepcopy(resolution['placement_zone']), resolution_id=resolution['resolution_id'],
                candidate_count=len(resolution['candidates']), attempted_candidate_count=index + 1,
                candidate_failures=failures)
            result['selection_evidence'].update(candidate_id=candidate['candidate_id'],
                candidate_index=index, rule='decisions_accepted_candidates_then_validated_path',
                candidate_count=len(resolution['candidates']), resolution=deepcopy(resolution['selection_evidence']))
            if 'semantic_selection' in resolution:
                result['selection_evidence']['semantic_selection'] = deepcopy(resolution['semantic_selection'])
            return result
        raise AlgorithmError('No resolved placement candidate produced a checked path for its bound release mode',
            code='place_no_valid_candidate', details={'candidate_failures': failures,
                'resolution_id': resolution['resolution_id'], 'whole_destination_infeasible': False})

    def plan_place(self, observation, target_id, arm, scene, holding, destination_id, *,
                   planning_settings=None, placement_request=None, progress=None, cancelled=None,
                   placement_resolution=None):
        if placement_resolution is not None:
            return self._plan_resolved_placement(observation, target_id, arm, scene, holding, destination_id,
                planning_settings=planning_settings, placement_request=placement_request,
                resolution=placement_resolution, progress=progress, cancelled=cancelled)
        raise AlgorithmError('fixed placement must be bound before plan_place',
                             code='placement_resolution_required')

    def _resolve_place_goal(self, observation, target_id, arm, scene, holding, destination_id, *,
                            planning_settings=None, placement_request=None, progress=None, cancelled=None):
        self._check_cancelled(cancelled)
        self._binding(observation, target_id, scene)
        if not holding.get('verified') or not holding.get('attached'):
            raise AlgorithmError('place requires verified attachment', code='place_requires_attachment')
        if holding['arm'] != arm or holding['instance_id'] != target_id:
            raise AlgorithmError('held object mismatch', code='place_holding_mismatch')
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        destination = observed_object(observation, destination_id)
        request = deepcopy(placement_request)
        prepared = holding.get('attachment_evidence', {})
        if 'attachment_spheres_tool_m' not in prepared:
            prepared = self.prepare_carrying(observation, arm, scene, holding, planning_settings=settings)
        if prepared.get('scene_token', scene['scene_token']) != scene['scene_token']:
            raise AlgorithmError('attachment belongs to another scene', code='planning_stale_scene')
        bottom_clearance = settings['place_bottom_clearance_m']
        if request['release_mode'] == 'container_release':
            from .container_release import container_release_reference

            rim = container_release_reference(destination, scene, observation, settings)
            request.update(release_surface_z_m=rim['rim_z_m'], rim_provenance=rim['rim_provenance'],
                           scene_token=scene['scene_token'])
            bottom_clearance = settings['container_release_bottom_clearance_m']
        goal = fixed_release_goal(destination, holding, request, prepared['attachment_spheres_tool_m'],
            self.gripper_release_spheres(observation, arm), clearance_m=settings['trajectory_collision_clearance_m'],
            observation_id=observation['observation_id'], bottom_clearance_m=bottom_clearance)
        goal['release_reference']['goal_workspace'] = check_goal_workspace(self.workspace_lookup, arm,
            np.asarray(goal['T_world_tcp'])[:3, 3].tolist())
        binding = self._held_binding(observation, target_id, arm, scene, holding)
        payload = self._motion_request(observation, arm, scene, goal['T_world_tcp'], settings)
        payload.update(carrying_binding=binding, approach_axis=settings['place_approach_axis'],
            approach_offset_m=settings['place_approach_offset_m'], approach_in_tool_frame=settings['place_approach_in_tool_frame'],
            lift_axis=settings['place_retreat_axis'], lift_offset_m=settings['place_retreat_offset_m'],
            lift_in_tool_frame=settings['place_retreat_in_tool_frame'],
            trajectory_collision_clearance_m=settings['trajectory_collision_clearance_m'],
            optimizer_collision_activation_distance_m=settings['optimizer_collision_activation_distance_m'])
        return {'goal': goal, 'payload': payload, 'binding': binding, 'settings': settings}


    def _plan_place_at_reference(self, observation, target_id, arm, scene, holding, destination_id, *,
                    planning_settings=None, placement_request=None, progress=None, cancelled=None, resolved=None):
        self._check_cancelled(cancelled)
        self._binding(observation, target_id, scene)
        resolved = (self._resolve_place_goal(observation, target_id, arm, scene, holding, destination_id,
            planning_settings=planning_settings, placement_request=placement_request,
            progress=progress, cancelled=cancelled) if resolved is None else deepcopy(resolved))
        goal, payload, binding, settings = (resolved[key] for key in ('goal', 'payload', 'binding', 'settings'))
        if placement_request['release_mode'] != goal['release_reference']['release_mode']:
            raise ValueError('placement_release_mode_changed')
        # The pose stays frozen; trajectory planning starts from current joint feedback.
        payload.update(self._motion_request(observation, arm, scene, goal['T_world_tcp'], settings))
        if progress:
            progress("planning_start", "cuRobo 固定单目标放置开始", {"arm": arm, "candidate_count": 1,
                     "release_reference": goal["release_reference"]})
        self._check_cancelled(cancelled)
        started = time.monotonic()
        try:
            response = self.request('/tools/plan_place', payload)
        except AlgorithmError:
            self._check_cancelled(cancelled)
            raise
        self._check_cancelled(cancelled)
        if response.get("scene_token") != scene["scene_token"]:
            raise AlgorithmError("place plan returned a stale scene token", code="planning_stale_scene")
        if response.get("collision_evidence", {}).get("optimizer_attachment") is not True:
            raise AlgorithmError("place plan did not optimize the carried object", code="attachment_not_applied")
        if not isinstance(response.get('phases'), dict) or set(response['phases']) != {'approach', 'place'}:
            raise AlgorithmError('place plan has incomplete phases', code='algorithm_invalid_response')
        raw = self._native_phases(response['phases'], arm)
        phases, checks = self._complete_place_candidate(
            observation, arm, scene, binding, raw, settings, cancelled=cancelled)
        selection = {"candidate_index": 0, "rule": "single_fixed_object_reference_and_grasp_rotation",
                     "retreat_checked": True, "release_mode": goal["release_reference"]["release_mode"], "yaw_delta_deg": 0., "candidate_count": 1}
        if progress:
            progress("planning_complete", "cuRobo 放置模式对应轨迹校验完成",
                     {**selection, "duration_s": time.monotonic() - started})
        return {"arm": arm, "target_id": target_id, "destination_id": destination_id,
            "observation_id": observation["observation_id"], "generation": observation["generation"],
            "scene_token": scene["scene_token"], "phases": phases, "goal": goal,
            "carrying_binding": binding, "collision_evidence": response["collision_evidence"],
            "holding_reference": deepcopy(holding["points_tool_m"]), "command_checks": checks,
            "retreat_requires_replan": False, "planning_settings": settings,
            "time_scale": settings["time_scale"], "candidate_count": 1, "candidate_failures": [],
            "selection_evidence": selection, "placement_request": deepcopy(placement_request)}
