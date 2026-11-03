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
    filter_fixed_slot_reach,
    filter_transport_reach,
    fixed_slot_reach_context,
    prepare_grasp_candidates,
)
from .demo_gripper_quality import predict_gripper_compatibility
from .demo_lift import source_clearance_lift
from .demo_placement import (
    apply_release_height,
    distributed_release_candidates,
    fixed_release_goal,
)
from .demo_sampling import grasp_candidate_count
from .geometry import TCP_FRAMES, front_tcp_offset, rigid_matrix
from .kinematics import robot_model
from .object_obb import held_geometry_points
from .placement import measured_object_geometry, observed_object
from .placement_candidates import measured_region_candidates
from .placement_constraints import release_configuration_clearance
from .planning_settings import PlanningSettings
from .robot_skills.placement_selection import (
    check_goal_workspace,
    check_height_workspace,
)


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
    entity = snapshot.get('inventory_state', {}).get('entities', {}).get(target_id, {})
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
    grasp_trajectory_batch_capacity = 2
    def evaluate_reachability(self, snapshot, target_id, destination_id=None, *, requested_arm=None):
        destination = next((item for item in snapshot.get('objects', [])
                            if item.get('id') == destination_id), {})
        if destination.get('kind') == 'bin':
            # Release is above the rim, not at the visible floor. Before scene
            # registration and grasp selection neither that volume nor attitude
            # is available; floor failure cannot certify transport infeasibility.
            result = super().evaluate_reachability(snapshot, target_id, requested_arm=requested_arm)
            for value in result.values():
                value.update(destination_id=destination_id,
                    destination={'reachable': None, 'status': 'pending_above_container_pose',
                                 'reason': 'requires_measured_region_and_grasp_attitude'},
                    destination_candidates=[], destination_status='pending_above_container_pose',
                    feasibility_scope='source_position_workspace_only')
            points = np.asarray(destination.get('points_world_m', []), float)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) and np.isfinite(points).all():
                # References on actual observed surfaces help select an arm.
                # They are not release poses and cannot exclude an arm merely
                # because a partially observed container has no positive sample.
                positions = points[np.linspace(0, len(points) - 1, min(64, len(points)), dtype=int)].tolist()
                for arm, value in result.items():
                    samples = []
                    for start in range(0, len(positions), 2):
                        samples.extend(self.workspace_lookup(arm, positions[start:start + 2]))
                    if (len(samples) != len(positions) or any(
                            not isinstance(row, dict) or type(row.get('reachable')) is not bool
                            or row.get('position_world_m') != position
                            for row, position in zip(samples, positions))):
                        raise AlgorithmError('invalid container workspace context')
                    reachable = [row['position_world_m'] for row in samples if row['reachable']]
                    value['destination_workspace_context'] = {
                        'source': 'observed_container_surface_samples', 'tested_count': len(samples),
                        'reachable_sample_count': len(reachable), 'reachable_examples': reachable[:2],
                        'reference_only_not_release_goals': True, 'absence_is_not_infeasibility_proof': True}
            return result
        if destination.get('kind') not in {'marker', 'bin'}:
            return super().evaluate_reachability(snapshot, target_id, destination_id,
                                                requested_arm=requested_arm)
        candidates = measured_region_candidates(destination)
        result = super().evaluate_reachability(snapshot, target_id, requested_arm=requested_arm)
        positions = [candidate['position_world_m'] for candidate in candidates]
        for arm, value in result.items():
            items = []
            # The offline atlas's public query accepts at most two positions.
            # These are cached CPU lookups, with no planner or network requests.
            for start in range(0, len(positions), 2):
                pair = self.workspace_lookup(arm, positions[start:start + 2])
                if not isinstance(pair, list):
                    raise AlgorithmError('invalid workspace lookup response')
                items.extend(pair)
            if (not isinstance(items, list) or len(items) != len(positions)
                    or any(not isinstance(item, dict) or type(item.get('reachable')) is not bool
                           or item.get('position_world_m') != position
                           for item, position in zip(items, positions))):
                raise AlgorithmError('invalid workspace lookup response')
            reachable = [candidate | {'workspace_evidence': item}
                         for candidate, item in zip(candidates, items) if item['reachable']]
            value.update(destination_id=destination_id,
                         destination={'reachable': bool(reachable), 'tested_count': len(candidates),
                                      'method': 'measured_support_region_workspace_samples'},
                         destination_candidates=reachable,
                         reachable=value['source']['reachable'] and bool(reachable),
                         destination_status='sampled_support_only_pose_and_path_pending')
        return result

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
        return {"accepted": not reasons, "reason": "accepted" if not reasons else ";".join(reasons),
                "position_error_m": position, "rotation_error_rad": rotation,
                "position_threshold_m": settings["ik_position_threshold_m"],
                "rotation_threshold_rad": settings["ik_rotation_threshold_rad"],
                "downward_dot": downward, "horizontal_away_from_robot_dot": away_dot,
                "maximum_tilt_deg": settings['grasp_maximum_tilt_deg'],
                "reference": "original_graspgen_tool_pose_with_arm_jaw_symmetry"}

    def _grasp_ik_candidate(self, candidate, snapshot, target_id, arm, scene, settings):
        item = candidate["ik"]
        achieved = self._bound_tool_fk(item.get("achieved_tool_pose"), item.get("joints_rad"),
                                       item.get("joint_names"), arm, scene)
        nominal = rigid_matrix(candidate["T_world_tcp"])
        evidence = {"candidate_index": candidate["candidate_index"],
                    **self._grasp_pose_validation(nominal, achieved, snapshot, target_id, arm, settings)}
        if not evidence["accepted"]:
            return None, evidence
        if verified_unpack_product(snapshot, target_id):
            evidence["gripper_prediction_applied"] = False
            evidence["gripper_prediction_reason"] = "verified_processing_product_uses_actual_attachment_checks"
        else:
            prediction = self._gripper_quality(snapshot, target_id, achieved, arm, settings)
            evidence["predicted_gripper_compatibility"] = prediction
            if not prediction["accepted"]:
                evidence.update(accepted=False, reason=prediction["reason"])
                return None, evidence
        return {**candidate, "nominal_T_world_tcp": nominal.tolist(),
                "T_world_tcp": achieved.tolist(), "source_ik_fk": deepcopy(item["achieved_tool_pose"]),
                "fk_goal_validation": evidence}, evidence

    def _grasp_destination_candidates(self, valid, snapshot, target_id, arm, scene, settings, destination_id):
        from itertools import product

        from .grasp_placement_screen import fixed_gripper_spheres, screen_grasp_placement
        from .object_obb import observed_object_obb

        if destination_id is None or not valid:
            return valid, {'applied': False, 'reason': 'no_destination_or_eligible_grasps'}
        destination = observed_object(snapshot, destination_id)
        reference = destination.get('placement_reference', {})
        if (reference.get('observation_id') != snapshot['observation_id']
                or reference.get('generation', snapshot['generation']) != snapshot['generation']
                or 'region_polygon_world_xy_m' not in reference or 'floor_z_m' not in reference):
            return valid, {'applied': False, 'reason': 'current_placement_region_unavailable',
                'destination_id': destination_id}
        self._binding(snapshot, target_id, scene)
        target = observed_object(snapshot, target_id)
        cad = target.get('cad')
        if cad is not None:
            points = self._completed_object_geometry(snapshot, target)
        else:
            measured_object_geometry(target)
            obb = observed_object_obb(target)
            transform = rigid_matrix(obb['T_frame_obb'])
            half = np.asarray(obb['half_extents_m'])
            corners = np.array(list(product(*[[-value, value] for value in half])))
            points = corners @ transform[:3, :3].T + transform[:3, 3]
        peer = 'B' if arm == 'A' else 'A'
        payload = {'arm': arm, 'scene_token': scene['scene_token'], 'planning_settings': settings,
            'observation_id': snapshot['observation_id'], 'generation': snapshot['generation'],
            'start_joints_deg': np.rad2deg(snapshot['arms'][arm]['joints_rad']).tolist(),
            'other_joints_deg': np.rad2deg(snapshot['arms'][peer]['joints_rad']).tolist(),
            'gripper_joint_positions_rad': {
                name: snapshot['arms'][name]['gripper_position_rad'] for name in ('A', 'B')}}

        def query(spheres):
            return self.request('/tools/query_world_spheres', {**payload, 'spheres_world_m': spheres.tolist()})

        kept, evidence = screen_grasp_placement(valid, points, reference, fixed_gripper_spheres(arm), query,
            scene_token=scene['scene_token'], settings=settings, uncertainty_m=target.get('uncertainty_m', .005))
        evidence.update(arm=arm, target_id=target_id, destination_id=destination_id,
            observation_id=snapshot['observation_id'], generation=snapshot['generation'],
            placement_reference=deepcopy(reference), object_geometry_point_count=len(points),
            object_geometry_source='registered_cad' if cad is not None else 'observed_obb')
        if cad is not None:
            evidence.update({key: deepcopy(cad[key]) for key in
                ('asset_id', 'source_sha256', 'mesh_sha256', 'T_world_object')})
        return kept, evidence

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

    def _validate_fk_candidate(self, candidate, snapshot, target_id, arm, scene, settings):
        if "nominal_T_world_tcp" not in candidate or "source_ik_fk" not in candidate:
            raise AlgorithmError("IK evaluation is missing the accepted FK goal evidence",
                                 code="planning_fk_evidence_invalid")
        accepted, _ = self._grasp_ik_candidate(
            {**candidate, "T_world_tcp": candidate["nominal_T_world_tcp"]},
            snapshot, target_id, arm, scene, settings)
        if (accepted is None or accepted["source_ik_fk"] != candidate["source_ik_fk"]
                or not np.array_equal(accepted["T_world_tcp"], candidate["T_world_tcp"])):
            raise AlgorithmError("IK evaluation no longer matches its accepted FK goal",
                                 code="planning_fk_evidence_invalid")

    def _grasp_plan_evidence(self, candidate, item, phases, snapshot, target_id, arm, scene, settings):
        self._validate_fk_candidate(candidate, snapshot, target_id, arm, scene, settings)
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

    def _grasp_lift_settings(self, snapshot, target_id, arm, candidates, scene, settings):
        for candidate in candidates:
            self._validate_fk_candidate(candidate, snapshot, target_id, arm, scene, settings)
        try:
            self._completed_object_geometry(snapshot, observed_object(snapshot, target_id))
            return source_clearance_lift(snapshot, target_id, arm, candidates, scene, settings)
        except ValueError as error:
            raise AlgorithmError(str(error), code="grasp_lift_geometry_invalid", details={
                "reason": str(error), "stage": "source_clearance_lift", "target_id": target_id,
                "observation_id": snapshot.get("observation_id"), "scene_token": scene.get("scene_token"),
                **getattr(error, "details", {})}) from error

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
        object_geometry = self._completed_object_geometry(snapshot, target)
        input_source = 'registered_cad' if target.get('cad') is not None else 'measured'
        transport = bool((placement_request or {}).get('transport_region'))
        if transport:
            if ((placement_request or {}).get('object_id') != target_id
                    or (placement_request or {}).get('transport_region', {}).get('observation_id')
                    != snapshot.get('observation_id') or not placement_request.get('transport_region')):
                raise ValueError('grasp_requires_current_transport_region')
            reach_context = {"robot_base_world_m": arm_wrist_reach(arm)["robot_base_world_m"],
                             "measured_reference_world_m": np.asarray(target["points_world_m"]).mean(axis=0)}
        else:
            reach_context = fixed_slot_reach_context(snapshot, target_id, arm, placement_request, settings)
            reach_context.update(nominal_object_min_world_z_m=float(object_geometry[:, 2].min()),
                                 nominal_object_geometry_source=input_source)
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
                "synthetic_point_count": len(input_points),
                "object_obb": deepcopy(result['object_obb'])}
        all_generated = result.pop('all_generated_candidates', result['candidates'])
        candidates, accepted, evidence = prepare_grasp_candidates(all_generated,
            planning_settings=settings.model_dump(),
            robot_base_world_m=reach_context["robot_base_world_m"],
            target_reference_world_m=reach_context["measured_reference_world_m"],
            reference_tcp=(snapshot.get('arms', {}).get(arm, {}).get('tcp')
                           if transport else None))
        replay = {"stage": "before_ik", "target_id": target_id, "sampling_round": sampling_round,
            "all_generated_candidates": all_generated, "filter_all_generated": True,
            "requested_candidate_count": candidate_count, "raw_candidate_count": result["raw_candidate_count"],
            "measured_points_world_m": np.asarray(target["points_world_m"]).copy(),
            "T_world_cloud": result.get("T_world_cloud"), "candidates_before_ik": candidates,
            "observation_id": snapshot.get("observation_id"), "arm": arm,
            "graspgen_request": graspgen_request, "graspgen_input_source": input_source,
            "object_obb": deepcopy(result.get('object_obb')),
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
        if transport:
            accepted, reach_evidence = filter_transport_reach(accepted, object_geometry, arm, placement_request)
        else:
            accepted, reach_evidence = filter_fixed_slot_reach(accepted, reach_context)
        replay['pre_budget_eligible_indices'] = [candidate['index'] for candidate in accepted]
        accepted = accepted[:candidate_count]
        replay['pre_ik_eligible_indices'] = [candidate['index'] for candidate in accepted]
        replay['reach_filter'] = deepcopy(reach_evidence)
        if not accepted and not transport:
            raise AlgorithmError("no downward grasp passes the fixed-slot nominal 3D reach screen",
                                 code="grasp_no_fixed_slot_reachable_candidate",
                                 details={"downward_filter": evidence,
                                          "fixed_slot_reach_filter": reach_evidence,
                                          "candidate_replay": replay})
        return {**result, "candidates": accepted, "candidate_count": len(accepted),
                "downward_filter": evidence, "fixed_slot_reach_filter": reach_evidence,
                "candidate_replay": replay}


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
                or source not in {"mock_vla", "verified_drop_action_bounds", "drop_release_action_bounds"}):
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
                buffer = kin["collision_sphere_buffer"].get(link, 0.)
                for sphere in kin["collision_spheres"][link]:
                    center_world = data.xpos[body] + rotation @ np.asarray(sphere["center"])
                    center_tool = tool_rotation.T @ (center_world - data.xpos[tool_id]) - front_tcp_offset(arm)
                    radius = float(sphere["radius"]) + buffer
                    spheres.append([*center_tool.tolist(), radius])
        return spheres

    def placement_xy_candidates(self, observation, target_id, holding, request, zone, settings, *,
                                scene, subdivision_cells):
        from .robot_skills.placement_resolution import candidate_requests
        return candidate_requests(observation, target_id, holding, request, zone, settings,
            scene=scene, subdivision_cells=subdivision_cells)

    def resolve_placement(self, observation, target_id, arm, scene, holding, destination_id, *,
                          placement_request, placement_zone, planning_settings=None, previous_resolution=None,
                          subdivision_cells=None, candidate_indices=None,
                          progress=None, cancelled=None):
        from .robot_skills.placement_resolution import (
            RETRYABLE_PLACEMENT_ERRORS,
            candidate_requests,
            digest,
            resolution_input,
            revalidation_candidate_indices,
        )

        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        self._binding(observation, target_id, scene)
        if placement_zone['destination_id'] != destination_id:
            raise ValueError('placement_destination_changed')
        requests, evidence = candidate_requests(observation, target_id, holding, placement_request,
                                               placement_zone, settings, scene=scene,
                                               subdivision_cells=subdivision_cells)
        allowed = list(range(len(requests))) if candidate_indices is None else candidate_indices
        if (not allowed or len(set(allowed)) != len(allowed)
                or any(type(i) is not int or not 0 <= i < len(requests) for i in allowed)):
            raise ValueError('placement_selected_xy_invalid')
        evidence['semantically_selected_xy_indices'] = allowed
        inputs = resolution_input(observation, target_id, arm, scene, holding, destination_id, placement_request, settings)
        seeds = [i for i in revalidation_candidate_indices(previous_resolution, inputs, placement_zone, requests)
                 if i in allowed]
        primary_count = evidence['primary_candidate_count']
        groups = [('previous', seeds),
            ('primary', [i for i in allowed if i < primary_count and i not in seeds]),
            ('remaining', [i for i in allowed if i >= primary_count and i not in seeds])]
        gripper = self.gripper_release_spheres(observation, arm)
        candidates, failures = [], []
        tested_indices = []
        for group, indices in groups:
            if candidates:
                break
            began = time.monotonic()
            for index in indices:
                request = requests[index]
                self._check_cancelled(cancelled)
                tested_indices.append(index)
                try:
                    resolved = self._resolve_place_goal(observation, target_id, arm, scene, holding, destination_id,
                        placement_request=request, planning_settings=settings, progress=progress, cancelled=cancelled,
                        gripper_spheres=gripper, workspace_prefilter=True)
                    pose = np.asarray(resolved['goal']['T_world_tcp'])
                    bottom = (held_geometry_points(holding) @ pose[:3, :3].T + pose[:3, 3])[:, 2].min()
                    gap = float(bottom - placement_zone['floor_z_m'])
                    if gap > settings['place_max_release_gap_m'] + 1e-9:
                        raise AlgorithmError('Clear pose is too high above the intended support',
                            code='placement_release_too_high', details={'object_floor_gap_m': gap,
                                'maximum_object_floor_gap_m': settings['place_max_release_gap_m']})
                    resolved['goal']['release_reference']['object_floor_gap_m'] = gap
                except AlgorithmError as error:
                    if error.code not in RETRYABLE_PLACEMENT_ERRORS:
                        raise
                    failures.append({'candidate_index': index, 'code': error.code, 'details': error.details})
                    continue
                candidates.append({'candidate_id': 'placement-' + str(index),
                                   'request': request, **resolved})
                if len(candidates) >= evidence.get('accepted_candidate_limit', 12):
                    break
            if group == 'previous' and seeds:
                evidence['candidate_revalidation'] = {'source_resolution_id': previous_resolution['resolution_id'],
                    'sampled_candidate_count': len(seeds), 'accepted_candidate_count': len(candidates),
                    'full_grid_fallback': not candidates, 'wall_s': time.monotonic() - began}
        height_proofs = [item['goal']['release_reference']['height_resolution'] for item in candidates]
        height_proofs.extend(item['details'] for item in failures if isinstance(item.get('details'), dict)
                             and 'tested_heights' in item['details'])
        evidence.update(occupied_candidate_count=sum(item['code'] == 'release_height_no_valid_goal' for item in failures),
            held_ik_rejected_candidate_count=sum(item['code'] == 'release_height_no_held_ik_goal' for item in failures),
            held_ik_required=True,
            held_ik_checked=any(proof.get('held_ik_sample_count', 0) for proof in height_proofs),
            height_world_sample_count=sum(len(proof['tested_heights']) for proof in height_proofs),
            world_colliding_sample_count=sum(bool(row['conflicting_sphere_indices'])
                for proof in height_proofs for row in proof['tested_heights']),
            held_ik_sample_count=sum(proof.get('held_ik_sample_count', 0) for proof in height_proofs),
            held_ik_batch_count=sum(len(proof.get('ik_batches', [])) for proof in height_proofs),
            height_world_query_s=sum(proof.get('world_query_s', 0.) for proof in height_proofs),
            held_ik_batch_wall_s=sum(batch['wall_s'] for proof in height_proofs for batch in proof.get('ik_batches', [])),
            world_checked_candidate_count=len(candidates) + sum(item['code'] != 'place_goal_workspace_rejected' for item in failures),
            sampled_candidate_count=len(tested_indices), tested_candidate_indices=tested_indices,
            scene_token=scene['scene_token'],
            remaining_grid_searched=any(index >= primary_count for index in tested_indices),
            unsearched_candidate_count=len(requests) - len(tested_indices),
            candidate_collision_checked=True, trajectory_checked=False)
        if not candidates:
            raise AlgorithmError('No sampled placement pose passed numerical endpoint checks',
                code='placement_no_candidate', details={'candidate_failures': failures,
                    'selection_evidence': evidence, 'whole_destination_infeasible': False})
        # Prefer the lowest valid release among the small accepted set. A center
        # that only clears an occupied spot after rising must not outrank an
        # equally reachable child center that releases next to the support.
        candidates.sort(key=lambda item: item['goal']['T_world_tcp'][2][3])
        evidence['candidate_order'] = 'lowest_valid_release_height_then_subdivision_order'
        positions = np.array([candidate['goal']['T_world_tcp'] for candidate in candidates])[:, :2, 3]
        selected = distributed_release_candidates(positions, limit=12)
        candidates = [candidates[next(i for i, xy in enumerate(positions) if np.array_equal(xy, point))]
                      for point in selected]
        result = {'status': 'success', 'candidates': candidates, 'candidate_failures': failures,
            'selection_evidence': evidence, 'placement_zone': deepcopy(placement_zone),
            'trajectory_checked': False, 'robot_feedback': deepcopy(inputs['robot_feedback']),
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
            feedback_check.update(authorization='current_three_phase_trajectory_checks',
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
        raise AlgorithmError('No resolved placement candidate produced a checked three-stage path',
            code='place_no_valid_candidate', details={'candidate_failures': failures,
                'resolution_id': resolution['resolution_id'], 'whole_destination_infeasible': False})

    def plan_place(self, observation, target_id, arm, scene, holding, destination_id, *,
                   planning_settings=None, placement_request=None, progress=None, cancelled=None,
                   placement_resolution=None):
        if placement_resolution is not None:
            return self._plan_resolved_placement(observation, target_id, arm, scene, holding, destination_id,
                planning_settings=planning_settings, placement_request=placement_request,
                resolution=placement_resolution, progress=progress, cancelled=cancelled)
        raise AlgorithmError('resolve_placement must select and validate the goal before plan_place',
                             code='placement_resolution_required')

    def _resolve_place_goal(self, observation, target_id, arm, scene, holding, destination_id, *,
                    planning_settings=None, placement_request=None, progress=None, cancelled=None,
                    gripper_spheres=None, workspace_prefilter=False):
        self._check_cancelled(cancelled)
        self._binding(observation, target_id, scene)
        if not holding.get("verified") or not holding.get("attached"):
            raise AlgorithmError("place requires an already attached held object", code="place_requires_attachment")
        if holding.get("arm") != arm or holding.get("instance_id") != target_id:
            raise AlgorithmError("held object differs from placement request", code="place_holding_mismatch")
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        destination = observed_object(observation, destination_id)
        prepared = holding.get("attachment_evidence", {})
        if "attachment_spheres_tool_m" not in prepared:
            prepared = self.prepare_carrying(observation, arm, scene, holding, planning_settings=settings)
        if prepared.get("scene_token", scene["scene_token"]) != scene["scene_token"]:
            raise AlgorithmError("attachment belongs to another scene", code="planning_stale_scene")
        self._check_cancelled(cancelled)
        goal = fixed_release_goal(destination, holding, placement_request,
            prepared["attachment_spheres_tool_m"], (self.gripper_release_spheres(observation, arm)
                if gripper_spheres is None else gripper_spheres),
            clearance_m=settings["trajectory_collision_clearance_m"], observation_id=observation["observation_id"],
            release_height_m=settings['place_release_height_m'])
        binding = self._held_binding(observation, target_id, arm, scene, holding)
        payload = self._motion_request(observation, arm, scene, goal["T_world_tcp"], settings)
        payload.update(carrying_binding=binding, approach_axis=settings["place_approach_axis"],
            approach_offset_m=settings["place_approach_offset_m"],
            approach_in_tool_frame=settings["place_approach_in_tool_frame"],
            lift_axis=settings["place_retreat_axis"], lift_offset_m=settings["place_retreat_offset_m"],
            lift_in_tool_frame=settings["place_retreat_in_tool_frame"],
            trajectory_collision_clearance_m=settings["trajectory_collision_clearance_m"],
            optimizer_collision_activation_distance_m=settings["optimizer_collision_activation_distance_m"])
        if workspace_prefilter:
            from .release_height_search import height_input_digest, height_positions, height_search_bounds

            try:
                bounds = height_search_bounds(goal['T_world_tcp'], held_geometry_points(holding),
                    goal['release_reference']['floor_z_m'], settings['place_max_release_gap_m'])
            except ValueError as error:
                if str(error) != 'placement_release_too_high':
                    raise
                raise AlgorithmError(str(error), code=error.code, details=error.details) from error
            workspace = check_height_workspace(self.workspace_lookup, arm, payload['goal_position_m'],
                                              positions=height_positions(bounds))
            search = {'bounds': bounds, 'workspace': workspace, 'arm': arm,
                'input_sha256': height_input_digest(payload),
                'ik_position_threshold_m': settings['ik_position_threshold_m'],
                'ik_rotation_threshold_rad': settings['ik_rotation_threshold_rad']}
            goal['release_reference']['height_search'] = deepcopy(search)
            payload['height_search'] = search
        if progress:
            progress("release_height_start", "按当前世界碰撞场计算固定XY最低释放高度", {
                "arm": arm, "analytic_goal_position_m": payload["goal_position_m"]})
        try:
            resolution = self.request("/tools/resolve_release_height", {
                **payload, "gripper_spheres_tool_m": goal["release_reference"]["gripper_spheres_tool_m"]})
        except AlgorithmError:
            self._check_cancelled(cancelled)
            raise
        self._check_cancelled(cancelled)
        if resolution.get("scene_token") != scene["scene_token"] or resolution.get("attached") is not True:
            raise AlgorithmError("release height returned stale or detached evidence", code="planning_stale_scene")
        try:
            goal = apply_release_height(goal, resolution, scene_token=scene["scene_token"], holding=holding)
        except ValueError as error:
            raise AlgorithmError(str(error), code="algorithm_invalid_response") from error
        payload["goal_position_m"] = np.asarray(goal["T_world_tcp"])[:3, 3].tolist()
        goal['release_reference']['goal_workspace'] = check_goal_workspace(
            self.workspace_lookup, arm, payload['goal_position_m'])
        configurations = destination.get('placement_reference', {}).get('required_robot_configurations', [])
        if configurations:
            proof = release_configuration_clearance(goal, holding, configurations,
                surface_voxel_size_m=scene.get('voxel_size_m', .02))
            if not proof['compatible']:
                raise AlgorithmError('release conflicts with a required destination robot configuration',
                    code='placement_configuration_conflict', details=proof)
            goal['release_reference']['required_robot_configuration_clearance'] = proof
        if destination.get('kind') == 'bin':
            points = np.asarray(destination.get('points_world_m', []), float)
            if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
                raise ValueError('placement_container_rim_unobserved')
            rim = float(points[:, 2].max())
            pose = np.asarray(goal['T_world_tcp'])
            held_points = held_geometry_points(holding) @ pose[:3, :3].T + pose[:3, 3]
            if held_points[:, 2].min() >= rim - settings['trajectory_collision_clearance_m']:
                raise AlgorithmError('Resolved release does not enter the measured container',
                    code='placement_release_above_container')
            # Enter from above the measured rim and leave it with the open gripper.
            offset = max(settings['place_approach_offset_m'],
                rim + settings['trajectory_collision_clearance_m'] - held_points[:, 2].min())
            gripper = np.asarray(goal['release_reference']['gripper_spheres_tool_m'])
            gripper_bottom = np.min((gripper[:, :3] @ pose[:3, :3].T)[:, 2] - gripper[:, 3]) + pose[2, 3]
            retreat = max(settings['place_retreat_offset_m'],
                rim + settings['trajectory_collision_clearance_m'] - gripper_bottom)
            settings.update(place_approach_axis='z', place_approach_in_tool_frame=False,
                place_approach_offset_m=float(offset), place_retreat_axis='z',
                place_retreat_in_tool_frame=False, place_retreat_offset_m=float(retreat))
            payload.update(approach_axis='z', approach_in_tool_frame=False, approach_offset_m=float(offset),
                           lift_axis='z', lift_in_tool_frame=False, lift_offset_m=float(retreat))
            goal['container_entry'] = {'measured_rim_z_m': rim,
                'approach_offset_m': float(offset), 'retreat_offset_m': float(retreat),
                'source': 'current_identity_bound_container_surface', 'collision_checked': False}
        return {'goal': goal, 'payload': payload, 'binding': binding, 'settings': settings}

    def _plan_place_at_reference(self, observation, target_id, arm, scene, holding, destination_id, *,
                    planning_settings=None, placement_request=None, progress=None, cancelled=None, resolved=None):
        self._check_cancelled(cancelled)
        self._binding(observation, target_id, scene)
        resolved = (self._resolve_place_goal(observation, target_id, arm, scene, holding, destination_id,
            planning_settings=planning_settings, placement_request=placement_request,
            progress=progress, cancelled=cancelled) if resolved is None else deepcopy(resolved))
        goal, payload, binding, settings = (resolved[key] for key in ('goal', 'payload', 'binding', 'settings'))
        # The pose stays frozen; trajectory planning starts from current joint feedback.
        payload.update(self._motion_request(observation, arm, scene, goal['T_world_tcp'], settings))
        if progress:
            progress("planning_start", "cuRobo 固定单目标放置开始", {"arm": arm, "candidate_count": 1,
                     "release_reference": goal["release_reference"]})
        self._check_cancelled(cancelled)
        started = time.monotonic()
        try:
            response = self.request("/tools/plan_place", payload)
        except AlgorithmError:
            self._check_cancelled(cancelled)
            raise
        self._check_cancelled(cancelled)
        if response.get("scene_token") != scene["scene_token"]:
            raise AlgorithmError("place plan returned a stale scene token", code="planning_stale_scene")
        if response.get("collision_evidence", {}).get("optimizer_attachment") is not True:
            raise AlgorithmError("place plan did not optimize the carried object", code="attachment_not_applied")
        if not isinstance(response.get("phases"), dict) or set(response["phases"]) != {"approach", "place"}:
            raise AlgorithmError("place plan has incomplete phases", code="algorithm_invalid_response")
        raw = self._native_phases(response["phases"], arm)
        phases, checks = self._complete_place_candidate(
            observation, arm, scene, binding, raw, settings, cancelled=cancelled)
        selection = {"candidate_index": 0, "rule": "single_fixed_object_reference_and_grasp_rotation",
                     "retreat_checked": True, "yaw_delta_deg": 0., "candidate_count": 1}
        if progress:
            progress("planning_complete", "cuRobo 固定放置及撤离校验完成",
                     {**selection, "duration_s": time.monotonic() - started})
        return {"arm": arm, "target_id": target_id, "destination_id": destination_id,
            "observation_id": observation["observation_id"], "generation": observation["generation"],
            "scene_token": scene["scene_token"], "phases": phases, "goal": goal,
            "carrying_binding": binding, "collision_evidence": response["collision_evidence"],
            "holding_reference": deepcopy(holding["points_tool_m"]), "command_checks": checks,
            "retreat_requires_replan": False, "planning_settings": settings,
            "time_scale": settings["time_scale"], "candidate_count": 1, "candidate_failures": [],
            "selection_evidence": selection, "placement_request": deepcopy(placement_request)}
