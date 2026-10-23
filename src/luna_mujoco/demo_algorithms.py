"""Demo policies composed over the existing service and execution boundaries."""

import hashlib
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .algorithms import AlgorithmError, Algorithms
from .carrying import CarryingBinding
from .demo_grasp import (
    MAX_GRASP_TILT_DEG,
    arm_wrist_reach,
    filter_fixed_slot_reach,
    filter_transport_reach,
    fixed_slot_reach_context,
    prepare_grasp_candidates,
)
from .demo_grasp_input import sample_completed_side_surface, surface_sampling_evidence
from .demo_gripper_quality import predict_gripper_compatibility
from .demo_lift import source_clearance_lift
from .demo_placement import apply_release_height, fixed_release_goal
from .demo_sampling import grasp_candidate_count
from .geometry import rigid_matrix
from .kinematics import robot_model
from .placement import observed_object
from .planning_settings import PlanningSettings
from .trajectory_postprocessing import build_execution_phases


class DemoAlgorithms(Algorithms):
    grasp_trajectory_batch_capacity = 4
    def evaluate_reachability(self, snapshot, target_id, destination_id=None, *, requested_arm=None):
        if snapshot.get("stage") != "unpacked":
            return super().evaluate_reachability(snapshot, target_id, destination_id,
                                                requested_arm=requested_arm)
        result = super().evaluate_reachability(snapshot, target_id, requested_arm=requested_arm)
        for value in result.values():
            value.update(destination_id=destination_id,
                         destination_status="pending_actual_lift_transport_goal")
        return result

    def verify_release(self, capture, binding, reference, instruction=""):
        return self.perception.verify_release(capture, binding, reference, instruction)

    @staticmethod
    def _bound_tool_fk(evidence, joints_rad, joint_names, arm, scene):
        side = "L" if arm == "A" else "R"
        names = [f"Joint{i}_{side}" for i in range(1, 8)]
        try:
            expected = {"source": "curobo_public_compute_kinematics", "frame": "world",
                        "arm": arm, "tool_frame": f"TCP_Link_{side}",
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
        if not np.isfinite(away).all() or length <= 1e-9:
            raise AlgorithmError("FK direction has no measured robot-to-target reference",
                                 code="planning_fk_evidence_invalid")
        away /= length
        position = float(np.linalg.norm(achieved[:3, 3] - nominal[:3, 3]))
        rotation = float(np.arccos(np.clip((np.trace(nominal[:3, :3].T @ achieved[:3, :3]) - 1) / 2, -1, 1)))
        downward = float(-achieved[2, 2])
        away_dot = float(achieved[:2, 2] @ away)
        reasons = []
        if position > settings["ik_position_threshold_m"] + 1e-9:
            reasons.append("nominal_position_tolerance_exceeded")
        if rotation > settings["ik_rotation_threshold_rad"] + 1e-9:
            reasons.append("nominal_rotation_tolerance_exceeded")
        if downward < np.cos(np.deg2rad(MAX_GRASP_TILT_DEG)) - 1e-9:
            reasons.append("achieved_tool_tilt_exceeded")
        if away_dot < -1e-9:
            reasons.append("achieved_tool_points_toward_robot")
        return {"accepted": not reasons, "reason": "accepted" if not reasons else ";".join(reasons),
                "position_error_m": position, "rotation_error_rad": rotation,
                "position_threshold_m": settings["ik_position_threshold_m"],
                "rotation_threshold_rad": settings["ik_rotation_threshold_rad"],
                "downward_dot": downward, "horizontal_away_from_robot_dot": away_dot,
                "maximum_tilt_deg": MAX_GRASP_TILT_DEG,
                "reference": "original_graspgen_tool_pose_after_single_1cm_insertion"}

    def _grasp_ik_candidate(self, candidate, snapshot, target_id, arm, scene, settings):
        item = candidate["ik"]
        achieved = self._bound_tool_fk(item.get("achieved_tool_pose"), item.get("joints_rad"),
                                       item.get("joint_names"), arm, scene)
        nominal = rigid_matrix(candidate["T_world_tcp"])
        evidence = {"candidate_index": candidate["candidate_index"],
                    **self._grasp_pose_validation(nominal, achieved, snapshot, target_id, arm, settings)}
        if not evidence["accepted"]:
            return None, evidence
        if snapshot.get("stage") == "unpacked":
            evidence["gripper_prediction_applied"] = False
            evidence["gripper_prediction_reason"] = "unpacked_stage_uses_actual_attachment_checks"
        else:
            prediction = self._gripper_quality(snapshot, target_id, achieved, arm, settings)
            evidence["predicted_gripper_compatibility"] = prediction
            if not prediction["accepted"]:
                evidence.update(accepted=False, reason=prediction["reason"])
                return None, evidence
        return {**candidate, "nominal_T_world_tcp": nominal.tolist(),
                "T_world_tcp": achieved.tolist(), "source_ik_fk": deepcopy(item["achieved_tool_pose"]),
                "fk_goal_validation": evidence}, evidence

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
        if snapshot.get("stage") == "unpacked":
            validation["gripper_prediction_applied"] = False
            validation["gripper_prediction_reason"] = "unpacked_stage_uses_actual_attachment_checks"
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
            raise AlgorithmError(str(error), code="grasp_lift_geometry_invalid") from error

    @staticmethod
    def _completed_object_geometry(snapshot, target):
        measured = np.asarray(target["points_world_m"])
        completed = target.get("completed_geometry_points_world_m")
        if completed is None:
            return measured
        completion = target.get("geometry_completion", {})
        completed = np.asarray(completed, dtype=float)
        if (not isinstance(completion, dict)
                or completion.get("source") != "measured_surface_and_user_height_prior"
                or completion.get("synthetic_points_are_measured") is not False
                or completion.get("observation_id") != snapshot.get("observation_id")
                or completion.get("measured_point_count") != len(measured)
                or type(completion.get("synthetic_point_count")) is not int
                or completion["synthetic_point_count"] <= 0
                or completed.ndim != 2 or completed.shape[1] != 3
                or len(completed) != len(measured) + completion["synthetic_point_count"]
                or not np.isfinite(completed).all()
                or not np.allclose(completed[:len(measured)], measured, atol=1e-7, rtol=0)):
            raise AlgorithmError("grasp_height_completion_provenance_invalid",
                                 code="grasp_height_completion_provenance_invalid")
        return completed

    @staticmethod
    def _graspgen_object_points(snapshot, target):
        geometry = DemoAlgorithms._completed_object_geometry(snapshot, target)
        if target.get("completed_geometry_points_world_m") is None:
            return geometry
        try:
            return sample_completed_side_surface(target["points_world_m"], geometry)
        except ValueError as error:
            raise AlgorithmError(str(error), code="graspgen_surface_sampling_invalid") from error

    def generate_grasps(self, snapshot, target_id, *, arm=None, placement_request=None,
                        planning_settings=None, sampling_round=1, progress=None, cancelled=None):
        self._check_cancelled(cancelled)
        candidate_count = grasp_candidate_count(sampling_round)
        if arm not in ("A", "B"):
            raise ValueError("demo_grasps_require_selected_arm")
        settings = PlanningSettings.model_validate(planning_settings or {})
        target = observed_object(snapshot, target_id)
        object_geometry = self._completed_object_geometry(snapshot, target)
        completed = target.get("completed_geometry_points_world_m")
        input_source = "measured" if completed is None else "height_completed"
        unpacked = snapshot.get("stage") == "unpacked"
        if unpacked:
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
        result["raw_candidate_count"] = sum(batch["candidate_count"] for batch in result["batches"])
        provenance = {} if completed is None else deepcopy(target["geometry_completion"])
        graspgen_request = result.pop("graspgen_input", None)
        input_sampling = None
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
            if completed is not None:
                input_sampling = surface_sampling_evidence(len(object_geometry), len(input_points))
            result["cloud"] = {
                "id": hashlib.sha256(input_points.tobytes()).hexdigest(),
                "frame": f"cloud:{target_id}", "hash_frame": f"cloud:{target_id}",
                "point_count": len(input_points),
                "source": ("rgbd_visible_surface" if completed is None else
                           "registered_rgbd_surface_with_explicit_height_completion"),
                "measured_point_count": len(target["points_world_m"]),
                "synthetic_point_count": len(input_points) - len(target["points_world_m"]),
                "completion_provenance": deepcopy(provenance),
                "graspgen_input_sampling": input_sampling}
        all_generated = result.pop('all_generated_candidates', result['candidates'])
        candidates, accepted, evidence = prepare_grasp_candidates(all_generated,
            robot_base_world_m=reach_context["robot_base_world_m"],
            target_reference_world_m=reach_context["measured_reference_world_m"],
            reference_tcp=(snapshot.get('arms', {}).get(arm, {}).get('tcp')
                           if unpacked else None))
        replay = {"stage": "before_ik", "target_id": target_id, "sampling_round": sampling_round,
            "all_generated_candidates": all_generated, "filter_all_generated": True,
            "requested_candidate_count": candidate_count, "raw_candidate_count": result["raw_candidate_count"],
            "measured_points_world_m": np.asarray(target["points_world_m"]).copy(),
            "T_world_cloud": result.get("T_world_cloud"), "candidates_before_ik": candidates,
            "observation_id": snapshot.get("observation_id"), "arm": arm,
            "graspgen_request": graspgen_request, "graspgen_input_source": input_source,
            "graspgen_input_sampling": input_sampling,
            "completion_provenance": provenance,
            "graspgen_input_cloud": deepcopy(result.get("cloud")),
            "measured_cloud_reference": deepcopy(result.get("measured_cloud_reference")),
            "direction_filter": evidence}
        if completed is not None:
            replay["completed_geometry_points_world_m"] = np.asarray(completed).copy()
        if not accepted:
            raise AlgorithmError("no grasp satisfies the 45 degree downward and away-from-robot rules",
                code="grasp_no_downward_candidate", details={**evidence, "candidate_replay": replay})
        if unpacked:
            accepted, reach_evidence = filter_transport_reach(accepted, object_geometry, arm, placement_request)
        else:
            accepted, reach_evidence = filter_fixed_slot_reach(accepted, reach_context)
        replay['pre_budget_eligible_indices'] = [candidate['index'] for candidate in accepted]
        accepted = accepted[:candidate_count]
        replay['pre_ik_eligible_indices'] = [candidate['index'] for candidate in accepted]
        replay['reach_filter'] = deepcopy(reach_evidence)
        if not accepted and not unpacked:
            raise AlgorithmError("no downward grasp passes the fixed-slot nominal 3D reach screen",
                                 code="grasp_no_fixed_slot_reachable_candidate",
                                 details={"downward_filter": evidence,
                                          "fixed_slot_reach_filter": reach_evidence,
                                          "candidate_replay": replay})
        return {**result, "candidates": accepted, "candidate_count": len(accepted),
                "downward_filter": evidence, "fixed_slot_reach_filter": reach_evidence,
                "candidate_replay": replay}

    def plan_attached_pose(self, observation, target_id, arm, scene, holding, goal, *,
                           planning_settings=None, cancelled=None):
        """One ordinary pose plan with the existing held body and current world."""
        self._check_cancelled(cancelled)
        self._binding(observation, target_id, scene)
        if not holding.get("verified") or not holding.get("attached"):
            raise AlgorithmError("transport requires a verified attached object", code="transport_requires_attachment")
        if holding.get("arm") != arm or holding.get("instance_id") != target_id:
            raise AlgorithmError("held object differs from transport request", code="transport_holding_mismatch")
        settings = PlanningSettings.model_validate(planning_settings or {}).model_dump()
        binding = self._held_binding(observation, target_id, arm, scene, holding)
        CarryingBinding.from_mapping(binding, scene=scene, arm=arm)
        prepared = holding.get("attachment_evidence", {})
        if prepared.get("scene_token", scene["scene_token"]) != scene["scene_token"]:
            raise AlgorithmError("attachment belongs to another scene", code="planning_stale_scene")
        goal = rigid_matrix(goal)
        position = goal[:3, 3].tolist()
        workspace = self.workspace_lookup(arm, [position])
        if (not isinstance(workspace, list) or len(workspace) != 1 or not isinstance(workspace[0], dict)
                or workspace[0].get("position_world_m") != position
                or workspace[0].get("reachable") is not True):
            raise AlgorithmError("actual transport goal is outside or lacks a valid workspace result",
                                 code="transport_goal_workspace_rejected", details={"workspace": workspace})
        self._check_cancelled(cancelled)
        payload = self._motion_request(observation, arm, scene, goal, settings)
        response = self.request("/tools/plan_carried_pose", {**payload, "carrying_binding": binding})
        self._check_cancelled(cancelled)
        if response.get("scene_token") != scene["scene_token"]:
            raise AlgorithmError("transport plan returned a stale scene", code="planning_stale_scene")
        if (response.get("attached") is not True
                or response.get("collision_evidence", {}).get("optimizer_attachment") is not True):
            raise AlgorithmError("transport plan did not optimize the carried body", code="attachment_not_applied")
        raw = self._native_phases({"transport": response.get("trajectory", {})}, arm)
        command = build_execution_phases(raw, settings["time_scale"])["transport"]
        check = self.validate_execution(observation, arm, scene, "transport", command, binding,
                                        planning_settings=settings)
        self._check_cancelled(cancelled)
        return {"arm": arm, "target_id": target_id, "scene_token": scene["scene_token"],
            "observation_id": observation["observation_id"], "generation": observation["generation"],
            "goal": goal.tolist(), "command": command, "command_check": check,
            "raw_trajectory": raw["transport"], "carrying_binding": binding,
            "goal_workspace": workspace[0],
            "collision_evidence": response["collision_evidence"], "planning_settings": settings,
            "time_scale": settings["time_scale"]}

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

    def publish_local_object(self, scene, capture, target_id, work_area, appearance):
        from .local_object_obstacle import measure_object_obstacle

        if capture['generation'] != scene['generation']:
            raise ValueError('local_object_capture_stale')
        cameras = ['3', '2', '1'] if appearance == 'unpacked' else ['3']
        for camera_id in cameras:
            try:
                measured = measure_object_obstacle(capture, self.perception.detector, work_area,
                                                    appearance, camera_id=camera_id)
                break
            except ValueError as error:
                if str(error) != 'local_object_depth_missing' or camera_id == cameras[-1]:
                    raise
        result = self.request('/perception/update', {
            'scene_token': scene['scene_token'], 'generation': scene['generation'],
            'operation': 'local_object', 'exclusion_state': 'released', 'target_id': target_id,
            'replaces_entity_id': scene['target_id'], 'camera_id': measured['camera_id'],
            'after_capture_id': capture['observation_id'], 'source': measured['source'],
            'bounds': measured['bounds']})
        return {'scene': {**scene, **result}, 'measurement': measured}

    def omit_unpack_obstacle(self, scene, target_id, action_id):
        """User-authorized exception for the single unpack visibility return."""
        result = self.request('/perception/update', {
            'scene_token': scene['scene_token'], 'generation': scene['generation'],
            'operation': 'unpack_view_motion', 'exclusion_state': 'released',
            'target_id': target_id, 'action_id': action_id,
            'source': 'user_authorized_unpack_visibility_fallback'})
        return {**scene, **result}

    def publish_action_envelope(self, scene, target_id, bounds, action_id, source="mock_vla"):
        """Retain the measured map and add explicitly action-derived geometry."""
        if not isinstance(bounds, dict):
            raise TypeError("invalid action envelope bounds")
        lower = np.asarray(bounds.get("minimum_world_m"), dtype=float)
        upper = np.asarray(bounds.get("maximum_world_m"), dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
                or not (upper > lower).all() or not isinstance(target_id, str) or not target_id
                or not isinstance(action_id, str) or not action_id
                or source not in {"mock_vla", "verified_drop_action_bounds"}):
            raise ValueError("invalid action envelope bounds or provenance")
        result = self.request("/perception/update", {
            "scene_token": scene["scene_token"], "generation": scene["generation"],
            "target_id": target_id, "operation": "action_envelope", "exclusion_state": "released",
            "bounds": {"minimum_world_m": lower.tolist(), "maximum_world_m": upper.tolist()},
            "action_id": action_id, "source": source})
        if not isinstance(result.get("scene_token"), str) or not result["scene_token"]:
            raise AlgorithmError("action envelope did not publish a scene token")
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
        """Current closed and fully open gripper endpoint spheres, in mount frame."""
        from .carrying import configure_gripper_feedback

        root = Path(__file__).resolve().parents[2]
        urdf_path = root / "assets/marvin_urdf/marvin_with_grippers.urdf"
        config = yaml.safe_load((root / "resources/curobo/marvin_gui_dual.yml").read_text(encoding="utf-8"))
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
                    center_tool = tool_rotation.T @ (center_world - data.xpos[tool_id])
                    radius = float(sphere["radius"]) + buffer
                    spheres.append([*center_tool.tolist(), radius])
        return spheres

    def plan_place(self, observation, target_id, arm, scene, holding, destination_id, *,
                   planning_settings=None, placement_request=None, progress=None, cancelled=None):
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
            prepared["attachment_spheres_tool_m"], self.gripper_release_spheres(observation, arm),
            clearance_m=settings["trajectory_collision_clearance_m"], observation_id=observation["observation_id"])
        binding = self._held_binding(observation, target_id, arm, scene, holding)
        payload = self._motion_request(observation, arm, scene, goal["T_world_tcp"], settings)
        payload.update(carrying_binding=binding, approach_axis=settings["place_approach_axis"],
            approach_offset_m=settings["place_approach_offset_m"],
            approach_in_tool_frame=settings["place_approach_in_tool_frame"],
            lift_axis=settings["place_retreat_axis"], lift_offset_m=settings["place_retreat_offset_m"],
            lift_in_tool_frame=settings["place_retreat_in_tool_frame"],
            trajectory_collision_clearance_m=settings["trajectory_collision_clearance_m"],
            optimizer_collision_activation_distance_m=settings["optimizer_collision_activation_distance_m"])
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
            goal = apply_release_height(goal, resolution, scene_token=scene["scene_token"])
        except ValueError as error:
            raise AlgorithmError(str(error), code="algorithm_invalid_response") from error
        payload["goal_position_m"] = np.asarray(goal["T_world_tcp"])[:3, 3].tolist()
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
