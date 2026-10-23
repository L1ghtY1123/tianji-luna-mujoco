"""Nominal release/approach screening from measured support and own robot geometry."""
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .geometry import front_tcp_offset, rigid_matrix
from .kinematics import robot_model
from .placement import observed_object

MEASUREMENT_RESERVE_M = .005
GEOMETRY_PADDING_M = .000001
MAX_GRASP_TILT_DEG = 45.


def prepare_grasp_candidates(candidates, *, robot_base_world_m, target_reference_world_m,
                             reference_tcp=None):
    """Filter directions and choose jaw symmetry without translating GraspGen poses."""
    away = (np.asarray(target_reference_world_m, dtype=float)
            - np.asarray(robot_base_world_m, dtype=float))[:2]
    length = np.linalg.norm(away)
    if not np.isfinite(away).all() or length <= 1e-9:
        raise ValueError("grasp_requires_horizontal_robot_target_direction")
    away /= length
    minimum_down = float(np.cos(np.deg2rad(MAX_GRASP_TILT_DEG)))
    reference = rigid_matrix(reference_tcp)[:3, :3] if reference_tcp is not None else None
    shifted, accepted, evidence = [], [], []
    for candidate in candidates:
        prepared = deepcopy(candidate)
        pose = rigid_matrix(candidate["T_world_tcp"]).copy()
        downward = float(-pose[2, 2])
        away_dot = float(pose[:2, 2] @ away)
        tilt_ok = downward >= minimum_down - 1e-9
        direction_ok = away_dot >= -1e-9
        if tilt_ok and direction_ok:
            # Parallel jaws have a half-turn equivalent about their approach Z.
            # Choose in the EXECUTING arm's frame, before IK/collision planning.
            # Never flip a solved trajectory or alter the approach direction.
            if reference is not None:
                alternative = pose[:3, :3] @ np.diag([-1., -1., 1.])
                flip = np.trace(reference.T @ alternative) > np.trace(reference.T @ pose[:3, :3]) + 1e-9
                prepared['T_world_tcp_before_symmetry'] = pose.tolist()
                prepared['jaw_symmetry_half_turn'] = bool(flip)
                if flip:
                    pose[:3, :3] = alternative
            prepared["T_world_tcp"] = pose.tolist()
            accepted.append(prepared)
        shifted.append(prepared)
        evidence.append({"candidate_index": candidate["index"], "downward_dot": downward,
                         "horizontal_away_from_robot_dot": away_dot,
                         "tilt_passed": tilt_ok, "away_from_robot_passed": direction_ok})
    return shifted, accepted, {
        "axis": "actual_tool_positive_z_in_world", "maximum_tilt_deg": MAX_GRASP_TILT_DEG,
        "minimum_downward_dot": minimum_down, "robot_base_world_m": list(robot_base_world_m),
        "horizontal_robot_to_target_unit_vector": away.tolist(), "insertion_m": 0.,
        "operation_order": ["direction_filter", "arm_jaw_symmetry", "batched_ik"],
        "input_count": len(candidates), "accepted_count": len(accepted),
        "rejected_count": len(candidates) - len(accepted),
        "rejected_tilt_count": sum(not row["tilt_passed"] for row in evidence),
        "rejected_toward_robot_count": sum(not row["away_from_robot_passed"] for row in evidence),
        "candidate_evidence": evidence}


@lru_cache(maxsize=2)
def arm_wrist_reach(arm):
    """Bound the two arm segments analytically; it does not assert IK success."""
    if arm not in ("A", "B"):
        raise ValueError("demo_grasps_require_selected_arm")
    side = "L" if arm == "A" else "R"
    root = Path(__file__).resolve().parents[2]
    joints = {joint.get("name"): joint for joint in
              ET.parse(root / "assets/marvin_urdf/marvin_with_grippers.urdf").getroot().findall("joint")}
    forearm = sum(np.linalg.norm(np.fromstring(
        joints[f"Joint{index}_{side}"].find("origin").get("xyz"), sep=" "))
        for index in range(5, 8))
    model = robot_model()
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    shoulder = model.body(f"Link1_{side}").id
    upper_start = model.body(f"Link2_{side}").id
    twist = model.body(f"Link3_{side}").id
    elbow = model.body(f"Link4_{side}").id
    wrist = model.body(f"Link7_{side}").id
    tool = model.body(f"TCP_Link_{side}").id
    rotation = data.xmat[twist].reshape(3, 3)
    upper_vector = rotation.T @ (data.xpos[twist] - data.xpos[upper_start])
    elbow_offset = rotation.T @ (data.xpos[elbow] - data.xpos[twist])
    axis = model.jnt_axis[model.joint(f"Joint3_{side}").id].copy()
    axis /= np.linalg.norm(axis)
    # Rotation about Joint3 preserves axial components. The transverse vectors
    # align at their maximum sum, including the authored tiny axis deviations.
    axial = upper_vector @ axis + elbow_offset @ axis
    transverse = (np.linalg.norm(upper_vector - (upper_vector @ axis) * axis)
                  + np.linalg.norm(elbow_offset - (elbow_offset @ axis) * axis))
    upper_maximum = float(np.hypot(axial, transverse))
    shoulder_offset = float(np.linalg.norm(data.xpos[upper_start] - data.xpos[shoulder]))
    maximum = shoulder_offset + upper_maximum + forearm + GEOMETRY_PADDING_M
    wrist_offset = data.xmat[tool].reshape(3, 3).T @ (data.xpos[wrist] - data.xpos[tool])
    return {"robot_base_world_m": data.xpos[model.body("Link_Base").id].tolist(),
            "shoulder_world_m": data.xpos[shoulder].tolist(),
            "tcp_to_wrist_tool_m": wrist_offset.tolist(),
            "maximum_shoulder_wrist_distance_m": float(maximum),
            "upper_arm_maximum_m": upper_maximum, "forearm_chain_length_m": float(forearm),
            "shoulder_offset_m": shoulder_offset, "geometry_padding_m": GEOMETRY_PADDING_M,
            "geometry_source": "authored_urdf_two_segment_axis_bound_and_own_robot_fk"}


def transport_tcp_height(rotation, lowest_held_z, occupied_top_z, arm):
    """Use loose drop Z to shorten shoulder-to-wrist reach above the bin."""
    geometry = arm_wrist_reach(arm)
    minimum = float(occupied_top_z + .10 - lowest_held_z)
    wrist_offset_z = float((np.asarray(rotation) @ geometry['tcp_to_wrist_tool_m'])[2])
    shoulder_level = geometry['shoulder_world_m'][2] - wrist_offset_z
    return max(minimum, shoulder_level)


def filter_transport_reach(candidates, object_points, arm, request):
    """Screen the retained grasp attitude at the eventual above-bin TCP goal."""
    geometry = arm_wrist_reach(arm)
    region = request['transport_region']
    center = np.asarray(region['center_world_xy_m'], float)
    top = float(region['occupied_top_z_m'])
    points = np.asarray(object_points, float)
    if (center.shape != (2,) or points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(center).all() or not np.isfinite(top) or not np.isfinite(points).all()):
        raise ValueError('grasp_requires_finite_transport_region')
    shoulder = np.asarray(geometry['shoulder_world_m'])
    limit = geometry['maximum_shoulder_wrist_distance_m'] - MEASUREMENT_RESERVE_M
    accepted, evidence = [], []
    for candidate in candidates:
        pose = rigid_matrix(candidate['T_world_tcp'])
        height = transport_tcp_height(pose[:3, :3], points[:, 2].min() - pose[2, 3], top, arm)
        mount = np.r_[center - (pose[:3, :3] @ front_tcp_offset(arm))[:2], height]
        wrist = mount + pose[:3, :3] @ geometry['tcp_to_wrist_tool_m']
        distance = float(np.linalg.norm(wrist - shoulder))
        passed = distance <= limit
        evidence.append({'candidate_index': candidate['index'], 'nominal_mount_world_m': mount.tolist(),
            'nominal_transport_wrist_distance_m': distance,
            'horizontal_wrist_distance_m': float(np.linalg.norm((wrist - shoulder)[:2])),
            'reach_screening_passed': passed})
        if passed:
            accepted.append(candidate)
    return accepted, {**geometry, 'applied': True, 'screening_scope': 'nominal_transport_wrist_reach',
        'observation_id': region['observation_id'], 'reach_screening_limit_m': limit,
        'measurement_reserve_m': MEASUREMENT_RESERVE_M, 'input_count': len(candidates),
        'accepted_count': len(accepted), 'rejected_count': len(candidates) - len(accepted),
        'candidate_evidence': evidence}


def fixed_slot_reach_context(snapshot, target_id, arm, placement_request, settings):
    request = placement_request or {}
    layout = request.get("layout", {})
    cells = [cell for cell in layout.get("cells", [])
             if cell.get("slot_id", cell.get("index", -1) + 1) == request.get("slot_id")]
    if request.get("object_id") != target_id or len(cells) != 1:
        raise ValueError("grasp_requires_bound_placement")
    raw_floor = cells[0].get("floor_z_m", layout.get("floor_z_m"))
    try:
        floor = float(raw_floor)
    except (TypeError, ValueError):
        raise ValueError("grasp_requires_finite_bound_floor") from None
    if isinstance(raw_floor, bool) or not np.isfinite(floor):
        raise ValueError("grasp_requires_finite_bound_floor")
    center = np.asarray(cells[0]["center_world_xy_m"], dtype=float)
    points = np.asarray(observed_object(snapshot, target_id)["points_world_m"], dtype=float)
    if (center.shape != (2,) or not np.isfinite(center).all() or points.ndim != 2
            or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all()):
        raise ValueError("grasp_requires_finite_measured_reference")
    return {**arm_wrist_reach(arm), "arm": arm, "object_id": target_id,
            "slot_id": request["slot_id"], "layout_revision": request.get("layout_revision"),
            "target_reference_world_xy_m": center.tolist(),
            "measured_reference_world_m": points.mean(axis=0).tolist(),
            "measured_cloud_min_world_z_m": float(points[:, 2].min()),
            "bound_floor_z_m": floor,
            "required_clearance_m": settings.trajectory_collision_clearance_m,
            "endpoint_allowance_m": 0.,
            "approach_axis": settings.place_approach_axis,
            "approach_offset_m": settings.place_approach_offset_m,
            "approach_in_tool_frame": settings.place_approach_in_tool_frame,
            "planning_settings_revision": settings.revision(),
            "measurement_reserve_m": MEASUREMENT_RESERVE_M,
            "reference_source": "current_measured_points_world_centroid"}


def filter_fixed_slot_reach(candidates, context):
    accepted, evidence = [], []
    center = np.asarray(context["target_reference_world_xy_m"])
    shoulder = np.asarray(context["shoulder_world_m"])
    wrist_offset = np.asarray(context["tcp_to_wrist_tool_m"])
    # Reserve measurement uncertainty inside the bound. This intentionally
    # screens out some boundary-feasible grasps; rejection is not an IK proof.
    limit = context["maximum_shoulder_wrist_distance_m"] - context["measurement_reserve_m"]
    for candidate in candidates:
        grasp = rigid_matrix(candidate["T_world_tcp"])
        mount_xy = center - (grasp[:3, :3] @ front_tcp_offset(context['arm']))[:2]
        # Use the validated full geometry's bottom when available. A top-only
        # measurement would underestimate nominal release Z by object height.
        # This screens quality only; actual attachment/map checks still set Z.
        object_min_z = context.get("nominal_object_min_world_z_m", context["measured_cloud_min_world_z_m"])
        mount_z = (grasp[2, 3] - object_min_z
                   + context["bound_floor_z_m"] + context["required_clearance_m"]
                   + context["endpoint_allowance_m"])
        mount = np.r_[mount_xy, mount_z]
        wrist = mount + grasp[:3, :3] @ wrist_offset
        approach_offset = np.zeros(3)
        approach_offset["xyz".index(context["approach_axis"])] = context["approach_offset_m"]
        if context["approach_in_tool_frame"]:
            approach_offset = grasp[:3, :3] @ approach_offset
        approach_wrist = wrist + approach_offset
        release_distance = float(np.linalg.norm(wrist - shoulder))
        approach_distance = float(np.linalg.norm(approach_wrist - shoulder))
        passes_screen = max(release_distance, approach_distance) <= limit
        evidence.append({"candidate_index": candidate["index"],
                         "target_mount_world_xy_m": mount_xy.tolist(),
                         "target_wrist_world_xy_m": wrist[:2].tolist(),
                         "horizontal_wrist_distance_m": float(np.linalg.norm(wrist[:2] - shoulder[:2])),
                         "nominal_mount_world_m": mount.tolist(),
                         "nominal_release_wrist_world_m": wrist.tolist(),
                         "nominal_approach_wrist_world_m": approach_wrist.tolist(),
                         "nominal_release_wrist_distance_m": release_distance,
                         "nominal_approach_wrist_distance_m": approach_distance,
                         "reach_screening_passed": passes_screen})
        if passes_screen:
            accepted.append(candidate)
    return accepted, {**context, "input_count": len(candidates), "accepted_count": len(accepted),
                      "rejected_count": len(candidates) - len(accepted),
                      "candidate_evidence": evidence,
                      "horizontal_screening_limit_m": limit,
                      "reach_screening_limit_m": limit,
                      "nominal_height_source": (
                          "height_completed_minimum_and_bound_floor"
                          if context.get("nominal_object_geometry_source") == "height_completed"
                          else "measured_cloud_minimum_and_bound_floor"),
                      "screening_scope": "nominal_release_and_approach_3d_with_inward_measurement_reserve",
                      "criterion": "max_nominal_release_approach_wrist_distance_le_two_segment_bound_minus_measurement_reserve"}
