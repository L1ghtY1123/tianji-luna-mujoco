"""Nominal release/approach screening from measured support and own robot geometry."""
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .geometry import front_tcp_offset, rigid_matrix
from .kinematics import robot_model
from .planning_settings import PlanningSettings

MEASUREMENT_RESERVE_M = .005
GEOMETRY_PADDING_M = .000001


def grasp_direction_evidence(axis_world, maximum_lateral_angle_deg, *, applied=True):
    axis = np.asarray(axis_world, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1., atol=1e-5):
        raise ValueError('grasp_requires_unit_tool_z_axis')
    if not np.isfinite(maximum_lateral_angle_deg) or not 0 <= maximum_lateral_angle_deg <= 45:
        raise ValueError('invalid_grasp_lateral_angle')
    axis = axis / np.linalg.norm(axis)
    lateral = float(np.rad2deg(np.arcsin(np.clip(abs(axis[1]), 0., 1.))))
    sector = bool(axis[0] >= -1e-6 and axis[2] <= 1e-6)
    side = lateral <= maximum_lateral_angle_deg + 1e-6
    return {'applied': applied, 'tool_z_world': axis.tolist(), 'forward_dot': float(axis[0]),
            'downward_dot': float(-axis[2]), 'lateral_angle_deg': lateral,
            'maximum_lateral_angle_deg': maximum_lateral_angle_deg,
            'forward_downward_passed': sector, 'lateral_passed': side,
            'accepted': not applied or (sector and side)}


def prepare_grasp_candidates(candidates, *, robot_base_world_m, target_reference_world_m,
                             reference_tcp=None, planning_settings=None):
    """Use the converted front TCP, filter directions, then choose jaw symmetry."""
    settings = PlanningSettings.model_validate(planning_settings or {})
    away = (np.asarray(target_reference_world_m, dtype=float)
            - np.asarray(robot_base_world_m, dtype=float))[:2]
    length = np.linalg.norm(away)
    if not np.isfinite(away).all() or (length <= 1e-9 and settings.grasp_require_away_from_robot):
        raise ValueError("grasp_requires_horizontal_robot_target_direction")
    away = away / length if length > 1e-9 else np.zeros(2)
    minimum_down = float(np.cos(np.deg2rad(settings.grasp_maximum_tilt_deg)))
    reference = rigid_matrix(reference_tcp)[:3, :3] if reference_tcp is not None else None
    shifted, accepted, evidence = [], [], []
    for candidate in candidates:
        prepared = deepcopy(candidate)
        # Reprocessing starts at the original front TCP. Only the fixed CAD
        # library permits a configured depth adjustment; GraspGen never inserts.
        pose = rigid_matrix(candidate.get("T_world_tcp_before_insertion",
                                          candidate["T_world_tcp"])).copy()
        prepared["T_world_tcp_before_insertion"] = pose.tolist()
        offline = candidate.get('source') == 'cad_offline_library'
        insertion_m = settings.cad_grasp_insertion_m if offline else 0.
        insertion = insertion_m * pose[:3, 2]
        pose[:3, 3] += insertion
        prepared.update(insertion_applied=bool(insertion_m), insertion_m=insertion_m,
                        insertion_offset_world_m=insertion.tolist(), T_world_tcp=pose.tolist())
        downward = float(-pose[2, 2])
        away_dot = float(pose[:2, 2] @ away)
        tilt_ok = downward >= minimum_down - 1e-9
        direction_ok = not settings.grasp_require_away_from_robot or away_dot >= -1e-9
        sector = grasp_direction_evidence(pose[:3, 2], settings.grasp_maximum_lateral_angle_deg, applied=not offline)
        accepted_direction = tilt_ok and direction_ok and sector['accepted']
        if accepted_direction:
            # Parallel jaws have a half-turn equivalent about their approach Z.
            # Choose in the EXECUTING arm's frame, before IK/collision planning.
            # Never flip a solved trajectory or alter the approach direction.
            if reference is not None and not offline:
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
                         "tilt_passed": tilt_ok, "away_from_robot_passed": direction_ok,
                         'sector': sector, 'accepted': accepted_direction})
    return shifted, accepted, {
        "axis": "actual_tool_positive_z_in_world", "maximum_tilt_deg": settings.grasp_maximum_tilt_deg,
        "require_away_from_robot": settings.grasp_require_away_from_robot,
        'graspgen_sector': 'world_positive_x_forward_to_negative_z_down',
        'maximum_lateral_angle_deg': settings.grasp_maximum_lateral_angle_deg,
        "minimum_downward_dot": minimum_down, "robot_base_world_m": list(robot_base_world_m),
        "horizontal_robot_to_target_unit_vector": away.tolist(), "insertion_m": 0.,
        "cad_insertion_m": settings.cad_grasp_insertion_m,
        "operation_order": ["arm_front_tcp_conversion", "cad_depth_only", "direction_filter",
                            "arm_jaw_symmetry", "official_plan_grasp_goalset"],
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
    wrist_offset = data.xmat[tool].reshape(3, 3).T @ (data.xpos[wrist] - data.xpos[tool]) - front_tcp_offset(arm)
    return {"robot_base_world_m": data.xpos[model.body("Link_Base").id].tolist(),
            "shoulder_world_m": data.xpos[shoulder].tolist(),
            "tcp_to_wrist_tool_m": wrist_offset.tolist(),
            "maximum_shoulder_wrist_distance_m": float(maximum),
            "upper_arm_maximum_m": upper_maximum, "forearm_chain_length_m": float(forearm),
            "shoulder_offset_m": shoulder_offset, "geometry_padding_m": GEOMETRY_PADDING_M,
            "geometry_source": "authored_urdf_two_segment_axis_bound_and_own_robot_fk"}
