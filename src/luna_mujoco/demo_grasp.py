"""Nominal release/approach screening from measured support and own robot geometry."""
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import front_tcp_offset, rigid_matrix
from .kinematics import link_poses, zero_configuration
from .planning_settings import PlanningSettings

MEASUREMENT_RESERVE_M = .005
GEOMETRY_PADDING_M = .000001


# Open Link3 inner surfaces in the RAG100-050 front TCP frame, metres.
# The left/right meshes are slightly asymmetric. Both arms use this pair.
RAG_PAD_X = (-.050105, .049481)
RAG_PAD_HALF_Y = .008
RAG_PAD_Z = (-.0466, -.0041)


def rag_contact_geometry(pose, points, cell_size):
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    half_cell = .5 * cell_size
    band = ((np.abs(local[:, 1]) <= RAG_PAD_HALF_Y + half_cell)
            & (local[:, 2] >= RAG_PAD_Z[0] - half_cell)
            & (local[:, 2] <= RAG_PAD_Z[1] + half_cell))
    faces = local[band, 0]
    if not len(faces):
        return {'accepted': False, 'reason': 'no_surface_at_rag_pads'}
    lo, hi = float(faces.min()), float(faces.max())
    margin = min(lo - RAG_PAD_X[0], RAG_PAD_X[1] - hi) - half_cell
    # Two opposed observed sides must actually straddle the intended jaw centre.
    accepted = margin >= 0 and lo < -half_cell and hi > half_cell
    projected = float(np.ptp(local[:, 0]))
    # Gravity loads a grasp through the horizontal offset from the measured object centre.
    # Estimate the pad contact midpoint at the depth occupied by the TSDF surface.
    middle = np.array([(lo + hi) / 2, np.median(local[band, 1]), np.median(local[band, 2])])
    contact_xy = (pose[:3, 3] + pose[:3, :3] @ middle)[:2]
    centre_xy = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
    offset = float(np.linalg.norm(contact_xy - centre_xy))
    return {'accepted': bool(accepted), 'reason': 'pad_span_fit' if accepted else
            'rag_opening_exceeded' if margin < 0 else 'opposed_contact_unobserved',
            'pad_band_points': len(faces), 'opening_margin_m': margin,
            'centre_offset_m': offset, 'centre_band': int(offset / max(.010, 2 * cell_size)),
            'pad_span_m': hi-lo, 'projected_span_m': projected,
            'width_band': int(np.ceil(projected / .010 - 1e-4))}


def prepare_grasp_candidates(candidates, *, reference_tcp=None, planning_settings=None,
                             target_surface=None, cell_size=None, prefer_top_entry=False):
    """Screen the actual arm-front approach in Link_Base before ranking grasps."""
    settings = PlanningSettings.model_validate(planning_settings or {})
    reference = rigid_matrix(reference_tcp)[:3, :3] if reference_tcp is not None else None
    prepared_candidates, rejected = [], []
    for candidate in candidates:
        prepared = deepcopy(candidate)
        # Reprocessing always starts at the unadjusted arm-front TCP.
        pose = rigid_matrix(candidate.get("T_world_tcp_before_insertion",
                                          candidate["T_world_tcp"])).copy()
        prepared["T_world_tcp_before_insertion"] = pose.tolist()
        offline = candidate.get('source') == 'cad_offline_library'
        if not offline:
            forward, lateral, vertical = pose[:3, 2]
            # Pitch spans forward through down. Limit departure from that plane independently.
            top_entry = -vertical >= forward
            if forward < -1e-6 or vertical > 1e-6 or abs(lateral) > np.sqrt(.5) + 1e-6:
                rejected.append({'index': candidate['index'], 'reason': 'approach_outside_forward_down',
                                 'approach_z_base': pose[:3, 2].tolist()})
                continue
        insertion_m = settings.cad_grasp_insertion_m if offline else settings.rag_grasp_insertion_m
        insertion = insertion_m * pose[:3, 2]
        pose[:3, 3] += insertion
        prepared.update(insertion_applied=bool(insertion_m), insertion_m=insertion_m,
                        insertion_offset_world_m=insertion.tolist(), T_world_tcp=pose.tolist())
        # Parallel jaws have a half-turn equivalent about their approach Z.
        # Choose before IK/collision planning without changing that direction.
        if reference is not None and not offline:
            alternative = pose[:3, :3] @ np.diag([-1., -1., 1.])
            flip = np.trace(reference.T @ alternative) > np.trace(reference.T @ pose[:3, :3]) + 1e-9
            prepared['T_world_tcp_before_symmetry'] = pose.tolist()
            prepared['jaw_symmetry_half_turn'] = bool(flip)
            if flip:
                pose[:3, :3] = alternative
        prepared["T_world_tcp"] = pose.tolist()
        if not offline:
            if target_surface is None or not cell_size or cell_size <= 0:
                raise ValueError('rag_contact_requires_current_target_surface')
            contact = rag_contact_geometry(pose, target_surface, cell_size)
            if prefer_top_entry:
                contact['top_entry'] = bool(top_entry)
            prepared.update(execution_gripper='tianji_rag100_050', rag_contact=contact)
            if not contact['accepted']:
                rejected.append({'index': candidate['index'], **contact})
                continue
        prepared_candidates.append(prepared)
    if prepared_candidates and 'rag_contact' in prepared_candidates[0]:
        prepared_candidates.sort(key=lambda c: (not c['rag_contact'].get('top_entry', False),
                                                c['rag_contact']['width_band'], c['rag_contact']['centre_band'],
                                                -c['score'], c['index']))
    return prepared_candidates, {
        "insertion_m": (settings.cad_grasp_insertion_m if candidates and candidates[0].get("source") == "cad_offline_library"
                        else settings.rag_grasp_insertion_m), "rejected_candidates": rejected,
        "cad_insertion_m": settings.cad_grasp_insertion_m,
        "prefer_top_entry": prefer_top_entry,
        "contact_profile": "tianji-rag100-050-pad-fit-centering-v2",
        "operation_order": ["arm_front_tcp_conversion", "forward_down_filter_for_tsdf", "local_z_insertion", "arm_jaw_symmetry",
                            "rag_pad_fit_for_tsdf", "official_plan_grasp_goalset"],
        "input_count": len(candidates), "prepared_count": len(prepared_candidates)}


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
    names = [f"Link{i}_{side}" for i in (1, 2, 3, 4, 7)] + [f"TCP_Link_{side}", "Link_Base"]
    poses = link_poses(zero_configuration(), names)
    shoulder, upper_start, twist, elbow, wrist = [f"Link{i}_{side}" for i in (1, 2, 3, 4, 7)]
    tool = f"TCP_Link_{side}"
    rotation = poses[twist][:3, :3]
    upper_vector = rotation.T @ (poses[twist][:3, 3] - poses[upper_start][:3, 3])
    elbow_offset = rotation.T @ (poses[elbow][:3, 3] - poses[twist][:3, 3])
    axis = np.fromstring(joints[f"Joint3_{side}"].find("axis").get("xyz"), sep=" ")
    axis /= np.linalg.norm(axis)
    # Rotation about Joint3 preserves axial components. The transverse vectors
    # align at their maximum sum, including the authored tiny axis deviations.
    axial = upper_vector @ axis + elbow_offset @ axis
    transverse = (np.linalg.norm(upper_vector - (upper_vector @ axis) * axis)
                  + np.linalg.norm(elbow_offset - (elbow_offset @ axis) * axis))
    upper_maximum = float(np.hypot(axial, transverse))
    shoulder_offset = float(np.linalg.norm(poses[upper_start][:3, 3] - poses[shoulder][:3, 3]))
    maximum = shoulder_offset + upper_maximum + forearm + GEOMETRY_PADDING_M
    wrist_offset = poses[tool][:3, :3].T @ (poses[wrist][:3, 3] - poses[tool][:3, 3]) - front_tcp_offset(arm)
    return {"robot_base_world_m": poses["Link_Base"][:3, 3].tolist(),
            "shoulder_world_m": poses[shoulder][:3, 3].tolist(),
            "tcp_to_wrist_tool_m": wrist_offset.tolist(),
            "maximum_shoulder_wrist_distance_m": float(maximum),
            "upper_arm_maximum_m": upper_maximum, "forearm_chain_length_m": float(forearm),
            "shoulder_offset_m": shoulder_offset, "geometry_padding_m": GEOMETRY_PADDING_M,
            "geometry_source": "authored_urdf_two_segment_axis_bound_and_own_robot_fk"}
