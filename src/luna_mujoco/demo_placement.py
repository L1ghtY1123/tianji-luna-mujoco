"""One measured-object release goal with bounded measured-world height resolution."""

import hashlib
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .geometry import front_tcp_offset, rigid_matrix
from .release_clearance import CLEARANCE_COST_EPSILON

RELEASE_HEIGHT_STEP_M = .001
MAX_RELEASE_HEIGHT_INCREASE_M = .1
RELEASE_PLANNING_RESERVE_M = .0001
RELEASE_PLANNING_RESERVE_REASON = "observed_loaded_position_servo_tracking_error"


class ReleaseHeightError(ValueError):
    code = "release_height_no_valid_goal"

    def __init__(self, details):
        super().__init__(self.code)
        self.details = details


def sphere_array(value):
    spheres = np.asarray(value, dtype=float)
    if (spheres.ndim != 2 or spheres.shape[1] != 4 or not len(spheres)
            or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)):
        raise ValueError("release_requires_finite_positive_spheres")
    return spheres


def lowest_sphere_z(spheres, rotation):
    return float(np.min((spheres[:, :3] @ rotation.T)[:, 2] - spheres[:, 3]))


def release_geometry_digest(attachment, gripper):
    return hashlib.sha256(np.concatenate((attachment, gripper)).astype("<f8").tobytes()).hexdigest()


def resolve_release_height(attachment_spheres, gripper_spheres, request, query_world_cost, *, scene_token):
    """Choose the first clear 1 mm Z sample using the unchanged hard world checker."""
    attachment, gripper = sphere_array(attachment_spheres), sphere_array(gripper_spheres)
    position = np.asarray(request.goal_position_m, dtype=float)
    quaternion = np.asarray(request.goal_quaternion_wxyz, dtype=float)
    if (position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0):
        raise ValueError("release_requires_finite_world_goal")
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    spheres = np.concatenate((attachment, gripper))
    world = spheres.copy()
    world[:, :3] = spheres[:, :3] @ rotation.T + position
    # The world checker applies the configured hard clearance directly.
    evidence = {"scene_token": scene_token, "initial_goal_position_m": position.tolist(),
        "goal_quaternion_wxyz": quaternion.tolist(), "resolution_m": RELEASE_HEIGHT_STEP_M,
        "maximum_height_increase_m": MAX_RELEASE_HEIGHT_INCREASE_M,
        "endpoint_margin_m": 0.,
        "required_clearance_m": float(request.trajectory_collision_clearance_m),
        "attachment_sphere_count": len(attachment), "gripper_sphere_count": len(gripper),
        "geometry_sha256": release_geometry_digest(attachment, gripper),
        "collision_source": "configured_hard_world_checker",
        "distance_reporting": "activation_cost_only", "tested_heights": []}
    count = round(MAX_RELEASE_HEIGHT_INCREASE_M / RELEASE_HEIGHT_STEP_M) + 1
    for index in range(count):
        increase = index * RELEASE_HEIGHT_STEP_M
        query = world.copy()
        query[:, 2] += increase
        costs = np.asarray(query_world_cost(query[None, None]), dtype=float)
        if (costs.shape != (1, 1, len(spheres)) or not np.isfinite(costs).all() or np.any(costs < 0)):
            raise ValueError("release_world_cost_requires_finite_nonnegative_per_sphere_values")
        conflicts = np.flatnonzero(costs[0, 0] > CLEARANCE_COST_EPSILON)
        z = float(position[2] + increase)
        evidence["tested_heights"].append({"z_m": z, "conflicting_sphere_indices": conflicts.tolist(),
                                           "maximum_world_collision_cost": float(costs.max())})
        if not len(conflicts):
            return {**evidence, "selected_goal_position_m": [*position[:2].tolist(), z],
                "height_increase_m": increase, "status": "first_clear_height_sample",
                "last_conflicting_z_m": None if index == 0 else float(z - RELEASE_HEIGHT_STEP_M),
                "final_world_collision_costs": costs[0, 0].tolist()}
    raise ReleaseHeightError(evidence)


def valid_height_resolution(reference, expected, minimum_z, *, scene_token):
    """Bind the reported first feasible sample to the exact goal and envelopes."""
    proof = reference.get("height_resolution", {})
    try:
        attachment = sphere_array(reference["attachment_spheres_tool_m"])
        gripper = sphere_array(reference["gripper_spheres_tool_m"])
        initial = np.asarray(proof["initial_goal_position_m"], dtype=float)
        selected = np.asarray(proof["selected_goal_position_m"], dtype=float)
        costs = np.asarray(proof["final_world_collision_costs"], dtype=float)
        rotation = Rotation.from_quat(np.asarray(proof["goal_quaternion_wxyz"])[[1, 2, 3, 0]]).as_matrix()
        increase = float(proof["height_increase_m"])
        samples = proof["tested_heights"]
        index = round(increase / RELEASE_HEIGHT_STEP_M)
        return bool(proof["scene_token"] == scene_token
            and reference.get("planning_reserve_m") == RELEASE_PLANNING_RESERVE_M
            and reference.get("planning_reserve_reason") == RELEASE_PLANNING_RESERVE_REASON
            and proof["status"] == "first_clear_height_sample"
            and proof["collision_source"] == "configured_hard_world_checker"
            and proof["geometry_sha256"] == release_geometry_digest(attachment, gripper)
            and proof["attachment_sphere_count"] == len(attachment)
            and proof["gripper_sphere_count"] == len(gripper)
            and proof["endpoint_margin_m"] == 0.
            and proof["resolution_m"] == RELEASE_HEIGHT_STEP_M
            and proof["maximum_height_increase_m"] == MAX_RELEASE_HEIGHT_INCREASE_M
            and proof["required_clearance_m"] == reference["required_clearance_m"]
            and 0 <= increase <= MAX_RELEASE_HEIGHT_INCREASE_M
            and abs(increase - index * RELEASE_HEIGHT_STEP_M) < 1e-9
            and initial.shape == selected.shape == (3,)
            and np.allclose(initial, [*expected[:2, 3], minimum_z], atol=1e-9, rtol=0)
            and np.allclose(selected, expected[:3, 3], atol=1e-9, rtol=0)
            and abs(selected[2] - minimum_z - increase) < 1e-9
            and np.allclose(rotation, expected[:3, :3], atol=1e-9, rtol=0)
            and len(samples) == index + 1
            and all(abs(item["z_m"] - minimum_z - i * RELEASE_HEIGHT_STEP_M) < 1e-9
                    and bool(item["conflicting_sphere_indices"]) == (i < index)
                    for i, item in enumerate(samples))
            and (proof["last_conflicting_z_m"] is None if index == 0 else
                 abs(proof["last_conflicting_z_m"] - selected[2] + RELEASE_HEIGHT_STEP_M) < 1e-9)
            and costs.shape == (len(attachment) + len(gripper),)
            and np.isfinite(costs).all() and np.all((costs >= 0) & (costs <= CLEARANCE_COST_EPSILON)))
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False


def apply_release_height(goal, proof, *, scene_token):
    result = deepcopy(goal)
    transform = rigid_matrix(result["T_world_tcp"]).copy()
    minimum_z = transform[2, 3]
    try:
        if not np.allclose(proof["initial_goal_position_m"], transform[:3, 3], atol=1e-9, rtol=0):
            raise ValueError("reported analytic goal differs from the submitted goal")
        transform[:3, 3] = proof["selected_goal_position_m"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("release_height_invalid_response") from error
    reference = result["release_reference"]
    reference["height_resolution"] = deepcopy(proof)
    if not valid_height_resolution(reference, transform, minimum_z, scene_token=scene_token):
        raise ValueError("release_height_invalid_response")
    increase = transform[2, 3] - minimum_z
    for field in ("attachment_floor_gap_m", "gripper_floor_gap_m", "release_height_m", "requested_release_height_m"):
        reference[field] += increase
    reference["height_source"] = "first_clear_measured_world_z_above_analytic_minimum"
    if increase > 0:
        reference["height_limiting_geometry"] = "measured_world"
    result["scene_token"] = scene_token
    result["T_world_tcp"] = transform.tolist()
    result["object_reference_world_m"] = (
        transform[:3, :3] @ np.asarray(reference["tool_reference_point_m"]) + transform[:3, 3]).tolist()
    return result


def fixed_release_goal(destination, holding, placement_request, attachment_spheres, gripper_spheres,
                       *, clearance_m, observation_id):
    request = placement_request or {}
    layout = request.get("layout")
    if not isinstance(layout, dict):
        raise TypeError("placement_requires_bound_layout")
    slot_id = request.get("slot_id")
    cells = [cell for cell in layout["cells"] if cell.get("slot_id", cell.get("index", -1) + 1) == slot_id]
    if len(cells) != 1 or request.get("object_id") != holding["instance_id"]:
        raise ValueError("placement_slot_not_bound")
    cell = cells[0]
    support = destination["placement_reference"]
    if support.get("observation_id", observation_id) != observation_id:
        raise ValueError("destination_support_reference_stale")
    floor = float(cell.get("floor_z_m", layout.get("floor_z_m", support["floor_z_m"])))
    if not np.isfinite(floor) or abs(floor - float(support["floor_z_m"])) > .005:
        raise ValueError("placement_support_movement_unresolved")
    center = np.asarray(cell["center_world_xy_m"], dtype=float)
    points = np.asarray(holding["points_tool_m"], dtype=float)
    if (center.shape != (2,) or not np.isfinite(center).all()
            or points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or not np.isfinite(clearance_m) or clearance_m < 0):
        raise ValueError("placement_requires_finite_measured_reference")
    attachment = sphere_array(attachment_spheres)
    gripper = sphere_array(gripper_spheres)
    grasp = rigid_matrix(holding["grasp_support_reference"]["T_world_tcp_at_grasp"])
    rotation = grasp[:3, :3]
    object_reference = front_tcp_offset(holding['arm'])
    goal = grasp.copy()
    goal[:2, 3] = center - (rotation @ object_reference)[:2]
    for polygon in (cell["polygon_world_xy_m"], layout["region_polygon_world_xy_m"]):
        vertices = np.asarray(polygon, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all():
            raise ValueError("placement_requires_measured_region_polygon")
        hull = ConvexHull(vertices)
        if np.any(center @ hull.equations[:, :2].T + hull.equations[:, 2] > 1e-9):
            raise ValueError("placement_front_outside_assigned_region")
    attachment_bottom = lowest_sphere_z(attachment, rotation)
    gripper_bottom = lowest_sphere_z(gripper, rotation)
    # Explicit commanded-height reserve for measured loaded servo error; this
    # does not alter the hard world or actual-feedback clearance requirement.
    planned_gap = float(clearance_m) + RELEASE_PLANNING_RESERVE_M
    attachment_min_z = floor + planned_gap - attachment_bottom
    gripper_min_z = floor + planned_gap - gripper_bottom
    goal[2, 3] = max(attachment_min_z, gripper_min_z)
    attachment_gap = goal[2, 3] + attachment_bottom - floor
    gripper_gap = goal[2, 3] + gripper_bottom - floor
    reference = {
        "position_constraint": "finger_front_world_xy",
        "visible_reference_tool_m": object_reference.tolist(),
        "tool_reference_point_m": object_reference.tolist(),
        "visible_reference_source": "calibrated_finger_front",
        "target_reference_world_xy_m": center.tolist(),
        "position_world_m": [*center.tolist(), floor],
        "floor_z_m": floor, "floor_source": support.get("source"),
        "T_world_tcp_at_grasp": grasp.tolist(),
        "required_clearance_m": float(clearance_m),
        "planning_reserve_m": RELEASE_PLANNING_RESERVE_M,
        "planning_reserve_reason": RELEASE_PLANNING_RESERVE_REASON,
        "release_height_m": float(attachment_gap),
        "requested_release_height_m": float(attachment_gap),
        "attachment_floor_gap_m": float(attachment_gap),
        "gripper_floor_gap_m": float(gripper_gap),
        "attachment_spheres_tool_m": attachment.tolist(),
        "gripper_spheres_tool_m": gripper.tolist(),
        "height_source": "minimum_fixed_rotation_attachment_and_gripper_support_clearance",
        "height_limiting_geometry": "gripper" if gripper_min_z > attachment_min_z else "attachment",
        "minimum_mount_z_m": {"attachment": attachment_min_z, "gripper": gripper_min_z},
        "region_id": request.get("region_id"), "slot_id": slot_id,
        "region_position_index": slot_id - 1, "layout_revision": request.get("layout_revision"),
        "region_polygon_world_xy_m": layout["region_polygon_world_xy_m"],
        "slot_polygon_world_xy_m": cell["polygon_world_xy_m"],
        "object_id": holding["instance_id"], "placement_target_count": request.get("target_count"),
        "region_constraint": "finger_front_in_configured_region",
        "collision_policy": "three_dimensional_attachment_gripper_and_scene",
        "assumptions": ["rigid_hold_without_slip", "grasp_world_rotation_preserved",
                        "commanded_positions_reference_calibrated_finger_front"],
    }
    return {"T_world_tcp": goal.tolist(), "destination_id": destination["id"],
        "candidate_index": 0, "yaw_delta_world_rad": 0., "frame": "world",
        "source": "fixed_demo_slot_and_measured_holding",
        "object_reference_world_m": (rotation @ object_reference + goal[:3, 3]).tolist(),
        "release_reference": reference}


def release_endpoint_evidence(goal, holding, actual_tcp):
    """Validate the same measured reference and minimum-Z construction at release."""
    reference = goal["release_reference"]
    actual = rigid_matrix(actual_tcp)
    expected = rigid_matrix(goal["T_world_tcp"])
    front_target = reference.get('position_constraint') == 'finger_front_world_xy'
    measured_reference = (front_tcp_offset(holding['arm']) if front_target else np.asarray(
        (holding.get("measurement_provenance") or {}).get("measured_reference_tool_m",
        np.asarray(holding["points_tool_m"], dtype=float).mean(axis=0))))
    if not np.allclose(reference["tool_reference_point_m"], measured_reference, atol=1e-12, rtol=0):
        raise ValueError("release_tool_reference_changed")
    floor, clearance = float(reference["floor_z_m"]), float(reference["required_clearance_m"])
    attachment = sphere_array(reference["attachment_spheres_tool_m"])
    gripper = sphere_array(reference["gripper_spheres_tool_m"])
    actual_reference = actual[:3, :3] @ measured_reference + actual[:3, 3]
    expected_reference = expected[:3, :3] @ measured_reference + expected[:3, 3]
    position_error = float(np.linalg.norm(actual_reference - expected_reference))
    xy_error = float(np.linalg.norm(actual_reference[:2] - reference["target_reference_world_xy_m"]))
    angle = float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ expected[:3, :3]) - 1) / 2, -1, 1)))
    attachment_gap = actual[2, 3] + lowest_sphere_z(attachment, actual[:3, :3]) - floor
    gripper_gap = actual[2, 3] + lowest_sphere_z(gripper, actual[:3, :3]) - floor
    minimum_z = floor + clearance + RELEASE_PLANNING_RESERVE_M - min(
        lowest_sphere_z(attachment, expected[:3, :3]), lowest_sphere_z(gripper, expected[:3, :3]))
    finite = np.isfinite([floor, clearance, attachment_gap, gripper_gap, xy_error]).all()
    height_valid = valid_height_resolution(reference, expected, minimum_z, scene_token=goal.get("scene_token"))
    world_sphere_xy = (attachment[:, :3] @ actual[:3, :3].T + actual[:3, 3])[:, :2]
    footprint_inside = True
    front_inside = True
    for name in ("slot_polygon_world_xy_m", "region_polygon_world_xy_m"):
        hull = ConvexHull(np.asarray(reference[name], dtype=float))
        distance = world_sphere_xy @ hull.equations[:, :2].T + hull.equations[:, 2]
        footprint_inside = footprint_inside and bool(np.all(distance + attachment[:, 3, None] <= 1e-9))
        front_inside = front_inside and bool(np.all(actual_reference[:2] @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9))
    region_valid = front_inside if front_target else footprint_inside
    valid = bool(finite and clearance >= 0 and height_valid and region_valid
        and attachment_gap >= clearance - 1e-9 and gripper_gap >= clearance - 1e-9)
    return {"at_release_goal": valid, "attachment_floor_gap_m": float(attachment_gap),
        "gripper_floor_gap_m": float(gripper_gap), "position_error_m": position_error,
        "rotation_error_rad": angle, "reference_xy_error_m": xy_error,
        "actual_reference_world_m": actual_reference.tolist(),
        "target_reference_world_xy_m": reference["target_reference_world_xy_m"],
        "tool_reference_point_m": measured_reference.tolist(), "required_clearance_m": clearance,
        "planning_reserve_m": reference.get("planning_reserve_m"),
        "planning_reserve_reason": reference.get("planning_reserve_reason"),
        "acceptance_policy": ("finger_front_in_region_and_actual_geometry_clearance" if front_target else
                              "object_footprint_in_region_and_actual_geometry_clearance"),
        "front_inside_region": front_inside,
        "footprint_inside_region": footprint_inside,
        "position_tolerance_m": None, "rotation_tolerance_deg": None,
        "actual_tcp_world": actual.tolist(), "floor_z_m": floor,
        "source": ("calibrated_finger_front_attachment_gripper_and_actual_tcp" if front_target else
                   "measured_object_reference_attachment_gripper_and_actual_tcp")}
