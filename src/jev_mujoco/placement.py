"""Measured visible surfaces and placement references shared by pick and place."""
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .geometry import front_tcp_offset, rigid_matrix
from .placement_layout import partition_region
from .planning_settings import RELEASE_ENDPOINT_POSITION_TOLERANCE_M
from .task_progress import SOURCE_BIN_MIN_XY_FRACTION

OBJECT_FIELDS = ("id", "display_id", "kind", "color", "position", "visible_centroid_world",
                 "confidence", "coverage", "coverage_fraction", "uncertainty_m",
                 "estimated", "source", "color_evidence")
PLACEMENT_REFERENCE_FIELDS = ("position_world_m", "floor_z_m", "region_polygon_world_xy_m", "source",
                              "measurement_sources", "observation_id", "generation")
PUBLIC_COVERAGE_FIELDS = ("observed_surface_only", "camera_ids", "point_count", "limitations")
PUBLIC_SOURCE_FIELDS = ("camera_ids", "point_count", "source", "observation_id")


class PlacementError(ValueError):
    def __init__(self, code, details):
        super().__init__(code)
        self.code = code
        self.details = details


def _public_sources(sources):
    if isinstance(sources, dict):
        return {key: sources[key] for key in PUBLIC_SOURCE_FIELDS if key in sources}
    if isinstance(sources, list):
        return [{key: source[key] for key in (*PUBLIC_SOURCE_FIELDS, "camera_id") if key in source}
                for source in sources if isinstance(source, dict)]
    return {}


def public_relationships(relationships):
    return [{key: deepcopy(value) for key, value in item.items()
             if key in {"relation", "object_id", "visible_xy_fraction", "floor_z_m",
                        "visible_source_lowest_z_m", "source", "complete_containment_verified"}}
            for item in relationships if isinstance(item, dict)]


def public_observation(observation):
    """White-list perception products; pixels, clouds and hidden metadata stay internal."""
    if observation is None:
        return None
    if observation.get("source") != "rgbd":
        raise ValueError("observation_requires_rgbd_source")
    result = {key: observation[key] for key in
              ("observation_id", "generation", "time", "source") if key in observation}
    result["table"] = {key: observation.get("table", {})[key] for key in
                       ("position", "size", "surface_height_m", "uncertainty_m", "estimated", "source")
                       if key in observation.get("table", {})}
    result["objects"] = []
    for obj in observation.get("objects", []):
        visible = {key: obj[key] for key in OBJECT_FIELDS if key in obj}
        if isinstance(visible.get("coverage"), dict):
            visible["coverage"] = {key: visible["coverage"][key] for key in PUBLIC_COVERAGE_FIELDS
                                   if key in visible["coverage"]}
        if isinstance(visible.get("color_evidence"), dict):
            visible["color_evidence"] = {key: visible["color_evidence"][key] for key in ("median_rgb", "source")
                                         if key in visible["color_evidence"]}
        if "measurement_sources" in obj:
            visible["measurement_sources"] = _public_sources(obj["measurement_sources"])
        if isinstance(obj.get("placement_reference"), dict):
            visible["placement_reference"] = {key: obj["placement_reference"][key]
                                               for key in PLACEMENT_REFERENCE_FIELDS
                                               if key in obj["placement_reference"]}
            if "measurement_sources" in visible["placement_reference"]:
                visible["placement_reference"]["measurement_sources"] = _public_sources(
                    visible["placement_reference"]["measurement_sources"])
        if isinstance(obj.get("visual_semantics"), dict):
            visible["visual_semantics"] = {key: deepcopy(value) for key, value in obj["visual_semantics"].items()
                if key in {"label", "kind", "color", "visibility", "ambiguous", "on_table", "inside_id",
                           "on_marker_id", "evidence", "source", "observation_id", "generation", "camera_ids"}}
        if "measured_relationships" in obj:
            visible["measured_relationships"] = public_relationships(obj["measured_relationships"])
        result["objects"].append(visible)
    return result


def box_corners(pose, size):
    transform = rigid_matrix(pose)
    dimensions = np.asarray(size, dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("object_extent_requires_three_positive_dimensions")
    local = np.asarray(list(product((-.5, .5), repeat=3))) * dimensions
    return local @ transform[:3, :3].T + transform[:3, 3]


def _tool_reference_point(holding):
    arm = holding.get('arm')
    if arm not in ('A', 'B'):
        raise ValueError('release_requires_valid_arm')
    return front_tcp_offset(arm)


def release_goal(target, destination, holding, *, release_height_m=.015, offset_world_m=(0., 0., 0.)):
    """Place the TCP origin at the requested world XY, preserving grasp attitude.

    This nominal Z uses the measured source support and rigid hold assumption.
    Release candidates replace it with the exact uploaded-sphere bottom formula.
    A visible point minimum is not an object bottom; clearance is never added here.
    """
    reference = destination.get("placement_reference")
    if destination.get("kind") not in {"bin", "marker"} or not isinstance(reference, dict):
        raise ValueError("destination_requires_measured_placement_reference")
    position = np.asarray(reference.get("position_world_m"), dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or "floor_z_m" not in reference:
        raise ValueError("destination_requires_measured_floor_reference")
    support = holding.get("grasp_support_reference")
    if (not isinstance(support, dict) or "T_world_tcp_at_grasp" not in support
            or not {"table_height_m", "support_height_m"}.intersection(support)):
        raise ValueError("release_requires_grasp_support_reference")
    grasp_tcp = rigid_matrix(support["T_world_tcp_at_grasp"])
    support_height = support.get("support_height_m", support.get("table_height_m"))
    values = np.asarray([reference["floor_z_m"], support_height, release_height_m], dtype=float)
    offset = np.asarray(offset_world_m, dtype=float)
    if (not np.isfinite(values).all() or release_height_m < 0
            or offset.shape != (3,) or not np.isfinite(offset).all()):
        raise ValueError("release_reference_and_height_must_be_finite")
    if "points_tool_m" in holding:
        points = np.asarray(holding["points_tool_m"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("holding_requires_measured_visible_reference")
        visible_tool = points.mean(axis=0)
        visible_source = "holding_measured_points_tool_centroid"
    else:
        visible_world = np.asarray(target.get("visible_centroid_world", target.get("position")), dtype=float)
        if visible_world.shape != (3,) or not np.isfinite(visible_world).all():
            raise ValueError("target_requires_measured_visible_reference")
        visible_tool = grasp_tcp[:3, :3].T @ (visible_world - grasp_tcp[:3, 3])
        visible_source = "pregrasp_visible_centroid_world"
    tool_reference = _tool_reference_point(holding)
    target_xy = position[:2] + offset[:2]
    tcp_goal = grasp_tcp.copy()
    tcp_goal[:2, 3] = target_xy - (tcp_goal[:3, :3] @ tool_reference)[:2]
    tcp_goal[2, 3] = grasp_tcp[2, 3] + values[0] - values[1] + release_height_m + offset[2]
    return {"T_world_tcp": tcp_goal.tolist(), "destination_id": destination["id"],
            "source": "rgbd_floor_and_grasp_support_reference",
            "release_reference": {
                "position_world_m": position.tolist(), "floor_z_m": float(values[0]),
                "table_height_m": float(values[1]), "support_height_m": float(values[1]),
                "T_world_tcp_at_grasp": grasp_tcp.tolist(),
                "tool_reference_point_m": tool_reference.tolist(),
                "tool_reference_source": "calibrated_finger_front_center",
                "tcp_frame": "gripper_front_" + holding['arm'],
                "solver_frame": "TCP_Link_L" if holding['arm'] == 'A' else "TCP_Link_R",
                "position_constraint": "tcp_world_xy",
                "target_reference_world_xy_m": target_xy.tolist(),
                "visible_reference_tool_m": visible_tool.tolist(), "visible_reference_source": visible_source,
                "release_height_m": float(release_height_m), "offset_world_m": offset.tolist(),
                "nominal_object_floor_gap_m": float(release_height_m + offset[2]),
                "assumptions": ["horizontal_measured_support_at_grasp", "rigid_hold_without_slip", "grasp_attitude_preserved"],
                "floor_source": reference.get("source"), "support_source": support.get("source"),
                "measurement_sources": reference.get("measurement_sources", [])}}


def release_endpoint_evidence(goal, holding, actual_tcp):
    """Check the uploaded attachment envelope at the measured TCP before opening."""
    expected = rigid_matrix(goal['T_world_tcp'])
    actual = rigid_matrix(actual_tcp)
    reference = goal['release_reference']
    support = holding['grasp_support_reference']
    grasp = rigid_matrix(support['T_world_tcp_at_grasp'])
    source_height = float(support.get('support_height_m', support.get('table_height_m')))
    floor = float(reference['floor_z_m'])
    requested = float(reference.get('requested_release_height_m', reference['release_height_m']))
    clearance = float(reference['required_clearance_m'])
    tolerance = RELEASE_ENDPOINT_POSITION_TOLERANCE_M
    tool_reference = _tool_reference_point(holding)
    if not np.array_equal(reference['tool_reference_point_m'], tool_reference):
        raise ValueError('release_tool_reference_changed')
    actual_reference = actual[:3, :3] @ tool_reference + actual[:3, 3]
    expected_reference = expected[:3, :3] @ tool_reference + expected[:3, 3]
    position_error = float(np.linalg.norm(actual_reference - expected_reference))
    target_xy = np.asarray(reference['target_reference_world_xy_m'])
    reference_xy_error = float(np.linalg.norm(actual_reference[:2] - target_xy))
    angle = float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ expected[:3, :3]) - 1) / 2, -1, 1)))
    spheres = _release_spheres(reference.get('attachment_spheres_tool_m'))
    centers = spheres[:, :3] @ actual[:3, :3].T + actual[:3, 3]
    gap = float(np.min(centers[:, 2] - spheres[:, 3]) - floor)
    nominal_gap = source_height + actual[2, 3] - grasp[2, 3] - floor
    valid = bool(np.isfinite([source_height, floor, requested, gap, clearance, reference_xy_error]).all())
    return {'at_release_goal': valid and clearance >= 0 and abs(requested - clearance - tolerance) <= 1e-9
            and clearance - 1e-9 <= gap <= clearance + 2 * tolerance + 1e-9
            and position_error <= tolerance and angle <= np.deg2rad(2.)
            and reference_xy_error <= tolerance
            and abs(gap - requested) <= tolerance,
        'requested_attachment_floor_gap_m': requested, 'attachment_floor_gap_m': gap,
        'estimated_object_floor_gap_m': float(nominal_gap),
        'position_error_m': position_error, 'rotation_error_rad': angle,
        'tool_reference_point_m': tool_reference.tolist(),
        'actual_reference_world_m': actual_reference.tolist(),
        'target_reference_world_xy_m': target_xy.tolist(),
        'reference_xy_error_m': reference_xy_error,
        'position_tolerance_m': tolerance, 'rotation_tolerance_deg': 2.,
        'required_clearance_m': clearance,
        'maximum_attachment_floor_gap_m': clearance + 2 * tolerance,
        'actual_tcp_world': actual.tolist(), 'floor_z_m': floor,
        'source': 'uploaded_attachment_spheres_and_actual_tcp'}


def _release_spheres(value):
    spheres = np.asarray(value, dtype=float)
    if (spheres.ndim != 2 or spheres.shape[1] != 4 or not 1 <= len(spheres) <= 16
            or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)):
        raise ValueError('release_requires_uploaded_attachment_spheres')
    return spheres


def observed_object(observation, identifier):
    matches = [obj for obj in observation.get("objects", []) if obj["id"] == identifier]
    if len(matches) != 1:
        raise ValueError("observed_instance_missing_or_ambiguous:" + str(identifier))
    return matches[0]


def release_candidate_groups(target, destination, holding, *, spheres_tool_m, clearance_m,
                       release_height_m=.015, map_resolution_m=0., observation=None,
                       placement_request=None, progress=None):
    """Eighteen world-Z rotations at fixed TCP XY and attachment floor clearance.

    Use the exact uploaded sphere envelope. Legacy references without a measured
    polygon retain 18 rotations about their single measured reference.
    Exhausting this slot never authorizes a different region or slot.
    """
    spheres = _release_spheres(spheres_tool_m)
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError("release_requires_nonnegative_clearance")
    if not np.isfinite(map_resolution_m) or map_resolution_m < 0:
        raise ValueError("release_requires_nonnegative_map_resolution")
    release_height_m = float(clearance_m) + RELEASE_ENDPOINT_POSITION_TOLERANCE_M
    reference_frame = destination.get('placement_reference', {}).get('observation_id')
    if observation and reference_frame is not None and reference_frame != observation.get('observation_id'):
        raise ValueError('destination_support_reference_stale')
    nominal = release_goal(target, destination, holding, release_height_m=release_height_m)
    reference = nominal['release_reference']
    grasp = np.asarray(reference['T_world_tcp_at_grasp'])
    centers_at_grasp = spheres[:, :3] @ grasp[:3, :3].T + grasp[:3, 3]
    bottom_at_grasp = float(np.min(centers_at_grasp[:, 2] - spheres[:, 3]))
    overhang = max(0., reference['table_height_m'] - bottom_at_grasp)
    # The requested gap belongs to the uploaded sphere envelope, not the TCP
    # or an inferred physical object bottom. The existing endpoint tolerance is
    # added above the hard minimum; never spent below it. No rim/grid allowance.
    tcp = np.asarray(nominal['T_world_tcp'])
    relative_bottom = float(np.min((spheres[:, :3] @ tcp[:3, :3].T)[:, 2] - spheres[:, 3]))
    tcp[2, 3] = reference['floor_z_m'] + release_height_m - relative_bottom
    nominal['T_world_tcp'] = tcp.tolist()
    reference['nominal_object_floor_gap_m'] = float(
        reference['support_height_m'] + tcp[2, 3] - grasp[2, 3] - reference['floor_z_m'])
    rim_height = None
    if destination.get('kind') == 'bin' and len(destination.get('points_world_m', [])):
        points = np.asarray(destination['points_world_m'], dtype=float)
        rim_height = float(np.quantile(points[:, 2], .98))
    base = nominal
    base['release_reference']['assumptions'][-1] = 'grasp_roll_pitch_preserved_world_yaw_only'
    base['release_reference'].update(requested_release_height_m=float(release_height_m),
        attachment_bottom_overhang_m=overhang, required_clearance_m=float(clearance_m),
        map_resolution_allowance_m=float(map_resolution_m),
        attachment_spheres_tool_m=spheres.tolist(), attachment_floor_gap_m=float(release_height_m),
        height_source='uploaded_attachment_sphere_bottom')
    if rim_height is not None:
        base['release_reference'].update(bin_rim_height_m=rim_height, bin_rim_source='current_measured_bin_points')
    tcp = np.asarray(base['T_world_tcp'])
    local_reference = np.asarray(base['release_reference']['visible_reference_tool_m'])
    polygon = destination["placement_reference"].get("region_polygon_world_xy_m")
    if polygon is not None:
        yield from _region_release_candidates(base, destination, holding, spheres, clearance_m,
                                              map_resolution_m, observation, placement_request, progress)
        return
    if (placement_request or {}).get('target_count', 1) != 1:
        raise ValueError('placement_requires_measured_region_polygon')
    target_xy = np.asarray(reference['target_reference_world_xy_m'])
    candidates = []
    for degrees in range(0, 360, 20):
        yaw = float(np.deg2rad(degrees))
        c, s = np.cos(yaw), np.sin(yaw)
        world_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        rotated = tcp.copy()
        rotated[:3, :3] = world_rotation @ tcp[:3, :3]
        rotated[:2, 3] = target_xy - (rotated[:3, :3] @ _tool_reference_point(holding))[:2]
        relative_bottom = np.min((spheres[:, :3] @ rotated[:3, :3].T)[:, 2] - spheres[:, 3])
        rotated[2, 3] = reference['floor_z_m'] + release_height_m - relative_bottom
        candidate = deepcopy(base)
        candidate['release_reference']['collision_policy'] = 'three_dimensional_attachment_and_scene'
        candidate.update(T_world_tcp=rotated.tolist(), candidate_index=len(candidates),
            yaw_delta_world_rad=yaw, frame='world',
            object_reference_world_m=(rotated[:3, :3] @ local_reference + rotated[:3, 3]).tolist())
        candidates.append(candidate)
    yield candidates


def release_candidates(*args, **kwargs):
    """Materialize candidates for diagnostics; runtime consumes lazy position groups."""
    return [goal for group in release_candidate_groups(*args, **kwargs) for goal in group]


def _region_hull(polygon):
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("placement_requires_measured_region_polygon")
    try:
        return ConvexHull(points)
    except QhullError as error:
        raise ValueError("placement_requires_measured_region_polygon") from error


def region_contains_xy(points, polygon):
    """Test measured XY references against a measured convex region, without pose inference."""
    hull = _region_hull(polygon)
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.all(xy @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9, axis=1)


def measured_source_relationships(source, objects):
    """Partial geometric evidence for JEV/VLM, never a complete containment claim."""
    points = np.asarray(source.get("points_world_m"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        return []
    lowest = float(np.min(points[:, 2]))
    relationships = []
    for obj in objects:
        reference = obj.get("placement_reference", {})
        polygon = reference.get('observed_floor_hull_world_xy_m', reference.get("region_polygon_world_xy_m"))
        if obj.get("kind") not in {"bin", "marker"} or polygon is None or "floor_z_m" not in reference:
            continue
        fraction = float(np.mean(region_contains_xy(points[:, :2], polygon)))
        if fraction > 0 and lowest >= reference["floor_z_m"] - .01:
            relationships.append({"relation": "above_observed_bin_floor" if obj["kind"] == "bin" else "above_observed_marker",
                "object_id": obj["id"], "visible_xy_fraction": fraction, "floor_z_m": reference["floor_z_m"],
                "visible_source_lowest_z_m": lowest, "source": "registered_rgbd_visible_samples",
                'complete_containment_verified': False})
    return relationships


def _region_floor_points(destination, polygon):
    reference = destination["placement_reference"]
    points = np.asarray(destination.get("points_world_m"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("placement_region_floor_unobservable")
    floor = points[np.abs(points[:, 2] - reference["floor_z_m"]) <= .005]
    if not len(floor):
        raise ValueError("placement_region_floor_unobservable")
    floor = floor[region_contains_xy(floor[:, :2], polygon)]
    if not len(floor):
        raise ValueError("placement_region_floor_unobservable")
    return floor


def _region_floor_positions(destination, target_count=1):
    reference = destination['placement_reference']
    _region_floor_points(destination, reference['region_polygon_world_xy_m'])
    layout = partition_region(reference['region_polygon_world_xy_m'], target_count)
    return np.asarray([[*cell['center_world_xy_m'], reference['floor_z_m']] for cell in layout['cells']])


def _region_release_candidates(base, destination, holding, spheres, clearance_m, map_resolution_m, observation,
                               placement_request=None, progress=None):
    reference = base["release_reference"]
    request = placement_request or {}
    target_count = request.get('target_count', 1)
    layout = request.get('layout') or partition_region(destination['placement_reference']['region_polygon_world_xy_m'], target_count)
    slot_id = request.get('slot_id', 1)
    if type(slot_id) is not int or not 1 <= slot_id <= layout['section_count']:
        raise PlacementError('placement_slot_not_bound', {'slot_id': slot_id, 'target_count': target_count})
    cell = next((cell for cell in layout['cells'] if cell.get('slot_id', cell['index'] + 1) == slot_id), None)
    if cell is None:
        raise PlacementError('placement_slot_not_bound', {'slot_id': slot_id})
    floor_z = layout.get('floor_z_m', reference['floor_z_m'])
    if abs(floor_z - reference['floor_z_m']) > .005:
        raise PlacementError('placement_support_movement_unresolved', {'slot_id': slot_id})
    reference['floor_z_m'] = floor_z
    hull = _region_hull(layout['region_polygon_world_xy_m'])
    positions = np.asarray([[*cell['center_world_xy_m'], floor_z]])
    # Validate current measured samples in the assigned region. A newly chosen
    # inner rectangle must not discard visible support for the frozen slot.
    floor = _region_floor_points(destination, layout['region_polygon_world_xy_m'])
    if request.get('layout'):
        spacing = float(np.median(cKDTree(floor[:, :2]).query(floor[:, :2], k=min(2, len(floor)))[0][:, -1])) if len(floor) > 1 else 0.
        tolerance = max(.005, min(.015, spacing * 1.5))
        if cKDTree(floor[:, :2]).query(positions[0, :2])[0] > tolerance:
            raise PlacementError('placement_slot_support_unobserved', {'slot_id': slot_id,
                'region_id': request.get('region_id'), 'support_distance_limit_m': tolerance})
    diagnostics = {'destination_id': destination['id'], 'target_count': target_count,
        'region_id': request.get('region_id', 'region:' + destination['id']), 'slot_id': slot_id,
        'layout_revision': request.get('layout_revision'),
        'section_count': layout['section_count'], 'yaw_count_per_position': 18, 'yaw_step_deg': 20,
        'layout_method': 'equal_area_radial_partitions',
        'collision_policy': 'three_dimensional_attachment_and_scene',
        'position_constraint': reference['position_constraint'],
        'tool_reference_point_m': reference['tool_reference_point_m'],
        'tcp_frame': reference['tcp_frame'],
        'position_source': 'centroids_of_measured_support_polygon_partitions',
        'position_candidates_world_m': positions.tolist(), 'positions': []}
    if progress is not None:
        progress('placement_layout', '放置区域分区与位置候选', deepcopy(diagnostics))
    tcp = np.asarray(base["T_world_tcp"])
    local_reference = np.asarray(reference["visible_reference_tool_m"])
    count = 0
    for position in positions:
        position_index = slot_id - 1
        candidates = []
        row = {'position_index': position_index, 'position_world_m': position.tolist(),
               'generated_yaw_count': 18}
        for degrees in range(0, 360, 20):
            yaw = float(np.deg2rad(degrees))
            c, s = np.cos(yaw), np.sin(yaw)
            world_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            rotated = tcp.copy()
            rotated[:3, :3] = world_rotation @ tcp[:3, :3]
            rotated[:2, 3] = position[:2] - (rotated[:3, :3] @ _tool_reference_point(holding))[:2]
            relative_bottom = np.min((spheres[:, :3] @ rotated[:3, :3].T)[:, 2] - spheres[:, 3])
            rotated[2, 3] = reference['floor_z_m'] + reference['requested_release_height_m'] - relative_bottom
            placed_reference = rotated[:3, :3] @ local_reference + rotated[:3, 3]
            candidate = deepcopy(base)
            candidate["release_reference"].update(position_world_m=position.tolist(),
                target_reference_world_xy_m=position[:2].tolist(),
                collision_policy='three_dimensional_attachment_and_scene',
                region_polygon_world_xy_m=hull.points[hull.vertices].tolist(),
                region_constraint='tcp_world_xy_at_assigned_slot',
                region_position_index=position_index, slot_id=slot_id,
                region_id=diagnostics['region_id'], layout_revision=diagnostics['layout_revision'],
                object_id=request.get('object_id'), object_display_id=request.get('object_display_id'),
                placement_target_count=target_count, placement_section_count=layout['section_count'],
                position_source=diagnostics['position_source'],
                nominal_object_floor_gap_m=float(reference['support_height_m'] + rotated[2, 3]
                    - reference['T_world_tcp_at_grasp'][2][3] - reference['floor_z_m']))
            candidate.update(T_world_tcp=rotated.tolist(), candidate_index=count + len(candidates),
                yaw_delta_world_rad=yaw, frame="world", object_reference_world_m=placed_reference.tolist())
            candidates.append(candidate)
        row['accepted_yaw_count'] = len(candidates)
        diagnostics['positions'].append(row)
        if progress is not None:
            progress('placement_position_checked', '固定 TCP XY 的世界 Z 轴偏航候选', deepcopy(row))
        if candidates:
            count += len(candidates)
            yield candidates


def grasp_support_reference(observation, target_id, tcp):
    """Record measured table/bin support; visible point minima are not hidden bottoms."""
    target = observed_object(observation, target_id)
    points = np.asarray(target.get("points_world_m"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("source_support_unobservable")
    semantic = target.get("visual_semantics", {})
    inside_id = semantic.get("inside_id") if not semantic.get("ambiguous") else None
    lowest = float(np.min(points[:, 2]))
    supports = []
    for obj in observation.get("objects", []):
        if obj.get("kind") != "bin" or (inside_id and obj["id"] != inside_id):
            continue
        reference = obj.get("placement_reference", {})
        polygon = reference.get("region_polygon_world_xy_m")
        if polygon is None or "floor_z_m" not in reference:
            continue
        if (np.mean(region_contains_xy(points[:, :2], polygon)) >= SOURCE_BIN_MIN_XY_FRACTION
                and lowest >= reference["floor_z_m"] - .01):
            supports.append(obj)
    if len(supports) > 1:
        raise ValueError("source_support_ambiguous")
    if inside_id and not supports:
        raise ValueError("source_support_unobservable")
    if supports:
        support = supports[0]
        height = float(support["placement_reference"]["floor_z_m"])
        source = "rgbd_measured_bin_floor_and_joint_feedback"
    else:
        height = float(observation["table"]["surface_height_m"])
        source = "rgbd_table_and_joint_feedback"
    if not np.isfinite(height) or lowest < height - .01:
        raise ValueError("source_support_unobservable")
    return {"T_world_tcp_at_grasp": rigid_matrix(tcp).tolist(), "support_height_m": height,
            "table_height_m": height, "source": source,
            "support_container_id": supports[0]["id"] if supports else None,
            "visible_source_lowest_z_m": lowest,
            "assumptions": ["source_supported_by_measured_horizontal_surface", "visible_minimum_is_not_object_bottom"]}


def carrying_binding(observation, identifier, arm, scene_token, *, owner_id, state="held"):
    target = observed_object(observation, identifier)
    points = np.asarray(target["points_world_m"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16 or not np.isfinite(points).all():
        raise ValueError("holding_requires_observed_object_points")
    tcp = rigid_matrix(observation["arms"][arm]["tcp"])
    inverse = np.linalg.inv(tcp)
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    return {"arm": arm, "owner_id": owner_id, "observation_id": observation["observation_id"],
            "generation": observation["generation"], "scene_token": scene_token,
            "instance_id": identifier, "frame": "tool", "state": state,
            "points_tool_m": local.tolist(), "uncertainty_m": target.get("uncertainty_m", .005),
            "coverage": target.get("coverage_fraction"),
            "measurement_provenance": {"source": "registered_rgbd_visible_surface",
                                       "reference_observation_id": observation["observation_id"],
                                       "instance_id": identifier, "point_count": len(local)}}
