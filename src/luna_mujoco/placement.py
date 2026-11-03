"""Measured visible surfaces and placement references shared by pick and place."""
from copy import deepcopy
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .carrying import require_measured_carrying_provenance
from .geometry import rigid_matrix
from .task_progress import SOURCE_BIN_MIN_XY_FRACTION, confirmed_task_source

OBJECT_FIELDS = ("id", "display_id", "kind", "color", "position", "visible_centroid_world",
                 "confidence", "coverage", "coverage_fraction", "uncertainty_m",
                 "estimated", "source", "color_evidence")
PLACEMENT_REFERENCE_FIELDS = ("position_world_m", "floor_z_m", "region_polygon_world_xy_m", "source",
                              "measurement_sources", "observation_id", "generation", "observed_floor_hull_world_xy_m")
PUBLIC_COVERAGE_FIELDS = ("observed_surface_only", "camera_ids", "point_count", "limitations")
PUBLIC_SOURCE_FIELDS = ("camera_ids", "point_count", "source", "observation_id")


def measured_object_geometry(target):
    """Reject archived inferred surfaces instead of silently reusing their binding."""
    completion = target.get('geometry_completion') or {}
    if (target.get('completed_geometry_points_world_m') is not None
            or target.get('dimension_prior') is not None
            or completion.get('source') == 'measured_surface_and_user_height_prior'
            or completion.get('synthetic_point_count', 0)):
        raise ValueError('synthetic_scene_geometry_unsupported:reobserve_required')
    points = np.asarray(target['points_world_m'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('measured_object_geometry_required')
    return points


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
            visible["color_evidence"] = {key: visible["color_evidence"][key] for key in (
                "median_rgb", "source", "heuristic_label", "semantic_source")
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


def visible_reference_tool_m(holding):
    """Frozen measured surface centroid, excluding any synthetic completion.

    This is an observable object reference, not an estimate of its hidden
    volume center. The rigid hold preserves it in the captured TCP frame.
    """
    require_measured_carrying_provenance(holding.get('measurement_provenance'))
    reference = (holding.get('measurement_provenance') or {}).get('measured_reference_tool_m')
    if reference is None:
        points = np.asarray(holding['points_tool_m'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('holding_requires_measured_visible_reference')
        reference = points.mean(axis=0)
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError('holding_requires_measured_visible_reference')
    return reference


def observed_object(observation, identifier):
    matches = [obj for obj in observation.get("objects", []) if obj["id"] == identifier]
    if len(matches) != 1:
        raise ValueError("observed_instance_missing_or_ambiguous:" + str(identifier))
    return matches[0]


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


def grasp_support_reference(observation, target_id, tcp, *, source_reference=None, scene_token=None, source_record=None):
    """Record measured table/bin support; visible point minima are not hidden bottoms."""
    target = observed_object(observation, target_id)
    points = np.asarray(target.get("points_world_m"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("source_support_unobservable")
    semantic = target.get("visual_semantics", {})
    inside_id = semantic.get("inside_id") if not semantic.get("ambiguous") else None
    bound_source = source_reference is not None
    table_source = False
    if bound_source:
        if (not isinstance(source_reference, dict)
                or source_reference.get('target_id') != target_id
                or any(source_reference.get(key) != observation.get(key)
                       for key in ('observation_id', 'generation'))
                or not source_reference.get('scene_token')
                or source_reference.get('scene_token') != scene_token
                or source_reference.get('geometry_source') !=
                   'published_retained_cells_and_pregrasp_full_surface_sphere_fit'
                or source_reference.get('support_kind') not in {'bin', 'table'}):
            raise PlacementError('source_support_evidence_invalid', {
                'target_id': target_id, 'observation_id': observation.get('observation_id'),
                'source_reference': deepcopy(source_reference)})
        table_source = source_reference['support_kind'] == 'table'
        inside_id = source_reference.get('source_container_id')
        if (table_source and inside_id is not None) or (not table_source and not inside_id):
            raise PlacementError('source_support_evidence_invalid', {'source_reference': deepcopy(source_reference)})
        if not table_source:
            inventory = observation.get('inventory_state', {})
            current_support = inventory.get('entities', {}).get(target_id, {}).get('observed_support')
            confirmed_id = None
            if current_support is not None:
                if (current_support.get('source') == 'openai_decisions'
                        and current_support.get('observation_id') == observation.get('observation_id')
                        and current_support.get('request_hash')):
                    confirmed_id = current_support.get('support_entity_id')
            else:
                record = source_record if isinstance(source_record, dict) and source_record.get('id') == target_id else {}
                transfers = record.get('transfers', [])
                if transfers:
                    latest = transfers[-1]
                    if latest.get('observation_id') and latest.get('release_feedback_id'):
                        confirmed_id = latest.get('destination_id')
                else:
                    confirmed = confirmed_task_source(record)
                    if confirmed and confirmed['source_kind'] == 'bin':
                        confirmed_id = confirmed['source_id']
            if confirmed_id != inside_id:
                raise PlacementError('source_support_binding_unconfirmed', {
                    'target_id': target_id, 'planned_source_id': inside_id,
                    'confirmed_source_id': confirmed_id})
    lowest = float(np.min(points[:, 2]))
    supports = []
    for obj in observation.get("objects", []):
        if table_source or obj.get("kind") != "bin" or (inside_id and obj["id"] != inside_id):
            continue
        reference = obj.get("placement_reference", {})
        polygon = reference.get("region_polygon_world_xy_m")
        if polygon is None or "floor_z_m" not in reference:
            continue
        if bound_source and reference.get('observation_id') != observation.get('observation_id'):
            raise PlacementError('source_support_measurement_stale', {'support_id': obj['id'],
                'observation_id': observation.get('observation_id'),
                'measurement_observation_id': reference.get('observation_id')})
        # The current lift plan already binds the source using its published
        # obstacle bounds. A visible floor patch is not the whole container.
        if ((bound_source or np.mean(region_contains_xy(points[:, :2], polygon)) >= SOURCE_BIN_MIN_XY_FRACTION)
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
        if support['placement_reference'].get('source') == 'validated_cad_support_plane':
            source = 'registered_cad_floor_and_joint_feedback'
    else:
        height = float(observation["table"]["surface_height_m"])
        source = "rgbd_table_and_joint_feedback"
    if not np.isfinite(height) or lowest < height - .01:
        raise ValueError("source_support_unobservable")
    return {"T_world_tcp_at_grasp": rigid_matrix(tcp).tolist(), "support_height_m": height,
            "table_height_m": height, "source": source,
            "support_container_id": supports[0]["id"] if supports else None,
            **({'source_binding': deepcopy(source_reference)} if bound_source else {}),
            "visible_source_lowest_z_m": lowest,
            "assumptions": ["source_supported_by_registered_cad_surface"
                if source == 'registered_cad_floor_and_joint_feedback' else "source_supported_by_measured_horizontal_surface",
                "visible_minimum_is_not_object_bottom"]}


def carrying_binding(observation, identifier, arm, scene_token, *, owner_id, state="held"):
    from .object_obb import obb_in_tool, observed_object_obb

    target = observed_object(observation, identifier)
    points = measured_object_geometry(target)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 16 or not np.isfinite(points).all():
        raise ValueError("holding_requires_observed_object_points")
    tcp = rigid_matrix(observation["arms"][arm]["tcp"])
    inverse = np.linalg.inv(tcp)
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    provenance = {"source": "registered_rgbd_visible_surface",
                  "reference_observation_id": observation["observation_id"],
                  "instance_id": identifier, "point_count": len(local)}
    result = {"arm": arm, "owner_id": owner_id, "observation_id": observation["observation_id"],
            "generation": observation["generation"], "scene_token": scene_token,
            "instance_id": identifier, "frame": "gripper_front", "state": state,
            "points_tool_m": local.tolist(), "uncertainty_m": target.get("uncertainty_m", .005),
            "coverage": target.get("coverage_fraction"),
            "measurement_provenance": provenance}
    if target.get('cad') is not None:
        from .carrying import cad_binding_in_tool

        result['cad_tool'] = cad_binding_in_tool(target, observation, tcp, local)
    else:
        result['obb_tool'] = obb_in_tool(observed_object_obb(target), tcp, local)
    return result
