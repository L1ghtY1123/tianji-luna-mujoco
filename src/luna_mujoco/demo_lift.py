"""Source-container exit height from retained map bounds and the full carried surface."""
from copy import deepcopy

import numpy as np

from .carrying import CarryingBinding, carrying_geometry
from .geometry import rigid_matrix
from .placement import carrying_binding, observed_object, region_contains_xy
from .planning_settings import PlanningSettings


class LiftGeometryError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def source_clearance_lift(snapshot, target_id, arm, candidates, scene, settings):
    """One world-Z offset for surviving grasps; source inference never changes the map."""
    settings = PlanningSettings.model_validate(settings).model_dump()
    if settings["lift_axis"] != "z" or settings["lift_in_tool_frame"] or settings["lift_offset_m"] <= 0:
        raise ValueError("grasp_lift_requires_positive_world_z")
    bounds = scene.get("collision_layer_bounds", {})
    if (any(bounds.get(key) != scene.get(key) for key in ("scene_token", "observation_id", "generation"))
            or bounds.get("observation_id") != snapshot.get("observation_id")
            or bounds.get("generation") != snapshot.get("generation") or not isinstance(bounds.get("layers"), dict)):
        raise ValueError("grasp_lift_requires_current_collision_bounds")
    entities = snapshot.get("inventory_state", {}).get("entities") or snapshot.get("scene_entities", {})
    target_entity = entities.get(target_id, {})
    support = target_entity.get("observed_support")
    if support is not None:
        if (not isinstance(support, dict) or support.get('observation_id') != snapshot.get('observation_id')
                or support.get('source') != 'openai_decisions' or not support.get('request_hash')
                or not support.get('support_entity_id')):
            raise ValueError('grasp_lift_requires_current_support_evidence')
        source_id = support['support_entity_id']
    else:
        source_id = target_entity.get("current_region") or target_entity.get("initial_parent_id")
    source_entity = entities.get(source_id, {})
    if source_entity.get('kind') == 'marker':
        source_id = (source_entity.get('placement_reference', {}).get('support_entity_id')
                     or source_entity.get('parent_entity_id'))
    if entities.get(source_id, {}).get('kind') == 'table':
        layers, top = _table_support(snapshot, target_id, bounds, scene, source_id)
        return _lift_settings(snapshot, target_id, arm, candidates, scene, settings,
                              None, [], {}, layers, top, support_kind="table")
    if not source_id or entities.get(source_id, {}).get("kind") != "bin":
        raise ValueError("grasp_lift_requires_inventory_source_container")
    source_objects = [obj for obj in snapshot['objects'] if obj['id'] == source_id]
    polygon = None
    if source_objects:
        source = observed_object(snapshot, source_id)
        if source.get("kind") != "bin":
            raise ValueError("grasp_lift_source_container_mismatch")
        region = source.get("placement_reference", {})
        polygon = np.asarray(region.get("region_polygon_world_xy_m"), dtype=float)
        if (region.get("observation_id") != snapshot["observation_id"] or polygon.ndim != 2
                or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all()):
            raise ValueError("grasp_lift_requires_measured_source_region")
    elif support is None:
        raise ValueError('grasp_lift_requires_current_support_evidence')
    # A retained, validated native collision layer supplies the rim height.
    # A newly visible floor polygon is diagnostic, not needed for upward exit.
    target = observed_object(snapshot, target_id)

    layers = _reference_layers(bounds, scene, source_id)
    lower = np.min([layer["minimum_world_m"] for layer in layers], axis=0)
    upper = np.max([layer["maximum_world_m"] for layer in layers], axis=0)

    def centroid(obj):
        points = np.asarray(obj.get("points_world_m"), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("grasp_lift_requires_measured_object_reference")
        return points.mean(axis=0)

    def inside(obj):
        xy = centroid(obj)[:2]
        return bool(np.all(xy >= lower[:2]) and np.all(xy <= upper[:2]))

    reference = {"source_container_id": source_id, "target_id": target_id,
        "target_centroid_world_m": centroid(target).tolist(),
        "visible_floor_polygon_world_xy_m": polygon.tolist() if polygon is not None else None,
        "published_envelope_minimum_world_m": lower.tolist(),
        "published_envelope_maximum_world_m": upper.tolist(),
        "target_inside_visible_floor": (bool(region_contains_xy(centroid(target)[None, :2], polygon)[0])
                                        if polygon is not None else None),
        "target_inside_published_envelope": inside(target),
        "scope": "lift_height_reference_only"}
    if not inside(target):
        raise LiftGeometryError("grasp_lift_target_outside_source_region", reference)
    completed = set(snapshot.get("completed_target_ids", []))
    if target_id in completed:
        raise ValueError("grasp_lift_target_already_completed")
    neighbors, excluded = [], {}
    for identifier, entity in entities.items():
        if identifier == target_id or entity.get("initial_parent_id") != source_id:
            continue
        if identifier in completed:
            excluded[identifier] = "completed_target"
        elif inside(observed_object(snapshot, identifier)):
            neighbors.append(identifier)
        else:
            excluded[identifier] = "outside_measured_source_region"
    for identifier in neighbors:
        layers.extend(_reference_layers(bounds, scene, identifier))
    top = max(float(layer["maximum_world_m"][2]) for layer in layers)
    effective, proof = _lift_settings(snapshot, target_id, arm, candidates, scene, settings,
                          source_id, neighbors, excluded, layers, top, support_kind="bin")
    proof["source_reference"] = reference
    return effective, proof


def _reference_layers(bounds, scene, identifier):
    name = "semantic_instance::" + identifier
    if any(layer.startswith(name + '::bin_shell::') for layer in bounds['layers']):
        raise ValueError('scene_geometry_policy_requires_rebuild:cad_shell')
    if name + "::height_completion" in bounds["layers"]:
        raise ValueError('synthetic_scene_geometry_unsupported:reobserve_required')
    names = [name]
    result = []
    for layer_name in names:
        layer = bounds["layers"].get(layer_name, {})
        lower = np.asarray(layer.get("minimum_world_m"), dtype=float)
        upper = np.asarray(layer.get("maximum_world_m"), dtype=float)
        allowed = (layer.get('source') == 'native_surface_voxel_cell_bounds'
            or layer.get('source') == 'published_cuboid_bounds' and layer.get('kind') == 'cuboid'
                and layer.get('geometry_source') == 'observed_obb' and layer.get('instance_id') == identifier
            or layer.get('source') == 'published_cad_mesh_bounds' and layer.get('kind') == 'mesh'
                and layer.get('geometry_source') == 'registered_cad' and layer.get('instance_id') == identifier)
        if (layer.get("enabled") is not True or layer_name in scene.get("disabled_obstacle_names", [])
                or lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
                or not np.isfinite(upper).all() or np.any(upper <= lower)
                or not allowed):
            raise LiftGeometryError("grasp_lift_source_bounds_unavailable:" + layer_name,
                {"layer_name": layer_name, "published_bounds": deepcopy(layer),
                 "disabled": layer_name in scene.get("disabled_obstacle_names", [])})
        result.append({"name": layer_name, **deepcopy(layer)})
    return result


def _table_support(snapshot, target_id, bounds, scene, support_id):
    """Locate the current measured table, never the task's historic bin parent."""
    table = snapshot.get("table", {})
    floor = table.get("surface_height_m")
    position, size = np.asarray(table.get("position"), float), np.asarray(table.get("size"), float)
    if (type(floor) not in (int, float) or not np.isfinite(floor)
            or position.shape != (3,) or size.shape != (3,)
            or not np.isfinite([*position, *size]).all() or np.any(size <= 0)):
        raise ValueError("grasp_lift_requires_measured_table")
    target = observed_object(snapshot, target_id)
    points = np.asarray(target.get("points_world_m"), float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all()
            or np.any(np.abs(points[:, :2] - position[:2]) > size[:2] / 2)):
        raise ValueError("grasp_lift_target_outside_measured_table")
    layers = []
    for name, layer in bounds["layers"].items():
        if name not in {"semantic_table::" + support_id, "semantic_table_forbidden"}:
            continue
        lower, upper = np.asarray(layer.get("minimum_world_m"), float), np.asarray(layer.get("maximum_world_m"), float)
        if (layer.get("enabled") is not True or name in scene.get("disabled_obstacle_names", [])
                or layer.get("source") != "published_cuboid_bounds"
                or lower.shape != (3,) or upper.shape != (3,)
                or not np.isfinite([*lower, *upper]).all() or np.any(upper <= lower)):
            raise ValueError("grasp_lift_table_bounds_invalid")
        if (abs(upper[2] - floor) <= .006
                and np.all(points[:, :2] >= lower[:2]) and np.all(points[:, :2] <= upper[:2])):
            layers.append({"name": name, **deepcopy(layer)})
    if len(layers) != 1:
        raise ValueError("grasp_lift_requires_unique_current_table_bounds")
    return layers, float(layers[0]["maximum_world_m"][2])


def _lift_settings(snapshot, target_id, arm, candidates, scene, settings,
                   source_id, neighbors, excluded, layers, top, *, support_kind):
    target = observed_object(snapshot, target_id)
    if not candidates:
        raise ValueError("grasp_lift_requires_ik_survivors")
    if target.get("completed_geometry_points_world_m") is not None:
        raise ValueError('synthetic_scene_geometry_unsupported:reobserve_required')
    evidence, required = [], settings["lift_offset_m"]
    for candidate in candidates:
        tcp = rigid_matrix(candidate["T_world_tcp"])
        predicted = {**snapshot, "arms": {**snapshot["arms"], arm: {**snapshot["arms"][arm], "tcp": tcp.tolist()}}}
        value = carrying_binding(predicted, target_id, arm, scene["scene_token"], owner_id="pregrasp-lift-envelope")
        value.update(settings["attachment"])
        geometry = carrying_geometry(CarryingBinding.from_mapping(value, scene=scene, arm=arm))
        spheres = np.asarray(geometry["spheres"])
        lowest = float(np.min((spheres[:, :3] @ tcp[:3, :3].T)[:, 2] + tcp[2, 3] - spheres[:, 3]))
        offset = top + settings["trajectory_collision_clearance_m"] - lowest
        required = max(required, offset)
        evidence.append({"candidate_index": candidate["candidate_index"],
            "lowest_sphere_world_z_m": lowest, "required_lift_offset_m": float(offset),
            **{key: geometry[key] for key in ("sphere_count", "sphere_budget", "measurement_uncertainty_m",
                                             "margin_m", "minimum_partition_size_m", "fit_method")}})
    if required > .5:
        raise ValueError("grasp_lift_exceeds_supported_offset_range")
    effective = PlanningSettings.model_validate({**settings, "lift_offset_m": float(required)}).model_dump()
    proof = {"scene_token": scene["scene_token"], "observation_id": snapshot["observation_id"],
        "generation": snapshot["generation"], "target_id": target_id, "source_container_id": source_id,
        "support_kind": support_kind,
        "source_neighbor_ids": neighbors, "excluded_reference_ids": excluded, "retained_source_layers": layers,
        "source_occupied_top_world_z_m": top, "required_clearance_m": settings["trajectory_collision_clearance_m"],
        "configured_lift_offset_m": settings["lift_offset_m"], "effective_lift_offset_m": float(required),
        "lift_axis": "z", "lift_in_tool_frame": False, "candidates": evidence,
        "geometry_source": "published_retained_cells_and_pregrasp_full_surface_sphere_fit",
        "scope": "preplanned_source_exit_height_not_measured_holding_or_full_carried_path_proof",
        "source_exclusion_unchanged": True}
    return effective, proof
