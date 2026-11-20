"""Versioned updates to measured voxel, table and registered CAD layers."""
import hashlib
import json
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from .collision_objects import layer_entity, require_native_objects_geometry
from .geometry import rigid_matrix

SCENE_INVALIDATION_SOURCES = frozenset({
    'base_vla', 'verified_drop_action_bounds', 'drop_release_action_bounds',
    'verified_release_action',
})


def transformed_pose(pose, transform):
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
    frame[:3, 3] = pose[:3]
    result = transform @ frame
    return [*result[:3, 3].tolist(), *Rotation.from_matrix(result[:3, :3]).as_quat(scalar_first=True).tolist()]


def prepare_scene_update(scene, grids, model, request):
    if not scene or any(request.get(key) != scene.get(key) for key in ('scene_token', 'generation')):
        raise ValueError('scene_update_stale')
    require_native_objects_geometry(model)
    name = 'semantic_instance::' + request['target_id']
    operation = request.get('operation')
    if operation not in {'transfer', 'select', 'invalidate', 'local_object', 'container_entry'}:
        raise ValueError('scene_update_operation_invalid')
    if not any(name in model.get(kind, {}) for kind in ('voxel', 'cuboid', 'mesh')) and operation not in {'invalidate', 'local_object', 'container_entry'}:
        raise ValueError('scene_update_target_layer_missing')
    policy = request.get('exclusion_state')
    if policy not in {'selected', 'released', 'observation'} or operation == 'transfer' and policy != 'released':
        raise ValueError('scene_update_policy_invalid')
    updated_grids, updated_model = list(grids), deepcopy(model)
    evidence = {'target_id': request['target_id'], 'operation': operation}
    if operation == 'local_object':
        raise ValueError('local_object_requires_registered_rgbd_mapping')
    if operation == 'container_entry':
        container = request.get('container_id')
        record = scene.get('object_layers', {}).get('entities', {}).get(container, {})
        if (policy != 'released' or request.get('source') not in {'bound_container_release', 'visual_container_entry'}
                or record.get('status') not in {'observed', 'retained'}
                or not request.get('after_capture_id')):
            raise ValueError('container_occupancy_evidence_invalid')
        if not request.get('receipt_id') or not request.get('binding_id'):
            raise ValueError('container_entry_receipt_missing')
        evidence.update(source=request['source'], container_id=container,
                        entry_receipt_id=request['receipt_id'], measured=False,
                        object_pose_measured=False, geometry_invalidated=True)
    if operation == 'invalidate':
        if (not request.get('action_id')
                or request.get('source') not in SCENE_INVALIDATION_SOURCES):
            raise ValueError('scene_invalidation_evidence_invalid')
        # The effect bounds are not a sensor measurement. Preserve the last
        # measured occupied map, but invalidate it before any further motion.
        evidence.update(source=request['source'], action_id=request['action_id'],
                        measured=False, rgbd_remapped=False, geometry_invalidated=True,
                        invalidation_reason='action_requires_registered_rgbd_mapping')
    target_names = [layer for group in model.values() for layer in group
                    if layer_entity(layer) == request['target_id']]
    floor = request.get("support_floor_z_m")
    if floor is not None and (isinstance(floor, bool) or not np.isfinite(float(floor))):
        raise ValueError("scene_update_support_floor_invalid")
    if operation == 'transfer':
        if request.get('source') != 'release_feedback_rigid_prediction' or not request.get('after_capture_id'):
            raise ValueError('scene_update_release_evidence_missing')
        transform = rigid_matrix(request['T_new_world_from_old_world'])
        found = False
        for kind in ('cuboid', 'mesh'):
            for layer_name in target_names:
                metadata = updated_model.get(kind, {}).get(layer_name)
                if metadata is not None:
                    metadata['pose'] = transformed_pose(metadata['pose'], transform)
                    metadata['position_source'] = 'action_prediction'
                    found = True
        if not found:
            raise ValueError('scene_update_target_layer_missing')
        evidence.update(position_source='action_prediction', source=request['source'],
            after_capture_id=request['after_capture_id'], T_new_world_from_old_world=transform.tolist(),
            transferred_layers=target_names, final_settled_pose_measured=False,
            geometry_invalidated=True, invalidation_reason='release_requires_registered_rgbd_mapping')
    content = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    identity = deepcopy(scene) | {'scene_token': 'rgbd-update:' + hashlib.sha256(content).hexdigest()[:24],
        'target_id': request['target_id'], 'exclusion_state': policy,
        'disabled_obstacle_names': tuple(target_names) if policy == 'selected' else (), 'layer_evidence': evidence}
    if operation == 'select':
        points = np.asarray(request.get('selected_surface_world_m', []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('scene_update_selected_surface_missing')
        identity['selected_surface_world_m'] = points.tolist()
    bounds = scene.get("collision_layer_bounds")
    if bounds is not None:
        identity.pop("collision_layer_bounds", None)
        if all(bounds.get(key) == scene.get(key) for key in ("scene_token", "observation_id", "generation")):
            bounds = deepcopy(bounds)
            bounds["scene_token"] = identity["scene_token"]
            for layer_name, layer in bounds["layers"].items():
                layer["enabled"] = layer_name not in identity["disabled_obstacle_names"]
                if operation == "transfer" and layer_name in target_names:
                    layer.pop("minimum_world_m", None)
                    layer.pop("maximum_world_m", None)
                    layer.update(status="unavailable", invalidation_reason="release_prediction_requires_fresh_bounds")
            identity["collision_layer_bounds"] = bounds
    return identity, updated_grids, updated_model


def updated_preview(preview, identity, request, model=None):
    result = deepcopy(preview)
    if result is None:
        return None
    result['scene_token'] = identity['scene_token']
    collision = result.get('collision_scene', {})
    collision['scene_token'] = identity['scene_token']
    if request['operation'] == 'container_entry' and model is not None:
        disabled = set(identity['disabled_obstacle_names'])
        layers = [*collision.get('mesh_layers', {}).values(), *collision.get('voxel_layers', []),
                  *collision.get('cuboid_layers', [])]
        for layer in layers:
            layer['enabled'] = layer['name'] not in disabled
        collision['coverage'] = deepcopy(identity['coverage'])
    return result
