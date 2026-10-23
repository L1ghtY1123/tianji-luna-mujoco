"""Rigid reuse of one measured voxel layer; no hidden geometry or landing reconstruction."""
import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import rigid_matrix


def transformed_pose(pose, transform):
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
    frame[:3, 3] = pose[:3]
    result = transform @ frame
    return [*result[:3, 3].tolist(), *Rotation.from_matrix(result[:3, :3]).as_quat(scalar_first=True).tolist()]


def transformed_points(points, transform):
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    return (values @ transform[:3, :3].T + transform[:3, 3]).tolist()


def prepare_scene_update(scene, grids, model, request):
    if not scene or any(request.get(key) != scene.get(key) for key in ('scene_token', 'generation')):
        raise ValueError('scene_update_stale')
    name = 'semantic_instance::' + request['target_id']
    if name not in model.get('voxel', {}) and request.get('operation') not in {'action_envelope', 'local_object', 'unpack_view_motion'}:
        raise ValueError('scene_update_target_layer_missing')
    operation = request.get('operation')
    if operation not in {'transfer', 'select', 'action_envelope', 'local_object', 'unpack_view_motion'}:
        raise ValueError('scene_update_operation_invalid')
    policy = request.get('exclusion_state')
    if policy not in {'selected', 'released', 'observation'} or operation == 'transfer' and policy != 'released':
        raise ValueError('scene_update_policy_invalid')
    updated_grids, updated_model = list(grids), deepcopy(model)
    evidence = {'target_id': request['target_id'], 'operation': operation}
    removed_names = set()
    if operation == 'unpack_view_motion':
        if (request['target_id'] != scene.get('target_id') or policy != 'released'
                or request.get('source') != 'user_authorized_unpack_visibility_fallback'
                or not request.get('action_id')):
            raise ValueError('unpack_view_motion_binding_invalid')
        removed_names = {name, name + '::height_completion',
                         'local_object::' + request['target_id'], 'action_envelope::' + request['target_id']}
        updated_grids = [grid for grid in grids if grid.name not in removed_names]
        for group in ('voxel', 'cuboid', 'mesh'):
            for old_name in removed_names:
                updated_model.get(group, {}).pop(old_name, None)
        evidence.update(source=request['source'], action_id=request['action_id'],
            removed_layers=sorted(removed_names), background_remapped=False,
            requires_fresh_grasp_map=True)
    if operation == 'local_object':
        bounds = request.get('bounds', {})
        lower, upper = np.asarray(bounds.get('minimum_world_m'), float), np.asarray(bounds.get('maximum_world_m'), float)
        previous = request.get('replaces_entity_id')
        sources = {'3': 'head_rgbd_object_surface_with_height_prior',
                   '1': 'wrist_rgbd_object_surface_with_height_prior',
                   '2': 'wrist_rgbd_object_surface_with_height_prior'}
        if (previous != scene['target_id'] or policy != 'released' or request.get('camera_id') not in sources
                or request.get('source') != sources.get(request.get('camera_id'))
                or not request.get('after_capture_id') or request['after_capture_id'] == scene.get('observation_id')
                or lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
                or not (upper > lower).all()):
            raise ValueError('scene_local_object_evidence_invalid')
        removed_names = {f'semantic_instance::{previous}', f'semantic_instance::{previous}::height_completion',
                         f'action_envelope::{previous}', f'local_object::{previous}'}
        updated_grids = [grid for grid in grids if grid.name not in removed_names]
        for group in ('voxel', 'cuboid', 'mesh'):
            for old_name in removed_names:
                updated_model.get(group, {}).pop(old_name, None)
        layer_name = 'local_object::' + request['target_id']
        updated_model.setdefault('cuboid', {})[layer_name] = {
            'dims': (upper - lower).tolist(), 'pose': [*((upper + lower) / 2).tolist(), 1., 0., 0., 0.],
            'position_source': request['source'], 'observation_id': request['after_capture_id']}
        evidence.update(source=request['source'], after_capture_id=request['after_capture_id'], camera_id=request['camera_id'],
            bounds=deepcopy(bounds), removed_layers=sorted(removed_names), added_layer=layer_name,
            background_remapped=False, local_depth_measured=True, full_surface_measured=False)
    if operation == 'action_envelope':
        bounds = request.get('bounds', {})
        lower = np.asarray(bounds.get('minimum_world_m'), float)
        upper = np.asarray(bounds.get('maximum_world_m'), float)
        if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
                or not (upper > lower).all() or not request.get('action_id')
                or request.get('source') not in {'mock_vla', 'verified_drop_action_bounds', 'drop_release_action_bounds'}):
            raise ValueError('scene_action_envelope_invalid')
        envelope = 'action_envelope::' + request['target_id']
        updated_model.setdefault('cuboid', {})[envelope] = {
            'dims': (upper - lower).tolist(), 'pose': [*((lower + upper) / 2).tolist(), 1., 0., 0., 0.],
            'position_source': 'authorized_action_bounds', 'action_id': request['action_id']}
        evidence.update(source=request['source'], action_id=request['action_id'],
            position_source='authorized_action_bounds', measured=False, rgbd_remapped=False,
            bounds=deepcopy(bounds), added_layer=envelope)
    completion_name = name + "::height_completion"
    target_names = [name] + ([completion_name] if completion_name in model.get("cuboid", {}) else [])
    floor = request.get("support_floor_z_m")
    if floor is not None and (isinstance(floor, bool) or not np.isfinite(float(floor))):
        raise ValueError("scene_update_support_floor_invalid")
    if operation == 'transfer':
        if request.get('source') != 'release_feedback_rigid_prediction' or not request.get('after_capture_id'):
            raise ValueError('scene_update_release_evidence_missing')
        transform = rigid_matrix(request['T_new_world_from_old_world'])
        found = False
        for index, grid in enumerate(grids):
            if grid.name == name:
                # Public VoxelGrid is a dataclass. Keep its native field, replace its pose.
                updated_grids[index] = replace(grid, pose=transformed_pose(grid.pose, transform), xyzr_tensor=None) \
                    if hasattr(grid, 'xyzr_tensor') else replace(grid, pose=transformed_pose(grid.pose, transform))
                found = True
        if not found:
            raise ValueError('scene_update_target_layer_missing')
        metadata = updated_model['voxel'][name]
        if 'pose' in metadata:
            metadata['pose'] = transformed_pose(metadata['pose'], transform)
        if 'centers_base_m' in metadata:
            metadata['centers_base_m'] = transformed_points(metadata['centers_base_m'], transform)
        # Prediction moves the native grid. Its sampled display points cannot
        # establish full world-cell bounds; require a fresh map for that evidence.
        metadata.pop("surface_cell_bounds_world_m", None)
        metadata.pop("surface_cell_bounds_source", None)
        evidence.update(position_source='action_prediction', source=request['source'],
            after_capture_id=request['after_capture_id'], T_new_world_from_old_world=transform.tolist(),
            final_settled_pose_measured=False)
        metadata['position_source'] = 'action_prediction'
        if completion_name in updated_model.get("cuboid", {}):
            completion = updated_model["cuboid"][completion_name]
            completion["pose"] = transformed_pose(completion["pose"], transform)
            completion["position_source"] = "action_prediction"
            if floor is not None:
                # Cover the release pose and downward settling to the observed
                # support without claiming to have measured the landed pose.
                rotation = Rotation.from_quat(completion["pose"][3:], scalar_first=True).as_matrix()
                center = np.asarray(completion["pose"][:3])
                half = np.abs(rotation) @ np.asarray(completion["dims"]) / 2
                lower, upper = center - half, center + half
                if float(floor) > upper[2]:
                    raise ValueError("scene_update_support_floor_above_object")
                lower[2] = min(lower[2], float(floor))
                completion["dims"] = (upper - lower).tolist()
                completion["pose"] = [*((upper + lower) / 2).tolist(), 1., 0., 0., 0.]
                completion["settling_envelope_floor_z_m"] = float(floor)
            evidence["transferred_layers"] = target_names
    content = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    identity = scene | {'scene_token': 'rgbd-update:' + hashlib.sha256(content).hexdigest()[:24],
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
            for old_name in removed_names:
                bounds['layers'].pop(old_name, None)
            for layer_name, layer in bounds["layers"].items():
                layer["enabled"] = layer_name not in identity["disabled_obstacle_names"]
                if operation == "transfer" and layer_name in target_names:
                    layer.pop("minimum_world_m", None)
                    layer.pop("maximum_world_m", None)
                    layer.update(status="unavailable", invalidation_reason="release_prediction_requires_fresh_bounds")
            identity["collision_layer_bounds"] = bounds
    return identity, updated_grids, updated_model


def updated_preview(preview, identity, request):
    result = deepcopy(preview)
    if result is None:
        return None
    result['scene_token'] = identity['scene_token']
    collision = result.get('collision_scene', {})
    collision['scene_token'] = identity['scene_token']
    if request['operation'] in {'local_object', 'unpack_view_motion'}:
        removed = identity['layer_evidence']['removed_layers']
        collision['voxel_layers'] = [layer for layer in collision.get('voxel_layers', []) if layer['name'] not in removed]
        collision['local_object_obstacle'] = deepcopy(identity['layer_evidence'])
    for layer in collision.get('voxel_layers', []):
        layer['enabled'] = layer['name'] not in identity['disabled_obstacle_names']
        if request['operation'] == 'transfer' and layer['name'] == 'semantic_instance::' + request['target_id']:
            transform = rigid_matrix(request['T_new_world_from_old_world'])
            layer.update(position_source='action_prediction', provenance='transformed_measured_voxels',
                         pose_wxyz=transformed_pose(layer['pose_wxyz'], transform),
                         surface_centers_world_m=transformed_points(layer['surface_centers_world_m'], transform))
    return result
