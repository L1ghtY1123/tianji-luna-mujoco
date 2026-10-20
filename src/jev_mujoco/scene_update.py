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
    if name not in model.get('voxel', {}):
        raise ValueError('scene_update_target_layer_missing')
    operation = request.get('operation')
    if operation not in {'transfer', 'select'}:
        raise ValueError('scene_update_operation_invalid')
    policy = request.get('exclusion_state')
    if policy not in {'selected', 'released', 'observation'} or operation == 'transfer' and policy != 'released':
        raise ValueError('scene_update_policy_invalid')
    updated_grids, updated_model = list(grids), deepcopy(model)
    evidence = {'target_id': request['target_id'], 'operation': operation}
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
        evidence.update(position_source='action_prediction', source=request['source'],
            after_capture_id=request['after_capture_id'], T_new_world_from_old_world=transform.tolist(),
            final_settled_pose_measured=False)
        metadata['position_source'] = 'action_prediction'
    content = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    identity = scene | {'scene_token': 'rgbd-update:' + hashlib.sha256(content).hexdigest()[:24],
        'target_id': request['target_id'], 'exclusion_state': policy,
        'disabled_obstacle_names': (name,) if policy == 'selected' else (), 'layer_evidence': evidence}
    if operation == 'select':
        points = np.asarray(request.get('selected_surface_world_m', []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('scene_update_selected_surface_missing')
        identity['selected_surface_world_m'] = points.tolist()
    return identity, updated_grids, updated_model


def updated_preview(preview, identity, request):
    result = deepcopy(preview)
    if result is None:
        return None
    result['scene_token'] = identity['scene_token']
    collision = result.get('collision_scene', {})
    collision['scene_token'] = identity['scene_token']
    for layer in collision.get('voxel_layers', []):
        layer['enabled'] = layer['name'] not in identity['disabled_obstacle_names']
        if request['operation'] == 'transfer' and layer['name'] == 'semantic_instance::' + request['target_id']:
            transform = rigid_matrix(request['T_new_world_from_old_world'])
            layer.update(position_source='action_prediction', provenance='transformed_measured_voxels',
                         pose_wxyz=transformed_pose(layer['pose_wxyz'], transform),
                         surface_centers_world_m=transformed_points(layer['surface_centers_world_m'], transform))
    return result
