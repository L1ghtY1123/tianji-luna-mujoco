"""Current observed object layers; temporal identity history stays in the database."""
import re
from copy import copy, deepcopy

import numpy as np

DEFAULT_MAX_RETAINED_VOXEL_BYTES = 256 * 1024 * 1024


def _valid_hybrid_geometry(group, name, geometry):
    instance = geometry.get('instance_id')
    if (not isinstance(instance, str) or not instance or name != 'semantic_instance::' + instance
            or not isinstance(geometry.get('observation_id'), str) or not geometry['observation_id']
            or type(geometry.get('generation')) is not int or geometry['generation'] < 1):
        return False
    try:
        pose = np.asarray(geometry.get('pose'), dtype=float)
        if (pose.shape != (7,) or not np.isfinite(pose).all()
                or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6, rtol=0)):
            return False
        history = geometry.get('geometry_observation_ids', [geometry['observation_id']])
        if (not isinstance(history, list) or not history
                or any(not isinstance(item, str) or not item for item in history)
                or len(set(history)) != len(history) or geometry['observation_id'] not in history):
            return False
        if group == 'mesh' and geometry.get('geometry_source') == 'registered_cad':
            return (isinstance(geometry.get('asset_id'), str) and bool(geometry['asset_id'])
                    and isinstance(geometry.get('file_path'), str) and bool(geometry['file_path'])
                    and all(isinstance(geometry.get(key), str)
                            and re.fullmatch(r'[0-9a-f]{64}', geometry[key]) is not None
                            for key in ('source_sha256', 'mesh_sha256')))
        if group == 'voxel' and geometry.get('geometry_source') == 'curobo_tsdf_esdf':
            dims = np.asarray(geometry.get('dims'), dtype=float)
            size = geometry.get('voxel_size')
            return (dims.shape == (3,) and np.isfinite(dims).all() and np.all(dims > 0)
                    and type(size) in (int, float) and np.isfinite(size) and size > 0)
    except (TypeError, ValueError):
        return False
    return False


def require_native_objects_geometry(model):
    """Reject obsolete scene primitives before mutation or historical reuse."""
    forbidden = []
    for group, layers in model.items():
        for name, geometry in layers.items():
            table = name == 'semantic_table_forbidden' or name.startswith('semantic_table::')
            hybrid = _valid_hybrid_geometry(group, name, geometry)
            if group == 'voxel' and name == 'workspace_environment':
                hybrid = (geometry.get('geometry_source') == 'workspace_tsdf_esdf'
                          and geometry.get('identity_scope') == 'anonymous'
                          and geometry.get('voxel_size', 0) > 0)
            if (not hybrid and (group != 'cuboid' or not table)
                    or any(marker in name for marker in ('::bin_shell::', '::height_completion'))
                    or name.startswith(('local_object::', 'action_envelope::'))):
                forbidden.append(f'{group}/{name}')
    if forbidden:
        raise ValueError('scene_geometry_policy_requires_rebuild:' + ','.join(sorted(forbidden)))


def mapping_strategy(requested_scope=None, strategy_source=None, scope_reason=None):
    """All callers rebuild the current workspace; local/subset updates are unsupported."""
    if requested_scope is None and strategy_source is None and scope_reason is None:
        return None
    if (requested_scope != 'global' or strategy_source != 'sole_permitted_scope'
            or not isinstance(scope_reason, str) or not scope_reason.strip()):
        raise ValueError('mapping_strategy_requires_global_rebuild')
    return {'requested_scope': 'global', 'strategy_source': strategy_source, 'scope_reason': scope_reason}


def copy_grid(grid, **changes):
    result = copy(grid)
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def voxel_bytes(grids):
    return sum(int(grid.feature_tensor.numel() * grid.feature_tensor.element_size())
               if hasattr(grid.feature_tensor, 'numel') else grid.feature_tensor.nbytes
               for grid in grids)


def layer_entity(name):
    prefix = 'semantic_instance::'
    if name.startswith(prefix):
        return name[len(prefix):].split('::', 1)[0]
    return None


def freeze_objects_scene(scene):
    """Freeze public VoxelGrid tensors before the Mapper reuses its buffers."""
    if scene is None:
        return None
    identity, grids, model = scene
    require_native_objects_geometry(model)
    frozen = []
    for grid in grids:
        tensor = grid.feature_tensor
        values = tensor.clone() if hasattr(tensor, 'clone') else np.array(tensor, copy=True)
        frozen_grid = copy_grid(grid, feature_tensor=values)
        if getattr(grid, 'xyzr_tensor', None) is not None:
            xyzr = grid.xyzr_tensor
            frozen_grid.xyzr_tensor = xyzr.clone() if hasattr(xyzr, 'clone') else np.array(xyzr, copy=True)
        frozen.append(frozen_grid)
    return deepcopy(identity), frozen, deepcopy(model)


def reconcile_objects(previous, identity, grids, model, *, required_entity_ids=(),
                        invalidated_entity_ids=(), retired_entity_ids=(), workspace_bounds=None,
                        max_retained_voxel_bytes=DEFAULT_MAX_RETAINED_VOXEL_BYTES,
                        requested_scope=None, strategy_source=None, scope_reason=None):
    """Publish current geometry; delivery receipts stay in the object database."""
    require_native_objects_geometry(model)
    identity, model, grids = deepcopy(identity), deepcopy(model), list(grids)
    strategy = mapping_strategy(requested_scope, strategy_source, scope_reason)
    if strategy:
        identity['mapping_strategy'] = strategy
    if not isinstance(max_retained_voxel_bytes, int) or max_retained_voxel_bytes <= 0:
        raise ValueError('objects_voxel_byte_budget_invalid')
    retired = set(retired_entity_ids)
    if any(not isinstance(item, str) or not item or '::' in item
           for item in (*required_entity_ids, *retired, *invalidated_entity_ids)):
        raise ValueError('objects_entity_ids_invalid')
    observed = {entity for layers in model.values() for name in layers
                if (entity := layer_entity(name)) is not None}
    if retired & observed:
        raise ValueError('retired_entity_still_observed')
    for layers in model.values():
        for name, layer in layers.items():
            if (layer_entity(name) is not None or name == 'workspace_environment') and any(
                    layer.get(key) != identity[key] for key in ('observation_id', 'generation')):
                raise ValueError('rebuilt_geometry_capture_mismatch')
    records = {entity: {'status': 'observed', 'source': 'registered_objects_rgbd',
                       'first_observation_id': identity['observation_id'],
                       'last_observation_id': identity['observation_id'],
                       'last_generation': identity['generation'],
                       'observation_ids': [identity['observation_id']],
                       'layer_names': [name for layers in model.values() for name in layers
                                       if layer_entity(name) == entity]}
               for entity in observed}
    published_bytes = voxel_bytes(grids)
    if published_bytes > max_retained_voxel_bytes:
        raise ValueError('objects_voxel_byte_budget_exceeded')
    required = set(required_entity_ids) or observed
    missing = sorted(required - set(records))
    identity['object_layers'] = {'entities': records, 'required_entity_ids': sorted(required),
        'retained_voxel_bytes': published_bytes, 'max_retained_voxel_bytes': max_retained_voxel_bytes,
        'retired_entity_ids': sorted(retired), 'rebuild_policy': 'replace_from_current_observation'}
    identity['coverage'] = {'objects_complete': not missing,
        'missing_entity_ids': missing, 'stale_entity_ids': [],
        'workspace_status': 'partial' if workspace_bounds is not None else 'unknown',
        'configured_workspace_bounds': deepcopy(workspace_bounds),
        'full_environment_coverage': False, 'camera_count_is_completeness_proof': False,
        'semantic_judgment_is_geometric_coverage': False}
    if identity.get('exclusion_state') in ('selected', 'held'):
        identity['disabled_obstacle_names'] = tuple(sorted(
            name for layers in model.values() for name in layers
            if layer_entity(name) == identity.get('target_id')))
    return identity, grids, model


def invalidate_objects(identity, entity_ids, reason):
    """Action evidence invalidates freshness; it does not remove obstacles."""
    for entity in entity_ids:
        record = identity.get('object_layers', {}).get('entities', {}).get(entity)
        if record is not None:
            record.update(status='stale', invalidation_reason=reason)
    if 'coverage' in identity:
        identity['coverage']['workspace_status'] = 'partial'
        identity['coverage']['stale_entity_ids'] = sorted(entity for entity, record
            in identity.get('object_layers', {}).get('entities', {}).items() if record['status'] == 'stale')
        identity['coverage']['objects_complete'] = not (
            identity['coverage']['stale_entity_ids'] or identity['coverage']['missing_entity_ids'])


def apply_objects_update(identity, request, model):
    """Actions invalidate geometry; only a native RGBD map refreshes it."""
    operation = request.get('operation')
    if operation == 'select':
        return
    if 'object_layers' not in identity:
        identifiers = {entity for group in model.values() for name in group
                       if (entity := layer_entity(name)) is not None}
        identity['object_layers'] = {
            'entities': {entity: {'status': 'retained', 'last_observation_id': identity.get('observation_id')}
                         for entity in identifiers}, 'required_entity_ids': sorted(identifiers)}
    identity.setdefault('coverage', {'objects_complete': False, 'missing_entity_ids': [],
                                    'stale_entity_ids': [], 'full_environment_coverage': False})
    entities = identity['object_layers']['entities']
    retired = set(request.get('retired_entity_ids', ()))
    for entity in retired:
        entities.pop(entity, None)
    identity['object_layers']['required_entity_ids'] = [entity for entity
        in identity['object_layers']['required_entity_ids'] if entity not in retired]
    target = request['target_id']
    if target not in identity['object_layers']['required_entity_ids']:
        identity['object_layers']['required_entity_ids'].append(target)
    if operation == 'container_entry':
        from .container_occupancy import occupancy_record

        entities[target] = occupancy_record(request, entities.get(target))
    else:
        invalidate_objects(identity, [target], operation)
    coverage = identity['coverage']
    coverage['missing_entity_ids'] = sorted(set(identity['object_layers']['required_entity_ids']) - set(entities))
    coverage['stale_entity_ids'] = sorted(entity for entity, row in entities.items() if row['status'] == 'stale')
    coverage['objects_complete'] = not (coverage['missing_entity_ids'] or coverage['stale_entity_ids'])


def objects_admission(identity, *, swept_workspace_bounds=None):
    """Check bookkeeping only; never substitute it for swept-volume evidence."""
    coverage = identity.get('coverage', {})
    if coverage.get('missing_entity_ids'):
        return {'accepted': False, 'reason': 'objects_geometry_missing', **coverage}
    if coverage.get('stale_entity_ids'):
        return {'accepted': False, 'reason': 'objects_geometry_stale', **coverage}
    if swept_workspace_bounds is not None:
        return {'accepted': False, 'reason': 'motion_coverage_required', **coverage}
    return {'accepted': True, 'reason': None, 'scope': 'objects_freshness_only', **coverage}


def certify_workspace(identity, motion_coverage):
    """Only a configured region and measured whole-volume proof yield complete."""
    coverage = identity['coverage']
    bounds = coverage.get('configured_workspace_bounds')
    if bounds is None:
        return
    lower = np.asarray(bounds.get('minimum_world_m'), dtype=float)
    upper = np.asarray(bounds.get('maximum_world_m'), dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(upper <= lower)):
        raise ValueError('configured_workspace_bounds_invalid')
    center, radius = (upper + lower) / 2, np.linalg.norm(upper - lower) / 2
    proof = motion_coverage.check(np.array([[[*center, radius]]]), np.zeros(1),
                                  scene_token=identity['scene_token'])
    coverage['workspace_status'] = 'complete' if proof['covered'] else 'partial'
    coverage['workspace_evidence'] = proof
    coverage['workspace_completeness_scope'] = 'configured_region_only'
