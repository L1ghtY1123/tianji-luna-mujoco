"""Incremental occupied inventory layers; observation coverage is separate evidence."""
import hashlib
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
        if group == 'cuboid' and geometry.get('geometry_source') == 'observed_obb':
            dims = np.asarray(geometry.get('dims'), dtype=float)
            return dims.shape == (3,) and np.isfinite(dims).all() and np.all(dims > 0)
        if group == 'mesh' and geometry.get('geometry_source') == 'registered_cad':
            return (isinstance(geometry.get('asset_id'), str) and bool(geometry['asset_id'])
                    and isinstance(geometry.get('file_path'), str) and bool(geometry['file_path'])
                    and all(isinstance(geometry.get(key), str)
                            and re.fullmatch(r'[0-9a-f]{64}', geometry[key]) is not None
                            for key in ('source_sha256', 'mesh_sha256')))
    except (TypeError, ValueError):
        return False
    return False


def require_native_inventory_geometry(model):
    """Reject obsolete scene primitives before mutation or historical reuse."""
    forbidden = []
    for group, layers in model.items():
        for name, geometry in layers.items():
            table = name == 'semantic_table_forbidden' or name.startswith('semantic_table::')
            hybrid = _valid_hybrid_geometry(group, name, geometry)
            if (not hybrid and (group not in {'voxel', 'cuboid'} or group == 'cuboid' and not table)
                    or any(marker in name for marker in ('::bin_shell::', '::height_completion'))
                    or name.startswith(('local_object::', 'action_envelope::'))):
                forbidden.append(f'{group}/{name}')
    if forbidden:
        raise ValueError('scene_geometry_policy_requires_rebuild:' + ','.join(sorted(forbidden)))


def mapping_strategy(requested_scope=None, mapping_entity_ids=(), strategy_source=None, scope_reason=None):
    """Validate a model's choice; absence is allowed only for low-level API callers."""
    if requested_scope is None and not mapping_entity_ids and strategy_source is None and scope_reason is None:
        return None
    if (requested_scope not in {'global', 'local'} or strategy_source not in {'responses', 'decisions'}
            or not isinstance(scope_reason, str) or not scope_reason.strip()
            or not isinstance(mapping_entity_ids, (list, tuple))
            or any(not isinstance(item, str) or not item or '::' in item for item in mapping_entity_ids)
            or len(set(mapping_entity_ids)) != len(mapping_entity_ids)
            or requested_scope == 'local' and not mapping_entity_ids):
        raise ValueError('mapping_strategy_invalid')
    return {'requested_scope': requested_scope, 'mapping_entity_ids': list(mapping_entity_ids),
            'strategy_source': strategy_source, 'scope_reason': scope_reason}


def copy_grid(grid, **changes):
    result = copy(grid)
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def compatible_lattice(left, right):
    """Only exactly equal public grids can be unioned without resampling."""
    return (all(hasattr(left, key) and hasattr(right, key)
                and np.array_equal(getattr(left, key), getattr(right, key))
                for key in ('pose', 'dims', 'voxel_size'))
            and left.feature_tensor.shape == right.feature_tensor.shape
            and left.feature_tensor.dtype == right.feature_tensor.dtype)


def voxel_bytes(grids):
    return sum(int(grid.feature_tensor.numel() * grid.feature_tensor.element_size())
               if hasattr(grid.feature_tensor, 'numel') else grid.feature_tensor.nbytes
               for grid in grids)


def layer_entity(name):
    prefix = 'semantic_instance::'
    if name.startswith(prefix):
        return name[len(prefix):].split('::', 1)[0]
    return None


def freeze_inventory_scene(scene):
    """Freeze public VoxelGrid tensors before the Mapper reuses its buffers."""
    if scene is None:
        return None
    identity, grids, model = scene
    require_native_inventory_geometry(model)
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


def reconcile_inventory(previous, identity, grids, model, *, required_entity_ids=(),
                        invalidated_entity_ids=(), retired_entity_ids=(), workspace_bounds=None,
                        max_retained_voxel_bytes=DEFAULT_MAX_RETAINED_VOXEL_BYTES,
                        requested_scope=None, mapping_entity_ids=(), strategy_source=None, scope_reason=None):
    """Keep every unretired measured layer unless its pose was invalidated.

    New observations of a static entity add surfaces, not evidence that older
    surfaces disappeared. A fresh measurement after explicit motion replaces
    that entity's obsolete pose. Missing invalidated entities remain occupied
    but stale and cannot authorize motion.
    """
    require_native_inventory_geometry(model)
    if previous is not None:
        require_native_inventory_geometry(previous[2])
    identity, model = deepcopy(identity), deepcopy(model)
    grids = list(grids)
    strategy = mapping_strategy(requested_scope, mapping_entity_ids, strategy_source, scope_reason)
    if strategy:
        identity['mapping_strategy'] = strategy
    if requested_scope == 'local':
        # Never refresh an unselected entity's timestamp merely because it was
        # also visible. Its existing geometry and freshness remain independent.
        selected = set(mapping_entity_ids)
        grids = [grid for grid in grids if layer_entity(grid.name) in selected or layer_entity(grid.name) is None]
        model = {group: {name: layer for name, layer in layers.items()
                        if layer_entity(name) in selected or layer_entity(name) is None}
                 for group, layers in model.items()}
    if not isinstance(max_retained_voxel_bytes, int) or max_retained_voxel_bytes <= 0:
        raise ValueError('inventory_voxel_byte_budget_invalid')
    old_identity, old_grids, old_model = previous or ({}, [], {})
    if previous is not None:
        version = (old_identity['scene_token'] + '\0' + identity['scene_token']).encode()
        identity['scene_token'] = 'rgbd-inventory:' + hashlib.sha256(version).hexdigest()[:24]
    records = deepcopy(old_identity.get('inventory_map', {}).get('entities', {}))
    if any(record.get('status') == 'bounded_action' for record in records.values()):
        raise ValueError('scene_geometry_policy_requires_rebuild:unmeasured_action_geometry')
    retired, invalidated = set(retired_entity_ids), set(invalidated_entity_ids)
    if any(not isinstance(item, str) or not item or '::' in item
           for item in (*required_entity_ids, *retired, *invalidated)):
        raise ValueError('inventory_entity_ids_invalid')
    observed = {entity for group in model.values() for name in group
                if (entity := layer_entity(name)) is not None}
    current_geometry = {}
    for layers in model.values():
        for name, layer in layers.items():
            if layer.get('geometry_source') in {'observed_obb', 'registered_cad'}:
                current_geometry[layer_entity(name)] = layer
                if layer['geometry_source'] == 'observed_obb':
                    layer.setdefault('geometry_observation_ids', [layer['observation_id']])
    if retired & observed:
        raise ValueError('retired_entity_still_observed')
    for entity in retired:
        records.pop(entity, None)
    for entity in invalidated:
        if entity in records:
            records[entity]['status'] = 'stale'
    stale = {entity for entity, record in records.items() if record['status'] == 'stale'}
    replaced = observed & (invalidated | stale)
    rename = {}
    current_names = {name for group in model.values() for name in group}
    old_grid_by_name = {grid.name: grid for grid in old_grids}
    for group, layers in old_model.items():
        for name, layer in layers.items():
            entity = layer_entity(name)
            if entity in retired or entity in replaced:
                continue
            current = current_geometry.get(entity)
            if current is not None:
                # Each permanent ID has one current measured OBB or validated CAD.
                # Old measurements remain in the archive, not in the new geometry.
                continue
            # A newly validated rigid CAD pose supersedes the prior registration.
            # Retaining both poses creates a fictitious second copy of the object.
            if entity in observed and layer.get('geometry_source') == 'registered_cad':
                continue
            if entity is None:
                model.setdefault(group, {}).setdefault(name, deepcopy(layer))
                continue
            base_name = name.split('::history::', 1)[0]
            old_grid = old_grid_by_name.get(name)
            merged = False
            if old_grid is not None:
                for index, grid in enumerate(grids):
                    if (grid.name.split('::history::', 1)[0] == base_name
                            and compatible_lattice(grid, old_grid)):
                        # Public ESDF is negative inside. Minimum keeps all occupied
                        # samples; interpolating these minima is conservative too.
                        values = (grid.feature_tensor.minimum(old_grid.feature_tensor)
                                  if hasattr(grid.feature_tensor, 'minimum')
                                  else np.minimum(grid.feature_tensor, old_grid.feature_tensor))
                        grids[index] = copy_grid(grid, feature_tensor=values)
                        metadata = model[group][grid.name]
                        incoming_bounds = np.asarray(metadata.get('surface_cell_bounds_world_m', []), float)
                        older_bounds = np.asarray(layer.get('surface_cell_bounds_world_m', []), float)
                        if (incoming_bounds.shape == older_bounds.shape == (2, 3)
                                and np.isfinite([incoming_bounds, older_bounds]).all()
                                and metadata.get('surface_cell_bounds_source')
                                == layer.get('surface_cell_bounds_source')
                                == 'all_native_surface_cells_before_preview_sampling'):
                            metadata['surface_cell_bounds_world_m'] = [
                                np.minimum(incoming_bounds[0], older_bounds[0]).tolist(),
                                np.maximum(incoming_bounds[1], older_bounds[1]).tolist()]
                            metadata['surface_cell_count'] = (metadata.get('surface_cell_count', 0)
                                                              + layer.get('surface_cell_count', 0))
                            metadata['surface_cell_count_is_upper_bound'] = True
                        else:
                            metadata.pop('surface_cell_bounds_world_m', None)
                            metadata.pop('surface_cell_bounds_source', None)
                        merged = True
                        break
            elif layer == model.get(group, {}).get(base_name):
                merged = True
            if merged:
                continue
            if (entity in observed and '::history::' not in name
                    and records.get(entity, {}).get('last_observation_id') == identity['observation_id']):
                continue
            new_name = name
            if name in current_names:
                capture = records.get(entity, {}).get('last_observation_id', old_identity.get('observation_id', 'unknown'))
                new_name = name + '::history::' + capture
            rename[name] = new_name
            model.setdefault(group, {})[new_name] = deepcopy(layer)
    for grid in old_grids:
        if grid.name in rename:
            grids.append(copy_grid(grid, name=rename[grid.name]))
    for entity in observed:
        previous_record = records.get(entity, {})
        records[entity] = {'status': 'observed', 'last_observation_id': identity['observation_id'],
            'last_generation': identity['generation'], 'source': 'registered_inventory_rgbd',
            'first_observation_id': previous_record.get('first_observation_id', identity['observation_id']),
            'pose_replaced_after_invalidation': entity in replaced,
            'observation_ids': list(dict.fromkeys(
                ([] if entity in replaced else previous_record.get('observation_ids', []))
                + [identity['observation_id']]))}
    for entity, record in records.items():
        if entity not in observed and record['status'] != 'stale':
            record['status'] = 'retained'
        names = [name for group in model.values() for name in group if layer_entity(name) == entity]
        record['layer_names'] = names
    retained_bytes = voxel_bytes(grids)
    if retained_bytes > max_retained_voxel_bytes:
        raise ValueError('inventory_voxel_byte_budget_exceeded')
    required = set(required_entity_ids) or (set(records) | observed)
    missing = sorted(required - set(records))
    stale = sorted(entity for entity, record in records.items() if record['status'] == 'stale')
    identity['inventory_map'] = {'entities': records, 'required_entity_ids': sorted(required),
        'retained_voxel_bytes': retained_bytes, 'max_retained_voxel_bytes': max_retained_voxel_bytes,
        'retired_entity_ids': sorted(retired), 'rebuild_policy': 'retain_unseen_and_static_surface_history',
        'retained_geometry_assumption': 'no_unreported_entity_motion_since_recorded_observation'}
    identity['coverage'] = {'inventory_complete': not missing and not stale,
        'missing_entity_ids': missing, 'stale_entity_ids': stale,
        'workspace_status': 'partial' if workspace_bounds is not None else 'unknown',
        'configured_workspace_bounds': deepcopy(workspace_bounds),
        'full_environment_coverage': False, 'camera_count_is_completeness_proof': False,
        'semantic_judgment_is_geometric_coverage': False}
    if identity.get('exclusion_state') in ('selected', 'held'):
        disabled = set(identity.get('disabled_obstacle_names', ()))
        disabled.update(name for group in model.values() for name in group
                        if layer_entity(name) == identity.get('target_id'))
        identity['disabled_obstacle_names'] = tuple(sorted(disabled))
    return identity, grids, model


def invalidate_inventory(identity, entity_ids, reason):
    """Action evidence invalidates freshness; it does not remove obstacles."""
    for entity in entity_ids:
        record = identity.get('inventory_map', {}).get('entities', {}).get(entity)
        if record is not None:
            record.update(status='stale', invalidation_reason=reason)
    if 'coverage' in identity:
        identity['coverage']['workspace_status'] = 'partial'
        identity['coverage']['stale_entity_ids'] = sorted(entity for entity, record
            in identity.get('inventory_map', {}).get('entities', {}).items() if record['status'] == 'stale')
        identity['coverage']['inventory_complete'] = not (
            identity['coverage']['stale_entity_ids'] or identity['coverage']['missing_entity_ids'])


def apply_inventory_update(identity, request, model):
    """Actions invalidate geometry; only a native RGBD map refreshes it."""
    operation = request.get('operation')
    if operation == 'select':
        return
    if 'inventory_map' not in identity:
        identifiers = {entity for group in model.values() for name in group
                       if (entity := layer_entity(name)) is not None}
        identity['inventory_map'] = {
            'entities': {entity: {'status': 'retained', 'last_observation_id': identity.get('observation_id')}
                         for entity in identifiers}, 'required_entity_ids': sorted(identifiers)}
    identity.setdefault('coverage', {'inventory_complete': False, 'missing_entity_ids': [],
                                    'stale_entity_ids': [], 'full_environment_coverage': False})
    entities = identity['inventory_map']['entities']
    retired = set(request.get('retired_entity_ids', ()))
    for entity in retired:
        entities.pop(entity, None)
    identity['inventory_map']['required_entity_ids'] = [entity for entity
        in identity['inventory_map']['required_entity_ids'] if entity not in retired]
    target = request['target_id']
    if target not in identity['inventory_map']['required_entity_ids']:
        identity['inventory_map']['required_entity_ids'].append(target)
    invalidate_inventory(identity, [target], operation)
    coverage = identity['coverage']
    coverage['missing_entity_ids'] = sorted(set(identity['inventory_map']['required_entity_ids']) - set(entities))
    coverage['stale_entity_ids'] = sorted(entity for entity, row in entities.items() if row['status'] == 'stale')
    coverage['inventory_complete'] = not (coverage['missing_entity_ids'] or coverage['stale_entity_ids'])


def inventory_admission(identity, *, swept_workspace_bounds=None):
    """Check bookkeeping only; never substitute it for swept-volume evidence."""
    coverage = identity.get('coverage', {})
    if coverage.get('missing_entity_ids'):
        return {'accepted': False, 'reason': 'inventory_geometry_missing', **coverage}
    if coverage.get('stale_entity_ids'):
        return {'accepted': False, 'reason': 'inventory_geometry_stale', **coverage}
    if swept_workspace_bounds is not None:
        return {'accepted': False, 'reason': 'motion_coverage_required', **coverage}
    return {'accepted': True, 'reason': None, 'scope': 'inventory_freshness_only', **coverage}


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
