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


OBSERVATION_POSE_TOLERANCE_DEG = 1.0


def observation_pose_errors_deg(joints, targets, *, degrees=False):
    """Maximum absolute joint error on each arm against the configured observation pose."""
    errors = {}
    for arm in ('A', 'B'):
        measured = np.asarray(joints.get(arm), dtype=float)
        goal = np.asarray(targets.get(arm), dtype=float)
        if (measured.shape != (7,) or goal.shape != (7,)
                or not np.isfinite(measured).all() or not np.isfinite(goal).all()):
            raise ValueError('mapping_requires_measured_dual_arm_joints')
        measured_deg = measured if degrees else np.rad2deg(measured)
        errors[arm] = float(np.max(np.abs(measured_deg - np.rad2deg(goal))))
    return errors


def mapping_strategy(requested_scope=None, strategy_source=None, scope_reason=None):
    if requested_scope is None and strategy_source is None and scope_reason is None:
        return {'requested_scope': 'local', 'strategy_source': 'unverified_pose',
                'scope_reason': 'No complete observation-pose evidence supplied'}
    if (requested_scope not in {'global', 'local'} or (strategy_source != 'measured_observation_pose'
            and (requested_scope, strategy_source) != ('local', 'spatial_memory'))
            or not isinstance(scope_reason, str) or not scope_reason.strip()):
        raise ValueError('invalid_collision_mapping_scope')
    return {'requested_scope': requested_scope, 'strategy_source': strategy_source, 'scope_reason': scope_reason}


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


def require_parent_generation(previous, identity, stopped_parent):
    if previous['generation'] == identity['generation']:
        if stopped_parent is not None:
            raise ValueError('unexpected_collision_memory_stop_receipt')
        return
    world = identity.get('mapping_world')
    if (not isinstance(world, dict) or set(world) != {'run_id', 'session_id', 'bridge_instance_id', 'geometry_revision'}
            or not all(isinstance(v, str) and v for v in world.values()) or previous.get('mapping_world') != world
            or identity['generation'] - previous['generation'] not in (1, 2)
            or stopped_parent != {'scene_token': previous['scene_token'], 'world': world,
                'previous_generation': previous['generation'], 'generation': identity['generation']}):
        raise ValueError('collision_memory_parent_generation_changed')


def reconcile_objects(previous, identity, grids, model, *, required_entity_ids=(),
                        invalidated_entity_ids=(), changed_by_operation_ids=(), retired_entity_ids=(), workspace_bounds=None,
                        max_retained_voxel_bytes=DEFAULT_MAX_RETAINED_VOXEL_BYTES,
                        requested_scope=None, strategy_source=None, scope_reason=None, previous_scene_token=None,
                        retained_visible_sources=None, stopped_parent=None, mapping_world=None):
    """Publish current geometry; delivery receipts stay in the object database."""
    require_native_objects_geometry(model)
    identity, model, grids = deepcopy(identity), deepcopy(model), list(grids)
    strategy = mapping_strategy(requested_scope, strategy_source, scope_reason)
    identity['mapping_strategy'] = strategy
    if mapping_world is not None:
        identity['mapping_world'] = deepcopy(mapping_world)
    local = strategy['requested_scope'] == 'local'
    if not isinstance(max_retained_voxel_bytes, int) or max_retained_voxel_bytes <= 0:
        raise ValueError('objects_voxel_byte_budget_invalid')
    retired = set(retired_entity_ids)
    if any(not isinstance(item, str) or not item or '::' in item
           for item in (*required_entity_ids, *retired, *invalidated_entity_ids, *changed_by_operation_ids)):
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
    required = (set(required_entity_ids) | observed) - retired
    retained, stale = [], []
    previous_identity = {}
    invalidated = set(invalidated_entity_ids) | set(changed_by_operation_ids)
    retained_visible = retained_visible_sources or {}
    if not isinstance(retained_visible, dict) or retained_visible and (
            not local or not previous_scene_token or set(retained_visible) & (observed | retired | invalidated)):
        raise ValueError('local_retained_geometry_invalid')
    if previous_scene_token:
        if previous is None or previous[0].get('scene_token') != previous_scene_token:
            raise ValueError('collision_memory_parent_changed')
        previous_identity, old_grids, old_model = previous
        require_parent_generation(previous_identity, identity, stopped_parent)
        if local:
            for name, table in old_model.get('cuboid', {}).items():
                identifier = name.removeprefix('semantic_table::')
                if name.startswith('semantic_table::') and identifier not in retired:
                    model.setdefault('cuboid', {}).setdefault(name, deepcopy(table))
        old_layers = previous_identity.get('object_layers', {})
        old_records = old_layers.get('entities', {})
        for identifier, origin in retained_visible.items():
            record = old_records.get(identifier, {})
            if (not origin or record.get('last_observation_id') != origin
                    or record.get('last_generation') != identity['generation']
                    or record.get('status') not in {'observed', 'retained'}
                    or not any('semantic_instance::' + identifier in layers for layers in old_model.values())):
                raise ValueError('local_retained_geometry_source_changed')
        required.update((set(old_layers.get('required_entity_ids', ())) | set(old_records)) - retired)
        for entity in observed & old_records.keys():
            records[entity]['first_observation_id'] = old_records[entity].get('first_observation_id', identity['observation_id'])
        for entity in sorted(required - observed - retired):
            record = old_records.get(entity)
            if not record:
                continue
            names = {name for group in old_model.values() for name in group if layer_entity(name) == entity}
            if not names:
                continue
            for kind, layers in old_model.items():
                model.setdefault(kind, {}).update({name: deepcopy(layers[name]) for name in names & layers.keys()})
            grids.extend(grid for grid in old_grids if grid.name in names)
            records[entity] = deepcopy(record)
            retained.append(entity)
            if entity in invalidated or record.get('status') not in {'observed', 'retained'}:
                records[entity].update(status='stale', invalidation_reason='location_requires_new_measurement')
                stale.append(entity)
            else:
                records[entity].update(status='retained', source='collision_memory')
                records[entity].pop('invalidation_reason', None)
    published_bytes = voxel_bytes(grids)
    if published_bytes > max_retained_voxel_bytes:
        raise ValueError('objects_voxel_byte_budget_exceeded')
    missing = sorted(required - set(records))
    identity['object_layers'] = {'entities': records, 'required_entity_ids': sorted(required),
        'retained_voxel_bytes': published_bytes, 'max_retained_voxel_bytes': max_retained_voxel_bytes,
        'retained_visible_sources': dict(retained_visible), 'retired_entity_ids': sorted(retired), 'retained_obstacle_ids': retained,
        'observed_entity_ids': sorted(observed), 'previous_scene_token': previous_scene_token,
        'rebuild_policy': ('refresh_global_scene' if not local else
                           'patch_local_scene' if previous_scene_token else 'initialize_local_scene')}
    identity['global_observation_id'] = (previous_identity.get('global_observation_id') if local
                                         else identity['observation_id'])
    identity['coverage'] = {'objects_complete': not missing and not stale,
        'missing_entity_ids': missing, 'stale_entity_ids': stale,
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
