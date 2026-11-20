"""Published collision scene and its source, version and validity state."""
from copy import deepcopy

import numpy as np

from .memory_contracts import (
    CollisionLifecycle,
    CollisionSnapshot,
    CollisionUpdate,
)


def local_geometry_change(anchor, points, views):
    """Compare current depth to the last published geometry of this same physical ID."""
    from .association_geometry import projected_depth_masks

    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)] if points.ndim == 2 and points.shape[1] == 3 else []
    # A missing face, changed centroid or a different number of pixels is common in a wrist view.
    tolerance_m, minimum_fraction, minimum_pixels = .02, .60, 30
    if len(points) < minimum_pixels:
        return None
    lower, upper = np.asarray(anchor['bounds'])
    beyond_face = np.maximum(np.mean(points < lower - tolerance_m, axis=0),
                             np.mean(points > upper + tolerance_m, axis=0)).max()
    if beyond_face >= minimum_fraction:
        return {'reason': 'surface_beyond_published_bounds', 'fraction': float(beyond_face),
                'reference_observation_id': anchor['observation_id'], 'tolerance_m': tolerance_m}
    for camera, view in views.items():
        projected, visible, supported, _ = projected_depth_masks(anchor['surface_samples'], view, tolerance_m)
        seen = int(projected.sum())
        cleared = int((visible & ~supported).sum())
        if seen >= minimum_pixels and cleared >= minimum_pixels and cleared / seen >= minimum_fraction:
            return {'reason': 'depth_sees_through_previous_surface', 'camera_id': camera,
                    'fraction': cleared / seen, 'reference_observation_id': anchor['observation_id'],
                    'tolerance_m': tolerance_m}
    return None


class SpatialMemory:
    def __init__(self):
        self._scene: dict = {}
        self._source_observation: dict = {}
        self._initialized = False
        self._refresh_required = False
        self._validated_capture_id: str | None = None
        self._lifecycle = CollisionLifecycle()
        self._database_revision = 0
        self._object_states = {}
        self._geometry_anchors = {}

    def geometry_history(self, generation):
        if self._scene.get('generation') != generation:
            return {}
        invalid = set(self._lifecycle.invalidated_entity_ids) | set(self._lifecycle.changed_by_operation_ids)
        records = self._scene.get('object_layers', {}).get('entities', {})
        return {key: deepcopy(anchor) for key, anchor in self._geometry_anchors.items()
                if anchor.get('generation') == generation and key not in invalid
                and records.get(key, {}).get('status') in {'observed', 'retained'}}


    @property
    def database_revision(self):
        return self._database_revision

    def synchronize_database(self, changes, *, world_key=None):
        """Apply committed physical state; changed geometry stays blocked until publication."""
        geometry_fields = ('kind', 'status', 'asset_id', 'geometry_policy',
                           'last_motion_receipt', 'transport_status', 'container_entry', 'input_entity_id', 'output_entity_id')
        invalid = []
        physically_changed = []
        for change in changes:
            if change['revision'] <= self._database_revision:
                continue
            identifier, state = change['entity_id'], deepcopy(change['state'])
            previous = change['previous']
            self._database_revision = change['revision']
            if world_key is not None and state.get('world_key') != world_key:
                continue
            self._object_states[identifier] = state
            geometry_changed = previous is None or any(
                previous.get(key) != state.get(key) for key in geometry_fields)
            if geometry_changed and state.get('kind') != 'marker':
                physically_changed.append(identifier)
            reference = state.get('geometry_reference') or {}
            change = reference.get('local_geometry_change') or {}
            anchor = self._geometry_anchors.get(identifier)
            if anchor and change.get('reference_observation_id') == anchor['observation_id']:
                geometry_changed = True
            if state.get('kind') != 'marker' and geometry_changed:
                invalid.append(identifier)
        if self._scene:
            self._scene.update(database_revision=self._database_revision,
                               object_states=deepcopy(self._object_states))
        if invalid:
            self.invalidate(invalid, changed_by_operation=physically_changed)

    def accept_attachment(self, identifier, evidence):
        if (evidence.get('attached') is not True or evidence.get('instance_id') != identifier
                or evidence.get('scene_token') != self._scene.get('scene_token')):
            raise ValueError('collision_attachment_source_mismatch')
        record = self._scene.get('object_layers', {}).get('entities', {}).get(identifier)
        if record is not None:
            record.update(status='carried', source='visually_verified_attachment',
                          database_revision=self._database_revision)
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': [key for key in self._lifecycle.invalidated_entity_ids if key != identifier]})
        self._refresh_required = (self._scene.get('coverage', {}).get('objects_complete') is not True
                                  or bool(self._lifecycle.invalidated_entity_ids))

    @property
    def scene(self):
        return deepcopy(self._scene)

    @property
    def initialized(self):
        return self._initialized

    @property
    def refresh_required(self):
        return self._refresh_required

    @property
    def validated_capture_id(self):
        return self._validated_capture_id

    def replace_geometry(self, scene, *, previous_scene_token):
        """Commit a bound prediction/invalidation result, without claiming a new observation."""
        if (previous_scene_token != self._scene.get('scene_token')
                or not scene.get('scene_token') or scene.get('generation') != self._scene.get('generation')):
            raise ValueError('collision_geometry_update_source_mismatch')
        self._scene = deepcopy(scene)
        self._scene.update(database_revision=self._database_revision,
                           object_states=deepcopy(self._object_states))

    def lifecycle(self):
        return self._lifecycle.model_dump(mode='json', exclude_none=True)

    def accept_container_occupancy(self, scene, receipt, *, previous_scene_token):
        record = scene.get('object_layers', {}).get('entities', {}).get(receipt['target_id'], {})
        if (record.get('source') != 'container_contents_pending' or record.get('status') != 'stale'
                or record.get('entry_receipt_id') != receipt['receipt_id']
                or record.get('container_id') != receipt['destination_id']):
            raise ValueError('container_occupancy_publication_mismatch')
        self.replace_geometry(scene, previous_scene_token=previous_scene_token)
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': sorted(set(self._lifecycle.invalidated_entity_ids)
                                             | {receipt['target_id']})})
        self._refresh_required = (scene.get('coverage', {}).get('objects_complete') is not True
                                  or bool(self._lifecycle.invalidated_entity_ids))

    def collision_snapshot(self):
        return CollisionSnapshot(scene=deepcopy(self._scene),
            observation=deepcopy(self._source_observation),
            refresh_required=self._refresh_required)

    def request_update(self, request: CollisionUpdate):
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | request.model_dump())

    def require_entities(self, required_ids, retired_ids):
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'required_entity_ids': sorted(set(required_ids) - set(retired_ids)),
            'retired_entity_ids': sorted(set(retired_ids))})

    def invalidate(self, identifiers, *, changed_by_operation=None):
        physical = identifiers if changed_by_operation is None else changed_by_operation
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': sorted(set(self._lifecycle.invalidated_entity_ids) | set(identifiers)),
            'changed_by_operation_ids': sorted(set(self._lifecycle.changed_by_operation_ids) | set(physical))})
        self._refresh_required = True

    def accept_publication(self, scene, observation, *, exclusion_state, database_revision=None):
        if database_revision is not None and database_revision != self._database_revision:
            raise ValueError('collision_database_revision_changed')
        if (scene.get('observation_id') != observation['observation_id']
                or scene.get('generation') != observation['generation'] or not scene.get('scene_token')
                or scene.get('exclusion_state') != exclusion_state):
            raise ValueError('collision_publication_source_mismatch')
        scope = self._lifecycle.requested_scope
        if (scene.get('mapping_strategy', {}).get('requested_scope') != scope
                or scene.get('object_layers', {}).get('previous_scene_token') != self._scene.get('scene_token')):
            raise ValueError('collision_publication_scope_or_parent_changed')
        records = scene.get('object_layers', {}).get('entities', {})
        measured_tables = {identifier for identifier, state in self._object_states.items()
                           if state.get('kind') == 'table' and scene.get('table_collision_enabled') is True}
        remaining = [identifier for identifier in self._lifecycle.invalidated_entity_ids
                     if identifier not in self._lifecycle.retired_entity_ids
                     and identifier not in measured_tables
                     and records.get(identifier, {}).get('status') != 'observed']
        layers = scene.get('collision_layer_bounds', {}).get('layers', {})
        anchors = {key: value for key, value in self._geometry_anchors.items() if key in records}
        for obj in observation.get('objects', []):
            key = obj['id']
            if records.get(key, {}).get('status') != 'observed':
                continue
            layer = layers.get('semantic_instance::' + key, {})
            bounds = np.asarray([layer.get('minimum_world_m'), layer.get('maximum_world_m')], dtype=float)
            points = np.asarray(obj.get('points_world_m', []))
            anchors.pop(key, None)
            if (bounds.shape != (2, 3) or not np.isfinite(bounds).all() or points.ndim != 2
                    or points.shape[1] != 3 or not len(points)):
                continue
            samples = points[::max(1, (len(points) + 511) // 512)]
            anchors[key] = {'observation_id': observation['observation_id'], 'generation': observation['generation'],
                'bounds': bounds.tolist(), 'surface_samples': samples.tolist(),
                'cad': deepcopy(obj.get('cad')), 'placement_reference': deepcopy(obj.get('placement_reference'))}
        self._geometry_anchors = anchors
        self._scene = deepcopy(scene)
        self._scene.update(database_revision=self._database_revision,
                           object_states=deepcopy(self._object_states))
        self._source_observation = {key: deepcopy(observation[key]) for key in
                                    ('observation_id', 'generation', 'time') if key in observation}
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': remaining,
            'changed_by_operation_ids': sorted(set(self._lifecycle.changed_by_operation_ids) & set(remaining))})
        self._refresh_required = scene.get('coverage', {}).get('objects_complete') is False or bool(remaining)
        if self._lifecycle.requested_scope == 'global' and not self._refresh_required:
            self._initialized = True
        if exclusion_state != 'observation':
            self._validated_capture_id = observation['observation_id']
