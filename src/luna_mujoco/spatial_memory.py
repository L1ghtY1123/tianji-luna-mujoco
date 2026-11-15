"""Published collision scene and its source, version and validity state."""
from copy import deepcopy

from .memory_contracts import (
    CollisionLifecycle,
    CollisionSnapshot,
    CollisionUpdate,
)


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

    @property
    def database_revision(self):
        return self._database_revision

    def synchronize_database(self, changes, *, world_key=None):
        """Apply committed physical state; changed geometry stays blocked until publication."""
        geometry_fields = ('kind', 'status', 'asset_id', 'geometry_policy',
                           'last_motion_receipt', 'transport_status', 'container_entry', 'input_entity_id', 'output_entity_id')
        invalid = []
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
            if previous is not None:
                # A new capture identifier alone does not move an obstacle.
                before = previous.get('geometry_reference') or {}
                after = state.get('geometry_reference') or {}
                old_cad, new_cad = before.get('cad') or {}, after.get('cad') or {}
                if old_cad or new_cad:
                    geometry_changed |= old_cad.get('T_world_object') != new_cad.get('T_world_object')
                else:
                    geometry_changed |= before.get('points_sha256') != after.get('points_sha256')
                geometry_changed |= (previous.get('last_position_measurement') or {}).get('position_world_m') != (
                    state.get('last_position_measurement') or {}).get('position_world_m')
            if state.get('kind') != 'marker' and geometry_changed:
                invalid.append(identifier)
        if self._scene:
            self._scene.update(database_revision=self._database_revision,
                               object_states=deepcopy(self._object_states))
        if invalid:
            self.invalidate(invalid)

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
                                             | {receipt['target_id'], receipt['destination_id']})})
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

    def invalidate(self, identifiers):
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': sorted(set(self._lifecycle.invalidated_entity_ids) | set(identifiers))})
        self._refresh_required = True

    def accept_publication(self, scene, observation, *, exclusion_state, database_revision=None):
        if database_revision is not None and database_revision != self._database_revision:
            raise ValueError('collision_database_revision_changed')
        if (scene.get('observation_id') != observation['observation_id']
                or scene.get('generation') != observation['generation'] or not scene.get('scene_token')
                or scene.get('exclusion_state') != exclusion_state):
            raise ValueError('collision_publication_source_mismatch')
        records = scene.get('object_layers', {}).get('entities', {})
        measured_tables = {identifier for identifier, state in self._object_states.items()
                           if state.get('kind') == 'table' and scene.get('table_collision_enabled') is True}
        remaining = [identifier for identifier in self._lifecycle.invalidated_entity_ids
                     if identifier not in self._lifecycle.retired_entity_ids
                     and identifier not in measured_tables
                     and records.get(identifier, {}).get('status') != 'observed'
                     and records.get(identifier, {}).get('source') != 'container_contents_tsdf']
        if (scene.get('object_layers', {}).get('rebuild_policy') == 'replace_from_current_observation'
                and scene.get('coverage', {}).get('objects_complete') is True):
            remaining = []
        self._scene = deepcopy(scene)
        self._scene.update(database_revision=self._database_revision,
                           object_states=deepcopy(self._object_states))
        self._source_observation = {key: deepcopy(observation[key]) for key in
                                    ('observation_id', 'generation', 'time') if key in observation}
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': remaining})
        self._refresh_required = scene.get('coverage', {}).get('objects_complete') is False or bool(remaining)
        if self._lifecycle.requested_scope == 'global' and not self._refresh_required:
            self._initialized = True
        if exclusion_state != 'observation':
            self._validated_capture_id = observation['observation_id']
