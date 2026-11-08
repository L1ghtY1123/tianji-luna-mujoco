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

    def lifecycle(self):
        return self._lifecycle.model_dump(mode='json', exclude_none=True)

    def accept_container_occupancy(self, scene, receipt, *, previous_scene_token):
        record = scene.get('inventory_map', {}).get('entities', {}).get(receipt['target_id'], {})
        if (record.get('source') != 'container_occupancy' or record.get('status') != 'retained'
                or record.get('entry_receipt_id') != receipt['receipt_id']
                or record.get('container_id') != receipt['destination_id']):
            raise ValueError('container_occupancy_publication_mismatch')
        self.replace_geometry(scene, previous_scene_token=previous_scene_token)
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': [identifier for identifier in self._lifecycle.invalidated_entity_ids
                                       if identifier != receipt['target_id']]})
        self._refresh_required = (scene.get('coverage', {}).get('inventory_complete') is not True
                                  or bool(self._lifecycle.invalidated_entity_ids))

    def collision_snapshot(self):
        return CollisionSnapshot(scene=deepcopy(self._scene),
            observation=deepcopy(self._source_observation),
            refresh_required=self._refresh_required)

    def request_update(self, request: CollisionUpdate):
        if request.requested_scope == 'local' and not self.initialized:
            raise ValueError('local_mapping_requires_global_base')
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | request.model_dump())

    def require_entities(self, required_ids, retired_ids):
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'required_entity_ids': sorted(set(required_ids) - set(retired_ids)),
            'retired_entity_ids': sorted(set(retired_ids))})

    def invalidate(self, identifiers):
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': sorted(set(self._lifecycle.invalidated_entity_ids) | set(identifiers))})
        self._refresh_required = True

    def accept_publication(self, scene, observation, *, exclusion_state):
        if (scene.get('observation_id') != observation['observation_id']
                or scene.get('generation') != observation['generation'] or not scene.get('scene_token')
                or scene.get('exclusion_state') != exclusion_state):
            raise ValueError('collision_publication_source_mismatch')
        records = scene.get('inventory_map', {}).get('entities', {})
        remaining = [identifier for identifier in self._lifecycle.invalidated_entity_ids
                     if identifier not in self._lifecycle.retired_entity_ids
                     and records.get(identifier, {}).get('status') not in {'observed', 'retained'}]
        self._scene = deepcopy(scene)
        self._source_observation = {key: deepcopy(observation[key]) for key in
                                    ('observation_id', 'generation', 'time') if key in observation}
        self._lifecycle = CollisionLifecycle.model_validate(self._lifecycle.model_dump() | {
            'invalidated_entity_ids': remaining})
        self._refresh_required = scene.get('coverage', {}).get('inventory_complete') is False or bool(remaining)
        if self._lifecycle.requested_scope == 'global' and not self._refresh_required:
            self._initialized = True
        if exclusion_state != 'observation':
            self._validated_capture_id = observation['observation_id']
