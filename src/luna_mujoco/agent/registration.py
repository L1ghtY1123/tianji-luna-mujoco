"""Commit accepted RGBD physical identities atomically, outside perception/model calls."""
from copy import deepcopy

from ..core.grounding import identity_provenance
from ..object_memory import restored_object_record
from ..visual_identity import mask_box


class ObjectRegistrar:
    def __init__(self, memory, judgments=None):
        self.memory = memory
        self.world_key = None

    def entities(self):
        return {row['id']: restored_object_record(row).model_dump(mode='json', exclude_none=True)
                for row in self.memory.entities(world_key=self.world_key) if row.get('status', 'active') == 'active'}

    def reference(self):
        return self.memory.reference(self.entities())

    def commit_grounding(self, capture, grounding, *, expected_revision):
        """Caller fences robot command/generation/cancellation; SQLite fences memory revision."""
        observation = grounding.observation
        if (observation.observation_id != capture['observation_id']
                or observation.generation != capture['generation']):
            raise ValueError('grounding_capture_mismatch')
        groups = {group.group_id: group for group in grounding.groups}
        matches = {match.group_id: match for match in grounding.history_matches}
        selected = {match.group_id for match in grounding.history_matches if match.outcome in {'existing', 'new'}}
        metadata = grounding.evidence.get('scene', {}).get('physical_metadata', {})
        owners, candidate_owners = {}, {}
        for identifier in selected:
            if identifier not in groups or identifier not in matches:
                raise ValueError('selected_group_without_history_match')
            group, match = groups[identifier], matches[identifier]
            if group.observation_id != observation.observation_id:
                raise ValueError('group_capture_mismatch')
            for index in group.candidate_indices:
                if index in candidate_owners:
                    raise ValueError('candidate_assigned_to_multiple_groups')
                candidate_owners[index] = identifier
                candidate = observation.candidates[index]
                if candidate['depth_evidence'].get('status') != 'measured':
                    raise ValueError('registration_requires_measured_view')
                if (candidate['observation_id'] != observation.observation_id
                        or candidate['generation'] != observation.generation
                        or candidate['silhouette_mask'].shape != capture['views'][candidate['camera_id']]['depth_m'].shape):
                    raise ValueError('candidate_capture_mismatch')
            if (match.outcome not in {'existing', 'new'} or group.geometry_evidence.get('status') != 'measured'
                    or group.geometry_evidence.get('correspondence_status') != 'confirmed'):
                raise ValueError('selected_group_without_measured_identity')
            if match.entity_id is not None:
                if match.entity_id in owners:
                    raise ValueError('duplicate_permanent_group_identity')
                owners[match.entity_id] = identifier
            cameras = [observation.candidates[i]['camera_id'] for i in group.candidate_indices]
            if len(cameras) != len(set(cameras)):
                raise ValueError('group_camera_conflict')
        group_ids, entities, events, masks = {}, {}, [], {camera: {} for camera in capture['views']}
        with self.memory.transaction():
            prior = self.memory.grounding_ids(selected)
            if (not selected or set(prior) != selected) and self.memory.revision != expected_revision:
                raise ValueError('grounding_database_revision_changed')
            stored = self.entities()
            self.memory.store_capture(capture)
            for group_id in sorted(selected):
                group, match = groups[group_id], matches[group_id]
                if group_id in prior:
                    identifier = prior[group_id]
                    if match.entity_id is not None and match.entity_id != identifier:
                        raise ValueError('grounding_replay_identity_conflict')
                    group_ids[group_id], entities[identifier] = identifier, stored[identifier]
                    continue
                members = [observation.candidates[i] for i in group.candidate_indices]
                primary = next((row for row in members if row['camera_id'] == '3'
                                and row['depth_evidence']['status'] == 'measured'), None)
                if primary is None:
                    raise ValueError('registration_requires_measured_head_mask')
                identifier = match.entity_id
                if identifier is not None and identifier not in stored:
                    raise ValueError('historical_instance_no_longer_active')
                item = deepcopy(stored.get(identifier, {}))
                item.pop('id', None)
                if identifier is None:
                    attributes = metadata.get(group_id)
                    if attributes is None:
                        raise ValueError('physical_attributes_missing')
                    kind, color = attributes['kind'], attributes['color']
                    item.update(kind=kind, color=color, description=attributes.get('description', ''),
                        roles=['table', 'collision'] if kind == 'table' else
                              ['container', 'collision'] if kind == 'bin' else ['collision'],
                        world_key=self.world_key, appearance_keywords=[],
                        segmentation_query=attributes.get('segmentation_query',
                            {'kind': kind, 'color': color, 'prompt': color + ' ' + kind, 'aliases': []}))
                elif item.get('kind') == 'table':
                    current = metadata.get(group_id, {})
                    if current.get('attribute_evidence', {}).get('color_source') == 'measured_head_tabletop_rgb':
                        item.update(color=current['color'], description=current['description'])
                # Keep later wrist queries bound to this task's actual declared subject,
                # including when an existing ID came from a different description.
                scope_hint = next((hint for row in members for hint in row.get('retrieval_hypotheses', [])
                    if row.get('scene_quote') and hint.get('scene_quote') == row['scene_quote']), None)
                if scope_hint:
                    item['segmentation_query'] = {'kind': scope_hint['kind'], 'color': scope_hint['color'],
                        'scene_quote': scope_hint['scene_quote'], 'asset_id': scope_hint.get('asset_id'), 'prompt': scope_hint['prompt'], 'aliases': []}
                provenance = identity_provenance(match)
                item.update(reference_camera_id=primary['camera_id'], reference_observation_id=observation.observation_id,
                    bbox=mask_box(primary['silhouette_mask']), identity_source=provenance['source'],
                    registration_basis='scene_association_v3', visibility='visible', observation_id=observation.observation_id,
                    identity_request_hash=provenance['request_hash'],
                    current_identity_proposal={'source': 'confirmed_rgbd_group',
                        'observation_id': observation.observation_id, 'generation': observation.generation,
                        'boxes': {row['camera_id']: mask_box(row['silhouette_mask']) for row in members},
                        'group_id': group_id})
                item['last_position_measurement'] = {
                    'observation_id': observation.observation_id,
                    'position_world_m': group.geometry_evidence['visible_surface_center_world_m'],
                    'source': 'current_rgbd_group_visible_surface'}
                if identifier is None:
                    item['initial_reference'] = {key: deepcopy(item[key]) for key in (
                        'reference_camera_id', 'reference_observation_id', 'bbox', 'identity_request_hash',
                        'current_identity_proposal', 'world_key')}
                    identifier = self.memory.allocate(str(self.world_key) + ':' + group_id, item)
                entities[identifier] = {'id': identifier, **item}
                group_ids[group_id] = identifier
                for row in members:
                    masks[row['camera_id']][identifier] = row['silhouette_mask']
                events.append((identifier, 'rgbd-grounding:' + group_id, 'instance_rgbd_grounded', item,
                    {'registration_basis': 'scene_association_v3', 'group_id': group_id,
                     'observation_id': observation.observation_id, 'generation': observation.generation,
                     'candidate_ids': [row['candidate_id'] for row in members],
                     'geometry': group.geometry_evidence, 'history_match': match.evidence,
                     'physical_attributes': metadata.get(group_id)}))
            self.memory.store_instance_masks(capture, {'observation_id': observation.observation_id,
                'generation': observation.generation, 'views': {camera: {'identity_masks': value}
                                                               for camera, value in masks.items()}})
            visible_ids = set(group_ids.values())
            for identifier, previous in stored.items():
                if identifier not in visible_ids and previous.get('visibility') != 'unseen':
                    events.append((identifier, 'scene-unseen:' + observation.observation_id,
                        'scene_visibility', {'visibility': 'unseen'},
                        {'observation_id': observation.observation_id, 'reason': 'not_confirmed_in_current_scene',
                         'old_pose_is_current_measurement': False}))
            self.memory.record_many(events)
            entities = self.entities()
        return {'entities': entities, 'group_ids': group_ids,
                'unresolved_candidates': grounding.evidence.get('scene', {}).get('unresolved_groups', []),
                'missing_tracks': grounding.evidence.get('scene', {}).get('missing_tracks', []),
                'coverage': observation.coverage}

    def apply_geometry_policies(self, selectors, identifiers, *, expected_revision):
        policies = [s for s in selectors if s.geometry_policy == 'unknown_tsdf' and s.ref in identifiers]
        if not policies:
            return
        with self.memory.transaction():
            if self.memory.revision != expected_revision:
                raise ValueError('grounding_database_revision_changed')
            entities = self.entities()
            events = []
            for selector in policies:
                ids = identifiers[selector.ref]
                for identifier in ids if isinstance(ids, list) else [ids]:
                    policy = {'mode': 'unknown_tsdf', 'instance_id': identifier,
                              'source': 'user_instruction_interpreted_by_responses',
                              'instruction_quote': selector.geometry_policy_instruction}
                    if entities[identifier].get('geometry_policy') != policy:
                        events.append((identifier, 'instruction-geometry:' + selector.ref, 'geometry_policy_bound',
                            {'geometry_policy': policy}, {'selector_ref': selector.ref}))
            self.memory.record_many(events)
