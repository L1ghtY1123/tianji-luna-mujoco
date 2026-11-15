"""Model proposals update instance memory independently of executable task graphs."""
from copy import deepcopy

import numpy as np

from ..core.decisions import choice_value, predicate_status
from ..object_memory import restored_object_record
from ..visual_api import VisualDecisionError


class ObjectRegistrar:
    """Register model proposals and expose permanent memory independently of task control."""

    source = 'openai_responses_proposal'

    def __init__(self, memory, judgments):
        self.memory = memory
        self.judgments = judgments
        self.world_key = None
        self.audit = None

    def entities(self):
        return {row['id']: restored_object_record(row).model_dump(mode='json', exclude_none=True)
                for row in self.memory.entities(world_key=self.world_key) if row.get('status', 'active') == 'active'}

    def reference(self):
        return self.memory.reference(self.entities())

    @staticmethod
    def workspace_samples(capture, camera, *, subject=None, bbox=None):
        view = capture['views'][camera]
        valid = np.asarray(view['valid_depth'], dtype=bool)
        if 'workspace_mask' not in view:
            raise VisualDecisionError('registration_requires_workspace_filtered_capture')
        mask = np.asarray(view['workspace_mask'], dtype=bool) & valid
        if subject is not None:
            mask = mask & np.asarray(subject['silhouette_mask'], dtype=bool)
        elif bbox is not None:
            height, width = mask.shape
            left, top, right, bottom = np.asarray(bbox) * [width, height, width, height]
            mask = mask[max(0, int(top)):min(height, int(np.ceil(bottom))),
                        max(0, int(left)):min(width, int(np.ceil(right)))]
        return int(np.count_nonzero(mask))

    def bind_proposals(self, proposals, capture, *, instruction, proposal_observations=None):
        """Resolve proposed subjects against permanent memory before allocating IDs."""
        proposals = list(proposals)
        memory = self.entities()
        observations = proposal_observations or {}
        if observations and any(observations.get(key) != capture.get(key)
                                for key in ('observation_id', 'generation')):
            raise VisualDecisionError('registration_observation_binding_mismatch')
        subjects = observations.get('subjects', [])
        current_references = {}
        refs = {proposal.ref for proposal in proposals}
        if len(refs) != len(proposals):
            raise VisualDecisionError('duplicate_scene_reference')
        for proposal in proposals:
            if proposal.existing_id is not None and proposal.existing_id not in memory:
                raise VisualDecisionError('unknown_existing_instance:' + proposal.existing_id)
            if proposal.parent_ref is not None and (
                    proposal.parent_ref == proposal.ref or proposal.parent_ref not in refs | memory.keys()):
                raise VisualDecisionError('invalid_parent_reference')
            if (proposal.geometry_policy == 'unknown_tsdf'
                    and proposal.geometry_policy_instruction not in instruction):
                raise VisualDecisionError('geometry_policy_instruction_not_supplied')
        self.judgments.audit = self.audit
        groups, association_evidence, unassociated = self._associate_proposals(proposals, capture, subjects)
        identifiers, entities, unresolved = {}, {}, []
        new_identifiers = set()
        confirmations = {}
        resolved = []
        # Complete cross-camera association and historical comparison before allocating any ID.
        for proposal in proposals:
            group = groups.get(proposal.ref, [])
            if not group:
                unresolved.append({'proposal': proposal.model_dump(mode='json'), 'status': 'unknown',
                    'reason': 'initial_multiview_identity_unresolved',
                    'evidence': association_evidence.get(proposal.ref, [])})
                continue
            from ..visual_identity import mask_box
            subject = max(group, key=lambda row: row['geometry']['point_count'])
            proposal = proposal.model_copy(update={'bbox': mask_box(subject['silhouette_mask']),
                'reference_camera_id': subject['camera_id']})
            evidence = association_evidence[proposal.ref]
            judged = {'evidence': {**evidence[-1], 'compared_instance_ids': [],
                'identity_comparison_requests': {row['candidate_id']: row['request_hash'] for row in evidence},
                'cross_camera_candidate_ids': [row['candidate_id'] for row in group]}}
            identifier = None
            if memory:
                identifier, judged = self._confirm_proposal(proposal, capture, memory,
                    subject=subject, current_subjects=group,
                    excluded_ids=[row[3] for row in resolved if row[3] is not None])
            if identifier in {'unknown', 'reject'}:
                unresolved.append({'proposal': proposal.model_dump(mode='json'),
                    'status': 'rejected' if identifier == 'reject' else 'unknown',
                    'reason': 'instance_registration_unconfirmed', 'evidence': judged})
                continue
            if identifier is None and not any(self.workspace_samples(capture, row['camera_id'], subject=row)
                                              for row in group):
                unresolved.append({'proposal': proposal.model_dump(mode='json'), 'status': 'rejected',
                    'reason': 'new_instance_outside_workspace'})
                continue
            resolved.append((proposal, group, subject, identifier, judged))
        existing_matches = [row[3] for row in resolved if row[3] is not None]
        conflicts = {identifier for identifier in existing_matches if existing_matches.count(identifier) > 1}
        with self.memory.transaction():
            self.memory.store_capture(capture)
            for proposal, group, subject, identifier, judged in resolved:
                if identifier in conflicts:
                    unresolved.append({'proposal': proposal.model_dump(mode='json'), 'status': 'unknown',
                        'reason': 'multiple_initial_groups_match_one_existing_id', 'evidence': judged})
                    continue
                confirmation = judged['evidence']
                item = {'kind': proposal.kind, 'color': proposal.color,
                    'description': proposal.description, 'bbox': proposal.bbox,
                    'appearance_keywords': list(proposal.appearance_keywords),
                    'reference_camera_id': proposal.reference_camera_id,
                    'reference_observation_id': capture['observation_id'],
                    'identity_source': self.source, 'world_key': self.world_key,
                    'roles': ['table', 'collision'] if proposal.kind == 'table' else ['target_region']
                        if proposal.kind == 'marker' else ['container', 'collision']
                        if proposal.kind == 'bin' else ['collision']}
                if subject is not None and subject['geometry'].get('centroid_world_m') is not None:
                    item['last_position_measurement'] = {
                        'observation_id': capture['observation_id'],
                        'position_world_m': subject['geometry']['centroid_world_m'],
                        'source': 'current_sam3_depth_visible_surface_centroid'}
                if confirmation is not None:
                    item.update(identity_source='openai_decisions',
                                identity_request_hash=confirmation['request_hash'])
                if proposal.kind not in {'table', 'marker'}:
                    item['segmentation_query'] = {'kind': proposal.kind, 'color': proposal.color,
                        'prompt': proposal.segmentation_prompt, 'aliases': list(proposal.segmentation_aliases)}
                if identifier is None:
                    identifier = self.memory.allocate(self.world_key + ':' + capture['observation_id']
                        + ':' + proposal.ref, item)
                    new_identifiers.add(identifier)
                if identifier in entities:
                    raise VisualDecisionError('duplicate_instance_proposal:' + identifier)
                item = {**deepcopy(memory.get(identifier, item)), 'id': identifier,
                        'visual_membership': proposal.membership,
                        'current_identity_proposal': {'observation_id': capture['observation_id'],
                            'camera_id': proposal.reference_camera_id, 'bbox': list(proposal.bbox),
                            'description': proposal.description, 'source': self.source}}
                if subject is not None and subject['geometry'].get('centroid_world_m') is not None:
                    item['last_position_measurement'] = {
                        'observation_id': capture['observation_id'],
                        'position_world_m': subject['geometry']['centroid_world_m'],
                        'source': 'current_sam3_depth_visible_surface_centroid'}
                if confirmation is not None:
                    item.update(identity_source='openai_decisions',
                                identity_request_hash=confirmation['request_hash'])
                    confirmations[identifier] = {key: deepcopy(confirmation[key]) for key in
                        ('request_hash', 'compared_instance_ids', 'identity_comparison_requests') if key in confirmation}
                if proposal.asset_id is not None:
                    item.update(asset_id=proposal.asset_id, asset_source='configured_catalog_responses_proposal')
                if proposal.geometry_policy == 'unknown_tsdf':
                    item['geometry_policy'] = {'mode': 'unknown_tsdf', 'instance_id': identifier,
                        'source': 'user_instruction_interpreted_by_responses',
                        'instruction_quote': proposal.geometry_policy_instruction}
                identifiers[proposal.ref] = identifier
                entities[identifier] = item
                if subject is not None:
                    from ..workspace import retain_instance_masks
                    retain_instance_masks(capture, {row['camera_id']: {identifier: row['silhouette_mask']} for row in group})
                if confirmation is not None and subject is not None:
                    camera = subject['camera_id']
                    view = capture['views'][camera]
                    item.update(reference_observation_id=capture['observation_id'],
                                reference_camera_id=camera, bbox=list(proposal.bbox))
                    current_references[identifier] = {
                        'observation_id': capture['observation_id'], 'generation': capture['generation'],
                        'time': capture['time'], 'camera_id': camera, 'rgb': view['rgb'],
                        'bbox': list(proposal.bbox), 'source': 'openai_decisions',
                        'request_hash': confirmation['request_hash'],
                        'views': {row['camera_id']: {**{key: capture['views'][row['camera_id']][key] for key in (
                            'rgb', 'depth_m', 'valid_depth', 'intrinsics', 'T_world_camera')},
                            'mask': row['silhouette_mask']} for row in group}}
                memory[identifier] = deepcopy(item)
            for proposal in proposals:
                if proposal.ref not in identifiers:
                    continue
                identifier = identifiers[proposal.ref]
                parent = identifiers.get(proposal.parent_ref)
                if proposal.parent_ref in memory:
                    parent = proposal.parent_ref
                entities[identifier]['current_identity_proposal']['parent_id'] = parent
                if identifier in new_identifiers:
                    entities[identifier]['initial_parent_id'] = parent
                if identifier in confirmations:
                    self.memory.record(identifier, 'registration:' + capture['observation_id'] + ':'
                        + confirmations[identifier]['request_hash'], 'instance_registered',
                        restored_object_record(entities[identifier]).model_dump(mode='json', exclude={'id'}, exclude_none=True),
                        {'observation_id': capture['observation_id'], 'source': 'openai_decisions',
                         **confirmations[identifier]})
            if capture.get('registered_instance_masks'):
                self.memory.store_instance_masks(capture, {
                    'observation_id': capture['observation_id'], 'generation': capture['generation'],
                    'views': {camera: {'identity_masks': masks}
                              for camera, masks in capture['registered_instance_masks'].items()}})
            return {'identifiers': identifiers, 'entities': entities, 'unresolved_proposals': unresolved,
                    'identity_references': current_references, 'unassociated_candidates': unassociated}

    def _associate_proposals(self, proposals, capture, candidates):
        """Locate requested task subjects, then associate only their camera masks."""
        from ..subject_assignment import select_subject_groups
        from ..task_subjects import select_task_subject

        def select(identifier, entity, established, excluded, *, current_groups=None):
            return select_task_subject(self.judgments, capture, candidates, identifier, entity,
                established_subjects=established, excluded_indices=excluded, current_groups=current_groups)

        selected, selection_records, camera_groups, correspondence = select_subject_groups(
            self.judgments, capture, candidates,
            {proposal.ref: proposal.model_dump(mode='json') for proposal in proposals}, select)
        groups, evidence = {}, {}
        for proposal in proposals:
            index = selected.get(proposal.ref)
            indices = next((group for group in camera_groups if index in group), [])
            records = selection_records[proposal.ref]
            evidence[proposal.ref] = records
            if not indices:
                continue
            groups[proposal.ref] = [candidates[member] for member in indices]
            evidence[proposal.ref] = [{**records[-1], 'candidate_id': row['candidate_id'],
                'camera_id': row['camera_id'], 'description_ref': proposal.ref,
                'task_subject_selection_requests': records,
                'current_correspondence_requests': correspondence} for row in groups[proposal.ref]]
        return groups, evidence, []

    def _confirm_proposal(self, proposal, capture, memory, *, subject, current_subjects, excluded_ids=()):
        """Use the same temporal comparison and archive recovery as scene observations."""
        from ..temporal_identity import choose_identity
        from ..visual_api import merge_decision_evidence

        identifier, answer, records = choose_identity(self.judgments, capture, current_subjects,
            memory, self.memory.reference(memory), 'registration_identity',
            archive_loader=self.memory.identity_archive, excluded_ids=excluded_ids)
        evidence = merge_decision_evidence(records, capture, 'instance_registration')
        evidence.update(compared_instance_ids=list(memory),
            identity_comparison_requests={key: records[-1]['request_hash'] for key in memory},
            identity_comparison_records=records,
            original_answer=answer)
        return (None if identifier == 'new_instance' else identifier), {'evidence': evidence}

    def register_observed(self, capture, subject, evidence):
        """Allocate an ID only after Decisions confirms a new external physical instance."""
        from ..visual_identity import mask_box
        if not self.workspace_samples(capture, subject['camera_id'], subject=subject):
            raise VisualDecisionError('no_valid_workspace_depth')
        answer = evidence['answer']
        confirmed = (predicate_status(answer.get('probability')) == 'yes' if answer.get('type') == 'predicate'
                     else choice_value(answer) == 'new_instance' if answer.get('type') == 'choice' else False)
        if not confirmed or 'compared_instance_ids' not in evidence:
            raise VisualDecisionError('new_instance_confirmation_required')
        camera = subject['camera_id']
        box = mask_box(subject['silhouette_mask'])
        color = subject.get('color', 'unknown')
        item = {'kind': 'object', 'color': color,
            'description': f'Observed {color} object in camera {camera}, box {box}.',
            'reference_camera_id': camera, 'reference_observation_id': capture['observation_id'],
            'bbox': box, 'world_key': self.world_key, 'roles': ['collision'],
            'identity_source': 'openai_decisions', 'identity_request_hash': evidence['request_hash'],
            'segmentation_query': {'kind': 'object', 'color': color, 'prompt': subject['prompt']},
            'visibility': 'visible'}
        key = self.world_key + ':' + capture['observation_id'] + ':' + answer['name']
        self.memory.store_capture(capture)
        identifier = self.memory.allocate(key, item)
        self.memory.record(identifier, 'registered:' + capture['observation_id'], 'instance_registered',
                           item, {**evidence, 'observation_id': capture['observation_id']})
        return {'id': identifier, **item}
