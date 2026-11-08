"""Model proposals update instance memory independently of executable task graphs."""
from copy import deepcopy

from ..core.decisions import JudgmentRequest
from ..identity_correspondence import correspondence_instructions
from ..object_memory import restored_object_record
from ..visual_api import VisualInventoryError


class InventoryRegistrar:
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

    def bind_proposals(self, proposals, capture, *, instruction, proposal_observations=None):
        """Reuse named instances. New proposals require a separate novelty judgment."""
        memory = self.entities()
        observations = proposal_observations or {}
        if observations and any(observations.get(key) != capture.get(key)
                                for key in ('observation_id', 'generation')):
            raise VisualInventoryError('registration_observation_binding_mismatch')
        subjects = observations.get('subjects', {})
        current_references = {}
        refs = {proposal.ref for proposal in proposals}
        if len(refs) != len(proposals):
            raise VisualInventoryError('duplicate_scene_reference')
        for proposal in proposals:
            if proposal.existing_id is not None and proposal.existing_id not in memory:
                raise VisualInventoryError('unknown_existing_instance:' + proposal.existing_id)
            if proposal.parent_ref is not None and (
                    proposal.parent_ref == proposal.ref or proposal.parent_ref not in refs | memory.keys()):
                raise VisualInventoryError('invalid_parent_reference')
            if (proposal.geometry_policy == 'unknown_obb'
                    and proposal.geometry_policy_instruction not in instruction):
                raise VisualInventoryError('geometry_policy_instruction_not_supplied')
        identifiers, entities, unresolved = {}, {}, []
        new_identifiers = set()
        confirmations = {}
        self.memory.store_capture(capture)
        for proposal in proposals:
            identifier = proposal.existing_id
            confirmation = None
            if identifier is None or memory[identifier].get('identity_source') == self.source:
                self.judgments.audit = self.audit
                identifier, judged = self._confirm_proposal(proposal, capture, memory,
                    subject=subjects.get(proposal.ref), current_references=current_references)
                if identifier in {'unknown', 'reject'}:
                    unresolved.append({'proposal': proposal.model_dump(mode='json'),
                        'status': 'rejected' if identifier == 'reject' else 'unknown',
                        'reason': 'instance_registration_unconfirmed', 'evidence': judged})
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
                raise VisualInventoryError('duplicate_instance_proposal:' + identifier)
            item = {**deepcopy(memory.get(identifier, item)), 'id': identifier,
                    'visual_membership': proposal.membership,
                    'current_identity_proposal': {'observation_id': capture['observation_id'],
                        'camera_id': proposal.reference_camera_id, 'bbox': list(proposal.bbox),
                        'description': proposal.description, 'source': self.source}}
            if confirmation is not None:
                item.update(identity_source='openai_decisions',
                            identity_request_hash=confirmation['request_hash'])
                confirmations[identifier] = {key: deepcopy(confirmation[key]) for key in
                    ('request_hash', 'compared_instance_ids', 'identity_comparison_requests') if key in confirmation}
            if proposal.asset_id is not None:
                item.update(asset_id=proposal.asset_id, asset_source='configured_catalog_responses_proposal')
            if proposal.geometry_policy == 'unknown_obb':
                item['geometry_policy'] = {'mode': 'unknown_obb', 'instance_id': identifier,
                    'source': 'user_instruction_interpreted_by_responses',
                    'instruction_quote': proposal.geometry_policy_instruction}
            identifiers[proposal.ref] = identifier
            entities[identifier] = item
            # The next proposal must also compare with instances accepted in
            # this batch, including their dated reference boxes.
            subject = subjects.get(proposal.ref)
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
                    'views': {camera: {**{key: view[key] for key in (
                        'rgb', 'depth_m', 'valid_depth', 'intrinsics', 'T_world_camera')},
                        'mask': subject['mask']}}}
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
                self.memory.record(identifier, 'registration:' + capture['observation_id'], 'instance_registered',
                    restored_object_record(entities[identifier]).model_dump(mode='json', exclude={'id'}, exclude_none=True),
                    {'observation_id': capture['observation_id'], 'source': 'openai_decisions',
                     **confirmations[identifier]})
        return {'identifiers': identifiers, 'entities': entities, 'unresolved_proposals': unresolved,
                'identity_references': current_references}


    def _confirm_proposal(self, proposal, capture, memory, *, subject=None, current_references=None):
        """Use the same atomic correspondence then novelty pattern as scene sensing."""
        context = {'verification_scope': 'instance_registration',
                   'proposal': proposal.model_dump(mode='json'),
                   'subject_geometry': {'status': 'unavailable',
                       'reason': 'proposal_mask_or_registered_depth_unavailable'}}
        if subject is not None:
            if (subject['camera_id'] != proposal.reference_camera_id
                    or any(subject['geometry'].get(key) != capture.get(key)
                           for key in ('observation_id', 'generation'))):
                raise VisualInventoryError('registration_subject_binding_mismatch')
            context.update(proposal_mask=subject['silhouette_mask'], subject_geometry=subject['geometry'])
        uncertain = None
        comparisons = {}
        for identifier, entity in memory.items():
            reference = self.memory.reference({identifier: entity})
            if identifier in (current_references or {}):
                reference['identity_references'][identifier] = current_references[identifier]
            comparison = dict(context)
            if subject is not None:
                from ..identity_geometry import projected_reference_surfaces

                rows = projected_reference_surfaces(capture, [subject], reference.get('identity_references', {}))
                comparison['reference_projection'] = [
                    {**{key: value for key, value in row.items() if key != 'target_candidate'}, 'subject_label': 'P'}
                    for row in rows]
            judged = self.judgments.judge(capture, requests=[JudgmentRequest(name='instance',
                instructions=correspondence_instructions('P in camera ' + proposal.reference_camera_id,
                    'R' + identifier, entity['description'], subject_description=proposal.description))],
                context=comparison, reference=reference)
            status = judged['judgments']['instance']['status']
            comparisons[identifier] = judged['evidence']['request_hash']
            if status == 'yes':
                return identifier, judged
            if status == 'unknown':
                uncertain = judged
        if uncertain is not None:
            return 'unknown', uncertain
        judged = self.judgments.judge(capture, requests=[JudgmentRequest(name='instance',
            instructions='The CURRENT subject P in camera ' + proposal.reference_camera_id
                + ' denotes a real external physical scene entity. Movable objects, containers '
                'and tabletop supports qualify. Robot links, grippers, attached cameras, cables '
                'and an empty image region are outside this predicate. Judge only the entity '
                'located by P, using its visible appearance.')], context=context)
        judged['evidence']['compared_instance_ids'] = list(comparisons)
        judged['evidence']['identity_comparison_requests'] = comparisons
        status = judged['judgments']['instance']['status']
        return (None if status == 'yes' else 'reject' if status == 'no' else 'unknown'), judged


    def register_observed(self, capture, subject, evidence):
        """Allocate an ID only after Decisions confirms a new external physical instance."""
        from ..visual_inventory import mask_box
        answer = evidence['answer']
        if (answer.get('type') != 'predicate'
                or answer.get('probability', 0.) < self.judgments.minimum_confidence
                or 'compared_instance_ids' not in evidence):
            raise VisualInventoryError('new_instance_confirmation_required')
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
