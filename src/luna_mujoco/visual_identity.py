"""Associate current measured subjects with permanent instance memory."""
import hashlib
import math
from contextlib import nullcontext
from copy import deepcopy

import numpy as np

from .core.decisions import choice_value
from .visual_api import VisualAPI, VisualDecisionError


def candidate_query_terms(candidate):
    """Original retrieval membership survives SAM3 fallback and merged proposals."""
    return {str(query).strip().casefold() for query in [candidate.get('prompt', ''),
        *candidate.get('query_prompts', []), *candidate.get('prompt_proposals', [])] if query}


def choice(name, instructions, values):
    meanings = {
        'new_instance': 'A distinct physical object whose permanent identity has not yet been registered.',
        'reject': 'The queried segmentation represents background or robot geometry.',
        'unknown': 'The supplied evidence cannot determine the answer.',
    }
    return {'type': 'choice', 'name': name, 'instructions': instructions,
            'choices': [deepcopy(value) if isinstance(value, dict) else {
                'value': value, **({'description': meanings[value]} if value in meanings else {})} for value in values]}


def image_part(rgb):
    from .raw_visual_input import image_part as encode
    return encode(rgb)


def mask_box(mask):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise VisualDecisionError('visual_candidate_empty_mask')
    height, width = mask.shape
    return [float(cols.min()/width), float(rows.min()/height),
            float((cols.max()+1)/width), float((rows.max()+1)/height)]


def array_stamp(value):
    array = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'shape': list(array.shape), 'dtype': array.dtype.str}


def candidate_identity_evidence(candidate, identifier, evidence):
    """Keep each accepted candidate bound to the request that actually answered it."""
    name = evidence.get('candidate_question_names', {}).get(candidate, 'identity_' + candidate.removeprefix('candidate-'))
    answer = next(row for row in evidence['answers'] if row['name'] == name)
    comparison = next((row for row in reversed(evidence.get('identity_comparison_records', []))
        if any(question['name'] == name for question in row.get('questions', []))), {})
    return {'source': 'openai_decisions', 'entity_id': identifier, 'candidate': candidate,
        'camera': evidence['candidate_provenance'][candidate]['camera'],
        'observation_id': evidence['observation_id'], 'generation': evidence['generation'],
        'request_hash': evidence.get('question_requests', {}).get(name, {}).get('request_hash', evidence['request_hash']),
        'answer': deepcopy(answer),
        'identity_binding': {key: deepcopy(value) for key, value in comparison.items() if key in {
            'identity_stage', 'current_instance_label', 'previous_observation_id',
            'previous_instance_owners', 'permanent_id_inheritance', 'archive_owners'}},
        'original_decisions_answers': [deepcopy(row) for row in evidence['original_decisions_answers'] if row['name'] == name],
        'questions': [deepcopy(row) for row in evidence.get('questions', []) if row['name'] == name],
        'candidate_provenance': deepcopy(evidence['candidate_provenance'][candidate])}


def reference_content(reference, *, current_capture=None, per_instance=False):
    from .raw_visual_input import reference_content as raw_reference
    return raw_reference(reference, current_capture=current_capture)






def valid_placement_region(record):
    region = record.get('placement_region') or {}
    if not record.get('region_id') or not region.get('layout_revision'):
        return False
    try:
        polygon = np.asarray(region.get('polygon_world_xy_m'), dtype=float)
        center = np.asarray(region.get('center_world_xy_m'), dtype=float)
        return (polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3
                and np.isfinite(polygon).all() and center.shape == (2,) and np.isfinite(center).all()
                and math.isfinite(region.get('floor_z_m')))
    except (ValueError, TypeError):
        return False


class VisualIdentity(VisualAPI):
    endpoint = '/v1/decisions'
    source = 'openai_decisions'
    error_prefix = 'visual_decisions'

    def __init__(self, *, register_instance=None, load_identity_archive=None,
                 registration_transaction=nullcontext, **kwargs):
        super().__init__(**kwargs)
        self.register_instance = register_instance
        self.load_identity_archive = load_identity_archive
        self.registration_transaction = registration_transaction

    def resolve(self, capture, detections, *, instruction, entities=None, **kwargs):
        from .identity_association import associate_instances
        return associate_instances(self, capture, detections, instruction=instruction, entities=entities, **kwargs)

    def _groups(self, detections, names, entities, judgments, evidence, *, purpose=None, reference=None,
                progress=None):
        grouped, membership = {}, {key: item.get('visual_membership', 'unknown')
            for key, item in entities.items() if item.get('kind') not in {'table', 'bin', 'marker'}}
        confirmed_surfaces = []
        for index, detection in enumerate(detections):
            identity = judgments['identity_' + str(index)]
            if identity == 'reject':
                continue
            if identity == 'unknown':
                answer = evidence['answers'][index]
                raise VisualDecisionError('visual_identity_unknown', evidence=self._identity_failure_evidence(
                    evidence, detection, names[index], answer))
            identifier, kind = identity, entities[identity]['kind']
            if kind not in {'table', 'bin', 'marker'} and np.any(detection['mask']):
                for prior_name, prior_id, prior_detection in confirmed_surfaces:
                    if (prior_id != identifier and prior_detection['camera_id'] == detection['camera_id']
                            and np.array_equal(prior_detection['mask'], detection['mask'])):
                        raise VisualDecisionError('visual_identity_conflict', evidence={
                            **evidence, 'camera_id': detection['camera_id'],
                            'candidate_ids': [prior_name, names[index]], 'entity_ids': [prior_id, identifier],
                            'shared_pixel_count': int(np.count_nonzero(detection['mask'])),
                            'reason': 'identical_measured_surface_assigned_to_distinct_objects'})
                confirmed_surfaces.append((names[index], identifier, detection))
            membership[identifier] = entities[identity].get('visual_membership', 'unknown')
            entry = {**detection, 'entity_id': identifier, 'kind': kind,
                     'visual_membership': membership.get(identifier),
                     'successful_prompts': [(detection['kind'], detection['prompt'])],
                     'identity_evidence': deepcopy(detection.get('identity_evidence', []))}
            by_camera = grouped.setdefault(identifier, {})
            camera = detection['camera_id']
            if camera in by_camera:
                raise VisualDecisionError('identity_unresolved_conflict', evidence={
                    **evidence, 'entity_id': identifier, 'camera_id': camera,
                    'reason': 'multiple_current_masks_for_one_permanent_id'})
            by_camera[camera] = entry
        groups = [list(cameras.values()) for cameras in grouped.values()]
        for group in groups:
            prompts = [item for entry in group for item in entry['successful_prompts']]
            prompt = next((text for kind, text in prompts if kind == group[0]['kind']), prompts[0][1])
            for entry in group:
                entry['prompt'] = prompt
        self._assign_surface_pixels(groups)
        return {'groups': groups,
                'membership': membership, 'evidence': evidence}

    def _identity_failure_evidence(self, evidence, detection, candidate, answer):
        score = detection.get('score')
        selected = choice_value(answer) if answer.get('type') == 'choice' else None
        identifier = answer.get('identity_candidate', selected)
        context = {'candidate_id': candidate, 'camera_id': detection['camera_id'],
                   'prompt': detection['prompt'], 'entity_id': identifier
                   if identifier not in {'unknown', 'reject'} else None,
                   'sam3_score': float(score) if score is not None else None}
        relevant, other = [], []
        for check in evidence.get('diagnostic_checks', []):
            if check['name'] == answer.get('name'):
                relevant.append(dict(check, **context))
            else:
                other.append(deepcopy(check))
        checks = relevant + other
        return {**evidence, **context, 'candidate': candidate, 'diagnostic_checks': checks,
                'unknowns': [*evidence.get('unknowns', []), *(['sam3_score_not_recorded'] if score is None else [])],
                'choice': answer.get('choice'), 'selected_value': selected, 'confidence': answer.get('confidence'),
                'probability': answer.get('probability'),
                **({'predicate_probability_threshold': .5, 'predicate_comparison': '>'}
                   if answer.get('type') == 'predicate' else {'selection_rule': 'probability_argmax'})}

    @staticmethod
    def _assign_surface_pixels(groups):
        """One depth pixel represents one surface, never two entity geometries."""
        entries = [entry for group in groups for entry in group]
        original = [entry['mask'].copy() for entry in entries]
        for index, entry in enumerate(entries):
            remove = np.zeros_like(original[index])
            for other_index, other in enumerate(entries):
                if entry['entity_id'] == other['entity_id'] or entry['camera_id'] != other['camera_id']:
                    continue
                # A box mask can cover its contents. Keep the confirmed workpiece
                # surface; discard shared pixels between two ordinary objects.
                if entry['kind'] != 'bin' and other['kind'] == 'bin':
                    continue
                remove |= original[other_index]
            entry['mask'] = original[index] & ~remove
            entry['silhouette_mask'] = entry['silhouette_mask'] & ~remove
            rows, cols = np.asarray(entry['pixels_rc']).T
            keep = entry['mask'][rows, cols]
            for field in ('points', 'pixels', 'pixels_rc'):
                entry[field] = np.asarray(entry[field])[keep]
