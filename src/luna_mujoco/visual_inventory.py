"""Bind SAM3 candidates to fixed RGB-discovered identities; never discover IDs here."""
import base64
import hashlib
import json
import math
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image

from .observation_labels import project_placement_regions, raster_visual_annotations
from .visual_api import VisualAPI, VisualInventoryError

CURRENT_ANCHOR_IDENTITY_RULE = (
    'Compare any CURRENT confirmed container anchors for the proposed ID in the same capture, '
    'using their labeled RGB views and calibrated measured world bounds. Multiple cameras may '
    'show one physical object; simultaneously distinct current objects must keep distinct IDs. '
    'Movement between captures cannot explain two distinct subjects present in this same capture. '
    'Do not assign identity from camera number or position alone; keep unknown for ambiguous correspondence. ')


def choice(name, instructions, values):
    return {'type': 'choice', 'name': name, 'instructions': instructions,
            'choices': [deepcopy(value) if isinstance(value, dict) else {'value': value} for value in values]}


def image_part(rgb):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    image.thumbnail((768, 768))
    stream = BytesIO()
    image.save(stream, format='JPEG', quality=90)
    return {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' +
            base64.b64encode(stream.getvalue()).decode('ascii'), 'detail': 'auto'}


def mask_box(mask):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        raise VisualInventoryError('visual_candidate_empty_mask')
    height, width = mask.shape
    return [float(cols.min()/width), float(rows.min()/height),
            float((cols.max()+1)/width), float((rows.max()+1)/height)]


def array_stamp(value):
    array = np.ascontiguousarray(value)
    return {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
            'shape': list(array.shape), 'dtype': array.dtype.str}


def candidate_identity_evidence(candidate, identifier, evidence):
    """Keep each accepted candidate bound to the request that actually answered it."""
    name = 'identity_' + candidate.removeprefix('candidate-')
    correspondence = evidence.get('measured_view_identity_derivations', {}).get(name)
    source_name = correspondence['source_question'] if correspondence else name
    retained = next((row for row in evidence.get('retained_identity_answers', [])
                     if row['answer']['name'] == source_name), None)
    answer = next(row for row in evidence['answers'] if row['name'] == name)
    prefix = 'match_' + source_name.removeprefix('identity_') + '_'
    original = ([retained['answer']] if retained else [row for row in evidence.get(
        'original_decisions_answers', evidence['answers']) if row['name'] == source_name or row['name'].startswith(prefix)])
    questions = ([retained['question']] if retained else [row for row in evidence.get('questions', [])
        if row['name'] == source_name or row['name'].startswith(prefix)])
    provenance = deepcopy(evidence['candidate_provenance'][candidate])
    return {'source': 'openai_decisions', 'entity_id': identifier, 'candidate': candidate,
        'camera': provenance['camera'], 'observation_id': evidence['observation_id'],
        'generation': evidence['generation'],
        'request_hash': retained['request_hash'] if retained else evidence['request_hash'],
        'answer': deepcopy(answer), 'original_decisions_answers': deepcopy(original),
        'questions': deepcopy(questions), 'candidate_provenance': provenance,
        **({'view_correspondence': deepcopy(correspondence)} if correspondence else {})}


def reference_content(reference, *, current_capture=None):
    if not reference:
        return []
    content = [{'type': 'input_text', 'text': 'INITIAL RGB REFERENCE. These IDs are permanent; positions may change.'}]
    current = current_capture or {}
    if (current.get('observation_id') and current['observation_id'] == reference.get('observation_id')
            and type(current.get('generation')) is int and current['generation'] == reference.get('generation')):
        matching = []
        for camera, view in current.get('views', {}).items():
            original = reference.get('views', {}).get(camera, {}).get('rgb')
            if original is not None and 'rgb' in view:
                before, after = np.asarray(original), np.asarray(view['rgb'])
                if before.dtype == after.dtype and np.array_equal(before, after):
                    matching.append(camera)
        if matching:
            content.append({'type': 'input_text', 'text': json.dumps({'same_capture_reference': {
                'observation_id': current['observation_id'], 'generation': current['generation'],
                'verified_camera_ids': sorted(matching), 'identity_or_visibility_proof': False,
                'meaning': 'For these cameras only, INITIAL REFERENCE and CURRENT RGB are the SAME frozen '
                    'capture, with identical shape, dtype and pixels. Annotations are locating proposals '
                    'on those pixels, not a different time. Pixel equality does not prove box accuracy, '
                    'identity, visibility or task success; still judge each requested fact from RGB.'}})})
    entities = reference.get('entities', {})
    for camera, view in reference.get('views', {}).items():
        items = [{'id': key, 'box': obj['bbox']} for key, obj in entities.items()
                 if obj.get('reference_camera_id', '3') == camera and obj.get('bbox')]
        content.append({'type': 'input_text', 'text': 'INITIAL REFERENCE RGB camera ' + camera})
        content.append(image_part(view['rgb']))
        content.extend(annotation_content(view['rgb'], camera, items, role='initial_reference_annotations'))
    return content


def product_reference_content(reference, capture, entities):
    """Show a product's own verified RGB history, bound to its independent ID."""
    content = []
    reference = reference or {}
    for identifier, stored in reference.get('product_references', {}).items():
        if identifier not in entities:
            continue
        evidence = stored.get('evidence', {})
        proof = stored.get('verification_capture', {})
        entity = entities[identifier]
        if (not reference.get('owner_id') or evidence.get('owner_id') != reference['owner_id']
                or evidence.get('output_entity_id') != identifier or not evidence.get('request_hash')
                or stored.get('entity', {}).get('id') != identifier
                or stored['entity'].get('creation_evidence') != evidence
                or entity.get('creation_evidence') != evidence
                or any(value.get('generation') != capture.get('generation') for value in (stored, proof, evidence))
                or not stored.get('observation_id')
                or any(value.get('observation_id') != stored['observation_id'] for value in (proof, evidence))
                or type(proof.get('time')) not in (int, float) or not math.isfinite(proof['time'])
                or proof['time'] > capture['time']
                or set(stored.get('views', {})) != {'1', '2', '3'}):
            continue
        parts = []
        for camera, view in stored['views'].items():
            rgb = view.get('rgb')
            stamp = proof.get('images', {}).get(camera, {})
            if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                    or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size
                    or list(rgb.shape) != stamp.get('shape')
                    or hashlib.sha256(rgb.tobytes()).hexdigest() != stamp.get('rgb_sha256')):
                break
            metadata = {'image_role': 'verified_product_reference', 'entity_id': identifier,
                'description': entity.get('description'), 'camera_id': camera,
                'observation_id': stored['observation_id'], 'request_hash': evidence['request_hash'],
                'pixels_identical_to_current_same_camera': bool(np.array_equal(
                    rgb, capture.get('views', {}).get(camera, {}).get('rgb'))),
                'pixel_comparison_meaning': 'Exact RGB equality verified by code, '
                    'not identity inferred from image similarity.',
                'meaning': 'Historical RGB when this independent product was first visually confirmed. '
                    'Use its described appearance and visible surroundings to match current candidates. '
                    'It is not the consumed input in INITIAL images. Other subjects in this full frame '
                    'do not acquire this ID. This reference does not prove current location or success; '
                    'retain unknown for ambiguous correspondence.'}
            parts.extend([{'type': 'input_text', 'text': json.dumps(metadata)}, image_part(rgb)])
        else:
            content.extend(parts)
    return content


def annotation_content(rgb, camera, items, *, role, regions=()):
    metadata = {'image_role': role, 'camera': camera, 'items': items, 'regions': regions,
                'coordinates': 'normalized to original RGB; labels are in the added right margin'}
    return [{'type': 'input_text', 'text': json.dumps(metadata)},
            image_part(raster_visual_annotations(rgb, items, regions=regions))]


def valid_assigned_slot(record):
    slot = record.get('assigned_slot') or {}
    if (not record.get('region_id') or record.get('slot_id') is None
            or slot.get('slot_id') != record['slot_id'] or not slot.get('layout_revision')):
        return False
    try:
        polygon = np.asarray(slot.get('polygon_world_xy_m'), dtype=float)
        center = np.asarray(slot.get('center_world_xy_m'), dtype=float)
        return (polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3
                and np.isfinite(polygon).all() and center.shape == (2,) and np.isfinite(center).all()
                and math.isfinite(slot.get('floor_z_m')))
    except (ValueError, TypeError):
        return False


class VisualInventory(VisualAPI):
    endpoint = '/v1/decisions'
    source = 'openai_decisions'
    error_prefix = 'visual_decisions'

    def __init__(self, *, minimum_confidence=.8, **kwargs):
        super().__init__(**kwargs)
        if not .5 < minimum_confidence <= 1.:
            raise ValueError('visual_decisions_invalid_confidence')
        self.minimum_confidence = minimum_confidence

    def resolve(self, capture, detections, *, instruction, reference=None, entities=None, progress=None,
                landmarks=None, table_scope=None, placement_regions=(), purpose=None, source_inventory=None,
                identity_context=None, candidate_indices=None, progress_question_ids=None, reference_capture=None):
        if not entities:
            raise VisualInventoryError('visual_inventory_initial_rgb_required')
        progress = progress or {}
        progress_records = {key: record for key, record in progress.items()
                            if progress_question_ids is None or key in progress_question_ids}
        progress_questions = {key: record for key, record in progress_records.items()
                              if isinstance(record.get('destination_id'), str) and record['destination_id'].strip()}
        regions = deepcopy(list(placement_regions or ()))
        for record in progress_questions.values():
            if valid_assigned_slot(record):
                slot = record['assigned_slot']
                regions.append({'region_id': record['region_id'], 'layout_revision': slot['layout_revision'],
                    'layout': {'cells': [slot], 'floor_z_m': slot['floor_z_m'],
                               'support_entity_id': record.get('destination_id'), 'source': 'measured_assigned_slot'}})
        projections = project_placement_regions(capture, regions, require_visible_support=False)
        # Depth/calibration are consumed locally only to draw 2D destination overlays.
        projections = [{key: region[key] for key in ('camera_id', 'region_id', 'slot_id', 'polygon', 'center')
                        if key in region} for region in projections]
        # The full layout and a target's assigned slot can project the same cell.
        projections = list({json.dumps(region, sort_keys=True): region for region in projections}.values())
        semantic = {key: {field: value for field, value in obj.items() if field in
            {'id', 'kind', 'color', 'description', 'appearance_keywords', 'initial_parent_id', 'roles', 'display_id'}}
            for key, obj in entities.items()}
        status = {key: {field: value for field, value in record.items() if field in
            {'status', 'destination_id', 'region_id', 'slot_id', 'transfer_index', 'release_feedback_id'}}
            for key, record in progress.items()}
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture['generation'], 'purpose': purpose,
            'task': instruction, 'fixed_entities': semantic, 'transport_progress': status,
            **({'source_inventory': source_inventory} if source_inventory is not None else {}),
            **(identity_context or {}),
            'rules': 'Use RGB appearance, initial references and scene context to match physical identity. '
            'Current numbered boxes locate SAM3 proposals, not established identities. '
            'appearance_keywords are initial visible-description proposals, not verified facts or extra '
            'evidence. Compare them against current pixels; a contradictory view must remain uncertain. '
            'Use the original full RGB images and their annotated versions; '
            'no segmentation contours or depth measurements are supplied. SAM3 prompts are proposals, not truth. '
            'A box may enclose other objects, especially container contents. Match the proposed subject, not every pixel. '
            'Boxes only locate subjects; they do not show segmentation pixel ownership. Do not judge mask quality. '
            'Several camera candidates or repeated same-frame proposals may show one entity. '
            'Distinct identical objects must keep distinct IDs. A large movement is allowed. '
            'If identity cannot be established (including identical objects exchanged under occlusion), choose unknown. '
            'Only inventory entities on the designated working table or its containers are in scope, including '
            'known entities subsequently lifted or moved. Reject background, robot and objects outside this inventory. '
            'Visibility is the union of current cameras, not a requirement for all cameras to see each entity. '
            'Judge visibility and destination directly from full RGB even if SAM3 missed an object. '
            'Action history is context, never proof of success. Region overlays are locally projected references, '
            'not evidence of visible support or exact object pose.'}, ensure_ascii=False)}]
        content.extend(reference_content(reference))
        content.extend(product_reference_content(reference, capture, entities))
        indices = list(range(len(detections))) if candidate_indices is None else list(candidate_indices)
        if (len(indices) != len(detections) or len(set(indices)) != len(indices)
                or any(type(index) is not int or index < 0 for index in indices)):
            raise VisualInventoryError('visual_inventory_candidate_indices_invalid')
        names = ['candidate-' + str(index) for index in indices]
        boxes = [mask_box(detection['silhouette_mask']) for detection in detections]
        view_stamps = {camera: {key: array_stamp(view[key]) for key in
            ('rgb', 'depth_m', 'valid_depth', 'K', 'T_world_camera') if key in view}
            for camera, view in capture['views'].items()}
        candidate_provenance = {name: {
            'observation_id': capture['observation_id'], 'generation': capture['generation'],
            'candidate': name, 'camera': detection['camera_id'],
            'stage': 'segmentation_before_inventory_surface_assignment',
            'mask': array_stamp(detection['mask']), 'silhouette_mask': array_stamp(detection['silhouette_mask']),
            'view': view_stamps[detection['camera_id']],
            'calibration': {key: deepcopy(capture['views'][detection['camera_id']][key]) for key in
                ('calibration_id', 'intrinsics', 'frame_id', 'capture_time')
                if key in capture['views'][detection['camera_id']]}}
            for name, detection in zip(names, detections, strict=True)}
        anchors = (identity_context or {}).get('current_confirmed_container_anchors', [])
        if any(anchor.get('observation_id') != capture['observation_id']
                or anchor.get('generation') != capture['generation']
                or anchor.get('camera') not in capture['views']
                or anchor.get('entity_id') not in entities
                or anchor.get('source') != 'measured_container_correspondence' for anchor in anchors):
            raise VisualInventoryError('visual_current_container_anchor_binding_invalid')
        for camera, view in capture['views'].items():
            content.extend([{'type': 'input_text', 'text': 'CURRENT RGB camera ' + camera}, image_part(view['rgb'])])
            items = [{'id': names[i], 'box': boxes[i]} for i, detection in enumerate(detections)
                     if detection['camera_id'] == camera]
            items.extend({'id': anchor['entity_id'], 'candidate': anchor['candidate'], 'box': anchor['box'],
                'annotation_role': 'current_confirmed_container_anchor'}
                for anchor in anchors if anchor['camera'] == camera)
            content.extend(annotation_content(view['rgb'], camera, items, role='current_annotations',
                regions=[region for region in projections if region['camera_id'] == camera]))
        options = [{'value': key, 'description': obj.get('description') or
                    f"{obj.get('color', '')} {obj.get('kind', '')}"}
                   for key, obj in entities.items() if obj.get('kind') not in {'table', 'marker'}]
        questions = []
        for index, detection in enumerate(detections):
            subject = names[index] + ' in camera ' + detection['camera_id']
            exclusions = [row for row in (identity_context or {}).get('identity_geometry_exclusions', [])
                          if row['candidate'] == names[index] and row['camera'] == detection['camera_id']]
            if any(row['observation_id'] != capture['observation_id']
                   or row['generation'] != capture['generation'] for row in exclusions):
                raise VisualInventoryError('identity_geometry_exclusion_stale')
            excluded_ids = {row['excluded_instance_id'] for row in exclusions}
            eligible_options = [option for option in options if option['value'] not in excluded_ids]
            content.append({'type': 'input_text', 'text': json.dumps({'candidate': names[index],
                'camera': detection['camera_id'], 'box': boxes[index], 'sam3_prompt_proposal': detection['prompt'],
                'candidate_provenance': candidate_provenance[names[index]]})})
            question = choice('identity_' + str(indices[index]), f'Which permanent entity is the proposed subject of {subject}? '
                'Locate its numbered box in the annotated view and compare current RGB with that entity\'s '
                'INITIAL or verified PRODUCT reference. A created product has its own appearance and ID. '
                'Reject unrelated or outside-inventory subjects. '
                'Identify the proposed subject, not every object inside its rectangle. '
                'For a container proposal, identify the container even when its box includes contents. '
                'Unknown when the subject cannot be distinguished or matched to one permanent identity.',
                eligible_options + ['reject', 'unknown'])
            if exclusions:
                question['instructions'] += (
                    ' The supplied identity_geometry_exclusions prove that some current surfaces are too far '
                    'apart to belong to that same known rigid instance; those IDs are not choices. '
                    'This does not identify the subject or exclude another instance using the same CAD.')
            if anchors:
                question['instructions'] += ' ' + CURRENT_ANCHOR_IDENTITY_RULE
            questions.append(question)
        for identifier, entity in entities.items():
            if entity.get('kind') == 'table':
                continue
            questions.append(choice('visibility_' + identifier,
                f'Is declared entity {identifier} visible in ANY CURRENT RGB view? '
                'Look at full images regardless of SAM3 proposals. visible requires identifiable pixels; '
                'unseen means absent/occluded; unknown means uncertain identity.', ['visible', 'unseen', 'unknown']))
        for identifier, record in progress_questions.items():
            slot_rule = ''
            if 'region_id' in record or 'slot_id' in record:
                slot_rule = (f"Assigned region {record.get('region_id')}, slot {record.get('slot_id')}. "
                    'Slot XY specifies the TCP anchor, not object-centre or footprint containment. '
                    'Check visible support/destination consistency; do not infer millimetre TCP accuracy. ')
                if not valid_assigned_slot(record):
                    slot_rule += 'Assigned-slot reference is unavailable; return unknown. '
            questions.append(choice('progress_' + identifier,
                f"Where is entity {identifier} NOW relative to destination {record.get('destination_id')}? "
                + slot_rule + 'at_destination requires visible support on that region or inside that container, '
                'not held. elsewhere means visibly outside it or held; unknown for hidden, uncertain identity '
                'or unknown destination. Use current RGB even without a SAM3 candidate.',
                ['at_destination', 'elsewhere', 'unknown']))
        if len(questions) > 200 or sum(part['type'] == 'input_image' for part in content) > 128:
            raise VisualInventoryError('visual_inventory_input_limit')
        if not questions:
            raise VisualInventoryError('visual_inventory_no_candidates')
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': questions}

        def parse(body, evidence):
            answers = body.get('answers', [])
            if not isinstance(answers, list) or len(answers) != len(questions):
                raise VisualInventoryError('visual_decisions_invalid_response')
            judgments = {}
            for question, answer in zip(questions, answers, strict=True):
                if not isinstance(answer, dict) or answer.get('name') != question['name']:
                    raise VisualInventoryError('visual_decisions_invalid_response')
                value, confidence = answer.get('choice'), answer.get('confidence', 0.)
                if not isinstance(confidence, (int, float)) or not 0. <= confidence <= 1.:
                    raise VisualInventoryError('visual_decisions_invalid_response')
                if answer.get('type') != 'choice' or confidence < self.minimum_confidence:
                    value = 'unknown'
                if value not in [item['value'] for item in question['choices']]:
                    raise VisualInventoryError('visual_decisions_invalid_response')
                judgments[question['name']] = value
            evidence = {**evidence, 'answers': answers, 'generation': capture['generation'],
                        'candidate_provenance': candidate_provenance,
                        'identity_geometry_exclusions': deepcopy(
                            (identity_context or {}).get('identity_geometry_exclusions', []))}
            local_judgments = {key: value for key, value in judgments.items() if not key.startswith('identity_')}
            local_judgments.update({'identity_' + str(local): judgments['identity_' + str(original)]
                                    for local, original in enumerate(indices)})
            bound_detections = []
            for index, detection in enumerate(detections):
                identity = local_judgments['identity_' + str(index)]
                entry = dict(detection)
                if identity in entities:
                    entry['identity_evidence'] = [candidate_identity_evidence(names[index], identity, evidence)]
                bound_detections.append(entry)
            result = self._groups(bound_detections, names, entities, local_judgments, evidence,
                                  purpose=purpose, reference=reference, progress=progress)
            observed = {group[0]['entity_id'] for group in result['groups']}
            result['missing_geometry_ids'] = [key for key, item in entities.items()
                if item.get('kind') not in {'table', 'marker'} and key not in observed]
            result['visibility'] = {key: judgments['visibility_' + key] for key, item in entities.items()
                                    if item.get('kind') != 'table'}
            result['progress'] = {}
            for identifier, record in progress_records.items():
                state = judgments.get('progress_' + identifier, 'unknown')
                if ('region_id' in record or 'slot_id' in record) and not valid_assigned_slot(record):
                    state = 'unknown'
                result['progress'][identifier] = {'status': state, 'destination_id': record.get('destination_id'),
                    'region_id': record.get('region_id'), 'slot_id': record.get('slot_id'),
                    'layout_revision': (record.get('assigned_slot') or {}).get('layout_revision'),
                    'observation_id': capture['observation_id'], 'source': self.source}
                if identifier not in progress_questions:
                    result['progress'][identifier].update(source='unbound_destination',
                                                          reason='progress_destination_unbound')
            return result
        subjects = {name: {'silhouette_mask': detection['silhouette_mask'].copy(),
                           'candidate_provenance': deepcopy(candidate_provenance[name])}
                    for name, detection in zip(names, detections, strict=True)}
        return self.request(payload, capture, purpose, parse, identity_subjects=subjects)

    def request(self, payload, capture, purpose, parse, *, identity_subjects=None):
        # Raw masks stay local; the base categorical request keeps its existing input.
        return super().request(payload, capture, purpose, parse)

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
                raise VisualInventoryError('visual_identity_unknown', evidence=self._identity_failure_evidence(
                    evidence, detection, names[index], answer))
            identifier, kind = identity, entities[identity]['kind']
            if kind not in {'table', 'bin', 'marker'} and np.any(detection['mask']):
                for prior_name, prior_id, prior_detection in confirmed_surfaces:
                    if (prior_id != identifier and prior_detection['camera_id'] == detection['camera_id']
                            and np.array_equal(prior_detection['mask'], detection['mask'])):
                        raise VisualInventoryError('visual_identity_conflict', evidence={
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
            # Same camera aliases become one measured pixel set. Cross-camera
            # entries remain separate so source pixel provenance is retained.
            by_camera = grouped.setdefault(identifier, {})
            camera = detection['camera_id']
            if camera not in by_camera:
                by_camera[camera] = entry
            else:
                prior = by_camera[camera]
                pixels = np.vstack([prior['pixels_rc'], entry['pixels_rc']])
                _, indices = np.unique(pixels, axis=0, return_index=True)
                for field in ('points', 'pixels', 'pixels_rc'):
                    prior[field] = np.vstack([prior[field], entry[field]])[indices]
                prior['mask'] = prior['mask'] | entry['mask']
                prior['silhouette_mask'] = prior['silhouette_mask'] | entry['silhouette_mask']
                prior['successful_prompts'].extend(entry['successful_prompts'])
                prior['identity_evidence'].extend(entry['identity_evidence'])
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
        context = {'candidate_id': candidate, 'camera_id': detection['camera_id'],
                   'prompt': detection['prompt'], 'entity_id': answer.get('choice')
                   if answer.get('choice') not in {'unknown', 'reject'} else None,
                   'sam3_score': float(score) if score is not None else None}
        pair_prefix = 'match_' + str(answer.get('name', '')).removeprefix('identity_') + '_'
        relevant, other = [], []
        for check in evidence.get('diagnostic_checks', []):
            if check['name'] == answer.get('name') or check['name'].startswith(pair_prefix):
                relevant.append(dict(check, **context))
            else:
                other.append(deepcopy(check))
        checks = relevant + other
        return {**evidence, **context, 'candidate': candidate, 'diagnostic_checks': checks,
                'unknowns': [*evidence.get('unknowns', []), *(['sam3_score_not_recorded'] if score is None else [])],
                'choice': answer.get('choice'), 'confidence': answer.get('confidence'),
                'minimum_confidence': self.minimum_confidence}

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
