"""Associate current measured subjects with permanent instance memory."""
import base64
import hashlib
import json
import math
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image

from .decision_conditions import parse_answers
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
    original = ([retained['answer']] if retained else [row for row in evidence.get(
        'original_decisions_answers', evidence['answers']) if row['name'] == source_name])
    questions = ([retained['question']] if retained else [row for row in evidence.get('questions', [])
        if row['name'] == source_name])
    provenance = deepcopy(evidence['candidate_provenance'][candidate])
    return {'source': 'openai_decisions', 'entity_id': identifier, 'candidate': candidate,
        'camera': provenance['camera'], 'observation_id': evidence['observation_id'],
        'generation': evidence['generation'],
        'request_hash': retained['request_hash'] if retained else evidence.get('question_requests', {}).get(
            source_name, {}).get('request_hash', evidence['request_hash']),
        'answer': deepcopy(answer), 'original_decisions_answers': deepcopy(original),
        'questions': deepcopy(questions), 'candidate_provenance': provenance,
        **({'view_correspondence': deepcopy(correspondence)} if correspondence else {})}


def reference_content(reference, *, current_capture=None):
    if not reference:
        return []
    if reference.get('identity_references'):
        from .observation_labels import outline_masks, raster_visual_annotations

        content = []
        grouped = {}
        for identifier, stored in reference['identity_references'].items():
            if identifier not in reference.get('entities', {}):
                continue
            key = (stored['observation_id'], stored['camera_id'])
            group = grouped.setdefault(key, {'rgb': stored['rgb'], 'time': stored.get('time'), 'items': [], 'masks': []})
            mask = stored.get('views', {}).get(stored['camera_id'], {}).get('mask')
            if mask is not None:
                group['masks'].append(mask)
            group['items'].append({'id': identifier, 'image_label': 'R' + identifier,
                                   'display_label': 'R' + identifier, 'box': stored['bbox']})
        for (capture_id, camera), group in grouped.items():
            current_rgb = (current_capture or {}).get('views', {}).get(camera, {}).get('rgb')
            same_pixels = current_rgb is not None and np.array_equal(group['rgb'], current_rgb)
            content.extend([{'type': 'input_text', 'text': json.dumps({
                'image_role': 'instance_reference', 'camera': camera, 'observation_id': capture_id,
                'capture_time_s': group['time'],
                'mask_contours_supplied': bool(group['masks']),
                'pixels_identical_to_current_same_camera': bool(same_pixels),
                'pixel_comparison_meaning': 'Exact RGB equality checked by code. Boxes are locating '
                    'proposals on the same pixels, and still require an identity judgment.',
                'items': group['items'], 'meaning': 'Dated instance reference. Boxes belong to these '
                    'stored pixels. Current identity still requires correspondence with current RGB.'})},
                image_part(raster_visual_annotations(outline_masks(group['rgb'], group['masks']),
                    group['items'], label_layout='near_box'))])
        return content
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
        located = reference.get('identity_references', {}).get(identifier, {})
        if located.get('request_hash') and located.get('views'):
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
            same_pixels = bool(np.array_equal(rgb, capture.get('views', {}).get(camera, {}).get('rgb')))
            current_creation = stored['observation_id'] == capture['observation_id'] and same_pixels
            metadata = {'image_role': 'verified_product_reference', 'entity_id': identifier,
                'description': entity.get('description'), 'camera_id': camera,
                'observation_id': stored['observation_id'], 'request_hash': evidence['request_hash'],
                'capture_time_s': proof['time'],
                'pixels_identical_to_current_same_camera': same_pixels,
                'pixel_comparison_meaning': 'Exact RGB equality verified by code, '
                    'not identity inferred from image similarity.',
                'meaning': ('The product was visually confirmed in this CURRENT capture. '
                            if current_creation else 'This dated RGB records the visually confirmed product. ')
                    + 'Use the product description and its surroundings to identify its subject within this '
                    'full scene. Other visible subjects retain their separate identities. '
                    'Report uncertainty when several subjects fit the same reference.'}
            parts.extend([{'type': 'input_text', 'text': json.dumps(metadata)}, image_part(rgb)])
        else:
            before = stored.get('before_capture', {})
            if (before.get('observation_id') != evidence.get('before_observation_id')
                    or not before.get('observation_id') or before.get('owner_id') != reference['owner_id']
                    or before.get('target_id') != evidence['input_entity_id']
                    or before.get('generation') != capture['generation']
                    or type(before.get('time')) not in (int, float) or not math.isfinite(before['time'])
                    or before['time'] >= proof['time']
                    or not evidence.get('verification_request_hash')
                    or evidence.get('verified_question', {}).get('type') != 'predicate'
                    or evidence.get('verified_answer', {}).get('name') != evidence['verified_question']['name']
                    or set(stored.get('before_views', {})) != {'1', '2', '3'}):
                continue
            creation = {'entity_id': identifier, 'description': entity['description'],
                'observation_id': evidence['observation_id'],
                'request_hash': evidence['verification_request_hash'],
                'verified_question': deepcopy(evidence['verified_question']),
                'answer': deepcopy(evidence['verified_answer']),
                'meaning': 'This accepted observation established the new product reference, without locating a mask. '
                    'Judge whether the queried candidate is the subject of this observation. '
                    'BEFORE is the recorded pre-processing scene; the product creation references identify '
                    'the verified post-processing capture. Later current captures may show subsequent motion.'}
            if entity.get('processing_skill') == 'vla_unpack':
                from .vla_configuration import vla_processing_effect

                creation['skill_effect'] = vla_processing_effect()
                creation['skill_effect_scope'] = (
                    'Declared processing capability, not proof of this execution. The bound creation '
                    'observation above records whether its expected effect was actually confirmed.')
            before_parts = []
            for camera, view in stored['before_views'].items():
                rgb, stamp = view.get('rgb'), before.get('images', {}).get(camera, {})
                if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                        or list(rgb.shape) != stamp.get('shape')
                        or hashlib.sha256(rgb.tobytes()).hexdigest() != stamp.get('rgb_sha256')):
                    break
                metadata = {'image_role': 'product_creation_before', 'entity_id': identifier,
                    'camera': camera, 'observation_id': before['observation_id'],
                    'capture_time_s': before['time'], 'creation_reference': creation,
                    'meaning': 'BEFORE processing, after verified table release. This image precedes the '
                        'verified product creation; the new product is not expected in this earlier image.'}
                before_parts.extend([{'type': 'input_text', 'text': json.dumps(metadata)}, image_part(rgb)])
            else:
                content.extend([*parts, *before_parts])
    return content


def annotation_content(rgb, camera, items, *, role, regions=()):
    metadata = {'image_role': role, 'camera': camera, 'items': items, 'regions': regions,
                'coordinates': 'normalized to original RGB; labels are in the added right margin'}
    return [{'type': 'input_text', 'text': json.dumps(metadata)},
            image_part(raster_visual_annotations(rgb, items, regions=regions))]


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


class VisualInventory(VisualAPI):
    endpoint = '/v1/decisions'
    source = 'openai_decisions'
    error_prefix = 'visual_decisions'

    def __init__(self, *, minimum_confidence=.8, register_instance=None, **kwargs):
        super().__init__(**kwargs)
        if not .5 < minimum_confidence <= 1.:
            raise ValueError('visual_decisions_invalid_confidence')
        self.minimum_confidence = minimum_confidence
        self.register_instance = register_instance

    def resolve(self, capture, detections, *, instruction, reference=None, entities=None, progress=None,
                landmarks=None, table_scope=None, placement_regions=(), purpose=None, source_inventory=None,
                identity_context=None, candidate_indices=None, progress_question_ids=None, reference_capture=None,
                association_scope=None):
        if not entities:
            raise VisualInventoryError('visual_inventory_initial_rgb_required')
        progress = progress or {}
        progress_records = {key: record for key, record in progress.items()
                            if progress_question_ids is None or key in progress_question_ids}
        progress_questions = {key: record for key, record in progress_records.items()
                              if isinstance(record.get('destination_id'), str) and record['destination_id'].strip()}
        regions = deepcopy(list(placement_regions or ()))
        for record in progress_questions.values():
            if valid_placement_region(record):
                region = record['placement_region']
                regions.append({'region_id': record['region_id'], 'layout_revision': region['layout_revision'],
                    'layout': {'region_polygon_world_xy_m': region['polygon_world_xy_m'],
                               'center_world_xy_m': region['center_world_xy_m'], 'floor_z_m': region['floor_z_m'],
                               'support_entity_id': record.get('destination_id'), 'source': 'measured_placement_region'}})
        projections = project_placement_regions(capture, regions, require_visible_support=False)
        # Depth/calibration are consumed locally only to draw 2D destination overlays.
        projections = [{key: region[key] for key in ('camera_id', 'region_id', 'polygon', 'center')
                        if key in region} for region in projections]
        # The shared destination can appear in more than one transfer record.
        projections = list({json.dumps(region, sort_keys=True): region for region in projections}.values())
        semantic = {key: {field: value for field, value in obj.items() if field in
            {'id', 'kind', 'color', 'description', 'appearance_keywords', 'initial_parent_id', 'roles',
             'current_identity_proposal', 'last_position_measurement', 'observed_support', 'creation_evidence', 'segmentation_query'}}
            for key, obj in entities.items()}
        status = {key: {field: value for field, value in record.items() if field in
            {'status', 'destination_id', 'region_id', 'transfer_index', 'release_feedback_id'}}
            for key, record in progress.items()}
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'capture_time_s': capture.get('time'),
            'generation': capture['generation'], 'purpose': purpose,
            'task': instruction, 'memory_entities': semantic, 'transport_progress': status,
            'association_scope': {key: list((association_scope or {}).get(key, ()))
                                  for key in ('focus_entity_ids', 'deferred_entity_ids')},
            **({'source_inventory': source_inventory} if source_inventory is not None else {}),
            **(identity_context or {}),
            'rules': 'Use RGB appearance, initial references and scene context to match physical identity. '
            'Current numbered boxes locate SAM3 proposals, not established identities. '
            'appearance_keywords are initial visible-description proposals, not verified facts or extra '
            'evidence. Compare them against current pixels; a contradictory view must remain uncertain. '
            'Use supplied CURRENT RGB and the evidence identified in each view. The final single-question '
            'input declares whether candidate mask contours and measured surface locations are supplied. '
            'SAM3 prompts are proposals, not truth. '
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
        content.extend(reference_content(reference, current_capture=capture))
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
                'Choose new_instance for a distinct external physical object absent from memory when offered. '
                'Reject robot parts or non-object regions. '
                'Identify the proposed subject, not every object inside its rectangle. '
                'For a container proposal, identify the container even when its box includes contents. '
                'Unknown when the subject cannot be distinguished or matched to one permanent identity.',
                eligible_options + (['new_instance'] if self.register_instance else []) + ['reject', 'unknown'])
            if exclusions:
                question['instructions'] += (
                    ' The supplied identity_geometry_exclusions prove that some current surfaces are too far '
                    'apart to belong to that same known rigid instance; those IDs are not choices. '
                    'This does not identify the subject or exclude another instance using the same CAD.')
            if anchors:
                question['instructions'] += ' ' + CURRENT_ANCHOR_IDENTITY_RULE
            questions.append(question)
        for identifier, record in progress_questions.items():
            region_rule = ''
            if 'region_id' in record:
                region_rule = (f"Destination region {record.get('region_id')}. "
                    'Check visible support/destination consistency. ')
                if not valid_placement_region(record):
                    region_rule += 'Destination-region reference is unavailable; return unknown. '
            questions.append(choice('progress_' + identifier,
                f"Where is entity {identifier} NOW relative to destination {record.get('destination_id')}? "
                + region_rule + 'at_destination means visibly on the assigned support or inside the container. '
                'elsewhere means visibly outside that destination; unknown means the location cannot be resolved. '
                'Judge location only. Use current RGB even without a SAM3 candidate.',
                ['at_destination', 'elsewhere', 'unknown']))
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': questions}

        def parse(body, evidence):
            registered = evidence.get('registered_entities', {})
            entities.update(registered)
            typed_questions = deepcopy(questions)
            for question, answer in zip(typed_questions, body['answers'], strict=True):
                if question['name'].startswith('identity_'):
                    question['choices'].extend({'value': identifier} for identifier in registered)
                if answer.get('type') == 'predicate' and question['name'].startswith('identity_'):
                    if answer.get('identity_candidate') not in {row['value'] for row in question['choices']}:
                        raise VisualInventoryError('identity_proposal_binding_invalid')
                    question.update(type='predicate')
                    question.pop('choices')
            judgments, answers = parse_answers(body, typed_questions, self.minimum_confidence)
            for answer in answers:
                if answer.get('type') == 'predicate' and answer['name'].startswith('identity_'):
                    judgments[answer['name']] = (answer['identity_candidate']
                        if judgments[answer['name']] == answer.get('identity_accept_status', 'yes') else 'unknown')
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
            result['registered_entities'] = deepcopy(registered)
            observed = {group[0]['entity_id'] for group in result['groups']}
            result['missing_geometry_ids'] = [key for key, item in entities.items()
                if item.get('kind') not in {'table', 'marker'} and key not in observed]
            result['visibility'] = {key: 'visible' if key in result.get('identity_coverage', {}).get(
                'accepted_entity_ids', observed) else 'unknown' for key, item in entities.items()
                if item.get('kind') != 'table'}
            result['progress'] = {}
            for identifier, record in progress_records.items():
                state = judgments.get('progress_' + identifier, 'unknown')
                if 'region_id' in record and not valid_placement_region(record):
                    state = 'unknown'
                result['progress'][identifier] = {'status': state, 'location_judgment': judgments.get('progress_' + identifier),
                    'destination_id': record.get('destination_id'),
                    'region_id': record.get('region_id'),
                    'layout_revision': (record.get('placement_region') or {}).get('layout_revision'),
                    'observation_id': capture['observation_id'], 'source': self.source}
                if identifier not in progress_questions:
                    result['progress'][identifier].update(source='unbound_destination',
                                                          reason='progress_destination_unbound')
            return result
        subjects = {name: {**{key: deepcopy(detection[key]) for key in ('camera_id', 'kind', 'color', 'prompt') if key in detection},
                           'silhouette_mask': detection['silhouette_mask'].copy(),
                           'candidate_provenance': deepcopy(candidate_provenance[name])}
                    for name, detection in zip(names, detections, strict=True)}
        return self.request(payload, capture, purpose, parse, identity_subjects=subjects)

    def request(self, payload, capture, purpose, parse, *, identity_subjects=None):
        # Raw masks stay local; the base categorical request keeps its existing input.
        if sum(part['type'] == 'input_image' for message in payload['input']
               for part in message['content']) > 128:
            raise VisualInventoryError('visual_inventory_input_limit')
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
        identifier = answer.get('identity_candidate', answer.get('choice'))
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
                'choice': answer.get('choice'), 'confidence': answer.get('confidence'),
                'probability': answer.get('probability'),
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
