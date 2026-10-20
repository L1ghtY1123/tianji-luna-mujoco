"""Bind SAM3 candidates to fixed RGB-discovered identities; never discover IDs here."""
import base64
import json
import math
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image

from .observation_labels import project_placement_regions, raster_visual_annotations
from .visual_api import VisualAPI, VisualInventoryError


def choice(name, instructions, values):
    return {'type': 'choice', 'name': name, 'instructions': instructions,
            'choices': [value if isinstance(value, dict) else {'value': value} for value in values]}


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


def reference_content(reference):
    if not reference:
        return []
    content = [{'type': 'input_text', 'text': 'INITIAL RGB REFERENCE. These IDs are permanent; positions may change.'}]
    entities = reference.get('entities', {})
    for camera, view in reference.get('views', {}).items():
        items = [{'id': key, 'box': obj['bbox']} for key, obj in entities.items()
                 if obj.get('reference_camera_id', '3') == camera and obj.get('bbox')]
        content.append({'type': 'input_text', 'text': 'INITIAL REFERENCE RGB camera ' + camera})
        content.append(image_part(view['rgb']))
        content.extend(annotation_content(view['rgb'], camera, items, role='initial_reference_annotations'))
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
                landmarks=None, table_scope=None, placement_regions=(), purpose=None):
        if not entities:
            raise VisualInventoryError('visual_inventory_initial_rgb_required')
        progress = progress or {}
        regions = deepcopy(list(placement_regions or ()))
        for record in progress.values():
            if valid_assigned_slot(record):
                slot = record['assigned_slot']
                regions.append({'region_id': record['region_id'], 'layout_revision': slot['layout_revision'],
                    'layout': {'cells': [slot], 'floor_z_m': slot['floor_z_m'],
                               'support_entity_id': record.get('destination_id'), 'source': 'measured_assigned_slot'}})
        projections = project_placement_regions(capture, regions, require_visible_support=False)
        # Depth/calibration are consumed locally only to draw 2D destination overlays.
        projections = [{key: region[key] for key in ('camera_id', 'region_id', 'slot_id', 'polygon', 'center')
                        if key in region} for region in projections]
        semantic = {key: {field: value for field, value in obj.items() if field in
            {'id', 'kind', 'color', 'description', 'initial_parent_id', 'roles', 'display_id'}}
            for key, obj in entities.items()}
        status = {key: {field: value for field, value in record.items() if field in
            {'status', 'destination_id', 'region_id', 'slot_id', 'stage_index'}} for key, record in progress.items()}
        content = [{'type': 'input_text', 'text': json.dumps({
            'observation_id': capture['observation_id'], 'generation': capture['generation'], 'purpose': purpose,
            'task': instruction, 'fixed_entities': semantic, 'transport_progress': status,
            'rules': 'Use RGB appearance, initial references and scene context to match physical identity. '
            'Current numbered boxes locate SAM3 proposals, not established identities. '
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
        names = ['candidate-' + str(index) for index in range(len(detections))]
        boxes = [mask_box(detection['silhouette_mask']) for detection in detections]
        for camera, view in capture['views'].items():
            content.extend([{'type': 'input_text', 'text': 'CURRENT RGB camera ' + camera}, image_part(view['rgb'])])
            items = [{'id': names[i], 'box': boxes[i]} for i, detection in enumerate(detections)
                     if detection['camera_id'] == camera]
            content.extend(annotation_content(view['rgb'], camera, items, role='current_annotations',
                regions=[region for region in projections if region['camera_id'] == camera]))
        options = [{'value': key, 'description': obj.get('description') or
                    f"{obj.get('color', '')} {obj.get('kind', '')}"}
                   for key, obj in entities.items() if obj.get('kind') not in {'table', 'marker'}]
        questions = []
        for index, detection in enumerate(detections):
            subject = names[index] + ' in camera ' + detection['camera_id']
            content.append({'type': 'input_text', 'text': json.dumps({'candidate': names[index],
                'camera': detection['camera_id'], 'box': boxes[index], 'sam3_prompt_proposal': detection['prompt']})})
            questions.append(choice('identity_' + str(index), f'Which permanent entity is the proposed subject of {subject}? '
                'Locate its numbered box in the annotated view and compare the full RGB images with the initial reference. '
                'Reject unrelated or outside-inventory subjects. '
                'Identify the proposed subject, not every object inside its rectangle. '
                'For a container proposal, identify the container even when its box includes contents. '
                'Unknown when the subject cannot be distinguished or matched to one permanent identity.',
                options + ['reject', 'unknown']))
        for identifier, entity in entities.items():
            if entity.get('kind') == 'table':
                continue
            questions.append(choice('visibility_' + identifier,
                f'Is initial entity {identifier} visible in ANY CURRENT RGB view? '
                'Look at full images regardless of SAM3 proposals. visible requires identifiable pixels; '
                'unseen means absent/occluded; unknown means uncertain identity.', ['visible', 'unseen', 'unknown']))
        for identifier, record in progress.items():
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
            evidence = {**evidence, 'answers': answers}
            result = self._groups(detections, names, entities, judgments, evidence)
            observed = {group[0]['entity_id'] for group in result['groups']}
            result['missing_geometry_ids'] = [key for key, item in entities.items()
                if item.get('kind') not in {'table', 'marker'} and key not in observed]
            result['visibility'] = {key: judgments['visibility_' + key] for key, item in entities.items()
                                    if item.get('kind') != 'table'}
            result['progress'] = {}
            for identifier, record in progress.items():
                state = judgments['progress_' + identifier]
                if ('region_id' in record or 'slot_id' in record) and not valid_assigned_slot(record):
                    state = 'unknown'
                result['progress'][identifier] = {'status': state, 'destination_id': record.get('destination_id'),
                    'region_id': record.get('region_id'), 'slot_id': record.get('slot_id'),
                    'layout_revision': (record.get('assigned_slot') or {}).get('layout_revision'),
                    'observation_id': capture['observation_id'], 'source': self.source}
            return result
        return self.request(payload, capture, purpose, parse)

    def _groups(self, detections, names, entities, judgments, evidence):
        grouped, membership = {}, {key: item.get('visual_membership', 'unknown')
            for key, item in entities.items() if item.get('kind') not in {'table', 'bin', 'marker'}}
        for index, detection in enumerate(detections):
            identity = judgments['identity_' + str(index)]
            if identity == 'reject':
                continue
            if identity == 'unknown':
                answer = evidence['answers'][index]
                raise VisualInventoryError('visual_identity_unknown', evidence={
                    **evidence, 'candidate': names[index], 'camera_id': detection['camera_id'],
                    'choice': answer.get('choice'), 'confidence': answer.get('confidence'),
                    'minimum_confidence': self.minimum_confidence})
            identifier, kind = identity, entities[identity]['kind']
            membership[identifier] = entities[identity].get('visual_membership', 'unknown')
            entry = {**detection, 'entity_id': identifier, 'kind': kind,
                     'visual_membership': membership.get(identifier),
                     'successful_prompts': [(detection['kind'], detection['prompt'])]}
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
        groups = [list(cameras.values()) for cameras in grouped.values()]
        for group in groups:
            prompts = [item for entry in group for item in entry['successful_prompts']]
            prompt = next((text for kind, text in prompts if kind == group[0]['kind']), prompts[0][1])
            for entry in group:
                entry['prompt'] = prompt
        self._assign_surface_pixels(groups)
        return {'groups': groups,
                'membership': membership, 'evidence': evidence}

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
