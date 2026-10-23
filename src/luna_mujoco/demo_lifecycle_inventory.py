"""Associate current visual roles with uniquely recorded unpacking task identities."""
import hashlib
import json
from copy import deepcopy

import numpy as np

from .decision_conditions import condition, parse_answers
from .demo_container_tracking import merge_container_correspondence
from .demo_identity_presentation import _current_candidate_zooms
from .observation_labels import raster_visual_annotations
from .visual_api import VisualAPI, VisualInventoryError
from .visual_inventory import VisualInventory, annotation_content, choice, image_part, mask_box


def _initial_identity_references(reference, entities, bindings):
    if not reference or not reference.get('observation_id'):
        raise VisualInventoryError('visual_lifecycle_reference_required')
    initial = reference.get('entities', {})
    wrapped = {}
    for identifier, binding in bindings.items():
        if binding['role'] != 'wrapped_source':
            continue
        if (identifier not in initial or identifier not in entities
                or entities[identifier].get('display_id') != initial[identifier].get('display_id')):
            raise VisualInventoryError('visual_lifecycle_reference_identity_changed')
        entity = initial[identifier]
        camera = entity.get('reference_camera_id')
        box = np.asarray(entity.get('bbox', []), dtype=float)
        if (camera not in reference.get('views', {}) or 'rgb' not in reference['views'][camera]
                or box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0)
                or np.any(box > 1) or np.any(box[2:] <= box[:2])):
            raise VisualInventoryError('visual_lifecycle_reference_pixels_required')
        number = entity.get('display_id')
        wrapped[identifier] = {'entity_id': identifier, 'display_id': number,
            'display_label': '#' + str(number) if number is not None else identifier,
            'description': entity.get('description', ''), 'reference_camera_id': camera,
            'bbox': box.tolist(), 'reference_observation_id': reference['observation_id']}
    return wrapped


def _initial_reference_content(reference):
    content = [{'type': 'input_text', 'text': 'INITIAL RGB REFERENCES for still-wrapped identity comparisons only. '
        'All original source subjects retain their canonical IDs and display numbers. '
        'Do not compare a currently black workpiece to its obsolete pink appearance.'}]
    sources = set(reference.get('demo_task', {}).get('source_object_ids', []))
    for camera, view in reference['views'].items():
        items = []
        for identifier, entity in reference['entities'].items():
            if identifier not in sources or entity.get('reference_camera_id') != camera or not entity.get('bbox'):
                continue
            number = entity.get('display_id')
            items.append({'id': identifier, 'box': entity['bbox'],
                          'display_label': '#' + str(number) if number is not None else identifier})
        content.extend([{'type': 'input_text', 'text': 'INITIAL REFERENCE RGB camera ' + camera},
                        image_part(view['rgb'])])
        content.extend(annotation_content(view['rgb'], camera, items, role='initial_reference_annotations'))
    return content


def _current_role_request(capture, detections, tracking, containers, instruction):
    """Present current roles with explicit camera and candidate-label conventions."""
    scopes = tracking['candidate_scopes']
    bindings = tracking['bindings']
    context = {'purpose': 'associate_inventory', 'observation_id': capture['observation_id'],
        'generation': capture['generation'], 'task': instruction,
        'evidence_binding_hashes': {
            'lifecycle': hashlib.sha256(json.dumps(tracking, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            'containers': containers['binding_hash']},
        'user_prior': 'Pink foam wraps the black inner workpiece. VLA removes that wrapping. '
            'Finish one process before starting the next. Pink inputs and black products are distinct entities; '
            'the process record links input and output, not physical sameness.',
        'current_roles': {identifier: {'role': binding['role'], 'description': binding['current_description']}
                          for identifier, binding in bindings.items()},
        'rules': 'Use all three CURRENT views together to check the visible subject at each C label. '
            'The camera number locates the candidate; it does not restrict which views may inform the judgment. '
            'A clear view may support another partial view, but never invent an unseen correspondence. '
            'These are SAM proposals and may overlap or repeat the same physical object. '
            'The identity_N role questions judge workpiece appearance. The wrapped_identity_N questions '
            'separately match original pink IDs using INITIAL references; do not count INITIAL subjects as current objects. '
            'Pink inputs are pink foam-wrapped workpieces; unpacked outputs are black workpieces. '
            'Robot parts, the container, and background are not workpieces. '
            'The program checks measured 3D support separately; do not infer tabletop contact from RGB. '
            'A partial view is sufficient if the visible subject is recognizable; otherwise answer unknown. '
            'Each completed collection contains one or more black workpieces or their visible parts; '
            'do not assign individual member IDs. Camera 1 is the right wrist, 2 the left wrist, 3 the head.'}
    content = [{'type': 'input_text', 'text': json.dumps(context, ensure_ascii=False)}]
    raw, annotations, labels = {}, {}, {}
    for camera, view in capture['views'].items():
        part = image_part(view['rgb'])
        raw[('current', camera)] = part
        content.extend([{'type': 'input_text', 'text': 'CURRENT RGB camera ' + camera}, part])
        items = []
        for index, scope in scopes.items():
            if scope['camera_id'] != camera:
                continue
            identifier = 'candidate-' + str(index)
            labels[identifier] = 'C' + str(index)
            items.append({'id': identifier, 'display_label': labels[identifier],
                          'box': mask_box(detections[index]['silhouette_mask'])})
        metadata = {'image_role': 'current_annotations', 'camera': camera, 'items': items}
        annotations[('current', camera)] = metadata
        content.extend([{'type': 'input_text', 'text': json.dumps(metadata)},
                        image_part(raster_visual_annotations(view['rgb'], items))])
    content.extend(_current_candidate_zooms({**context, 'item_lifecycle': bindings},
        raw, annotations, labels, 2))
    questions = []
    for index, scope in scopes.items():
        subject = ('one or more black workpieces or their visible parts' if scope['role'] == 'completed_collection'
                   else 'a pink workpiece' if scope['role'] == 'wrapped_source' else 'a black workpiece')
        instructions = (f"C{index} in camera {scope['camera_id']} shows {subject}. "
            'Use all CURRENT views together, including its labelled full image and detail. '
            'Robot parts, container and background do not count.')
        questions.append(condition('identity_' + str(index), instructions))
    return {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': questions}


def _candidate_judgments(body, questions, minimum_confidence):
    return parse_answers(body, questions, minimum_confidence)


def _camera_comparison(capture, reference, containers, camera):
    tracks = reference.get('container_identity_tracks', {})
    if (tracks.get('reference_observation_id') != reference['observation_id']
            or tracks.get('generation') != capture['generation']):
        return {'status': 'unavailable'}
    for match in containers.get('matches', {}).values():
        if match['camera_id'] != camera:
            continue
        anchor = tracks.get('containers', {}).get(match['entity_id'], {}).get('views', {}).get(camera, {})
        current = capture['views'][camera]
        a, b = (np.asarray(view.get('T_world_camera', []), float) for view in (anchor, current))
        if a.shape != (4, 4) or b.shape != (4, 4) or not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        return {'status': 'available', 'source': 'registered_initial_container_anchor_and_current_camera',
            'reference_observation_id': reference['observation_id'], 'observation_id': capture['observation_id'],
            'initial_calibration_id': anchor.get('calibration_id'), 'current_calibration_id': current.get('calibration_id'),
            'same_calibration_id': anchor.get('calibration_id') == current.get('calibration_id'),
            'same_intrinsics': anchor.get('intrinsics') == current.get('intrinsics'),
            'translation_delta_m': float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            'rotation_delta_rad': float(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))}
    return {'status': 'unavailable'}


def _temporal_evidence(capture, detections, reference, containers, candidates):
    task = reference.get('demo_task', {})
    source_ids = task.get('source_object_ids', [])
    history = {}
    for identifier in source_ids:
        record = task.get('item_lifecycle', {}).get(identifier, {})
        summary = {key: record[key] for key in ('phase', 'appearance', 'appearance_revision') if key in record}
        summary['recorded_action_fields_present'] = [key for key in ('mock_action', 'drop_action') if key in record]
        summary['recorded_actions'] = []
        for key in summary['recorded_action_fields_present']:
            action = record[key]
            if (isinstance(action, dict) and action.get('target_id') == identifier
                    and action.get('generation') == capture['generation']
                    and action.get('owner_id') == reference.get('owner_id')):
                summary['recorded_actions'].append({field: action[field] for field in
                    ('target_id', 'action_id', 'source', 'generation', 'completed') if field in action})
        history[identifier] = summary
    proposals = []
    for index, scope in candidates.items():
        camera = scope['camera_id']
        box = mask_box(detections[index]['silhouette_mask'])
        references = []
        for identifier in source_ids:
            entity = reference['entities'][identifier]
            original = entity.get('bbox')
            same_camera = entity.get('reference_camera_id') == camera
            overlap = None
            if same_camera and original:
                intersection = max(0., min(box[2], original[2]) - max(box[0], original[0])) * max(
                    0., min(box[3], original[3]) - max(box[1], original[1]))
                union = ((box[2]-box[0])*(box[3]-box[1])
                         + (original[2]-original[0])*(original[3]-original[1]) - intersection)
                overlap = intersection / union if union > 0 else 0.
            references.append({'entity_id': identifier, 'display_id': entity.get('display_id'),
                'reference_camera_id': entity.get('reference_camera_id'), 'reference_box': original,
                'box_source': 'initial_responses_rgb_annotation_not_sam_measurement',
                'same_camera': same_camera, 'box_iou': overlap})
        proposals.append({'candidate': 'candidate-' + str(index), 'camera_id': camera,
            'current_sam_box': box, 'initial_source_references': references,
            'camera_comparison': _camera_comparison(capture, reference, containers, camera),
            'identity_assignment': False})
    return {'temporal_correspondence_proposals': proposals, 'recorded_task_history': history}


def _wrapped_identity_request(capture, detections, reference, containers, wrapped_references,
                              scopes, role_payload):
    candidates = {index: scope for index, scope in scopes.items()
                  if scope['role'] == 'wrapped_source'}
    context = {'purpose': 'wrapped_identity_in_joint_observation', 'observation_id': capture['observation_id'],
        'generation': capture['generation'], 'initial_reference_observation_id': reference['observation_id'],
        'wrapped_identity_references': wrapped_references,
        'candidate_ids': ['candidate-' + str(index) for index in candidates],
        **_temporal_evidence(capture, detections, reference, containers, candidates),
        'rules': 'Compare only the specified still-wrapped pink candidates to their INITIAL physical identities. '
            'The role questions in this same request do not establish original identities. '
            'Use combined INITIAL/CURRENT visible appearance, calibrated spatial continuity and recorded per-ID '
            'task history for persistent association. Unique intrinsic markings are not required. '
            'Reference rectangles are RGB annotations, not measured SAM masks; IoUs and camera comparisons '
            'are correspondence hints, never automatic identity. Compare all original source IDs, including competitors. '
            'Recorded actions describe what the task did; absence of recorded action does not guarantee no physical motion. '
            'Do not reassign by remaining count, role or pink existence. Canonical IDs and initial display numbers '
            'never change. Unknown if combined evidence leaves competing associations or unresolved contradictions. '
            'Other current candidates and old pink appearances of unpacked objects are context only, '
            'not additional identity questions. This request does not discover or count hidden objects.'}
    content = [{'type': 'input_text', 'text': json.dumps(context, ensure_ascii=False)}]
    content.extend(_initial_reference_content(reference))
    content.extend(deepcopy(role_payload['input'][0]['content'][1:]))
    questions = []
    for index, scope in candidates.items():
        identifier = scope['entity_id']
        if identifier is None:
            instructions = (f"Which immutable INITIAL pink source ID is the still-wrapped C{index} in camera "
                f"{scope['camera_id']}? Compare all INITIAL references with CURRENT visual features, registered "
                'spatial continuity and actual action history. Do not assign by count, location or remaining role alone. '
                'Choose unknown if the physical association is ambiguous; reject if it is not an original pink source.')
            questions.append(choice('wrapped_identity_' + str(index), instructions, [*wrapped_references, 'reject', 'unknown']))
            continue
        original = wrapped_references[identifier]
        instructions = (f"Does C{index} in camera {scope['camera_id']} show the same original physical "
            f"pink workpiece as canonical ID {identifier}, immutable display {original['display_label']}, "
            f"in INITIAL reference camera {original['reference_camera_id']}? "
            'Use the combined INITIAL/CURRENT visual appearance, registered box/view continuity and recorded '
            'per-ID manipulation history. Unique intrinsic markings are not required. Compare the competing '
            'original source references, not just this proposed match. Do not reassign IDs by remaining count, '
            'role or pink existence. Choose unknown if the combined evidence leaves competing identities, '
            'contradictory movement or unresolved occlusion. Never renumber or reassign IDs.')
        questions.append(choice('wrapped_identity_' + str(index), instructions, ['yes', 'no', 'unknown']))
    return {'context': context, 'content': content, 'questions': questions, 'candidates': candidates}


def resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers, *, instruction,
                                reference=None):
    """Confirm roles and immutable pink identities in one joint three-view request."""
    scopes, bindings = tracking['candidate_scopes'], tracking['bindings']
    wrapped_references = _initial_identity_references(reference, entities, bindings)
    payload = _current_role_request(capture, detections, tracking, containers, instruction)
    wrapped = (_wrapped_identity_request(capture, detections, reference, containers,
        wrapped_references, scopes, payload) if wrapped_references else None)
    if wrapped:
        context = json.loads(payload['input'][0]['content'][0]['text'])
        context['wrapped_identity'] = wrapped['context']
        payload['input'][0]['content'] = [
            {'type': 'input_text', 'text': json.dumps(context, ensure_ascii=False)}, *wrapped['content'][1:]]
        payload['questions'].extend(wrapped['questions'])
    required_left = {identifier for identifier, binding in bindings.items() if binding['role'] == 'wrapped_source'}

    def parse_observation(body, evidence):
        judgments, answers = _candidate_judgments(body, payload['questions'], client.minimum_confidence)
        evidence = {**evidence, 'answers': answers, 'lifecycle_role_scopes': tracking}
        accepted = {index: scope['entity_id'] for index, scope in scopes.items()
                    if judgments['identity_' + str(index)] == 'yes'}
        current_ids = {identifier for identifier in accepted.values() if identifier is not None}
        left_ids = {identifier for index, identifier in accepted.items() if detections[index]['camera_id'] == '2'}
        if (set(bindings) - required_left - current_ids
                or required_left and not any(scopes[index]['role'] == 'wrapped_source'
                    and detections[index]['camera_id'] == '2' for index in accepted)):
            raise VisualInventoryError('visual_lifecycle_role_unconfirmed', evidence={**evidence,
                'missing_current_role_ids': sorted(set(bindings) - current_ids),
                'missing_left_source_ids': sorted(required_left - left_ids)})
        roles = {'accepted': accepted, 'judgments': judgments, 'evidence': evidence}
        strict = None
        if wrapped:
            identities = {}
            for index, scope in wrapped['candidates'].items():
                if index not in accepted:
                    continue
                judgment = judgments['wrapped_identity_' + str(index)]
                if scope['entity_id'] is None and judgment in wrapped_references:
                    identities[index] = judgment
                elif scope['entity_id'] is not None and judgment == 'yes':
                    identities[index] = scope['entity_id']
            left = {identifier for index, identifier in identities.items() if scopes[index]['camera_id'] == '2'}
            proof = {**evidence, 'generation': capture['generation'],
                'decision_scope': 'joint_current_roles_and_original_identity',
                'question_prefix': 'wrapped_identity_', 'role_request_hash': evidence['request_hash'],
                'initial_reference_observation_id': reference['observation_id'],
                'wrapped_identity_references': deepcopy(wrapped_references),
                'temporal_correspondence_proposals': deepcopy(wrapped['context']['temporal_correspondence_proposals']),
                'recorded_task_history': deepcopy(wrapped['context']['recorded_task_history']),
                'accepted_candidate_ids': [str(index) for index in identities],
                'accepted_entity_ids': sorted(set(identities.values())), 'accepted_left_entity_ids': sorted(left)}
            if set(wrapped_references) - left:
                raise VisualInventoryError('visual_lifecycle_original_identity_unconfirmed', evidence={**proof,
                    'missing_left_source_ids': sorted(set(wrapped_references) - left)})
            strict = {'accepted': identities, 'judgments': judgments, 'evidence': proof}
        return {'roles': roles, 'strict': strict}

    joint = VisualAPI.request(client, payload, capture, 'associate_inventory', parse_observation)
    roles, strict = joint['roles'], joint['strict']
    # Both role and identity evidence refer to this one immutable request and its pixels.
    accepted = {index: identifier for index, identifier in roles['accepted'].items()
                if scopes[index]['role'] != 'wrapped_source'}
    if strict:
        accepted.update(strict['accepted'])
    accepted = dict(sorted(accepted.items()))
    judgments = {**roles['judgments'], **(strict['judgments'] if strict else {})}
    evidence = {**roles['evidence'], **({'wrapped_identity_verification': strict['evidence']} if strict else {})}
    selected = [deepcopy(detections[index]) for index in accepted]
    assigned = {'identity_' + str(local): identifier for local, identifier in enumerate(accepted.values())}
    result = VisualInventory._groups(client, selected, ['candidate-' + str(i) for i in accepted],
                                     entities, assigned, evidence)
    for group in result['groups']:
        for entry in group:
            identifier = entry['entity_id']
            wrapped = identifier in wrapped_references
            entry['identity_evidence'] = [{'source': 'openai_decisions',
                'method': ('initial_reference_visual_same_physical_identity' if wrapped
                           else 'current_visual_role_and_verified_lifecycle'),
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'request_hash': strict['evidence']['request_hash'] if wrapped else evidence['request_hash'],
                'entity_id': identifier, 'camera_id': entry['camera_id'],
                'lifecycle_binding': deepcopy(bindings[identifier]),
                **({'initial_reference': deepcopy(wrapped_references[identifier]),
                    'role_request_hash': evidence['request_hash']} if wrapped else {})}]
    current_ids = set(accepted.values())
    container_ids = set(containers['candidate_mapping'].values())
    left_ids = {identifier for index, identifier in accepted.items() if detections[index]['camera_id'] == '2'}
    left_ids |= {row['entity_id'] for row in containers['matches'].values() if row['camera_id'] == '2'}
    result.update(demo_inventory_check={'status': 'not_reassessed', 'source': 'frozen_initial_inventory',
                'observation_id': capture['observation_id'], 'request_hash': evidence['request_hash'],
                'initial_decisions_request_hash': containers['initial_request_hash'],
                'scope': 'known_current_roles_only_no_new_global_discovery'},
            identity_coverage={'source': 'mixed_decisions_lifecycle_measured_container',
                'observation_id': capture['observation_id'], 'request_hash': evidence['request_hash'],
                'purpose': 'associate_inventory', 'minimum_confidence': client.minimum_confidence,
                'required_entity_ids': sorted(set(bindings) | container_ids),
                'accepted_entity_ids': sorted(current_ids | container_ids),
                'accepted_left_entity_ids': sorted(left_ids), 'remaining_source_ids': sorted(required_left)},
            lifecycle_correspondence={**tracking, 'accepted_candidate_ids': [str(index) for index in accepted],
                                      'request_hash': evidence['request_hash']},
            unassociated_candidates=[{'candidate': 'candidate-' + str(index), 'camera_id': scope['camera_id'],
                'status': judgments[('wrapped_identity_' if scope['role'] == 'wrapped_source' else 'identity_')
                                    + str(index)], 'geometry_policy': 'discard_from_inventory_map'}
                for index, scope in scopes.items() if index not in accepted],
            visibility={identifier: 'visible' for identifier in current_ids | container_ids}, progress={})
    if strict:
        result['wrapped_identity_verification'] = deepcopy(strict['evidence'])
        for key in ('identity_coverage', 'lifecycle_correspondence'):
            result[key]['wrapped_identity_request_hash'] = strict['evidence']['request_hash']
    return merge_container_correspondence(client, detections, entities, containers, result)

