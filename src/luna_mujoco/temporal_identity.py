"""Match confirmed current instances to dated observations and archived identities."""
import json
from copy import deepcopy

import numpy as np

from .association_depth import DEPTH_INSTRUCTIONS, association_depth_state
from .decision_conditions import parse_answers
from .raw_visual_input import capture_content, reference_content, subject_description
from .visual_api import VisualAPI
from .visual_identity import choice, mask_box


def reference_description(label, entity, reference):
    views = reference.get('views') or {reference['camera_id']: {}}
    descriptions = []
    for camera, view in views.items():
        mask = view.get('mask')
        box = mask_box(mask) if mask is not None and np.any(mask) else (
            reference.get('bbox') if camera == reference['camera_id'] else None)
        if box is not None:
            descriptions.append(subject_description(label, camera, box))
    return (f'{label}. Previously confirmed {entity.get("color", "unknown-color")} {entity.get("kind", "object")}. '
            f'Reference capture {reference.get("observation_id")}. ' + ' '.join(descriptions))


def choose_identity(client, capture, members, entities, reference, name, *, archive_loader=None,
                    excluded_ids=(), established=()):
    masks = {row['camera_id']: {name: row['silhouette_mask']} for row in members}
    points = np.vstack([row['points'] for row in members])
    subject_cameras = list(masks)
    state = {'observation_id': capture['observation_id'], 'subject_label': name,
        'subject_camera_ids': subject_cameras,
        'current_group': [{'camera_id': row['camera_id'], 'bbox': mask_box(row['silhouette_mask']),
                           'sam3_query': row.get('prompt'), 'kind_hint': row.get('kind')}
                          for row in members],
        'current_measurement': {'frame': 'world (Link_Base)', 'visible_surface_centroid_world_m': points.mean(axis=0).tolist()},
        'reference_scope': 'Reference descriptions, regions and measurements are dated historical evidence. '
            'Current location and appearance come from CURRENT images. The P and archive labels refer to their dated frames.',
        'references': {identifier: {key: deepcopy(value) for key, value in entity.items() if key in {
            'description', 'kind', 'color', 'current_region', 'confirmed_observation_id', 'reference_observation_id',
            'last_position_measurement', 'last_motion_receipt', 'transfers', 'processing_history'}}
            for identifier, entity in entities.items()}}
    state['established_current_associations'] = list(established)
    state['excluded_permanent_ids'] = list(excluded_ids)
    available = {key: value for key, value in reference.get('identity_references', {}).items() if key in entities}
    previous = reference.get('previous_observation_id')
    if previous is None and available:
        latest = max(available.values(), key=lambda row: (row.get('generation', 0), row.get('time', 0)))
        previous = latest.get('observation_id')
    previous_ids = [key for key, value in available.items() if value.get('observation_id') == previous]
    observations = {'P' + str(index): identifier for index, identifier in enumerate(previous_ids)}
    recent = {'identity_references': {label: available[identifier] for label, identifier in observations.items()}}
    previous_context = {'previous_observation_id': previous, 'previous_instance_owners': observations,
        'references': {label: state['references'][identifier] for label, identifier in observations.items()}}
    eligible = {label: identifier for label, identifier in observations.items() if identifier not in excluded_ids}
    records = []

    def ask(stage, instructions, choices, visual_reference, extra=None):
        question = choice(name + '_' + stage, instructions + ' ' + DEPTH_INSTRUCTIONS, choices)
        modalities = ('rgb',)
        owners = (extra or {}).get('archive_owners') or {
            label: {'entity_id': identifier} for label, identifier in observations.items()}
        depth = association_depth_state(capture,
            {name + '_camera_' + row['camera_id']: row for row in members}, entities,
            {**visual_reference, 'archive_owners': owners})
        spatial_excluded = {identifier for relations in depth.get('candidate_history_relations', {}).values()
                            for identifier, comparison in relations.items() if comparison['same_capture_conflict']}
        question['choices'] = [option for option in question['choices']
            if owners.get(option['value'], {}).get('entity_id', option['value']) not in spatial_excluded]
        depth['same_capture_spatial_excluded_permanent_ids'] = sorted(spatial_excluded)
        content = [{'type': 'input_text', 'text': json.dumps({**state, **(extra or {}), **depth}, ensure_ascii=False)}]
        content.extend(capture_content(capture, masks=masks, modalities=modalities,
            camera_order=subject_cameras))
        content.extend(reference_content(visual_reference, current_capture=capture,
                                         modalities=modalities, labelled_only=True))
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [question]}
        body, evidence = VisualAPI.request(client, payload, capture, 'instance_' + stage,
                                          lambda body, evidence: (body, evidence))
        parsed, answers = parse_answers(body, [question])
        records.append({**evidence, 'answers': answers, 'identity_stage': stage,
            'compared_instance_ids': list(entities), 'current_instance_label': name,
            'current_group': state['current_group'],
            **depth,
            'excluded_permanent_ids': list(excluded_ids),
            'established_current_associations': deepcopy(list(established)),
            **{key: deepcopy(value) for key, value in (extra or {}).items()
               if key in {'previous_observation_id', 'archive_owners'}}})
        return parsed[question['name']], deepcopy(answers[0])

    identity, answer = 'no_match', None
    if eligible:
        queries = list(dict.fromkeys(row['prompt'] for row in members if row.get('prompt')))
        identity, answer = ask('temporal',
            'Which PREVIOUS instance is the same physical object as the CURRENT segmented surface '
            + name + ' in camera(s) ' + ', '.join(subject_cameras) + '? '
            'The current segmentation query hints are ' + json.dumps(queries) + '. '
            'Identify the labelled surface from its own visible appearance. Compare it with the P-labelled '
            'dated instances. Other objects visible in a camera or inside a bounding rectangle supply context. '
            'A container and its contents have independent identities. Use dated operations to account for '
            'movement, while a planned action alone supplies no identity proof. '
            'Choose no_match when no eligible previous instance corresponds, reject for a non-object '
            'segmentation, and unknown when identity evidence is insufficient.',
            [{'value': label, 'description': reference_description(label, entities[identifier], available[identifier])}
             for label, identifier in eligible.items()] + [
                {'value': 'no_match', 'description': 'None of the eligible P-labelled previous instances is this current object.'},
                {'value': 'reject', 'description': 'The current segmented subject is background or robot geometry.'},
                {'value': 'unknown', 'description': 'The temporal identity cannot be determined from the supplied evidence.'}], recent,
            previous_context)
        records[-1]['previous_instance_owners'] = observations
        if identity in eligible:
            identifier = eligible[identity]
            records[-1]['permanent_id_inheritance'] = {'previous_instance': identity,
                'previous_observation_id': previous, 'entity_id': identifier, 'source': 'program_inheritance'}
            return identifier, answer, records
    if identity in {'no_match', 'unknown'}:
        archived = archive_loader(entities) if archive_loader else {}
        archived_references = {'history_' + key: value for key, value in available.items() if key not in previous_ids}
        owners = {'history_' + key: {'entity_id': key, 'observation_id': value.get('observation_id')}
                  for key, value in available.items() if key not in previous_ids}
        archived_references.update(archived.get('identity_references', {}))
        owners.update(archived.get('archive_owners', {}))
        owners.update({label: {'entity_id': identifier, 'observation_id': previous}
                       for label, identifier in observations.items()})
        if archived_references:
            options = [{'value': identifier, 'description': 'Permanent ID ' + identifier
                + '. Dated image labels ' + (', '.join(label for label, owner in owners.items() if owner['entity_id'] == identifier) or 'none supplied')
                + '. Historical description ' + entity.get('description', '')}
                for identifier, entity in entities.items() if identifier not in excluded_ids]
            combined = {**recent, 'identity_references': {
                **recent['identity_references'], **archived_references}}
            identity, answer = ask('archive',
                'Recover the permanent identity of current instance ' + name + ' using the additional older '
                'confirmed reference images and dated operations. Archive image labels are mapped to permanent '
                'IDs in archive_owners. Choose new_instance only when the evidence establishes a distinct '
                'unregistered object. An object already owned by an established current association cannot be '
                'registered again. Similar indistinguishable objects or interrupted identity evidence require unknown.',
                options + ['new_instance', 'reject', 'unknown'], combined,
                {'archive_owners': owners})
        elif identity == 'no_match':
            identity, answer = ask('novelty',
                'No eligible previous-instance match has been accepted for current instance ' + name + '. '
                'Is it an independently established new physical object, background/robot, or unresolved? '
                'An occluded remembered object, a possible duplicate of an established current association, '
                'or an uncertain correspondence requires unknown.',
                ['new_instance', 'reject', 'unknown'], recent,
                {**previous_context, 'prior_comparison': {'asked': bool(eligible),
                    'result': identity if eligible else None}})
    return identity, answer, records
