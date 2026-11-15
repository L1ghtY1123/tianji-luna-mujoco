"""Select task subjects before grouping their other camera masks."""
import json

import numpy as np

from .association_depth import DEPTH_INSTRUCTIONS, association_depth_state
from .decision_conditions import parse_answers
from .raw_visual_input import capture_content, reference_content, subject_description
from .visual_api import VisualAPI, VisualDecisionError
from .visual_identity import candidate_query_terms, choice, mask_box


def candidate_side_evidence(candidate, side):
    """Initial robot-side filtering in the calibrated fixed-base world frame."""
    if side not in {'left', 'right'}:
        raise ValueError('invalid_robot_relative_side')
    point = np.asarray(candidate.get('geometry', {}).get('centroid_world_m'), dtype=float)
    valid = point.shape == (3,) and np.isfinite(point).all()
    # The existing camera/depth correspondence budget also defines the ambiguous
    # strip at the robot midplane. A picture's left/right is never used here.
    tolerance = .01
    signed = float(point[1]) * (1 if side == 'left' else -1) if valid else None
    return {'required_side': side, 'eligible': bool(valid and signed > tolerance),
            'visible_surface_centroid_world_m': point.tolist() if valid else None,
            'frame': 'world: +X robot-forward, +Y robot-left, +Z up',
            'midplane_tolerance_m': tolerance, 'source': 'same_capture_mask_depth_and_calibration'}


def select_task_subject(client, capture, detections, identifier, entity, *, reference=None,
                        excluded_indices=(), established_subjects=None, current_groups=None):
    """Return one current mask for a task description or a dated permanent identity."""
    preferred = entity.get('reference_camera_id', '3')
    cameras = sorted(capture['views'], key=lambda camera: (camera != preferred, camera != '3', camera))
    review = current_groups is not None
    scopes = [None] if review else cameras
    records = []
    established_subjects = established_subjects or {}
    excluded = set(excluded_indices) | set(established_subjects.values())
    description = (f'Previously confirmed {entity.get("color", "unknown-color")} '
                   f'{entity.get("kind", "object")}. Identity is shown by the labelled dated references.'
                   if reference else entity.get('description', 'Requested task object'))
    stored_query = entity.get('segmentation_query') or {}
    retrieval_queries = {str(query).strip().casefold() for query in
        [entity.get('segmentation_prompt', ''), *entity.get('segmentation_aliases', []),
         stored_query.get('prompt', ''), *stored_query.get('aliases', [])] if query}
    # A fallback mask still belongs to its original retrieval query. Matching
    # only the phrase SAM3 actually used silently discards valid fallback masks.
    candidate_queries = [candidate_query_terms(row) for row in detections]
    required_side = entity.get('robot_relative_side') if not reference else None
    side_evidence = {f'C{index}': candidate_side_evidence(row, required_side)
                     for index, row in enumerate(detections)} if required_side else {}
    for camera in scopes:
        visible = {f'C{index}': index for index, row in enumerate(detections)
                   if camera is None or row['camera_id'] == camera}
        available = {f'C{index}': index for index, row in enumerate(detections)
                     if (camera is None or row['camera_id'] == camera) and index not in excluded
                     and (not retrieval_queries or retrieval_queries & candidate_queries[index])
                     and (not required_side or side_evidence[f'C{index}']['eligible'])}
        depth_state = association_depth_state(capture, {label: detections[index] for label, index in visible.items()},
                                              {identifier: entity} if reference else None, reference)
        spatial_excluded = [label for label, relations in depth_state.get('candidate_history_relations', {}).items()
                            if label in available and relations.get(identifier, {}).get('same_capture_conflict')]
        available = {label: index for label, index in available.items() if label not in spatial_excluded}
        if not available and not spatial_excluded:
            continue
        labels = {}
        for label, index in visible.items():
            row = detections[index]
            labels.setdefault(row['camera_id'], {})[label] = row['silhouette_mask']
        question = choice('task_subject_' + identifier + ('_multiview_review' if review else ''),
            'Select the CURRENT mask for this subject: ' + description
            + ' Compare its own outlined surface with the dated labelled references when supplied. '
            'A container and its contents are different objects, even when their boxes overlap. '
            'Select no_match if the requested object has no offered mask; never substitute a nearby object. '
            'Select unknown if evidence cannot distinguish them. Locations in dated references are historical. '
            'Current multi-camera groups establish physical correspondence, not semantic identity. ' + DEPTH_INSTRUCTIONS,
            [{'value': label, 'description': subject_description(label, detections[index]['camera_id'],
                mask_box(detections[index]['silhouette_mask'])) + ' SAM3 query hint '
                + str(detections[index].get('prompt', 'object')) + '.'} for label, index in available.items()] + [
                {'value': 'no_match', 'description': 'This camera has no offered mask matching the requested subject.'},
                {'value': 'unknown', 'description': 'The supplied evidence cannot distinguish the requested subject.'}])
        state = {'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
            'task_subject_ref': identifier, 'description': description,
            'description_camera_id': preferred,
            'description_time': 'dated_reference' if reference else 'task_description',
            'current_camera_id': camera,
            'candidate_labels': list(available),
            'candidate_retrieval_queries': sorted(retrieval_queries),
            'robot_relative_side': required_side,
            'candidate_side_evidence': {label: side_evidence[label] for label in visible if label in side_evidence},
            'retrieval_interpretation': 'Choices come from the subject segmentation query and aliases. '
                'Query membership only retrieves candidates; it does not prove identity. Confirm the labelled surface visually.',
            'established_mask_owners': {f'C{index}': owner for owner, index in established_subjects.items()
                                       if camera is None or detections[index]['camera_id'] == camera},
            'excluded_candidate_labels': [label for label, index in visible.items() if index in excluded],
            'reference_instance_ids': list((reference or {}).get('identity_references', {}))}
        state.update(depth_state, same_capture_spatial_excluded_labels=spatial_excluded)
        if review:
            state['confirmed_current_camera_groups'] = [[f'C{index}' for index in group]
                                                       for group in current_groups]
            state['group_semantics'] = ('Groups establish same-capture physical correspondence only. '
                'They do not prove which group matches the requested identity. Ungrouped masks remain eligible.')
            question['instructions'] += (' Compare all CURRENT views jointly with the dated references. '
                'Choose one offered mask of the requested physical object in its clearest view, or no_match/unknown. '
                'A cross-camera group may be internally consistent yet belong to a different object. '
                'This is the final subject selection before permanent identity is committed.')
        content = [{'type': 'input_text', 'text': json.dumps(state, ensure_ascii=False)}]
        scoped_capture = capture if review else {**capture, 'views': {camera: capture['views'][camera]}}
        content.extend(capture_content(scoped_capture, masks=labels, modalities=('rgb',), camera_order=cameras))
        if not review and not reference and preferred != camera and preferred in capture['views']:
            content.extend(capture_content({**capture, 'views': {preferred: capture['views'][preferred]}},
                                           modalities=('rgb',)))
        if reference:
            content.extend(reference_content(reference, current_capture=capture, modalities=('rgb',), labelled_only=True))
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [question]}
        body, evidence = VisualAPI.request(client, payload, capture, 'task_subject_selection',
                                          lambda body, evidence: (body, evidence))
        parsed, answers = parse_answers(body, [question])
        selected = parsed[question['name']]
        record = {**evidence, 'answers': answers, 'task_subject_ref': identifier,
                  'reference_observation_ids': sorted({row['observation_id'] for row in
                      (reference or {}).get('identity_references', {}).values()}),
                  'selected_candidate_id': selected,
                  'selected_camera_id': detections[available[selected]]['camera_id'] if selected in available else camera,
                  'selection_stage': 'multiview_group_review' if review else 'tentative_single_view'}
        record['established_mask_owners'] = state['established_mask_owners']
        record['excluded_candidate_labels'] = state['excluded_candidate_labels']
        record['candidate_side_evidence'] = state['candidate_side_evidence']
        record['candidate_depth_evidence'] = state['candidate_depth_evidence']
        record['historical_spatial_evidence'] = state.get('historical_spatial_evidence', {})
        record['candidate_history_relations'] = state.get('candidate_history_relations', {})
        record['same_capture_spatial_excluded_labels'] = spatial_excluded
        records.append(record)
        if selected in available:
            return available[selected], records
    if any(record['selected_candidate_id'] == 'unknown' for record in records):
        raise VisualDecisionError('task_subject_unresolved', evidence={
            **records[-1], 'camera_selection_requests': records})
    return None, records


def select_known_subject(client, capture, detections, identifier, entity, reference, *,
                         established_subjects=None, excluded_indices=(), current_groups=None):
    """Use older confirmed views only when the latest reference cannot resolve the subject."""
    stored = reference.get('identity_references', {}).get(identifier)
    records, unresolved = [], None
    if stored:
        try:
            index, records = select_task_subject(client, capture, detections, identifier, entity,
                reference={'identity_references': {identifier: stored}}, established_subjects=established_subjects,
                excluded_indices=excluded_indices, current_groups=current_groups)
            if index is not None:
                return index, records
        except VisualDecisionError as error:
            if str(error) != 'task_subject_unresolved':
                raise
            unresolved = error
            records = error.evidence.get('camera_selection_requests', [error.evidence])
    archived = client.load_identity_archive({identifier: entity})
    if archived.get('identity_references'):
        index, requests = select_task_subject(client, capture, detections, identifier, entity,
                                              reference=archived, established_subjects=established_subjects,
                                              excluded_indices=excluded_indices, current_groups=current_groups)
        return index, [*records, *requests]
    if unresolved is not None:
        raise unresolved
    if not stored:
        raise VisualDecisionError('task_identity_reference_missing', evidence={'entity_id': identifier})
    return None, records
