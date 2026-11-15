"""Decisions correspondence between same-capture camera masks."""
import json
from itertools import combinations

from .association_depth import DEPTH_INSTRUCTIONS, association_depth_state, candidate_depth_evidence, spatial_relation
from .decision_conditions import parse_answers
from .raw_visual_input import capture_content, subject_description
from .visual_api import VisualAPI, VisualDecisionError
from .visual_identity import candidate_query_terms, choice, mask_box


def _correspondence(client, capture, subject, candidates, *, review=None, review_subjects=(),
                    confirmed_subjects=(), established=()):
    confirmed_labels = list(dict.fromkeys(row['candidate_id'] for row in [subject, *confirmed_subjects]))
    source_queries = candidate_query_terms(subject)
    offered = [row for row in candidates if not source_queries or not candidate_query_terms(row)
               or source_queries & candidate_query_terms(row)]
    offered_labels = {row['candidate_id'] for row in offered}
    labels = [{'value': row['candidate_id'], 'description':
        subject_description(row['candidate_id'], row['camera_id'], mask_box(row['silhouette_mask']))
        + ' SAM3 query hint ' + str(row.get('prompt', 'object')) + '.'} for row in offered]
    question = choice('camera_' + subject['candidate_id'] + '_to_' + candidates[0]['camera_id']
                      + ('_review' if review else ''),
        'Which offered mask is the SAME physical object as ' + subject['candidate_id']
        + ' in camera ' + subject['camera_id'] + '? Compare the labelled surfaces in the CURRENT RGB views. '
        'Similar-looking boxes on opposite robot sides are different objects. Select no_match if no candidate '
        'is the same physical object, or unknown if uncertain.',
        labels + [
            {'value': 'no_match', 'description': 'The source object has no corresponding offered mask in the destination view.'},
            {'value': 'segmentation_error', 'description': 'Duplicate or mixed-object masks prevent one unique mask correspondence.'},
            {'value': 'unknown', 'description': 'The supplied evidence cannot determine which correspondence is correct.'}])
    state = {'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
        'source_candidate_id': subject['candidate_id'], 'source_camera_id': subject['camera_id'],
        'source_annotation': subject_description(subject['candidate_id'], subject['camera_id'], mask_box(subject['silhouette_mask'])),
        'candidate_labels': [row['candidate_id'] for row in offered],
        'source_retrieval_queries': sorted(source_queries),
        'retrieval_excluded_candidate_ids': [row['candidate_id'] for row in candidates
                                            if row['candidate_id'] not in offered_labels],
        'retrieval_interpretation': 'Only masks retrieved for the source query or its aliases are offered. '
            'Other labelled surfaces remain context. Query membership does not prove identity.',
        'label_scope': 'C labels identify masks in this capture. SAM3 query text is a search hint. '
            'Use the labelled mask contour to identify the subject surface. Boxes enclose that contour and may include other objects.',
        'destination_camera_id': candidates[0]['camera_id']}
    state['confirmed_same_object_labels'] = confirmed_labels
    state['established_correspondences'] = list(established)
    question['instructions'] += (
        ' confirmed_same_object_labels are already confirmed views of the source object in this capture. '
        'Their contours are drawn in the corresponding images. Compare these views together. '
        'A box, its contents and its supporting table are distinct physical objects. '
        'Nearby coordinates and not_separated are neutral: they never prove identity. '
        'Compare the outlined surfaces themselves; choose no_match when offered masks omit the source object. '
        'Spatially separated same-capture surfaces cannot be the same object. '
        'Choose no_match if all offered candidates are separated from the source. ' + DEPTH_INSTRUCTIONS)
    if review:
        state['established_or_competing_correspondences'] = review
        question['instructions'] += (
            ' Consider the supplied correspondences jointly. They may expose a conflict or exclude '
            'candidates already assigned to another object in the source camera. '
            'Use unknown if the remaining alternatives cannot be distinguished from these images.')
    visible = {row['candidate_id']: row for row in [*review_subjects, *confirmed_subjects, subject, *candidates]}
    state['labelled_candidates'] = {label: {
        'camera_id': row['camera_id'], 'sam3_query': row.get('prompt'), 'bbox': mask_box(row['silhouette_mask']),
    } for label, row in visible.items()}
    state.update(association_depth_state(capture, visible))
    depth = state['candidate_depth_evidence']
    state['candidate_spatial_relations'] = {row['candidate_id']: {
        label: spatial_relation(depth[label], depth[row['candidate_id']])
        for label in confirmed_labels} for row in offered}
    masks = {}
    for row in visible.values():
        masks.setdefault(row['camera_id'], {})[row['candidate_id']] = row['silhouette_mask']
    content = [{'type': 'input_text', 'text': json.dumps(state)}]
    content.extend(capture_content(capture, masks=masks, modalities=('rgb',),
        camera_order=(subject['camera_id'], candidates[0]['camera_id'])))
    payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [question]}
    body, evidence = VisualAPI.request(client, payload, capture, 'current_instance_correspondence',
                                      lambda body, evidence: (body, evidence))
    parsed, answers = parse_answers(body, [question])
    selected = parsed[question['name']]
    model_selected = selected
    contradictions = [label for label, relation in state['candidate_spatial_relations'].get(selected, {}).items()
                      if relation['status'] == 'separated']
    if contradictions:
        selected = 'unknown'
    record = {**evidence, 'source': 'openai_decisions', 'answers': answers,
        'source_candidate_id': subject['candidate_id'], 'destination_camera_id': candidates[0]['camera_id'],
        'selected_candidate_id': selected,
        'model_selected_candidate_id': model_selected,
        'candidate_depth_evidence': depth,
        'candidate_spatial_relations': state['candidate_spatial_relations'],
        'source_retrieval_queries': state['source_retrieval_queries'],
        'retrieval_excluded_candidate_ids': state['retrieval_excluded_candidate_ids'],
        'spatial_acceptance': {'accepted': not contradictions, 'contradicting_source_labels': contradictions,
                               'reason': 'same_capture_spatial_separation' if contradictions else None},
        'confirmed_same_object_labels': confirmed_labels,
        'established_correspondences': list(established),
        'conflict_review': bool(review)}
    return selected, record


def _groups_and_conflicts(subjects, relations):
    groups = [{row['candidate_id']} for row in subjects]
    for relation in relations:
        source, target = relation['source'], relation['selected']
        if target in {'no_match', 'unknown', 'segmentation_error'}:
            continue
        source_group = next(group for group in groups if source in group)
        target_group = next(group for group in groups if target in group)
        if source_group is not target_group:
            source_group.update(target_group)
            groups.remove(target_group)
    cameras = {row['candidate_id']: row['camera_id'] for row in subjects}
    conflicts = set()
    for group in groups:
        group_cameras = [cameras[label] for label in group]
        if len(group_cameras) != len(set(group_cameras)):
            conflicts.update(group)
        measurements = {row['candidate_id']: row.get('depth_evidence', {}) for row in subjects}
        if any(spatial_relation(measurements[a], measurements[b])['status'] == 'separated'
               for a, b in combinations(group, 2)):
            conflicts.update(group)
    for relation in relations:
        group = next(group for group in groups if relation['source'] in group)
        expected = {label for label in group if cameras[label] == relation['destination_camera']}
        selected = relation['selected']
        if selected == 'no_match' and expected:
            conflicts.update(group)
    return groups, conflicts


def _unresolved_relations(subjects, groups, relations):
    """Propagate established identities and exclusions without changing model answers."""
    cameras = {row['candidate_id']: row['camera_id'] for row in subjects}
    unresolved = []
    for relation in relations:
        if relation['selected'] == 'segmentation_error':
            unresolved.append(relation)
            continue
        if relation['selected'] != 'unknown':
            continue
        source_group = next(group for group in groups if relation['source'] in group)
        source_cameras = {cameras[label] for label in source_group}
        established = [label for label in source_group if cameras[label] == relation['destination_camera']]
        possible = []
        for group in groups:
            if group is source_group or not any(cameras[label] == relation['destination_camera'] for label in group):
                continue
            group_cameras = {cameras[label] for label in group}
            if source_cameras & group_cameras:
                continue
            excluded = any(row['selected'] == 'no_match' and (
                row['source'] in source_group and row['destination_camera'] in group_cameras
                or row['source'] in group and row['destination_camera'] in source_cameras)
                for row in relations)
            if not excluded:
                possible.append(group)
        if possible:
            relation['remaining_candidate_ids'] = sorted(label for group in possible for label in group
                if cameras[label] == relation['destination_camera'])
            unresolved.append(relation)
        else:
            relation['record']['derived_correspondence'] = {
                'source': 'confirmed_correspondence_constraints',
                'value': established[0] if established else 'no_match',
                'source_group': sorted(source_group),
                **({'model_question_asked': False} if relation.get('derived_only') else {'model_answer_preserved': True})}
    return unresolved


def current_instance_groups(client, capture, detections, *, seed_indices=None):
    subjects = []
    for index, detection in enumerate(detections):
        subjects.append({**detection, 'candidate_id': 'C' + str(index), 'source_index': index,
                         'depth_evidence': candidate_depth_evidence(capture, detection)})
    cameras = [camera for camera in ('3', '2', '1') if camera in capture['views']]
    cameras.sort(key=lambda camera: -sum(row['camera_id'] == camera for row in subjects))
    relations, records, implied_relations = [], [], []
    seeds = None if seed_indices is None else {'C' + str(index) for index in seed_indices}
    pairs_to_compare = list(combinations(cameras, 2)) if seeds is None else list(dict.fromkeys(
        (camera, subjects[index]['camera_id']) for index in seed_indices
        for camera in cameras if camera != subjects[index]['camera_id']))
    for destination_camera, source_camera in pairs_to_compare:
        candidates = [row for row in subjects if row['camera_id'] == destination_camera]
        if not candidates:
            continue
        for subject in subjects:
            if subject['camera_id'] != source_camera or (seeds is not None and subject['candidate_id'] not in seeds):
                continue
            groups, conflicts = _groups_and_conflicts(subjects, relations)
            implied = {'source': subject['candidate_id'], 'destination_camera': destination_camera,
                       'selected': 'unknown', 'derived_only': True,
                       'record': {}}
            if not conflicts:
                unresolved = _unresolved_relations(subjects, groups, [*relations, implied])
                if implied not in unresolved:
                    implied_relations.append(implied)
                    continue
            confirmed_group = next(group for group in groups if subject['candidate_id'] in group)
            confirmed = [row for row in subjects if row['candidate_id'] in confirmed_group] if not conflicts else []
            established = [{key: row[key] for key in ('source', 'destination_camera', 'selected')}
                           for row in relations if row['source'] in confirmed_group
                           and row['selected'] not in {'unknown', 'segmentation_error'}]
            selected, record = _correspondence(client, capture, subject, candidates,
                                               confirmed_subjects=confirmed, established=established)
            records.append(record)
            relations.append({'source': subject['candidate_id'], 'destination_camera': destination_camera,
                              'selected': selected, 'record': record})
    groups, conflicts = _groups_and_conflicts(subjects, relations)
    if conflicts:
        review = [{key: row[key] for key in ('source', 'destination_camera', 'selected')}
                  for row in relations if row['source'] in conflicts]
        # Reverse the contested camera pairs so competing source masks appear in one choice.
        cameras_by_label = {row['candidate_id']: row['camera_id'] for row in subjects}
        pairs = {}
        for relation in relations:
            pair = (cameras_by_label[relation['source']], relation['destination_camera'])
            pairs.setdefault(pair, []).append(relation)
        review_pairs = set()
        for pair, pair_relations in pairs.items():
            targets = [row['selected'] for row in pair_relations
                       if row['selected'] not in {'no_match', 'unknown', 'segmentation_error'}]
            if len(targets) != len(set(targets)):
                review_pairs.add(pair)
        if not review_pairs:
            review_pairs = {pair for pair, rows in pairs.items()
                            if any(row['source'] in conflicts for row in rows)}
        for (source_camera, destination_camera), pair_relations in pairs.items():
            if (source_camera, destination_camera) not in review_pairs:
                continue
            candidates = [row for row in subjects if row['camera_id'] == source_camera]
            revised = []
            for subject in subjects:
                if (subject['camera_id'] != destination_camera
                        or (seeds is not None and subject['candidate_id'] not in conflicts)):
                    continue
                selected, record = _correspondence(client, capture, subject, candidates, review=review,
                                                   review_subjects=subjects)
                records.append(record)
                revised.append({'source': subject['candidate_id'], 'destination_camera': source_camera,
                                'selected': selected, 'record': record})
            for relation in pair_relations:
                relation['record']['superseded_by_reverse_pair_requests'] = [
                    row['record']['request_hash'] for row in revised]
            relations = [row for row in relations if row not in pair_relations] + revised
        groups, conflicts = _groups_and_conflicts(subjects, relations)
    # Recheck inferred edges against the final reviewed relations.
    relations.extend(implied_relations)
    unresolved = _unresolved_relations(subjects, groups, relations)
    if not conflicts:
        # A smaller candidate set supplies new information from the other confirmed correspondences.
        for relation in unresolved:
            remaining = relation.get('remaining_candidate_ids', [])
            available = [row for row in subjects if row['camera_id'] == relation['destination_camera']]
            if not remaining or (len(remaining) >= len(available) and not relation.get('derived_only')):
                continue
            subject = next(row for row in subjects if row['candidate_id'] == relation['source'])
            candidates = [row for row in available if row['candidate_id'] in remaining]
            review = [{key: row[key] for key in ('source', 'destination_camera', 'selected')}
                      for row in relations if row['selected'] not in {'unknown', 'segmentation_error'}]
            confirmed_group = next(group for group in groups if subject['candidate_id'] in group)
            confirmed = [row for row in subjects if row['candidate_id'] in confirmed_group]
            selected, record = _correspondence(client, capture, subject, candidates, review=review,
                                               review_subjects=subjects, confirmed_subjects=confirmed,
                                               established=review)
            record['candidate_reduction_source'] = 'confirmed_correspondence_constraints'
            relation.update(selected=selected, record=record)
            records.append(record)
        groups, conflicts = _groups_and_conflicts(subjects, relations)
        unresolved = _unresolved_relations(subjects, groups, relations)
    if conflicts or unresolved:
        raise VisualDecisionError('visual_identity_unknown', evidence={
            'reason': 'current_correspondence_conflict', 'conflicting_candidates': sorted(conflicts),
            'correspondence_requests': records})
    for relation in implied_relations:
        if relation['selected'] == 'unknown':
            records[-1].setdefault('implied_correspondences', []).append({
                'source_candidate_id': relation['source'], 'destination_camera_id': relation['destination_camera'],
                **relation['record']['derived_correspondence'],
                'supporting_request_hashes': [row['record']['request_hash'] for row in relations
                                             if 'request_hash' in row['record']]})
    return [sorted(row['source_index'] for row in subjects if row['candidate_id'] in group)
            for group in groups if seeds is None or group & seeds], records
