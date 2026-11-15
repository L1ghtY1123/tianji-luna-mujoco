"""Decisions correspondence between same-capture camera masks."""
import json
from itertools import combinations

import numpy as np

from .decision_conditions import parse_answers
from .initial_correspondence import projection_evidence
from .raw_visual_input import capture_content, subject_description
from .visual_api import VisualAPI, VisualDecisionError
from .visual_identity import choice, mask_box


def _correspondence(client, capture, subject, candidates, *, review=None, review_subjects=(),
                    confirmed_subjects=(), established=()):
    geometry = projection_evidence(subject, candidates, capture, confirmed_subjects=confirmed_subjects)
    excluded = [row['candidate_id'] for row in geometry['comparisons'] if row['status'] == 'conflict']
    eligible = [row for row in candidates if row['candidate_id'] not in excluded]
    labels = [{'value': row['candidate_id'], 'description':
        subject_description(row['candidate_id'], row['camera_id'], mask_box(row['silhouette_mask']))
        + ' SAM3 query hint ' + str(row.get('prompt', 'object')) + '.'} for row in eligible]
    question = choice('camera_' + subject['candidate_id'] + '_to_' + candidates[0]['camera_id']
                      + ('_review' if review else ''),
        'Which offered mask is the SAME physical object as ' + subject['candidate_id']
        + ' in camera ' + subject['camera_id'] + '? Compare current views, mask surfaces and world-coordinate evidence. '
        'Similar-looking boxes on opposite robot sides are different objects. Select no_match if no candidate '
        'is the same physical object, or unknown if uncertain.',
        labels + [
            {'value': 'no_match', 'description': 'The source object has no corresponding offered mask in the destination view.'},
            {'value': 'segmentation_error', 'description': 'Duplicate or mixed-object masks prevent one unique mask correspondence.'},
            {'value': 'unknown', 'description': 'The supplied evidence cannot determine which correspondence is correct.'}])
    state = {'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
        'source_candidate_id': subject['candidate_id'], 'source_camera_id': subject['camera_id'],
        'source_annotation': subject_description(subject['candidate_id'], subject['camera_id'], mask_box(subject['silhouette_mask'])),
        'candidate_labels': [row['candidate_id'] for row in eligible],
        'excluded_geometric_conflicts': excluded,
        'label_scope': 'C labels identify masks in this capture. SAM3 query text is a search hint. '
            'Use the labelled mask contour to identify the subject surface. Boxes enclose that contour and may include other objects.',
        'destination_camera_id': candidates[0]['camera_id'], 'geometry_correspondence': geometry}
    state['confirmed_same_object_labels'] = geometry['confirmed_subject_ids']
    state['established_correspondences'] = list(established)
    question['instructions'] += (
        ' confirmed_same_object_labels are already confirmed views of the source object in this capture. '
        'Their contours are drawn in the corresponding images. Use these views together, including the '
        'bidirectional geometric evidence. A candidate excluded by any confirmed view cannot be selected.')
    if review:
        state['established_or_competing_correspondences'] = review
        question['instructions'] += (
            ' Consider the supplied correspondences jointly. They may expose a conflict or exclude '
            'candidates already assigned to another object in the source camera. '
            'Use unknown if the remaining alternatives cannot be distinguished from these images.')
    visible = {row['candidate_id']: row for row in [*review_subjects, *confirmed_subjects, subject, *candidates]}
    state['labelled_candidates'] = {label: {
        'camera_id': row['camera_id'], 'sam3_query': row.get('prompt'), 'bbox': mask_box(row['silhouette_mask']),
        'visible_surface_centroid_world_m': row['geometry']['centroid_world_m'],
        'visible_surface_extent_world_m': np.ptp(row['points'], axis=0).tolist(),
    } for label, row in visible.items()}
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
    record = {**evidence, 'source': 'openai_decisions', 'answers': answers,
        'source_candidate_id': subject['candidate_id'], 'destination_camera_id': candidates[0]['camera_id'],
        'selected_candidate_id': selected, 'geometry_correspondence': geometry,
        'confirmed_same_object_labels': geometry['confirmed_subject_ids'],
        'established_correspondences': list(established),
        'excluded_geometric_conflicts': excluded,
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
    for relation in relations:
        group = next(group for group in groups if relation['source'] in group)
        expected = {label for label in group if cameras[label] == relation['destination_camera']}
        selected = relation['selected']
        if selected == 'no_match' and expected:
            conflicts.update(group)
        for comparison in relation['record']['geometry_correspondence']['comparisons']:
            if comparison['candidate_id'] == selected and comparison['status'] == 'conflict':
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
        excluded_labels = set(relation['record'].get('excluded_geometric_conflicts', []))
        for group in groups:
            if group is source_group or not any(cameras[label] == relation['destination_camera'] for label in group):
                continue
            if any(label in excluded_labels for label in group):
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
        points = np.asarray(detection['points'])
        subjects.append({**detection, 'candidate_id': 'C' + str(index), 'source_index': index,
            'geometry': {**detection.get('geometry', {}), 'centroid_world_m': points.mean(axis=0).tolist()}})
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
                       'record': {'geometry_correspondence': {'comparisons': []}}}
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
            geometric_conflict = any(comparison['candidate_id'] == row['selected']
                and comparison['status'] == 'conflict' for row in pair_relations
                for comparison in row['record']['geometry_correspondence']['comparisons'])
            if len(targets) != len(set(targets)) or geometric_conflict:
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
