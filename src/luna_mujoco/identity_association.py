"""Bind confirmed current camera groups through dated identity observations."""
from copy import deepcopy

import numpy as np

from .current_instances import current_instance_groups
from .subject_assignment import select_subject_groups
from .task_subjects import select_known_subject, select_task_subject
from .temporal_identity import choose_identity
from .visual_api import VisualDecisionError, merge_decision_evidence
from .visual_identity import array_stamp, candidate_identity_evidence, mask_box


def associate_instances(client, capture, detections, *, instruction, entities, reference=None,
                      progress=None, purpose=None, candidate_indices=None, modalities=('rgb',), association_scope=None, **unused):
    entities = deepcopy(entities or {})
    reference = reference or {}
    source_indices = list(range(len(detections))) if candidate_indices is None else list(candidate_indices)
    unique, indices, duplicates, seen = [], [], [], {}
    for index, detection in zip(source_indices, detections, strict=True):
        stamp = array_stamp(detection['silhouette_mask'])
        key = (detection['camera_id'], tuple(stamp['shape']), stamp['dtype'], stamp['sha256'])
        if key in seen:
            duplicates.append({'candidate_index': index, 'same_sensor_mask_as': seen[key]})
            continue
        seen[key] = index
        unique.append(detection)
        indices.append(index)
    deferred = set((association_scope or {}).get('deferred_entity_ids', []))
    candidates = {key: value for key, value in entities.items()
                  if value.get('kind') not in {'table', 'marker'} and key not in deferred}
    def select(identifier, entity, established, excluded, *, current_groups=None):
        return select_known_subject(client, capture, unique, identifier, entity, reference,
            established_subjects=established, excluded_indices=excluded, current_groups=current_groups)

    seeds, all_records, camera_groups, correspondence_records = select_subject_groups(
        client, capture, unique, candidates, select, reference=reference)
    selection_records, records = {}, []
    for identifier, requests in all_records.items():
        index = seeds.get(identifier)
        records.extend(requests if index is None else requests[:-1])
        if index is not None:
            selection_records[identifier] = requests[-1]
    if (association_scope or {}).get('discover_task_inputs'):
        assigned = {index for group in camera_groups for index in group}
        for _ in range(len(unique)):
            index, requests = select_task_subject(client, capture, unique, 'new_task_input',
                {'description': association_scope['target_selector'], 'reference_camera_id': '3'},
                excluded_indices=assigned)
            records.extend(requests)
            if index is None:
                break
            available = [i for i in range(len(unique)) if i not in assigned]
            new_groups, new_records = current_instance_groups(client, capture,
                [unique[i] for i in available], seed_indices=[available.index(index)])
            groups = [[available[i] for i in group] for group in new_groups]
            camera_groups.extend(groups)
            correspondence_records.extend(new_records)
            assigned.update(i for group in groups for i in group)
    decisions, established = [], []
    used_ids = set()
    for local, group in enumerate(camera_groups):
        members = [unique[index] for index in group]
        owners = [identifier for identifier, index in seeds.items() if index in group]
        if len(owners) > 1:
            raise VisualDecisionError('multiple_task_identities_match_one_camera_group')
        if owners:
            identity = owners[0]
            record = deepcopy(selection_records[identity])
            record['current_group'] = [{'camera_id': row['camera_id'], 'bbox': mask_box(row['silhouette_mask'])}
                                       for row in members]
            record['permanent_id_inheritance'] = {'entity_id': identity,
                'previous_observation_id': reference.get('identity_references', {}).get(identity, {}).get('observation_id'),
                'source': 'dated_identity_selected_current_mask'}
            record['permanent_id_inheritance']['reference_observation_ids'] = record['reference_observation_ids']
            answer, requests = record['answers'][0], [record]
        else:
            identity, answer, requests = choose_identity(client, capture, members, candidates, reference,
                'identity_' + str(local), archive_loader=client.load_identity_archive,
                excluded_ids=sorted(used_ids), established=established)
        decisions.append({'identity': identity, 'answer': answer, 'evidence': requests[-1], 'members': members})
        records.extend(requests)
        if identity in candidates:
            used_ids.add(identity)
            established.append({'current_instance': 'identity_' + str(local), 'entity_id': identity,
                'previous_instance': requests[-1].get('permanent_id_inheritance', {}).get('previous_instance'),
                'request_hash': requests[-1]['request_hash'], 'views': requests[-1]['current_group']})
    active_indices = sorted(index for group in camera_groups for index in group)
    remap = {old: new for new, old in enumerate(active_indices)}
    unique = [unique[index] for index in active_indices]
    indices = [indices[index] for index in active_indices]
    camera_groups = [[remap[index] for index in group] for group in camera_groups]
    selected = [row['identity'] for row in decisions]
    conflicts = {identifier for identifier in selected if identifier in entities and selected.count(identifier) > 1}
    if conflicts or any(identifier in {'unknown', 'no_match'} for identifier in selected):
        evidence = merge_decision_evidence(records, capture, purpose or 'associate_objects')
        evidence.update(conflicting_instance_ids=sorted(conflicts), current_correspondence_requests=correspondence_records)
        raise VisualDecisionError('identity_unresolved_conflict' if conflicts else 'visual_identity_unknown', evidence=evidence)
    registered = {}
    with client.registration_transaction():
        for local, row in enumerate(decisions):
            if row['identity'] != 'new_instance':
                continue
            eligible = [member for member in row['members'] if np.any(member['silhouette_mask']
                & capture['views'][member['camera_id']]['workspace_mask']
                & capture['views'][member['camera_id']]['valid_depth'])]
            if not eligible:
                row['identity'] = 'reject'
                continue
            if client.register_instance is None:
                raise VisualDecisionError('new_instance_registrar_unavailable')
            subject = max(eligible, key=lambda member: len(member['points']))
            entity = client.register_instance(capture, subject, {'answer': row['answer'],
                'request_hash': row['evidence']['request_hash'], 'compared_instance_ids': list(candidates)})
            row['identity'] = entity['id']
            entities[entity['id']] = registered[entity['id']] = deepcopy(entity)
    names = ['candidate-' + str(index) for index in indices]
    answers, judgments, provenance, question_names = [], {}, {}, {}
    by_member = {index: row for group, row in zip(camera_groups, decisions, strict=True) for index in group}
    for local, (name, detection) in enumerate(zip(names, unique, strict=True)):
        row = by_member[local]
        answer = deepcopy(row['answer'])
        answer['identity_candidate'] = row['identity']
        answers.append(answer)
        judgments['identity_' + str(local)] = row['identity']
        question_names[name] = answer['name']
        provenance[name] = {'camera': detection['camera_id'], 'observation_id': capture['observation_id'],
            'generation': capture['generation'], 'mask': array_stamp(detection['silhouette_mask']),
            'sam3_prompt': detection.get('prompt'), 'bbox': mask_box(detection['silhouette_mask'])}
    evidence = merge_decision_evidence(records, capture, purpose or 'associate_objects')
    evidence.update(answers=answers,
        original_decisions_answers=[deepcopy(answer) for record in records for answer in record['answers']],
        candidate_provenance=provenance, registered_entities=registered,
        identity_derivation='confirmed_observation_identity_inheritance',
        identity_comparison_records=deepcopy(records),
        identical_mask_aliases=duplicates, candidate_question_names=question_names,
        current_camera_groups=camera_groups, current_correspondence_requests=correspondence_records)
    if not records:
        import hashlib
        evidence.update(source='current_segmentation', component='perception',
            identity_derivation='no_current_identity_candidates',
            request_hash=hashlib.sha256(capture['observation_id'].encode()).hexdigest())
    bound = []
    for local, (name, detection) in enumerate(zip(names, unique, strict=True)):
        entry = dict(detection, raw_identity_mask=detection['silhouette_mask'].copy())
        identifier = judgments['identity_' + str(local)]
        if identifier in entities:
            entry['identity_evidence'] = [candidate_identity_evidence(name, identifier, evidence)]
        bound.append(entry)
    result = client._groups(bound, names, entities, judgments, evidence,
                            purpose=purpose, reference=reference, progress=progress)
    observed = {group[0]['entity_id'] for group in result['groups']}
    result.update(registered_entities=registered,
        missing_geometry_ids=[key for key, entity in entities.items()
                              if entity.get('kind') not in {'table', 'marker'} and key not in observed],
        visibility={key: 'visible' if key in observed else 'unknown' for key in entities}, progress={})
    return result
