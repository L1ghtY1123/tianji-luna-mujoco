"""Resolve tentative subject ownership before any permanent identity is committed."""
import json
from copy import deepcopy

from .association_depth import DEPTH_INSTRUCTIONS, association_depth_state
from .current_instances import current_instance_groups
from .decision_conditions import parse_answers
from .raw_visual_input import capture_content, reference_content
from .visual_api import VisualAPI, VisualDecisionError
from .visual_identity import choice


def _review_owner(client, capture, detections, group, owners, entities, records, reference):
    name = 'subject_owner_' + str(min(group))
    question = choice(name,
        'The independent subject selections conflict: these CURRENT masks were associated as one physical '
        'object, but multiple task subjects claimed them. Which ONE subject actually owns this object? '
        'Compare all competing descriptions and their labelled dated references together. Earlier choices '
        'are tentative and may be wrong; selection order gives no priority. A container and its contents '
        'are distinct objects. Choose reject if none owns this object, or unknown if the conflict cannot '
        'be resolved from the supplied evidence. Do not choose by elimination without positive evidence. ' + DEPTH_INSTRUCTIONS,
        [{'value': owner, 'description': owner + ': ' + entities[owner].get('description', '')}
         for owner in owners] + [
            {'value': 'reject', 'description': 'None of the competing subjects owns this physical object.'},
            {'value': 'unknown', 'description': 'The competing identities cannot be distinguished.'}])
    state = {'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'conflict_scope': 'tentative_current_mask_ownership',
        'competing_subjects': {owner: {key: entities[owner][key] for key in (
            'description', 'kind', 'color', 'robot_relative_side', 'reference_camera_id') if key in entities[owner]}
            for owner in owners},
        'tentative_selections': {owner: {key: records[owner][-1].get(key) for key in (
            'request_hash', 'selected_candidate_id', 'selected_camera_id', 'reference_observation_ids')}
            for owner in owners}}
    masks = {}
    for index in group:
        row = detections[index]
        masks.setdefault(row['camera_id'], {})['C' + str(index)] = row['silhouette_mask']
    visual_reference = {}
    if reference:
        refs = {key: value for key, value in reference.get('identity_references', {}).items() if key in owners}
        # If a proposal relied on an older view, the conflict review must see it.
        if any(set(records[owner][-1].get('reference_observation_ids', []))
               - {refs.get(owner, {}).get('observation_id')} for owner in owners):
            archived = client.load_identity_archive({key: entities[key] for key in owners})
            refs.update(archived['identity_references'])
            visual_reference['archive_owners'] = archived.get('archive_owners', {})
        visual_reference['identity_references'] = refs
    state.update(association_depth_state(capture, {'C' + str(index): detections[index] for index in group},
                                        {owner: entities[owner] for owner in owners}, visual_reference))
    spatial_excluded = {owner for relations in state.get('candidate_history_relations', {}).values()
                        for owner, comparison in relations.items() if comparison['same_capture_conflict']}
    question['choices'] = [option for option in question['choices'] if option['value'] not in spatial_excluded]
    state['same_capture_spatial_excluded_permanent_ids'] = sorted(spatial_excluded)
    content = [{'type': 'input_text', 'text': json.dumps(state, ensure_ascii=False, sort_keys=True)}]
    content.extend(capture_content(capture, masks=masks, modalities=('rgb',)))
    if reference:
        content.extend(reference_content(visual_reference, current_capture=capture,
                                         modalities=('rgb',), labelled_only=True))
    payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [question]}
    body, evidence = VisualAPI.request(client, payload, capture, 'subject_ownership_review',
                                      lambda body, evidence: (body, evidence))
    parsed, answers = parse_answers(body, [question])
    winner = parsed[name]
    evidence = {**evidence, 'answers': answers, 'ownership_review': state,
                'resolved_owner': winner, 'current_candidate_indices': list(group)}
    if winner == 'unknown':
        raise VisualDecisionError('subject_ownership_unresolved', evidence=evidence)
    return winner, evidence


def select_subject_groups(client, capture, detections, entities, select, *, reference=None):
    """Tentative views, one multiview identity review, then bounded ownership reconciliation."""
    selected, records = {}, {}
    for identifier, entity in entities.items():
        index, requests = select(identifier, entity, {}, ())
        records[identifier] = requests
        if index is not None:
            selected[identifier] = index
    groups, correspondence = current_instance_groups(client, capture, detections,
        seed_indices=sorted(set(selected.values())))
    regroup = False
    for identifier, previous in list(selected.items()):
        # New correspondence evidence permits a single joint-view review.
        # Do not tell the model that the provisional seed already owns this ID.
        index, requests = select(identifier, entities[identifier], {}, (), current_groups=groups)
        records[identifier].extend(requests)
        requests[-1]['supersedes_tentative_candidate_id'] = f'C{previous}'
        requests[-1]['identity_review_performed'] = True
        previous_group = next(group for group in groups if previous in group)
        if index is None:
            selected.pop(identifier)
            regroup = True
        else:
            selected[identifier] = index
            regroup |= index not in previous_group
    if regroup:
        groups, revised = current_instance_groups(client, capture, detections,
            seed_indices=sorted(set(selected.values())))
        correspondence.extend(revised)
    losers, excluded, reviews = [], set(), []
    for group in groups:
        owners = [identifier for identifier, index in selected.items() if index in group]
        excluded.update(group)
        if len(owners) < 2:
            continue
        winner, review = _review_owner(client, capture, detections, group, owners, entities, records, reference)
        reviews.append(review)
        for owner in owners:
            previous = records[owner][-1]
            records[owner].append({**previous, **deepcopy(review), 'task_subject_ref': owner,
                'original_selection_request_hash': previous['request_hash']})
            if owner != winner:
                selected.pop(owner)
                losers.append(owner)
    for identifier in losers:
        index, requests = select(identifier, entities[identifier], selected, tuple(sorted(excluded)),
                                 current_groups=groups)
        records[identifier].extend(requests)
        if index is not None:
            selected[identifier] = index
            excluded.add(index)
    if losers:
        groups, revised = current_instance_groups(client, capture, detections,
            seed_indices=sorted(set(selected.values())))
        correspondence.extend(revised)
    if (len(set(selected.values())) != len(selected)
            or any(sum(index in group for index in selected.values()) > 1 for group in groups)):
        raise VisualDecisionError('subject_ownership_conflict_after_review', evidence={
            'observation_id': capture['observation_id'], 'selected_subjects': selected,
            'selection_requests': records, 'ownership_reviews': reviews})
    return selected, records, groups, correspondence
