"""Match confirmed current instances to dated observations and archived identities."""
import json
from copy import deepcopy

import numpy as np

from .association_depth import DEPTH_INSTRUCTIONS, association_depth_state
from .association_geometry import temporal_overlap_evidence
from .decision_conditions import parse_answers
from .raw_visual_input import capture_content, reference_content, subject_description, subject_detail_content
from .visual_api import VisualAPI
from .visual_identity import choice, mask_box


def reference_description(label, entity, reference):
    views = reference.get('views') or {reference['camera_id']: {}}
    descriptions = []
    for camera, view in views.items():
        mask = view.get('mask')
        box = mask_box(mask) if mask is not None and np.any(mask) else view.get('bbox',
            reference.get('bbox') if camera == reference['camera_id'] else None)
        if box is not None:
            descriptions.append(subject_description(label, camera, box))
    return (f'{label}. Previously confirmed {entity.get("color", "unknown-color")} {entity.get("kind", "object")}. '
            f'Reference capture {reference.get("observation_id")}. ' + ' '.join(descriptions))


def resolve_identity_conflict(client, capture, competing, identifier, entities, reference):
    """One joint comparison of competing current groups, not repeated independent votes."""
    dated = reference.get('identity_references', {}).get(identifier)
    if dated is None:
        return None, None
    labels = {'K' + str(index): group_id for index, group_id in enumerate(competing)}
    masks, candidates, choices = {}, {}, []
    for label, group_id in labels.items():
        descriptions = []
        for row in competing[group_id]:
            camera = row['camera_id']
            masks.setdefault(camera, {})[label] = row['silhouette_mask']
            candidates[label + '_camera_' + camera] = row
            descriptions.append(subject_description(label, camera, mask_box(row['silhouette_mask'])))
        choices.append({'value': label, 'description': ' '.join(descriptions)})
    historical = {'identity_references': {'P': dated}, 'archive_owners': {'P': {'entity_id': identifier}}}
    state = {'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'reason': 'Independent historical comparisons assigned one permanent ID to multiple distinct current groups.',
        'permanent_id': identifier, 'current_group_labels': labels,
        'dated_reference': reference_description('P', entities[identifier], dated),
        **association_depth_state(capture, candidates, {identifier: entities[identifier]}, historical)}
    question = choice('resolve_history_conflict',
        'Which ONE of the mutually distinct CURRENT K-labelled physical groups is the same physical '
        'entity as dated reference P? Compare all competing groups jointly, including calibrated positions '
        'and executed motion history. Matching color/category alone does not prove identity. '
        'Tables and containers are valid entities. Choose no_match only if none is that entity; '
        'choose unknown if the evidence cannot distinguish them. ' + DEPTH_INSTRUCTIONS,
        [*choices, 'no_match', 'unknown'])
    content = [{'type': 'input_text', 'text': json.dumps(state, ensure_ascii=False)}]
    content.extend(capture_content(capture, masks=masks, camera_order=('3', '2', '1'), modalities=('rgb', 'mask')))
    content.extend(reference_content(historical, current_capture=capture, modalities=('rgb', 'mask'), labelled_only=True))
    payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [question]}
    body, evidence = VisualAPI.request(client, payload, capture, 'instance_conflict_review', lambda body, evidence: (body, evidence))
    values, answers = parse_answers(body, [question])
    return labels.get(values[question['name']]), {**evidence, 'answers': answers,
        'identity_stage': 'joint_current_conflict_review', 'competing_group_ids': list(competing),
        'permanent_id': identifier, 'current_group_labels': labels}


def choose_identity(client, capture, members, entities, reference, name, *, archive_loader=None, rivals=None, reserved=None):
    """One joint competition: remembered IDs, genuinely new, invalid, or unknown."""
    repeated = []
    for identifier, dated in reference.get('identity_references', {}).items():
        if (identifier not in entities or dated.get('observation_id') != capture['observation_id']
                or dated.get('generation') != capture['generation'] or not members):
            continue
        same = True
        for member in members:
            view = dated.get('views', {}).get(member['camera_id'], {})
            current = member['view']
            if (any(key not in view for key in ('mask', 'rgb', 'depth_m', 'valid_depth'))
                    or not np.array_equal(view['mask'], member['silhouette_mask'])
                    or view.get('intrinsics') != current['intrinsics']
                    or not np.array_equal(view.get('T_world_camera'), current['T_world_camera'])
                    or not np.array_equal(view.get('rgb'), current['rgb'])
                    or not np.array_equal(view['valid_depth'], current['valid_depth'])
                    or not np.array_equal(view.get('depth_m'), current['depth_m'], equal_nan=True)):
                same = False
                break
        if same:
            repeated.append(identifier)
    if repeated:
        identity = repeated[0] if len(repeated) == 1 else 'unknown'
        return identity, {}, [{'source': 'confirmed_same_capture_masks',
            'identity_stage': 'immutable_reference_reuse', 'compared_instance_ids': repeated,
            'accepted_identity': identity, 'observation_id': capture['observation_id'],
            'generation': capture['generation'], 'model_called': False}]
    archived = archive_loader(entities) if entities and archive_loader else {}
    refs = dict(archived.get('identity_references', {}))
    owners = dict(archived.get('archive_owners', {}))
    for identifier, dated in reference.get('identity_references', {}).items():
        if identifier in entities:
            label = 'latest_' + identifier
            refs[label] = dated
            owners[label] = {'entity_id': identifier, 'observation_id': dated.get('observation_id')}
    # Display short labels. Capture hashes remain in provenance, not badges
    # painted over small objects. One ID/capture needs only one reference.
    labelled, compact_owners, seen = {}, {}, set()
    for label, dated in reversed(list(refs.items())):
        if not dated.get('observation_id') or (dated.get('rgb') is None and not any(
                view.get('rgb') is not None for view in dated.get('views', {}).values())):
            continue
        owner = owners[label]
        key = (owner['entity_id'], dated.get('observation_id'), dated.get('generation'))
        if key in seen:
            continue
        seen.add(key)
        compact = 'P' + str(len(labelled))
        labelled[compact], compact_owners[compact] = dated, owner
    refs, owners = labelled, compact_owners
    represented = {owner['entity_id'] for owner in owners.values()}
    missing = sorted(set(entities) - represented)
    visual_reference = {'identity_references': refs, 'archive_owners': owners}
    masks, current = {}, {}
    subject = 'Q'
    for label, rows in {subject: members, **(rivals or {})}.items():
        for row in rows:
            masks.setdefault(row['camera_id'], {})[label] = row['silhouette_mask']
            current[label + '_camera_' + row['camera_id']] = row
    depth = association_depth_state(capture, current, entities, visual_reference)
    temporal_overlap = temporal_overlap_evidence(members, reference)
    state = {'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'subject_label': subject, 'current_group_label': name, 'rival_current_groups': list(rivals or {}),
        'archive_owners': owners, 'missing_reference_ids': missing,
        'physical_history': entities, 'candidate_pose_hypotheses': {
            label: row.get('pose_hypothesis') for label, row in current.items()},
        'temporal_overlap': temporal_overlap, **depth}
    choices = [{'value': identifier, 'description': 'Existing permanent ID ' + identifier + '; historical '
        + entity.get('description', '') + '; dated labels ' + ', '.join(label for label, owner in owners.items()
                                                                      if owner['entity_id'] == identifier)}
        for identifier, entity in entities.items()]
    instructions = ('Which remembered physical object is CURRENT ' + subject + '? Compare the dated initial/latest '
        'references, current rival objects, measured positions and executed motion. A moved object keeps its ID '
        'despite a different position. Shared color/category/CAD alone does not establish identity. Choose '
        'new_instance only for a distinct unregistered object, reject for a mixed/background/robot mask or '
        'duplicate of another current group, unknown for ambiguous identity or missing evidence. '
        'Binary masks identify the precise subject pixels in WHITE; enclosed BLACK holes are excluded.')
    outcomes = ['new_instance', 'reject', 'unknown']
    if reserved:
        instructions = ('A joint same-capture conflict review has already assigned the reserved IDs to other '
            'current groups. Compare Q with the remaining dated IDs. Is Q one of these distinct historical '
            'objects? Compare robot-frame coordinates (+Y left), current RGB/masks and actions. Choose reject '
            'for a duplicate or fragment, unknown for an unlisted, unseen or uncertain instance.')
        outcomes = ['reject', 'unknown']
    query = choice(name, instructions, [*choices, *outcomes])
    shown = {'subject': subject, 'world_frame': depth['spatial_frame'], 'missing_reference_ids': missing,
        'visible_surface_centers': {label: item.get('visible_surface_center_world_m')
            for label, item in depth['candidate_depth_evidence'].items()}, 'archive_owners': owners,
        'history': {identifier: {key: history[key] for key in ('last_measured_position', 'last_motion_receipt',
            'current_region', 'placement_evidence')} for identifier, history in depth.get('historical_spatial_evidence', {}).items()}}
    if reserved:
        shown['reserved_ids_to_current_labels'] = reserved
    if reference.get('stationary_capture'):
        shown['robot_continuity'] = reference['stationary_capture']
        shown['continuity_scope'] = 'Robot feedback only. A person or an object may move between distinct captures.'
    shown['current_confirmed_kinds'] = sorted({row['physical_kind'] for row in members if row.get('physical_kind')})
    shown['history_kinds'] = {identifier: row.get('kind') for identifier, row in entities.items()}
    shown['unchanged_rgbd_surface'] = [row['scope_measurement'] for row in members if row.get('scope_measurement')]
    shown['dated_overlap_with_each_id'] = {identifier: {
        'reference_observation_id': item['reference_observation_id'],
        'same_view_mask_iou': [round(row['same_view_mask_iou'],4) for row in item['comparisons']
                              if row['same_view_mask_iou'] is not None],
        'projected_old_surface_inside_current_mask': [round(row['projection']['projected_surface_in_target_fraction'],4)
            for row in item['comparisons'] if row['projection']['status']=='available']}
        for identifier,item in temporal_overlap.items()}
    shown['overlap_scope'] = 'Dated geometry: high overlap supports continuity; low or unavailable overlap does not reject a moved object.'
    content = [{'type': 'input_text', 'text': json.dumps(shown, ensure_ascii=False)}]
    content.extend(capture_content(capture, masks=masks, camera_order=('3', '2', '1'), modalities=('rgb',)))
    for row in members:
        content.extend(subject_detail_content(capture, row['camera_id'], subject, row['silhouette_mask']))
    for label, dated in refs.items():
        masked = {camera:view for camera,view in dated.get('views',{}).items()
                  if view.get('mask') is not None and np.any(view['mask'])}
        if masked:
            for camera,view in masked.items():
                content.extend(subject_detail_content(dated,camera,label,view['mask'],role='dated_reference'))
        else:
            content.extend(reference_content({'identity_references':{label:dated}},current_capture=capture,labelled_only=True))
    body, evidence = VisualAPI.request(client, {'model': 'gpt-6-luna',
        'input': [{'role': 'user', 'content': content}], 'questions': [query]}, capture,
        'instance_temporal', lambda body, evidence: (body, evidence))
    values, answers = parse_answers(body, [query])
    identity = values[name]
    model_identity = identity
    subject_depth = {label: relations for label, relations in depth.get('candidate_history_relations', {}).items()
                     if label.startswith(subject + '_camera_')}
    contradiction = any(relations.get(identity, {}).get('same_capture_conflict') for relations in subject_depth.values())
    if contradiction or identity in missing or (identity == 'new_instance' and missing):
        identity = 'unknown'
    record = {**evidence, 'answers': answers, 'identity_stage': 'joint_history_and_novelty',
        'compared_instance_ids': list(entities), 'current_instance_label': name,
        'association_state': state, 'model_identity': model_identity, 'accepted_identity': identity,
        'same_capture_contradiction': contradiction, 'missing_reference_ids': missing}
    return identity, deepcopy(answers[0]), [record]
