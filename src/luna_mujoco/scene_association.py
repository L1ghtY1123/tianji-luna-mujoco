"""Scene-wide physical identities are independent of task selection success."""
from dataclasses import replace

import numpy as np

from .association_geometry import DEPTH_TOLERANCE_M, dominant_rgb_color, measured_mask_color, measured_table_top
from .core.grounding import COLOR_CATEGORIES, HistoryMatch
from .current_instances import _ask, group_current_candidates
from .visual_api import VisualDecisionError
from .temporal_identity import choose_identity, resolve_identity_conflict


def measured_head_table_color(members):
    head = next((r for r in members if r['camera_id'] == '3'), None)
    plane = measured_table_top(head['points']) if head is not None else None
    if plane is None:
        return 'unknown'
    on_top = np.abs(head['points'][:, 2] - plane[0]) <= DEPTH_TOLERANCE_M
    return dominant_rgb_color(head['pixels'][on_top])


def compatible_kind(kind, known_kinds):
    return not known_kinds or kind not in {'table', 'bin', 'marker', 'object'} or kind in known_kinds


def associate_scene(client, capture, observation, *, entities, reference, archive_loader=None, pose_resolver=None, initial_supports=()):
    groups, correspondence = group_current_candidates(client, capture, observation, pose_resolver=pose_resolver,
                                                       initial_supports=initial_supports)
    candidates = observation.candidates
    members = {g.group_id: [candidates[i] for i in g.candidate_indices] for g in groups}
    labels = {g.group_id: 'G' + str(i) for i, g in enumerate(groups)}
    matches, metadata = {}, {}
    decisions = [row for row in correspondence if row.get("request_hash")]
    for group in groups:
        if group.geometry_evidence.get('physical_status') == 'invalid':
            matches[group.group_id] = HistoryMatch(group.group_id, 'reject', None,
                {'source': 'same_image_entity_ownership', 'records': []})
            continue
        if (group.geometry_evidence.get('status') != 'measured'
                or group.geometry_evidence.get('correspondence_status') != 'confirmed'):
            matches[group.group_id] = HistoryMatch(group.group_id, 'unknown', None,
                {'reason': 'measured_current_group_required', 'records': []})
            continue
        if not entities:
            # Novelty is a known database fact here. The category decision below
            # still has to confirm a distinct valid physical entity before commit.
            matches[group.group_id] = HistoryMatch(group.group_id, 'new', None,
                {'source': 'empty_registry_and_confirmed_current_group',
                 'compared_instance_ids': [], 'records': []})
            continue
        current = members[group.group_id]
        known_kinds = {row['physical_kind'] for row in current if row.get('physical_kind')}
        offered = {key: row for key, row in entities.items() if compatible_kind(row.get('kind'), known_kinds)}
        relevant_ref = {**reference, 'identity_references': {key: value for key, value in
            reference.get('identity_references', {}).items() if key in offered}}
        rivals = {labels[key]: rows for key, rows in members.items() if key != group.group_id
                  and all(compatible_kind(row.get('physical_kind'), known_kinds) for row in rows)}
        try:
            identity, answer, records = choose_identity(client, capture, current, offered, relevant_ref,
                labels[group.group_id], archive_loader=archive_loader, rivals=rivals)
        except VisualDecisionError as error:
            identity, answer = 'unknown', {}
            records = [{**error.evidence, 'source': 'temporal_decision_unavailable', 'error': str(error)}]
        decisions.extend(record for record in records if record.get('request_hash'))
        outcome = 'existing' if identity in offered else 'new' if identity == 'new_instance' else 'reject' if identity == 'reject' else 'unknown'
        matches[group.group_id] = HistoryMatch(group.group_id, outcome, identity if outcome == 'existing' else None,
            {'records': records, 'answer': answer, 'compared_instance_ids': list(offered),
             'excluded_incompatible_kinds': sorted(entities.keys() - offered.keys()),
             'source': records[-1].get('source', 'openai_decisions') if records else 'unknown'})
    owners = {}
    for group_id, match in matches.items():
        if match.entity_id:
            owners.setdefault(match.entity_id, []).append(group_id)
    for identifier, conflicts in owners.items():
        if len(conflicts) < 2:
            continue
        try:
            winner, record = resolve_identity_conflict(client, capture, {key: members[key] for key in conflicts},
                                                       identifier, entities, reference)
        except VisualDecisionError as error:
            winner, record = None, {**error.evidence, 'source': 'temporal_conflict_unavailable', 'error': str(error)}
        if record:
            decisions.append(record)
        for group_id in conflicts:
            previous = matches[group_id]
            matches[group_id] = replace(previous,
                outcome='existing' if group_id == winner else 'unknown',
                entity_id=identifier if group_id == winner else None,
                evidence={**previous.evidence, 'records': [*previous.evidence['records'], *([record] if record else [])],
                          'joint_conflict_winner': winner})
    # A duplicate claim can hide another remembered subject. After a joint review has established
    # ownership, Decisions may compare the losing group with the remaining unclaimed historical IDs.
    reserved = {match.entity_id: labels[key] for key, match in matches.items() if match.entity_id}
    alternatives = {}
    for key, match in matches.items():
        if match.outcome != 'unknown' or not match.evidence.get('joint_conflict_winner'):
            continue
        offered = {i: entities[i] for i in match.evidence['compared_instance_ids'] if i not in reserved}
        if not offered:
            continue
        dated = {i: v for i, v in reference.get('identity_references', {}).items() if i in offered}
        rivals = {labels[other]: rows for other, rows in members.items() if other != key}
        try:
            identity, answer, records = choose_identity(client, capture, members[key], offered,
                {**reference, 'identity_references': dated}, labels[key], archive_loader=archive_loader,
                rivals=rivals, reserved=reserved)
        except VisualDecisionError as error:
            identity, answer = 'unknown', {}
            records = [{**error.evidence, 'source': 'temporal_decision_unavailable', 'error': str(error)}]
        decisions.extend(record for record in records if record.get('request_hash'))
        matches[key] = replace(match, outcome='reject' if identity == 'reject' else 'unknown',
            evidence={**match.evidence, 'records': [*match.evidence['records'], *records],
                      'remaining_history_answer': answer, 'reserved_ids': reserved})
        if identity in offered:
            alternatives.setdefault(identity, []).append(key)
    for identifier, claimants in alternatives.items():
        if len(claimants) == 1:
            key = claimants[0]
            match = matches[key]
            matches[key] = replace(match, outcome='existing', entity_id=identifier,
                evidence={**match.evidence, 'original_history_answer': match.evidence.get('answer'),
                          'answer': match.evidence['remaining_history_answer'], 'accepted_identity': identifier})
    for group in groups:
        match = matches[group.group_id]
        if match.outcome not in {'existing', 'new'} or group.geometry_evidence['correspondence_status'] != 'confirmed':
            continue
        if match.entity_id:
            item = entities[match.entity_id]
            if item.get('kind') == 'table':
                color = measured_head_table_color(members[group.group_id])
                if color != 'unknown':
                    item = {**item, 'color': color, 'description': color + ' table',
                            'attribute_evidence': {'color_source': 'measured_head_tabletop_rgb', 'color': color}}
            metadata[group.group_id] = item
            continue
        # Semantics can exclude an unrelated candidate even when its depth or
        # identity is unknown. This does not upgrade the unknown identity or
        # permit registration without measured, confirmed correspondence.
        colors = (*COLOR_CATEGORIES, 'unknown')
        state = {'group_members': list(group.candidate_indices), 'visible_indices': list(group.candidate_indices),
                 'subject_details': True, 'include_context': True,
                 'attribute_hypotheses': [hint for i in group.candidate_indices
                     for hint in candidates[i].get('retrieval_hypotheses', [])]}
        subject = ', '.join('C' + str(i) for i in group.candidate_indices)
        classified = [candidates[i] for i in group.candidate_indices if candidates[i].get('physical_kind')]
        kinds = {row['physical_kind'] for row in classified}
        if len(kinds) == 1:
            kind = classified[0]['physical_kind']
            kind_record = classified[0]['physical_kind_evidence']
        else:
            kind, kind_record = _ask(client, capture, candidates, 'category_' + labels[group.group_id],
                f'What kind of physical entity is represented by {subject}? RGB context and matched RGB/mask '
                'details show the subject. WHITE pixels belong to it; BLACK pixels do not. Classify the entity '
                'as a whole, not only its visible face. The bounding rectangle is not the subject. '
                'Scattered edge pixels without a distinct object are invalid. '
                'Retrieval hints are hypotheses, not confirmed facts.',
                [{'value': 'table', 'description': 'Tabletop or fixed supporting work surface.'},
                 {'value': 'bin', 'description': 'Hollow container/receptacle with an interior for other objects.'},
                 {'value': 'object', 'description': 'Other physical item, such as a solid block, part or package.'},
                 {'value': 'marker', 'description': 'Physical marker or label.'},
                 {'value': 'invalid', 'description': 'Mixed objects, background, robot, or a fragment without a distinct object.'},
                 'unknown'], state)
            decisions.append(kind_record)
        matches[group.group_id] = replace(match,
            evidence={**match.evidence, 'records': [*match.evidence['records'], kind_record],
                      'physical_category': kind})
        if kind in {'invalid', 'unknown'}:
            matches[group.group_id] = replace(matches[group.group_id],
                outcome='reject' if kind == 'invalid' else 'unknown')
            continue
        measured_colors = {measured_mask_color(r) for r in members[group.group_id]} - {'unknown'}
        tabletop_color = measured_head_table_color(members[group.group_id]) if kind == 'table' else 'unknown'
        if tabletop_color != 'unknown':
            color, color_record = tabletop_color, {'source': 'measured_head_tabletop_rgb'}
        elif len(measured_colors) == 1:
            color = measured_colors.pop()
            color_record = {'source': 'dominant_mask_rgb'}
        else:
            color, color_record = _ask(client, capture, candidates, 'color_' + labels[group.group_id],
                f'What is the main visible color of the {kind} labelled {subject}? '
                'Use RGB colors at WHITE pixels of the corresponding binary mask. '
                'BLACK pixels in the mask are excluded from this subject.', list(colors), state)
            decisions.append(color_record)
        attributes = {'kind': kind, 'color': color}
        scope_hint = next((hint for row in classified for hint in row.get('retrieval_hypotheses', [])
                           if row.get('scene_quote') and hint.get('scene_quote') == row['scene_quote']), None)
        metadata[group.group_id] = {**attributes, 'description': attributes['color'] + ' ' + attributes['kind'],
            'segmentation_query': {'kind': attributes['kind'], 'color': attributes['color'],
                'prompt': scope_hint['prompt'] if scope_hint else (attributes['color'] + ' ' if attributes['color'] != 'unknown' else '')
                    + ('container' if attributes['kind'] == 'bin' else attributes['kind']), 'aliases': []},
            'attribute_evidence': {'category_source': kind_record.get('source'), 'category_request_hash': kind_record.get('request_hash'),
                'color_source': color_record['source'], 'color_request_hash': color_record.get('request_hash'), 'kind': kind, 'color': color}}
    return groups, tuple(matches[g.group_id] for g in groups), {
        'current_correspondence': correspondence, 'decision_records': decisions,
        'physical_metadata': metadata, 'coverage': observation.coverage,
        'missing_tracks': sorted(set(entities) - {m.entity_id for m in matches.values() if m.entity_id}),
        'unresolved_groups': [m.group_id for m in matches.values() if m.outcome == 'unknown']}
