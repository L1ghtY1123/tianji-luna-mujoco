"""Simultaneous camera correspondences, committed only as consistent components."""
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from .association_geometry import CROSS_VIEW_TOLERANCE_M, DEPTH_TOLERANCE_M, SURFACE_MATCH, head_wrist_pair, pair_geometry, table_contact
from .association_geometry import strong_surface_match as _strong_surface_match
from .core.grounding import CurrentGroup
from .decision_conditions import parse_answers
from .instance_masks import consolidate_masks, separate_container_contents
from .raw_visual_input import capture_content, subject_description, subject_detail_content
from .visual_api import VisualAPI
from .visual_identity import choice, mask_box


def _display_numbers(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _display_numbers(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_display_numbers(child) for child in value]
    return value


def _geometry_summary(evidence):
    result = {'visible_bounds_gap_m': evidence.get('spatial', {}).get('visible_bounds_gap_m')}
    for key in ('same_image_overlap', 'nearest_surface_distance_m', 'visible_bounds_iou', 'shared_voxel_fraction_of_smaller'):
        if key in evidence:
            result[key] = evidence[key]
    for key in ('left_to_right', 'right_to_left', 'cad_left_to_right', 'cad_right_to_left'):
        if key in evidence:
            result[key] = {name: evidence[key][name] for name in ('status',
                'projected_surface_in_target_fraction', 'projected_mask_iou',
                'target_surface_coverage', 'depth_consistent_fraction', 'visible_depth_consistent_fraction') if name in evidence[key]}
    if 'cad_world_bounds_iou' in evidence:
        result['cad_world_bounds_iou'] = evidence['cad_world_bounds_iou']
    return result


def _ask(client, capture, candidates, name, question, choices, state):
    visible = set(state.get('visible_indices', range(len(candidates))))
    masks = {}
    for index, row in enumerate(candidates):
        if index not in visible:
            continue
        masks.setdefault(row['camera_id'], {})['C' + str(index)] = row['silhouette_mask']
    state = {**state, 'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'candidate_labels': {'C' + str(i): {'camera_id': row['camera_id'],
            'bbox': mask_box(row['silhouette_mask']), 'query_hints': row['query_prompts'],
            'depth': row['depth_evidence'], 'pose_hypothesis': row.get('pose_hypothesis')}
            for i, row in enumerate(candidates) if i in visible}}
    # Keep full calibration/provenance in the audit record. Present only the
    # quantities needed by this question, so hashes and repeated metadata do
    # not obscure the actual correspondence evidence.
    shown = {key: state[key] for key in ('left', 'right', 'anchor', 'target_camera', 'subject',
        'stage', 'representative_preference',
        'attribute_hypotheses', 'group_members', 'current_groups') if key in state}
    shown['frame'] = 'Same instant. World metres: +X robot forward, +Y robot left, +Z up.'
    shown['confirmed_physical_kinds'] = {'C' + str(i): candidates[i]['physical_kind'] for i in visible
                                       if candidates[i].get('physical_kind')}
    if 'group_members' in state:
        shown['mask_support'] = {label: {key: row['depth'].get(key) for key in (
            'status', 'mask_pixel_count', 'mask_component_count', 'largest_mask_component_pixels',
            'mask_bbox_fill_fraction', 'valid_point_count')} for label,row in state['candidate_labels'].items()}
    if 'target_camera' in state or 'pair' in state or 'current_groups' in state:
        shown['visible_surface_centers'] = {label: row['depth'].get('visible_surface_center_world_m')
            for label, row in state['candidate_labels'].items()}
        shown['visible_surface_bounds'] = {label: row['depth'].get('visible_bounds_world_m')
            for label, row in state['candidate_labels'].items()}
        shown['cad_pose_hypotheses'] = {label: {key: row['pose_hypothesis'][key]
            for key in ('asset_id', 'bounds_world_m')} for label, row in state['candidate_labels'].items()
            if (row['pose_hypothesis'] or {}).get('status') == 'estimated'}
    if 'pair' in state:
        shown['pair'] = _geometry_summary(state['pair'])
    if 'pair_geometry' in state:
        shown['comparisons'] = {key: _geometry_summary(value) for key, value in state['pair_geometry'].items()}
    content = [{'type': 'input_text', 'text': json.dumps(_display_numbers(shown), ensure_ascii=False)}]
    if state.get('subject_details'):
        if state.get('include_context'):
            scoped = {**capture, 'views': {camera: capture['views'][camera] for camera in masks}}
            content.extend(capture_content(scoped, masks=masks, modalities=('rgb',)))
        for camera, subjects in masks.items():
            for label, mask in subjects.items():
                if 'detail_indices' in state and int(label[1:]) not in state['detail_indices']:
                    continue
                content.extend(subject_detail_content(capture, camera, label, mask))
    elif state.get('separate_masks'):
        # Overlaid near-identical masks hide each other's labels. Show each on
        # its own copy of the SAME image, with unchanged image coordinates.
        for camera, subjects in masks.items():
            scoped = {**capture, 'views': {camera: capture['views'][camera]}}
            for label, mask in subjects.items():
                content.extend(capture_content(scoped, masks={camera: {label: mask}}, modalities=('rgb', 'mask')))
    else:
        scoped = {**capture, 'views': {camera: capture['views'][camera] for camera in masks}}
        content.extend(capture_content(scoped, masks=masks,
            camera_order=('3', '2', '1'), modalities=('rgb',)))
    query = choice(name, question, choices)
    body, evidence = VisualAPI.request(client, {'model': 'gpt-6-luna',
        'input': [{'role': 'user', 'content': content}], 'questions': [query]}, capture,
        'current_scene_association', lambda body, evidence: (body, evidence))
    answers, records = parse_answers(body, [query])
    return answers[name], {**evidence, 'answers': records, 'association_state': state}


def _consistent_components(indices, candidates, choices, unknown, distinct):
    """Require mutual head/wrist ownership and a distinct camera for every group member."""
    neighbors = {i: set() for i in indices}
    conflicts = []
    ambiguous = set(unknown)
    for (source, camera), target in choices.items():
        if not head_wrist_pair(candidates[source]['camera_id'], camera):
            raise ValueError('cross_camera_link_requires_head_and_wrist')
        if target == 'unknown':
            ambiguous.add(source)
        elif target is not None:
            neighbors[source].add(target)
            neighbors[target].add(source)
            reverse = choices.get((target, candidates[source]['camera_id']), 'unknown')
            if reverse != source:
                conflicts.append({'source': source, 'target': target,
                    'target_camera': camera, 'reverse_choice': reverse,
                    'reason': 'cross_camera_correspondence_not_mutual'})
                ambiguous.update((source, target))
    components, remaining = [], set(indices)
    while remaining:
        pending, component = [min(remaining)], set()
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(neighbors[index] - component)
        remaining.difference_update(component)
        cameras = [candidates[i]['camera_id'] for i in component]
        excluded = [sorted(pair) for pair in distinct if pair <= component]
        if len(cameras) != len(set(cameras)) or excluded:
            conflicts.append({'component': sorted(component), 'reason': 'same_camera_entity_conflict',
                              'distinct_pairs': excluded})
            ambiguous.update(component)
        if '3' not in cameras:
            ambiguous.update(component)
            conflicts.append({'component': sorted(component), 'reason': 'head_anchor_unobserved'})
        if ambiguous.intersection(component):
            ambiguous.update(component)
        components.append(tuple(sorted(component)))
    return components, ambiguous, conflicts


def review_declared_counts(candidates, components, ambiguous, rejected, records, *, initial_supports):
    by_quote = {}
    limits = {}
    for members in components:
        if rejected.intersection(members) or not any(candidates[i]['camera_id'] == '3' for i in members):
            continue
        subject = candidates[members[0]]
        quote = subject.get('scene_quote')
        for hint in subject.get('retrieval_hypotheses', []):
            if quote and hint.get('scene_quote') == quote:
                by_quote.setdefault(quote, []).append(members)
                limits[quote] = hint.get('declared_count')
                break
    for quote, groups in by_quote.items():
        count = limits[quote]
        if count is None or len(groups) <= count:
            continue
        indices = {i for group in groups for i in group}
        outcomes, witnesses = set(), []
        if count == 1 and not ambiguous.intersection(indices):
            for part, support in initial_supports:
                if support != quote or len(by_quote.get(part, [])) != 1:
                    continue
                members = by_quote[part][0]
                if ambiguous.intersection(members) or rejected.intersection(members):
                    continue
                part_points = np.vstack([candidates[i]['points'] for i in members])
                relations = []
                for group in groups:
                    table_points = np.vstack([candidates[i]['points'] for i in group])
                    relation, evidence = table_contact(table_points, part_points)
                    relations.append(relation)
                    witnesses.append({'table_group': list(group), 'part_group': list(members),
                                      'relation': relation, **evidence})
                if relations.count('supports') == 1 and relations.count('below') == len(groups) - 1:
                    outcomes.add(tuple(groups[relations.index('supports')]))
        if len(outcomes) == 1:
            rejected.update(indices - set(next(iter(outcomes))))
        else:
            ambiguous.update(indices)
        records.append({'stage': 'declared_count_competition', 'source': 'measured_support_geometry',
                        'scene_quote': quote, 'declared_count': count, 'witnesses': witnesses,
                        'winner': list(next(iter(outcomes))) if len(outcomes) == 1 else None})


def _discard_lower_plane_proposals(candidates, available, ambiguous, rejected, records, initial_supports):
    # A horizontal floor can be returned by a table text/box prompt. Compare it
    # with the independently measured part and tabletop before counting tables.
    for part_quote, table_quote in initial_supports:
        parts = [candidates[i]['points'] for i in available - rejected - ambiguous
                 if candidates[i]['camera_id'] == '3' and candidates[i].get('scene_quote') == part_quote
                 and len(candidates[i]['points']) >= 30]
        tables = [i for i in available - rejected if any(h.get('scene_quote') == table_quote
                  and h.get('kind') == 'table' and h.get('declared_count') == 1
                  for h in candidates[i]['retrieval_hypotheses'])]
        if not parts or len(tables) < 2:
            continue
        part_points = np.vstack(parts)
        supported = [i for i in tables if i not in ambiguous and
                     table_contact(candidates[i]['points'], part_points)[0] == 'supports']
        if not supported:
            continue
        height = float(np.quantile(part_points[:, 2], .05))
        for index in set(tables) - set(supported):
            points = np.asarray(candidates[index]['points'])
            if len(points) < 80:
                continue
            z = float(np.median(points[:, 2]))
            horizontal = points[np.abs(points[:, 2] - z) <= DEPTH_TOLERANCE_M]
            if (height - z > .08 and len(horizontal) >= .75 * len(points)
                    and np.min(np.ptp(horizontal[:, :2], axis=0)) > .04):
                rejected.add(index)
                ambiguous.discard(index)
                records.append({'stage': 'declared_table_support', 'source': 'distinct_lower_horizontal_plane',
                    'candidate_index': index, 'measured_z_m': z, 'supported_part_bottom_m': height,
                    'confirmed_support_candidates': supported, 'anonymous_depth_retained': True})


def shared_tabletop_matches(candidates, eligible):
    from scipy.ndimage import label
    from .association_geometry import measured_table_top
    from .cameras import deproject_view

    tables = {}
    for i in eligible:
        row = candidates[i]
        if row.get('physical_kind') != 'table' or not row.get('scene_quote'):
            continue
        if not any(h.get('scene_quote') == row['scene_quote'] and h.get('declared_count') == 1
                   for h in row['retrieval_hypotheses']):
            continue
        plane = measured_table_top(row['points'], tolerance_m=CROSS_VIEW_TOLERANCE_M)
        if plane is not None:
            tables.setdefault(row['scene_quote'], []).append((i, row['camera_id'], plane[0]))
    matches = []
    for rows in tables.values():
        heads = [(i, z) for i, camera, z in rows if camera == '3']
        if len(heads) != 1:
            continue
        head, z = heads[0]
        row = candidates[head]
        view = row['view']
        mask = row['mask'] & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
        r, c = np.where(mask)
        points = deproject_view(view, mask)
        plane_mask = np.zeros(mask.shape, dtype=bool)
        on_plane = np.abs(points[:, 2] - z) <= CROSS_VIEW_TOLERANCE_M
        plane_mask[r[on_plane], c[on_plane]] = True
        labels, count = label(plane_mask, structure=np.ones((3,3), dtype=bool))
        if not count:
            continue
        sizes = np.bincount(labels[plane_mask]); component = sizes.argmax()
        if sizes[component] < .65 * plane_mask.sum():
            continue
        surface = cKDTree(points[labels[r,c] == component])
        members, coverage = [head], {}
        for camera in ('1', '2'):
            supported = []
            for index, source, height in rows:
                if source != camera or abs(height - z) > CROSS_VIEW_TOLERANCE_M:
                    continue
                xyz = candidates[index]['points']
                xyz = xyz[np.abs(xyz[:, 2] - z) <= CROSS_VIEW_TOLERANCE_M]
                if len(xyz) < 80:
                    continue
                sampled = xyz[::max(1, len(xyz)//4096)]
                fraction = float(np.mean(surface.query(sampled)[0] <= CROSS_VIEW_TOLERANCE_M))
                if fraction >= .65:
                    supported.append((index, fraction))
            if len(supported) != 1:
                continue
            index, fraction = supported[0]
            members.append(index)
            coverage[camera] = fraction
        if len(members) > 1:
            matches.append((members, {'stage': 'cross_camera_correspondence',
                'source': 'same_connected_head_tabletop', 'candidate_indices': members,
                'height_m': z, 'wrist_surface_coverage': coverage}))
    return matches


def declared_kind_evidence(row, candidates, initial_supports=()):
    """Resolve a single declared SAM hypothesis using independent measured evidence."""
    from .association_geometry import measured_mask_color, measured_table_top

    hints = [h for h in row['retrieval_hypotheses'] if h.get('scene_quote')]
    color = measured_mask_color(row)
    if len(hints) == 1 and row['depth_evidence']['status'] == 'measured':
        hint = hints[0]
        # SAM3 text with a matching RGBD mask and an independently proposed
        # current box. A text label alone cannot establish physical kind.
        boxed = any(source.get('origin') in {'responses_box', 'reobserved_proposal_rgbd'} and source.get('scene_quote') == hint['scene_quote']
                    for source in row['retrieval_sources'])
        color_matches = color != 'unknown' and color == hint.get('color')
        if hint['kind'] == 'table' and hint.get('declared_count') == 1 and row['camera_id'] != '3':
            heads = [h for h in candidates if h['camera_id'] == '3' and h.get('physical_kind') == 'table'
                     and h.get('scene_quote') == hint['scene_quote']]
            head_plane = measured_table_top(heads[0]['points']) if len(heads) == 1 else None
            if head_plane is not None:
                z = np.asarray(row['points'])[:, 2]
                # A narrow vertical edge can belong to the table. Reject only a different,
                # lower horizontal surface whose measured returns do not reach the tabletop.
                low = float(np.median(z))
                if (len(z) >= 80 and low < head_plane[0] - .08
                        and np.mean(np.abs(z-low) <= DEPTH_TOLERANCE_M) >= .60
                        and np.quantile(z, .95) < head_plane[0] - .08
                        and np.quantile(z, .99) < head_plane[0] - DEPTH_TOLERANCE_M):
                    return 'invalid', {'source': 'wrist_plane_below_head_table',
                        'scene_quote': hint['scene_quote'], 'head_surface_z_m': head_plane[0],
                        'wrist_surface_z_m': low, 'stage': 'declared_scene_membership'}
        table_matches = False
        corroborated = False
        if hint['kind'] == 'table' and hint.get('color') in ('unknown', color):
            top = measured_table_top(row['points'])
            if top is not None:
                # A box around a horizontal floor also returns a coherent plane.
                # Confirm the intended tabletop from another calibrated view or named supported part.
                support_quotes = {part for part, table in initial_supports if table == hint['scene_quote']}
                for other in candidates:
                    if other is row or other['depth_evidence']['status'] != 'measured':
                        continue
                    if ((other['camera_id'] == row['camera_id'] or head_wrist_pair(other['camera_id'], row['camera_id']))
                            and support_quotes.intersection(h.get('scene_quote') for h in other['retrieval_hypotheses'])
                            and table_contact(row['points'], other['points'])[0] == 'supports'):
                        table_matches = True
                    if head_wrist_pair(other['camera_id'], row['camera_id']) and other.get('retrieval_hypotheses') == hints:
                        other_top = measured_table_top(other['points'])
                        if other_top is not None and abs(top[0] - other_top[0]) <= DEPTH_TOLERANCE_M:
                            overlap = np.minimum(top[1][1], other_top[1][1]) - np.maximum(top[1][0], other_top[1][0])
                            if np.all(overlap > .04):
                                table_matches = corroborated = True
                    if table_matches:
                        break
        if not boxed and color_matches and hint['kind'] != 'table':
            # A text-prompt mask can be independently supported by another
            # calibrated view even when Responses supplied no box here.
            corroborated = any(head_wrist_pair(other['camera_id'], row['camera_id'])
                and other.get('retrieval_hypotheses') == hints
                and other['depth_evidence']['status'] == 'measured'
                and measured_mask_color(other) == color and _strong_surface_match(pair_geometry(row, other))
                for other in candidates)
        if (boxed or corroborated) and (table_matches if hint['kind'] == 'table' else color_matches):
            return hint['kind'], {'source': 'declared_sam3_box_rgbd' if boxed else 'declared_sam3_multiview_rgbd', 'color': color,
                                  'scene_quote': hint['scene_quote'], 'stage': 'declared_scene_membership'}
    return None


def group_current_candidates(client, capture, observation, *, pose_resolver=None, initial_supports=()):
    """No historical ID, task role, or greedy camera ownership enters this stage."""
    if (observation.observation_id, observation.generation) != (capture['observation_id'], capture['generation']):
        raise ValueError('candidate_capture_mismatch')
    candidates = observation.candidates

    def classify(index):
        hints = [hint for hint in candidates[index]['retrieval_hypotheses'] if hint.get('scene_quote')]
        measured = declared_kind_evidence(candidates[index], candidates, initial_supports)
        if measured:
            kind, evidence = measured
            if kind != 'invalid':
                candidates[index]['scene_quote'] = evidence['scene_quote']
            return kind, evidence
        measured = candidates[index].get('scope_measurement')
        if measured:
            hint = next(h for h in hints if h['scene_quote'] == measured['scene_quote'])
            candidates[index]['scene_quote'] = hint['scene_quote']
            return hint['kind'], measured
        if hints:
            options = {'D' + str(i): hint for i, hint in enumerate(hints)}
            answer, record = _ask(client, capture, candidates, f'scene_subject_{index}',
                f'Which declared subject does the entity at the WHITE pixels C{index} visibly match? '
                'Inspect the RGB at the masked surface including context. Match the kind, color and '
                'distinguishing description. A coherent surface of a table or bin may belong to that '
                'entity. Separate contents and supported objects have their own identity. Choose invalid '
                'for an undeclared subject, the room floor, robot or mixed masks. Choose unknown if unclear.',
                [{'value': key, 'description': hint['scene_quote'] + ' / ' + hint['prompt']}
                 for key, hint in options.items()] + ['invalid', 'unknown'],
                {'stage': 'declared_scene_membership', 'subject': 'C' + str(index), 'group_members': [index],
                 'visible_indices': [index], 'subject_details': True, 'include_context': True})
            if answer in options:
                hint = options[answer]
                candidates[index]['scene_quote'] = hint['scene_quote']
                return hint['kind'], record
            return ('invalid' if answer == 'invalid' else 'unknown'), record
        return _ask(client, capture, candidates, f'physical_kind_{index}',
            f'What physical entity do the WHITE mask pixels C{index} belong to? '
            'Use the full RGB scene and the matched unpainted RGB/binary-mask details. '
            'A visible tabletop, side or edge belongs to the whole table; a container wall, floor '
            'or rim belongs to the whole container. A separate object ON a table or INSIDE a '
            'container is its own entity. Inspect the WHITE pixels, not everything enclosed by '
            'the bounding rectangle. Reject a mask that mixes several unrelated entities, '
            'the room floor/ground/background, or the robot/attached cameras/cables. '
            'A coherent partial surface of an identifiable entity is valid; scattered unrelated '
            'pixels are invalid. Choose unknown when physical ownership is not clear.',
            [{'value': 'table', 'description': 'One table or work surface, including its own sides/edges.'},
             {'value': 'bin', 'description': 'One container/receptacle, excluding its separate contents.'},
             {'value': 'object', 'description': 'One distinct physical item/part/package, excluding its support.'},
             {'value': 'marker', 'description': 'One physical marker or label.'},
             {'value': 'invalid', 'description': 'Robot, room floor/background, mixed entities, or stray pixels.'},
             'unknown'],
            {'stage': 'physical_mask_validation', 'subject': 'C' + str(index), 'group_members': [index],
             'visible_indices': [index], 'subject_details': True, 'include_context': True})

    def ownership(index, indices):
        return _ask(client, capture, candidates, f'owner_{index}',
            f'Which ONE mask is the best representative of the physical entity whose surface is C{index}? '
            'Consider ALL labelled masks in this SAME image. Select the mask showing the most complete '
            'surface of that SAME physical entity, including C' + str(index) + ' itself. '
            'A tabletop and its rim/edge are one entity; different walls/floor/rim of one container '
            'are one entity. Container contents and objects resting on a table are SEPARATE entities. '
            'Do not select a supporting table or enclosing container for an object on/in it. '
            'Two identical-looking but separate objects remain separate. White pixels define each mask, '
            'not its enclosing rectangle. If multiple masks are equally complete, use the first one '
            'in representative_preference. Choose invalid for robot/background/mixed objects or '
            'pixels without a physical entity; unknown if you cannot determine ownership.',
            [{'value': 'C' + str(i), 'description': subject_description('C' + str(i),
                candidates[i]['camera_id'], mask_box(candidates[i]['silhouette_mask']))} for i in indices]
            + ['invalid', 'unknown'],
            {'stage': 'same_image_entity_ownership', 'subject': 'C' + str(index),
             'representative_preference': ['C' + str(i) for i in indices],
             'visible_indices': indices, 'detail_indices': indices,
             'subject_details': True, 'include_context': True})

    available, ambiguous, rejected, distinct, duplicates, records = consolidate_masks(capture, candidates, ownership, classify)
    _discard_lower_plane_proposals(candidates, available, ambiguous, rejected, records, initial_supports)
    unmeasured = {i for i in available - rejected if candidates[i]['depth_evidence']['status'] != 'measured'}
    ambiguous.update(unmeasured)
    if unmeasured:
        records.append({'stage': 'cross_camera_measurement', 'status': 'unknown',
            'reason': 'each_associated_view_requires_measured_depth', 'candidate_indices': sorted(unmeasured)})
    eligible = sorted(available - ambiguous - rejected, key=lambda i: candidates[i]['candidate_id'])
    comparisons = {}

    def compare(a, b):
        if (a, b) not in comparisons:
            comparisons[a, b] = pair_geometry(candidates[a], candidates[b])
        return comparisons[a, b]

    alternatives = {(a, camera): [b for b in eligible if candidates[b]['camera_id'] == camera
        and (not candidates[a].get('physical_kind') or not candidates[b].get('physical_kind')
             or candidates[a]['physical_kind'] == candidates[b]['physical_kind'])
        and candidates[a].get('scene_quote') == candidates[b].get('scene_quote')
        and compare(a, b)['spatial']['status'] != 'separated']
        for a in eligible for camera in ('3', '2', '1') if head_wrist_pair(camera, candidates[a]['camera_id'])}
    strong = {key: [b for b in values if _strong_surface_match(compare(key[0], b))]
              for key, values in alternatives.items()}
    choices = {}
    for key, values in alternatives.items():
        a, camera = key
        if not values:
            choices[key] = None
        elif len(strong[key]) == 1:
            b = strong[key][0]
            if strong.get((b, candidates[a]['camera_id'])) == [a]:
                choices[key] = b
                records.append({'stage': 'cross_camera_correspondence', 'source': 'unique_bidirectional_depth',
                    'source_index': a, 'target_index': b, 'target_camera': camera,
                    'evidence': compare(a, b),
                    'policy': dict(SURFACE_MATCH)})

    for indices, evidence in shared_tabletop_matches(candidates, eligible):
        head, *wrists = indices
        for wrist in wrists:
            camera = candidates[wrist]['camera_id']
            if (choices.get((head, camera)) in (None, wrist) and choices.get((wrist, '3')) in (None, head)
                    and frozenset((head, wrist)) not in distinct):
                choices[head, camera], choices[wrist, '3'] = wrist, head
                records.append({**evidence, 'candidate_indices': [head, wrist]})

    def ensure_pose(index):
        if pose_resolver is not None and 'pose_hypothesis' not in candidates[index]:
            candidates[index]['pose_hypothesis'] = pose_resolver(capture, candidates[index])
            for key in list(comparisons):
                if index in key:
                    del comparisons[key]

    for (anchor, camera), possible in alternatives.items():
        if (anchor, camera) in choices:
            continue
        # Optional CAD evidence is requested only after redundant/partial masks
        # have been consolidated, and only for unresolved camera correspondence.
        for index in [anchor, *possible]:
            ensure_pose(index)
        # Weak contact between visible fragments is insufficient for identity.
        # A head detection may stand alone. A wrist detection cannot create an ID.
        supported = list(strong[anchor, camera])
        for other in possible:
            if other in supported:
                continue
            evidence = compare(anchor, other)
            if (evidence.get('cad_world_bounds_iou') or 0.) < SURFACE_MATCH['minimum_bounds_iou']:
                continue
            projections = [evidence.get(k, {}) for k in ('cad_left_to_right', 'cad_right_to_left')]
            if all(p.get('depth_supported_pixels', 0) >= SURFACE_MATCH['minimum_common_pixels']
                   and (p.get('projected_surface_in_target_fraction') or 0.) >= SURFACE_MATCH['minimum_surface_precision']
                   and (p.get('visible_depth_consistent_fraction') or 0.) >= SURFACE_MATCH['minimum_depth_consistency']
                   for p in projections):
                supported.append(other)
        choices[anchor, camera] = supported[0] if len(supported) == 1 else 'unknown' if supported else None
        records.append({'stage': 'cross_camera_correspondence', 'source': 'calibrated_surface_and_optional_cad',
                        'source_index': anchor, 'target_camera': camera, 'possible': possible,
                        'geometrically_supported': supported,
                        'pair_geometry': {str(i): _geometry_summary(compare(anchor, i)) for i in possible}})
    components, ambiguous, conflicts = _consistent_components(eligible, candidates, choices, ambiguous, distinct)
    # Unknown/invalid masks remain explicit, but cannot acquire an ID. Their
    # depth still contributes anonymous collision geometry independently.
    components.extend((i,) for i in sorted(available - set(eligible)))
    review_declared_counts(candidates, components, ambiguous, rejected, records, initial_supports=initial_supports)
    records.append({'stage': 'current_group_integrity', 'status': 'unknown' if ambiguous else 'success',
        'source': 'simultaneous_correspondence_constraints', 'ambiguous_indices': sorted(ambiguous),
        'rejected_indices': sorted(rejected), 'conflicts': conflicts,
        'choices': [{'source': a, 'target_camera': camera, 'target': target}
                    for (a, camera), target in choices.items()],
        'duplicate_representatives': {str(k): v for k, v in duplicates.items()}})
    confirmed = [group for group in components if not (ambiguous | rejected).intersection(group)]
    records.extend(separate_container_contents(capture, candidates, confirmed, unresolved=ambiguous - rejected))
    result = []
    for members in sorted(components, key=lambda group: sorted(candidates[i]['candidate_id'] for i in group)):
        identifiers = sorted(candidates[i]['candidate_id'] for i in members)
        group_id = observation.observation_id + ':group:' + hashlib.sha256('|'.join(identifiers).encode()).hexdigest()[:16]
        geometry = {'status': 'insufficient', 'frame': 'Link_Base', 'visible_surface_only': True,
            'correspondence_status': 'unknown' if ambiguous.intersection(members) else 'confirmed',
            'physical_status': 'invalid' if rejected.intersection(members) else 'candidate'}
        measured = [candidates[i] for i in members if candidates[i]['depth_evidence']['status'] == 'measured']
        head = next((row for row in measured if row['camera_id'] == '3'), None)
        if head is not None:
            points = np.vstack([row['points'] for row in measured])
            geometry.update(status='measured', primary_camera_id='3',
                            full_visible_bounds_world_m=[points.min(0).tolist(), points.max(0).tolist()],
                            visible_surface_center_world_m=np.median(head['points'], axis=0).tolist())
        result.append(CurrentGroup(group_id, observation.observation_id, tuple(members), geometry))
    return tuple(result), records
