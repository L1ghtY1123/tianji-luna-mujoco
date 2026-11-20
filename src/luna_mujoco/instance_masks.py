"""Consolidate segmentation hypotheses before cross-camera identity assignment."""
import hashlib
from itertools import combinations

import numpy as np

from .association_depth import candidate_depth_evidence, spatial_relation
from .association_geometry import DEPTH_TOLERANCE_M, head_wrist_pair
from .cameras import deproject_view
from .observation_overlap import mask_pair_metrics

# Symmetric pixel IoU, not box IoU or containment. This consolidates redundant
# segmentation hypotheses in ONE image; it never matches a historical ID.
REDUNDANT_MASK_IOU = .70
MAX_OWNERSHIP_CANDIDATES_PER_CAMERA = 64


def corroborated_target_masks(views, target_id, distance_m=.004):
    """Recover owned boundary returns only inside the original SAM silhouette."""
    from scipy.ndimage import binary_dilation
    from scipy.spatial import cKDTree

    surfaces = {}
    for camera, view in views.items():
        mask = view.get('instance_masks', {}).get(target_id)
        if mask is not None:
            points = deproject_view(view, mask)
            if len(points):
                surfaces[camera] = points
    result = {}
    for camera, view in views.items():
        raw = view.get('identity_masks', {}).get(target_id)
        if raw is None or camera not in surfaces or len(surfaces) < 2:
            continue
        mask = np.asarray(view['instance_masks'][target_id], dtype=bool)
        boundary = (np.asarray(raw, dtype=bool) & ~mask & binary_dilation(mask)
                    & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0))
        boundary &= ~np.asarray(view.get('robot_surface_mask', False), dtype=bool)
        for key, other in view['instance_masks'].items():
            if key != target_id:
                boundary &= ~np.asarray(other, dtype=bool)
        rows, cols = np.nonzero(boundary)
        if not len(rows):
            continue
        measured = deproject_view(view, boundary)
        witnesses = [pts for key, pts in surfaces.items() if head_wrist_pair(camera, key)]
        if not witnesses:
            continue
        other_surface = np.concatenate(witnesses)
        witnessed = cKDTree(other_surface).query(measured, distance_upper_bound=distance_m)[0] < distance_m
        if np.any(witnessed):
            expanded = mask.copy()
            expanded[rows[witnessed], cols[witnessed]] = True
            result[camera] = expanded
    return result


def _rank(candidates, index):
    row = candidates[index]
    return (-int(row['silhouette_mask'].sum()), -len(row['points']),
            -row['score'], row['candidate_id'])


def _merge_sources(candidates, winner, members):
    row = candidates[winner]
    for name in ('query_prompts', 'retrieval_sources', 'retrieval_hypotheses'):
        values = []
        for index in members:
            for value in candidates[index].get(name, []):
                if value not in values:
                    values.append(value)
        row[name] = values
    row['consolidated_candidate_ids'] = sorted({identifier for index in members for identifier in
        candidates[index].get('consolidated_candidate_ids', [candidates[index]['candidate_id']])})


def _merge_surfaces(capture, candidates, winner, members):
    """Only confirmed parts of ONE entity may contribute to a union silhouette."""
    row = candidates[winner]
    mask = np.logical_or.reduce([candidates[i]['silhouette_mask'] for i in members])
    _merge_sources(candidates, winner, members)
    if np.array_equal(mask, row['silhouette_mask']):
        return
    _set_surface(capture, row, mask, 'confirmed_same_image_entity_parts')


def _set_surface(capture, row, mask, source):
    view = row['view']
    valid = mask & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
    if 'robot_mask' in view:
        valid &= ~np.asarray(view['robot_mask'], dtype=bool)
    stamp = hashlib.sha256(mask.tobytes()).hexdigest()
    points = deproject_view(view, valid)
    for value in (mask, valid, points):
        value.setflags(write=False)
    row.update(silhouette_mask=mask, raw_identity_mask=mask, mask=valid, points=points,
        pixels=view['rgb'][valid], pixels_rc=np.column_stack(np.nonzero(valid)), mask_sha256=stamp,
        candidate_id=capture['observation_id'] + ':' + row['camera_id'] + ':' + stamp[:16],
        mask_source=source)
    row['depth_evidence'] = candidate_depth_evidence(capture, row)
    row.pop('scope_measurement', None)
    row.pop('pose_hypothesis', None)
    row.pop('_pose_world_points', None)


def _confirmed_pixels(points, witness, exclusive):
    """Sample the other camera at each 3D point and require matching visible depth."""
    view = witness['view']
    transform = np.asarray(view['T_world_camera'])
    xyz = (points - transform[:3, 3]) @ transform[:3, :3]
    depth = view['depth_m']
    k = view['intrinsics']
    valid = np.flatnonzero(np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0))
    x = np.rint(xyz[valid, 0] * k['fx'] / xyz[valid, 2] + k['cx']).astype(int)
    y = np.rint(xyz[valid, 1] * k['fy'] / xyz[valid, 2] + k['cy']).astype(int)
    inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
    indices, x, y = valid[inside], x[inside], y[inside]
    measured = exclusive[y, x] & (np.abs(depth[y, x] - xyz[indices, 2]) <= DEPTH_TOLERANCE_M)
    result = np.zeros(len(points), dtype=bool)
    result[indices[measured]] = True
    return result


def separate_container_contents(capture, candidates, components, *, unresolved=()):
    """Resolve measured shared pixels only after simultaneous groups agree."""
    objects, bins, by_camera = [], [], {}
    for group in components:
        rows = [candidates[i] for i in group]
        for row in rows:
            by_camera.setdefault(row['camera_id'], []).append(row)
        kinds = {row.get('physical_kind') for row in rows}
        if kinds == {'bin'}:
            bins.extend(rows)
        elif kinds == {'object'} and len(rows) > 1:
            objects.append(rows)
    for index in unresolved:
        row = candidates[index]
        by_camera.setdefault(row['camera_id'], []).append(row)
    changes = []
    for container in bins:
        mask = container['silhouette_mask'].copy()
        removed = np.zeros_like(mask)
        for group in objects:
            subject = next((r for r in group if r['camera_id'] == container['camera_id']), None)
            if subject is None:
                continue
            overlap = mask & subject['mask']
            if overlap.sum() < 30:
                continue
            points = deproject_view(subject['view'], overlap)
            confirmed = np.zeros(len(points), dtype=bool)
            for witness in group:
                if not head_wrist_pair(subject['camera_id'], witness['camera_id']):
                    continue
                exclusive = witness['mask'].copy()
                for other in by_camera[witness['camera_id']]:
                    if other is not witness:
                        exclusive &= ~other['silhouette_mask']
                confirmed |= _confirmed_pixels(points, witness, exclusive)
            if confirmed.sum() >= 30:
                y, x = np.nonzero(overlap)
                removed[y[confirmed], x[confirmed]] = True
        remaining = mask & ~removed
        measured = np.count_nonzero(container['mask'] & remaining)
        if removed.any() and measured >= max(30, .5 * remaining.sum()):
            changes.append((container, remaining, int(removed.sum())))
    records = []
    for row, mask, count in changes:
        original = row['candidate_id']
        _set_surface(capture, row, mask, 'depth_confirmed_container_contents_excluded')
        records.append({'stage': 'visible_surface_ownership', 'source': row['mask_source'],
                        'container_candidate': original, 'camera': row['camera_id'], 'removed_pixels': count})
    return records


def consolidate_masks(capture, candidates, ask, classify):
    """Return per-camera entity representatives, unknowns and rejected masks.

    Validate the physical subject before comparing ownership. Positive same-
    entity claims are equivalences: chains and cycles can share one physical
    entity. A deterministic representative is selected after collecting them.
    """
    records, duplicate_of, unresolved, rejected = [], {}, set(), set()
    for camera in ('3', '2', '1'):
        indices = sorted((i for i, row in enumerate(candidates) if row['camera_id'] == camera),
                         key=lambda i: _rank(candidates, i))
        declarations = {i: {hint['scene_quote'] for hint in candidates[i]['retrieval_hypotheses']
                            if 'scene_quote' in hint} for i in indices}
        available = set(indices)
        for winner in indices:
            if winner not in available:
                continue
            members = [winner]
            for index in indices:
                if index == winner or index not in available:
                    continue
                # Complete-link membership prevents a chain of near overlaps
                # from collapsing two masks that disagree with each other.
                if any(declarations[member] != declarations[index] for member in members):
                    continue
                if all(mask_pair_metrics(candidates[index]['silhouette_mask'],
                        candidates[member]['silhouette_mask'])['mask_iou'] >= REDUNDANT_MASK_IOU
                       for member in members):
                    members.append(index)
                    available.remove(index)
                    duplicate_of[index] = winner
            if len(members) > 1:
                _merge_sources(candidates, winner, members)
                records.append({'stage': 'mask_consolidation', 'source': 'same_image_mask_nms',
                    'camera_id': camera, 'representative': winner, 'members': members,
                    'minimum_pairwise_mask_iou': REDUNDANT_MASK_IOU,
                    'policy': 'complete_link_symmetric_mask_overlap_not_box_containment',
                    'permanent_identity_assigned': False})
        indices = [i for i in indices if i in available]
        if len(indices) < 2 and not any(
                hint.get('scene_quote') for i in indices for hint in candidates[i]['retrieval_hypotheses']):
            continue
        if len(indices) > MAX_OWNERSHIP_CANDIDATES_PER_CAMERA:
            unresolved.update(indices)
            records.append({'stage': 'same_image_entity_ownership', 'status': 'unknown',
                'camera_id': camera, 'reason': 'ownership_candidate_budget_exceeded'})
            continue
        for index in indices:
            kind, record = classify(index)
            records.append(record)
            candidates[index]['physical_kind'] = kind
            candidates[index]['physical_kind_evidence'] = record
            if kind == 'invalid':
                rejected.add(index)
            elif kind == 'unknown':
                unresolved.add(index)
        indices = [i for i in indices if i not in rejected and i not in unresolved]
        owners = {}
        for index in indices:
            compatible = [i for i in indices if candidates[i]['physical_kind'] == candidates[index]['physical_kind']
                and candidates[i].get('scene_quote') == candidates[index].get('scene_quote')
                and (i == index or spatial_relation(candidates[i]['depth_evidence'],
                     candidates[index]['depth_evidence'])['status'] != 'separated')]
            if len(compatible) == 1:
                owners[index] = index
                continue
            answer, record = ask(index, compatible)
            records.append(record)
            if answer == 'invalid':
                rejected.add(index)
            elif answer in {'C' + str(i) for i in compatible}:
                owners[index] = int(answer[1:])
            else:
                unresolved.add(index)
        neighbors = {i: set() for i in indices}
        for index, owner in owners.items():
            neighbors[index].add(owner)
            neighbors[owner].add(index)
        remaining = set(indices)
        while remaining:
            pending, component = [min(remaining)], set()
            while pending:
                index = pending.pop()
                if index not in component:
                    component.add(index)
                    pending.extend(neighbors[index] - component)
            remaining.difference_update(component)
            if component.intersection(unresolved | rejected):
                unresolved.update(component - rejected)
                continue
            members = sorted(component, key=lambda i: _rank(candidates, i))
            root = members[0]
            if len(members) < 2:
                continue
            _merge_surfaces(capture, candidates, root, members)
            for index in members:
                if index != root:
                    duplicate_of[index] = root
            records.append({'stage': 'same_image_entity_ownership', 'source': 'consistent_visual_ownership',
                'camera_id': camera, 'representative': root, 'members': sorted(members),
                'surface_policy': 'union_of_confirmed_parts_no_workspace_clipping'})
    representatives = set(range(len(candidates))) - duplicate_of.keys()
    unresolved.intersection_update(representatives)
    rejected.intersection_update(representatives)
    # Same-camera canonical representatives are distinct according to the joint
    # ownership task, even when their masks overlap (container and contents).
    distinct = {frozenset((a, b)) for a, b in combinations(sorted(representatives - unresolved - rejected), 2)
                if candidates[a]['camera_id'] == candidates[b]['camera_id']}
    return representatives, unresolved, rejected, distinct, duplicate_of, records
