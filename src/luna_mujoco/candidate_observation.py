"""Declared-scene SAM3 retrieval; every mask is a hypothesis without permanent ID."""
import hashlib
from itertools import combinations
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import find_objects, label

from .association_depth import candidate_depth_evidence
from .association_geometry import CROSS_VIEW_TOLERANCE_M, projected_depth_masks
from .cameras import deproject_view
from .core.grounding import CandidateObservation
from .observation_overlap import box_mask_metrics, mask_pair_metrics
from .visual_identity import mask_box

BOX_MASK_COVERAGE = .5
BOX_EXTENT_IOU = .5
MAX_CANDIDATES = 512


def normalized_query(text):
    return ' '.join(text.casefold().split())


def release_region_boxes(capture, identifier, entity, dated, quote):
    """Use the last bound release as a search region for a fresh SAM3 box."""
    from .demo_transfer_verification import valid_container_entry

    receipt = entity.get('container_entry') or {}
    destination = receipt.get('destination_id')
    region = receipt.get('container_geometry') or {}
    if (not destination or not valid_container_entry(receipt, identifier, destination, receipt.get('release_feedback_id'))
            or dated.get('observation_id') != receipt.get('pre_release_measurement_observation_id')
            or region.get('entity_id') != destination):
        return []
    lower, upper = (np.asarray(region.get(key), dtype=float) for key in ('minimum_world_m', 'maximum_world_m'))
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(upper < lower)):
        return []
    colors, clouds = [], []
    for old in dated.get('views', {}).values():
        if not all(key in old for key in ('rgb', 'mask', 'valid_depth', 'depth_m', 'T_world_camera', 'intrinsics')):
            continue
        valid = old['mask'] & old['valid_depth'] & np.isfinite(old['depth_m']) & (old['depth_m'] > 0)
        if valid.sum() >= 30:
            colors.append(old['rgb'][valid])
            clouds.append(deproject_view(old, valid))
    if not clouds:
        return []
    previous = np.vstack(colors).astype(float)
    color = np.median(previous, axis=0)
    if np.mean(np.max(np.abs(previous - color), axis=1) <= 24) < .65:
        return []
    span = np.linalg.norm(np.ptp(np.quantile(np.vstack(clouds), [.05, .95], axis=0), axis=0))
    if not np.isfinite(span) or span <= 0:
        return []
    low, high = lower - .015, upper + [.015, .015, span]
    hints = []
    for camera, view in capture['views'].items():
        depth = view['depth_m']
        valid = view['valid_depth'] & np.isfinite(depth) & (depth > 0)
        valid &= ~np.asarray(view.get('robot_mask', False), dtype=bool)
        valid &= np.max(np.abs(view['rgb'] - color), axis=2) <= 24
        points = deproject_view(view, valid)
        rows, columns = np.nonzero(valid)
        inside = np.all((points >= low) & (points <= high), axis=1)
        region_pixels = np.zeros_like(valid)
        region_pixels[rows[inside], columns[inside]] = True
        components, _ = label(region_pixels)
        component_sizes = np.bincount(components[region_pixels])
        height, width = depth.shape
        for component, window in enumerate(find_objects(components), 1):
            if window is None or component_sizes[component] < 30:
                continue
            y, x = window
            member = components[rows, columns] == component
            size = np.linalg.norm(np.ptp(np.quantile(points[member], [.05, .95], axis=0), axis=0))
            if min(size, span) < .5 * max(size, span):
                continue
            hints.append(SimpleNamespace(scene_quote=quote, camera_id=camera, asset_id=None,
                bbox=[x.start / width, y.start / height, x.stop / width, y.stop / height],
                proposal_id=f'release-{identifier}-{camera}-{component}', origin='release_region_measured_appearance'))
    return hints


def dated_box_prompts(capture, declared_objects, reference):
    """Project dated measurements into current depth to retrieve declared subjects."""
    allowed = {d.scene_quote for d in declared_objects}
    hints = []
    for identifier, dated in reference.get('identity_references', {}).items():
        entity = reference.get('entities', {}).get(identifier, {})
        quote = (entity.get('segmentation_query') or {}).get('scene_quote')
        if quote not in allowed or dated.get('observation_id') == capture['observation_id']:
            continue
        hints.extend(release_region_boxes(capture, identifier, entity, dated, quote))
        for camera, old in dated.get('views', {}).items():
            if camera not in capture['views'] or not all(k in old for k in
                    ('mask', 'valid_depth', 'depth_m', 'intrinsics', 'T_world_camera')):
                continue
            samples = deproject_view(old, old['mask'] & old['valid_depth'])
            projected, _, shared, _ = projected_depth_masks(samples, capture['views'][camera])
            if shared.sum() < 30 or shared.sum() < .5 * projected.sum():
                continue
            rows, cols = np.nonzero(shared)
            h, w = shared.shape
            box = [cols.min()/w, rows.min()/h, (cols.max()+1)/w, (rows.max()+1)/h]
            hints.append(SimpleNamespace(scene_quote=quote, camera_id=camera, bbox=[float(v) for v in box], asset_id=None,
                proposal_id=f'depth-{identifier}-{camera}', origin='dated_depth_projected_to_current_view'))
    return hints


def unchanged_subject_appearance(capture, candidate, reference):
    """Re-use only the declared class of a visually unchanged measured surface."""
    camera = candidate['camera_id']
    view = capture['views'][camera]
    quotes = {h['scene_quote'] for h in candidate['retrieval_hypotheses']}
    matches = []
    for identifier, dated in reference.get('identity_references', {}).items():
        query = (reference.get('entities', {}).get(identifier) or {}).get('segmentation_query') or {}
        old = dated.get('views', {}).get(camera, {})
        if (query.get('scene_quote') not in quotes or 'mask' not in old or 'depth_m' not in old
                or 'valid_depth' not in old or 'rgb' not in old or old.get('intrinsics') != view['intrinsics']
                or not np.allclose(old['T_world_camera'], view['T_world_camera'], rtol=0, atol=1e-6)
                or old['mask'].shape != candidate['silhouette_mask'].shape):
            continue
        overlap = mask_pair_metrics(old['mask'], candidate['silhouette_mask'])['mask_iou']
        shared = candidate['mask'] & old['mask'] & old['valid_depth']
        if overlap < .75 or shared.sum() < 30:
            continue
        depth_agree = np.abs(view['depth_m'][shared] - old['depth_m'][shared]) <= .015
        rgb_agree = np.max(np.abs(view['rgb'][shared].astype(int) - old['rgb'][shared]), axis=1) <= 12
        fraction = float(np.mean(depth_agree & rgb_agree))
        if fraction >= .8:
            matches.append({'scene_quote': query['scene_quote'], 'reference_entity_id': identifier, 'mask_iou': overlap,
                'rgb_depth_agreement': fraction, 'reference_observation_id': dated.get('observation_id')})
    if matches and len({m['scene_quote'] for m in matches}) == 1:
        return {'stage': 'declared_scene_membership', 'source': 'unchanged_measured_appearance',
                'scene_quote': matches[0]['scene_quote'], 'comparisons': matches, 'identity_assigned': False}
    return None


def projected_prompt_box(points, view):
    _, visible, shared, _ = projected_depth_masks(points, view, CROSS_VIEW_TOLERANCE_M)
    if shared.sum() < 30 or shared.sum() < .55 * visible.sum():
        return None
    x0, y0, x1, y1 = mask_box(shared)
    h, w = shared.shape
    dx, dy = max(3 / w, .08 * (x1-x0)), max(3 / h, .08 * (y1-y0))
    return [max(0., x0-dx), max(0., y0-dy), min(1., x1+dx), min(1., y1+dy)], shared


def head_projection_prompts(candidates, query, camera, view):
    for row in candidates:
        if (row['camera_id'] != '3' or query['hypotheses'][0] not in row['retrieval_hypotheses']
                or row['depth_evidence']['status'] != 'measured'):
            continue
        projected = projected_prompt_box(row['points'], view)
        if projected is None:
            continue
        bbox, shared = projected
        source = {'origin': 'current_head_depth_projection', 'scene_quote': query['scene_quote'],
                  'source_candidate': row['candidate_id'], 'target_camera': camera,
                  'depth_supported_pixels': int(shared.sum())}
        yield bbox, source, shared


def reobserved_proposal_hints(previous, capture, inventory):
    """Reuse a box only as a SAM prompt when its camera and measured patch still agree."""
    if (capture['observation_id'] == previous['observation_id'] or capture['generation'] != previous['generation']):
        return []
    hints = []
    for row in inventory:
        cam, box = row.camera_id, row.bbox
        if not box or cam not in capture['views'] or cam not in previous['views']:
            continue
        old, new = previous['views'][cam], capture['views'][cam]
        if (old['intrinsics'] != new['intrinsics'] or old.get('calibration_id') != new.get('calibration_id')
                or not np.allclose(old['T_world_camera'], new['T_world_camera'], rtol=0, atol=.001)):
            continue
        h, w = new['depth_m'].shape
        x0,y0,x1,y1 = np.array(box) * [w,h,w,h]
        patch = np.s_[int(y0):int(np.ceil(y1)):2, int(x0):int(np.ceil(x1)):2]
        old_valid = np.asarray(old['valid_depth'][patch], dtype=bool)
        new_valid = np.asarray(new['valid_depth'][patch], dtype=bool)
        visible, shared = old_valid | new_valid, old_valid & new_valid
        if np.count_nonzero(shared) < 30:
            continue
        agreement = (np.abs(old['depth_m'][patch][shared] - new['depth_m'][patch][shared]) <= .006)
        agreement &= np.max(np.abs(old['rgb'][patch][shared].astype(int) - new['rgb'][patch][shared]), axis=1) <= 12
        if np.count_nonzero(agreement) >= .90 * np.count_nonzero(visible):
            hints.append(dict(scene_quote=row.scene_quote, camera_id=cam, bbox=box, asset_id=None,
                proposal_id=row.proposal_id, origin='reobserved_proposal_rgbd'))
    return hints


def collect_candidates(capture, declared_objects, segment_query, *, inventory=(), proposal_observation_id=None,
                       reference=None, reobserved_hints=()):
    if set(capture.get('views', {})) != {'1', '2', '3'}:
        raise ValueError('grounding_requires_synchronized_three_rgbd_views')
    if not declared_objects:
        raise ValueError('explicit_scene_declarations_required')
    queries = {}
    for subject in declared_objects:
        hint = {'scene_quote': subject.scene_quote, 'kind': subject.kind, 'color': subject.color,
                'prompt': subject.segmentation_prompt, 'declared_count': subject.declared_count, 'asset_id': subject.asset_id}
        queries[subject.scene_quote] = {**hint, 'aliases': [], 'hypotheses': [hint],
            'sources': [{'origin': 'declared_scene', 'scene_quote': subject.scene_quote}]}
    candidates, attempts, seen, coverage = [], [], {}, {}
    current_inventory = (inventory if proposal_observation_id == capture['observation_id']
                         else dated_box_prompts(capture, declared_objects, reference or {}))

    if reobserved_hints and proposal_observation_id != capture['observation_id']:
        current_inventory = [*current_inventory, *(SimpleNamespace(**hint) for hint in reobserved_hints
                             if hint['scene_quote'] in queries)]

    def retrieve(camera, query, *, projected_surface=None):
        view = capture['views'][camera]
        proposals, prompt = segment_query(capture, camera, query)
        attempts.append({'camera_id': camera, 'query': query['prompt'], 'used_prompt': prompt,
            'bbox': query.get('bbox'), 'sources': query['sources'], 'mask_count': len(proposals),
            'outside_proposal_count': 0, 'absence_proven': False})
        for proposal in proposals:
            raw_mask = np.asarray(proposal['mask'], dtype=bool)
            if raw_mask.shape != view['depth_m'].shape:
                raise ValueError('candidate_mask_image_mismatch')
            mask = raw_mask.copy()
            if not mask.any():
                continue
            if query.get('bbox') is not None:
                metrics = box_mask_metrics(query['bbox'], mask)
                if ((projected_surface is None and (metrics['mask_coverage'] < BOX_MASK_COVERAGE
                                                    or metrics['box_iou'] < BOX_EXTENT_IOU))
                        or (projected_surface is not None and
                            (projected_surface & mask).sum() < .70 * projected_surface.sum())):
                    attempts[-1]['outside_proposal_count'] += 1
                    continue
            score = float(proposal['score'])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('invalid_sam3_score')
            stamp = hashlib.sha256(mask.tobytes()).hexdigest()
            key = (camera, stamp)
            terms = list(dict.fromkeys([query['prompt'], *query['aliases'], prompt]))
            hints = query.get('hypotheses', [
                {'kind': query['kind'], 'color': query['color'], 'prompt': query['prompt']}])
            if key in seen:
                existing = candidates[seen[key]]
                existing['query_prompts'] = list(dict.fromkeys([*existing['query_prompts'], *terms]))
                existing['retrieval_sources'].extend(query['sources'])
                for hint in hints:
                    if hint not in existing['retrieval_hypotheses']:
                        existing['retrieval_hypotheses'].append(hint)
                continue
            if len(candidates) >= MAX_CANDIDATES:
                raise ValueError('scene_evidence_budget_exceeded')
            valid = mask & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
            if 'robot_mask' in view:
                valid &= ~np.asarray(view['robot_mask'], dtype=bool)
            points = deproject_view(view, valid)
            candidate = {'candidate_id': capture['observation_id'] + ':' + camera + ':' + stamp[:16],
                'camera_id': camera, 'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'calibration_id': view.get('calibration_id'), 'mask': valid, 'silhouette_mask': mask,
                'raw_identity_mask': mask, 'raw_sam3_mask': raw_mask, 'mask_sha256': stamp, 'points': points,
                'pixels': view['rgb'][valid], 'pixels_rc': np.column_stack(np.nonzero(valid)), 'view': view,
                'score': score, 'kind': query['kind'], 'color': query['color'], 'prompt': prompt,
                'query_prompts': terms, 'retrieval_sources': list(query['sources']), 'retrieval_hypotheses': list(hints)}
            candidate['depth_evidence'] = candidate_depth_evidence(capture, candidate)
            for array in (mask, valid, points):
                array.setflags(write=False)
            seen[key] = len(candidates)
            candidates.append(candidate)

    def retrieve_box(camera, query, bbox, source, shared=None):
        found = False
        for row in candidates:
            if row['camera_id'] != camera or query['hypotheses'][0] not in row['retrieval_hypotheses']:
                continue
            mask = row['silhouette_mask']
            if shared is not None:
                matches = np.count_nonzero(shared & mask) >= .70 * shared.sum()
            else:
                metrics = box_mask_metrics(bbox, mask)
                matches = metrics['mask_coverage'] >= BOX_MASK_COVERAGE and metrics['box_iou'] >= BOX_EXTENT_IOU
            if matches:
                row['retrieval_sources'].append(source)
                found = True
        if not found:
            retrieve(camera, {**query, 'bbox': bbox, 'sources': [source]}, projected_surface=shared)

    for camera in ('3', '2', '1'):
        if camera == '3':
            for query in queries.values():
                retrieve(camera, query)
        for row in current_inventory:
            if row.camera_id == camera and row.bbox is not None:
                query = queries[row.scene_quote]
                source = {'origin': getattr(row, 'origin', 'responses_box'), 'scene_quote': row.scene_quote,
                          'proposal_id': row.proposal_id, 'asset_id': query['asset_id']}
                retrieve_box(camera, query, row.bbox, source)
        if camera != '3':
            for query in queries.values():
                for bbox, source, shared in head_projection_prompts(candidates, query, camera, capture['views'][camera]):
                    retrieve_box(camera, query, bbox, source, shared)
    for camera, view in capture['views'].items():
        valid = view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
        if 'robot_mask' in view:
            valid &= ~view['robot_mask']
        explained = np.zeros_like(valid)
        for row in candidates:
            if row['camera_id'] == camera:
                explained |= row['silhouette_mask']
        coverage[camera] = {'valid_depth_pixels': int(valid.sum()), 'unexplained_depth_pixels': int((valid & ~explained).sum()),
            'absence_proven': False, 'occluded_space': 'unobserved', 'anonymous_geometry': 'retained_in_capture_depth'}
    candidates.sort(key=lambda row: (('3', '2', '1').index(row['camera_id']), row['candidate_id']))
    for candidate in candidates:
        evidence = unchanged_subject_appearance(capture, candidate, reference or {})
        if evidence:
            candidate['scope_measurement'] = evidence
    overlaps = []
    for row in current_inventory:
        for candidate in candidates:
            if row.camera_id == candidate['camera_id'] and row.bbox is not None:
                overlaps.append(dict(proposal_id=row.proposal_id, candidate_id=candidate['candidate_id'],
                    camera_id=row.camera_id, observation_id=capture['observation_id'], bbox=row.bbox,
                    **box_mask_metrics(row.bbox, candidate['silhouette_mask'])))
    pairs = []
    for left, right in combinations(candidates, 2):
        if left['camera_id'] == right['camera_id']:
            metrics = mask_pair_metrics(left['silhouette_mask'], right['silhouette_mask'])
            if metrics['mask_iou'] > 0:
                pairs.append(dict(left=left['candidate_id'], right=right['candidate_id'],
                    camera_id=left['camera_id'], observation_id=capture['observation_id'], **metrics,
                    automatically_merged=False))
    return CandidateObservation(capture['observation_id'], capture['generation'], tuple(candidates),
        tuple(attempts), tuple(overlaps), tuple(pairs), coverage)
