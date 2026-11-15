"""Demo sensing: raw three-camera bootstrap and a permanent RGB scene objects."""
import hashlib
import time
from copy import deepcopy

import numpy as np

from .cameras import deproject_view
from .core.decisions import predicate_status
from .demo_geometry import fit_objects_table, supported_surface_points
from .memory_reference import current_memory_entities
from .perception import (
    Perception,
    PerceptionError,
    _readonly,
)
from .visual_api import VisualDecisionError
from .visual_identity import VisualIdentity, mask_box


class DemoVisualIdentity(VisualIdentity):
    """Associate current subjects while preserving unresolved identity evidence."""

    def resolve(self, capture, detections, *, reference=None, entities=None, **kwargs):
        entities = {key: value for key, value in current_memory_entities(reference, entities).items()
                    if value.get('source') != 'configured_demo_region'}
        return super().resolve(capture, detections, reference=reference, entities=entities, **kwargs)

    def _groups(self, detections, names, entities, judgments, evidence, *, purpose=None, reference=None,
                progress=None):
        accepted = {judgments['identity_' + str(index)] for index in range(len(detections))
                    if judgments['identity_' + str(index)] in entities}
        required = {identifier for identifier, entity in entities.items()
                    if entity.get('kind') not in {'table', 'marker'}
                    and entity.get('status') not in {'removed', 'consumed'}}
        pending_verification = {identifier for identifier, record in (progress or {}).items()
                                if record.get('release_feedback_id') and record.get('status') != 'completed'}
        if purpose == 'verify_transfer' and pending_verification:
            required = pending_verification
        coverage = {'source': evidence['source'], 'observation_id': evidence['observation_id'],
            'request_hash': evidence['request_hash'], 'purpose': purpose,
            'selection_rule': 'probability_argmax',
            'required_entity_ids': sorted(required), 'accepted_entity_ids': sorted(accepted),
            'complete': required <= accepted}
        unassociated = self._unresolved_identity_candidates(detections, names, judgments, evidence)
        retained = [index for index in range(len(detections))
                    if judgments['identity_' + str(index)] != 'unknown']
        if purpose == 'verify_transfer' and pending_verification - accepted:
            raise VisualDecisionError('visual_verification_target_unconfirmed', evidence={
                **evidence, 'missing_target_ids': sorted(pending_verification - accepted)})
        if not unassociated:
            result = super()._groups(detections, names, entities, judgments, evidence)
            result['identity_coverage'] = coverage
            return result
        # Only trusted candidates enter semantic geometry. Unassociated mask pixels
        # remain in the audit evidence but do not enter the objects-only map.
        retained_judgments = dict(judgments)
        for index, original in enumerate(retained):
            retained_judgments['identity_' + str(index)] = judgments['identity_' + str(original)]
        result = super()._groups([detections[index] for index in retained], [names[index] for index in retained],
                                 entities, retained_judgments, evidence)
        result['unassociated_candidates'] = unassociated
        result['identity_coverage'] = coverage
        return result

    def _unresolved_identity_candidates(self, detections, names, judgments, evidence):
        """Summarize existing verdicts without making another identity decision."""
        original = evidence.get('original_decisions_answers', evidence['answers'])
        answers = {row['name']: row for row in evidence['answers']}
        unresolved = []
        for index, detection in enumerate(detections):
            if judgments['identity_' + str(index)] != 'unknown':
                continue
            name = 'identity_' + names[index].removeprefix('candidate-')
            answer = answers[name]
            raw = next((row for row in original if row['name'] == name
                        and row.get('type') == answer.get('type')
                        and row.get('probability') == answer.get('probability')), answer)
            reason = ('correspondence_conflict' if answer.get('identity_conflict') else
                      'model_refusal' if raw.get('type') == 'refusal' else
                      'proposed_identity_rejected' if raw.get('type') == 'predicate'
                      and predicate_status(raw.get('probability')) == 'no' else
                      'proposed_identity_unconfirmed' if raw.get('type') == 'predicate' else
                      'choice_tied_or_correspondence_conflict')
            unresolved.append({'candidate_id': names[index], 'camera_id': detection['camera_id'],
                'source': self.source, 'observation_id': evidence['observation_id'],
                'generation': evidence.get('generation'), 'request_hash': evidence['request_hash'],
                'status': 'unknown', 'reason': reason,
                **({'predicate_probability_threshold': .5, 'predicate_comparison': '>'}
                   if raw.get('type') == 'predicate' else {'selection_rule': 'probability_argmax'}),
                'answer': deepcopy(answer), 'original_decisions_answers': [deepcopy(raw)]})
        return unresolved


class DemoPerception(Perception):

    def __init__(self, detector=None, *, robot_surface_classifier=None, visual_identity=None, cad_resolver=None):
        self.demo_tables = None
        super().__init__(detector, robot_surface_classifier=robot_surface_classifier,
            visual_identity=visual_identity if visual_identity is not None else DemoVisualIdentity.from_environment(),
            cad_resolver=cad_resolver)

    @classmethod
    def from_environment(cls):
        import os

        from .cad_perception import CadGeometryResolver

        return cls(cad_resolver=CadGeometryResolver.from_environment()
                   if os.getenv('LUNA_GEOMETRY_MODE') == 'cad_esdf' else None)

    def verify_release(self, capture, binding, reference, instruction=""):
        from .demo_transfer_verification import verify_release_result
        target = binding['target_id']
        entities = {target: reference.get('entities', {}).get(target, {})}
        evidence = self.prepare_gripper_evidence(capture, [binding['arm']], entities)
        return verify_release_result(self.visual_identity, capture, binding, reference, instruction,
                                     gripper_evidence=evidence)

    def bootstrap(self, capture):
        raise PerceptionError('demo_initial_observation_required')

    def _observe_entities(self, capture, instruction, checkpoint, hint, entities, reference, discovery_evidence):
        current_entities = current_memory_entities(reference, entities)
        return super()._observe_entities(capture, instruction, checkpoint, hint, current_entities,
                                         reference, discovery_evidence)

    def _record_segmentation_attempt(self, evidence, view, proposals):
        super()._record_segmentation_attempt(evidence, view, proposals)
        evidence['rgb_sha256'] = hashlib.sha256(np.ascontiguousarray(view['rgb']).tobytes()).hexdigest()
        evidence['proposals'] = []
        for proposal in proposals:
            mask = np.asarray(proposal['mask'], dtype=bool)
            if mask.shape != view['depth_m'].shape:
                raise PerceptionError('invalid_sam3_detection')
            valid = mask & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
            evidence['proposals'].append({'score': float(proposal['score']), 'mask': _readonly(mask),
                'valid_depth_mask': _readonly(valid), 'points_world_m': _readonly(deproject_view(view, valid)),
                'pixels_rc': _readonly(np.column_stack(np.nonzero(valid))),
                'identity_status': 'unassociated_proposal'})

    def _deduplicate_detections(self, detections):
        unique = []
        for detection in detections:
            mask = np.asarray(detection['silhouette_mask'], bool)
            duplicate = None
            for previous in unique:
                if previous['camera_id'] != detection['camera_id']:
                    continue
                other = np.asarray(previous['silhouette_mask'], bool)
                if np.array_equal(other, mask):
                    duplicate = previous
                    break
            if duplicate is None:
                unique.append(detection | {'prompt_proposals': [detection['prompt']]})
            else:
                duplicate['prompt_proposals'].append(detection['prompt'])
                duplicate['query_prompts'] = list(dict.fromkeys(
                    duplicate.get('query_prompts', []) + detection.get('query_prompts', [])))
        return unique


    def _filter_geometry_groups(self, groups, views, *, entities=None):
        for group in groups:
            for entry in group:
                points = np.asarray(entry['points'])
                keep = supported_surface_points(points)
                entity = (entities or {}).get(entry.get('entity_id'), {})
                evidence = entity.get('observed_support', {})
                support_id = evidence.get('support_entity_id')
                support = self.demo_tables['tables'].get(support_id)
                frame_id = views.get(entry['camera_id'], {}).get('frame_id')
                if (support and frame_id and evidence.get('observation_id') == frame_id
                        and evidence.get('source') == 'openai_decisions' and evidence.get('request_hash')):
                    model = support['model']
                    center = np.asarray(model['pose'][:2])
                    half = np.asarray(model['dims'][:2]) / 2
                    inside = (np.all(np.abs(points[:, :2] - center) <= half + .006, axis=1)
                              & (points[:, 2] >= support['surface_height_m'] - .006))
                    keep &= inside
                    entry['geometry_filter_evidence'] = {
                        'source': 'cached_measured_table_and_current_observed_support',
                        'support_entity_id': support_id, 'camera_id': entry['camera_id'],
                        'observation_id': frame_id, 'request_hash': evidence['request_hash'],
                        'surface_height_m': support['surface_height_m'],
                        'footprint_center_xy_m': center.tolist(), 'footprint_half_size_m': half.tolist(),
                        'measurement_margin_m': .006, 'raw_point_count': len(points),
                        'outside_support_scope_points': int(np.count_nonzero(~inside)),
                        'identity_assignment': False}
                for table in self.demo_tables['tables'].values():
                    model = table['model']
                    center, half = np.asarray(model['pose'][:2]), np.asarray(model['dims'][:2]) / 2
                    on_table = (np.all(np.abs(points[:, :2] - center) <= half + .006, axis=1)
                                & (np.abs(points[:, 2] - table['surface_height_m']) <= .006))
                    keep &= ~on_table
                if 'geometry_filter_evidence' in entry:
                    entry['geometry_filter_evidence']['retained_point_count'] = int(np.count_nonzero(keep))
                mask = np.array(entry['mask'], copy=True)
                removed = np.asarray(entry['pixels_rc'])[~keep]
                if len(removed):
                    mask[removed[:, 0], removed[:, 1]] = False
                entry['mask'] = mask
                for field in ('points', 'pixels', 'pixels_rc'):
                    entry[field] = np.asarray(entry[field])[keep]
        return groups

    def _table_for_entities(self, capture, hint, entities, reference, detections, *, background_points=()):
        reference_id = (reference or {}).get('observation_id')
        if self.demo_tables is None or self.demo_tables['source_observation_id'] != reference_id:
            measurement = capture
            historical = reference_id != capture['observation_id']
            if historical:
                measurement = hint.get('reference_capture')
                if (not measurement or measurement.get('observation_id') != reference_id
                        or measurement.get('generation') != capture.get('generation')):
                    raise PerceptionError('table_reference_capture_mismatch')
            excluded_by_camera = {}
            for camera, view in measurement['views'].items():
                excluded = np.zeros(view['depth_m'].shape, bool)
                for detection in detections if not historical else ():
                    if detection['camera_id'] == camera:
                        excluded |= detection['mask']
                if historical:
                    # These boxes only discard foreground from the historical table fit;
                    # they never generate object masks, identity or occupied-map geometry.
                    rows, columns = np.indices(excluded.shape)
                    height, width = excluded.shape
                    for entity in entities.values():
                        if entity.get('kind') == 'table' or entity.get('reference_camera_id') != camera:
                            continue
                        box = entity.get('bbox')
                        if box is not None:
                            excluded |= ((columns >= box[0] * width) & (columns < box[2] * width)
                                         & (rows >= box[1] * height) & (rows < box[3] * height))
                excluded_by_camera[camera] = excluded

            def without_robot(masks):
                ownership = self.table_robot_surface_classifier(measurement, masks)
                if ownership.get('status') == 'not_applied' and self.robot_surface_classifier is None:
                    return masks
                if ownership.get('status') != 'measured':
                    raise PerceptionError('demo_table_robot_feedback_required')
                return {key: mask & ~ownership.get('masks', {}).get(key, np.zeros(mask.shape, bool))
                        for key, mask in masks.items()}

            def supplemental_points(seed, normal, offset, seed_camera):
                # fuse_table_plane discards these points before constructing any
                # neighbourhood. Classify only the exact same input region.
                lower, upper = seed[:, :2].min(0) - .02, seed[:, :2].max(0) + .02
                masks = {}
                for key, view in measurement['views'].items():
                    if key == seed_camera:
                        continue
                    mask = (~excluded_by_camera[key] & view['valid_depth']
                            & np.isfinite(view['depth_m']) & (view['depth_m'] > 0))
                    pixels = np.column_stack(np.nonzero(mask))
                    points = deproject_view(view, mask)
                    relevant = (np.all((points[:, :2] >= lower) & (points[:, :2] <= upper), axis=1)
                                & (np.abs(points @ normal + offset) <= .03))
                    masks[key] = np.zeros(mask.shape, bool)
                    masks[key][tuple(pixels[relevant].T)] = True
                clean = without_robot(masks)
                return {key: deproject_view(measurement['views'][key], mask) for key, mask in clean.items()}

            tables = {}
            for identifier, entity in entities.items():
                if entity.get('kind') != 'table':
                    continue
                camera = entity['reference_camera_id']
                view = measurement['views'][camera]
                rows, columns = np.indices(view['depth_m'].shape)
                height, width = rows.shape
                box = entity['bbox']
                seed_mask = ((columns >= box[0] * width) & (columns < box[2] * width)
                    & (rows >= box[1] * height) & (rows < box[3] * height)
                    & ~excluded_by_camera[camera] & view['valid_depth'])
                clean_seed = without_robot({camera: seed_mask})[camera]
                tables[identifier] = fit_objects_table(view, box, ~clean_seed,
                    supplemental_points=lambda seed, normal, offset, seed_camera=camera:
                        supplemental_points(seed, normal, offset, seed_camera))
                tables[identifier].update(measurement_observation_id=measurement['observation_id'],
                    foreground_exclusion_source='reference_rgb_boxes' if historical else 'current_sam3_masks')
            primary = next(iter(tables), None)
            if primary not in tables:
                raise PerceptionError('support_table_entity_missing')
            self.demo_tables = {'source_observation_id': reference_id, 'primary_table_id': primary, 'tables': tables}
        primary = self.demo_tables['tables'][self.demo_tables['primary_table_id']]
        model = primary['model']
        return {'position': [*model['pose'][:2], primary['surface_height_m']],
                'size': [*model['dims'][:2], .006], 'surface_height_m': primary['surface_height_m'],
                'uncertainty_m': .006, 'estimated': True, 'source': primary['source']}

    def observe(self, capture, *, instruction='', checkpoint=None, source_hint=None):
        hint = deepcopy(source_hint or {})
        hint['camera_ids'] = ['1', '2', '3']
        observation = super().observe(capture, instruction=instruction, checkpoint=checkpoint, source_hint=hint)
        depth_started = time.monotonic()
        progress = hint.get('observation_progress')
        if progress:
            progress('observation_stage', '任务实例深度整理',
                     {'completed_timings_s': dict(observation['perception_timings_s'])})
        reference = observation['visual_reference']
        observation.update(object_memory_snapshot=deepcopy((reference or {}).get('object_memory_snapshot', {})),
                           demo_mapping=True, demo_tables=deepcopy(self.demo_tables))
        entities = {**current_memory_entities(reference, observation['scene_entities']),
                    **observation['visual_identity'].get('registered_entities', {})}
        observation['scene_entities'] = entities
        if capture['observation_id'] == reference.get('observation_id'):
            observation['visual_reference']['initial_instance_boxes'] = {
                camera: {identifier: mask_box(mask) for identifier, mask in view['instance_masks'].items()
                         if np.any(mask)} for camera, view in observation['views'].items()}
        for camera, view in observation['views'].items():
            view['coverage_depth_m'] = _readonly(np.array(view['depth_m'], copy=True))
            view['coverage_valid_depth'] = _readonly(np.array(view['valid_depth'], copy=True))
        observation['depth_policy'] = 'workspace_depth_with_task_instance_masks'
        observation['perception_timings_s']['task_depth_preparation'] = time.monotonic() - depth_started
        if progress:
            progress('observation_stage', '感知计算完成，更新观测记录',
                     {'completed_timings_s': dict(observation['perception_timings_s'])})
        return observation
