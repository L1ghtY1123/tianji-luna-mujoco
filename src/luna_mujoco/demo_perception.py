"""Demo sensing: raw three-camera bootstrap and a permanent RGB scene inventory."""
import hashlib
import json
from copy import deepcopy

import numpy as np

from .cameras import deproject_view
from .demo_container_tracking import (
    container_correspondence,
    merge_container_correspondence,
    reference_with_container_tracks,
    retain_container_placement_regions,
)
from .demo_geometry import fit_inventory_table, supported_surface_points
from .demo_identity_presentation import improve_identity_presentation, release_verification_context
from .inventory_reference import current_inventory_entities, visual_inventory_state
from .perception import (
    Perception,
    PerceptionError,
    _readonly,
)
from .visual_api import VisualInventoryError
from .visual_inventory import VisualInventory, mask_box


def _same_capture_context(capture, reference):
    if (not reference or not capture.get('observation_id')
            or reference.get('observation_id') != capture['observation_id']
            or reference.get('generation') != capture.get('generation')):
        return {}
    verified = {}
    for camera, view in capture['views'].items():
        initial = reference.get('views', {}).get(camera, {}).get('rgb')
        current = np.asarray(view['rgb'])
        if initial is not None and current.dtype == initial.dtype and np.array_equal(current, initial):
            verified[camera] = hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest()
    return {'same_capture_reference': {
        'initial_reference_observation_id': reference['observation_id'],
        'current_observation_id': capture['observation_id'], 'generation': capture.get('generation'),
        'verified_camera_ids': sorted(verified), 'raw_rgb_sha256_by_camera': verified}}



class DemoVisualInventory(VisualInventory):
    """Associate current subjects while preserving unresolved identity evidence."""

    def resolve(self, capture, detections, *, reference=None, entities=None, reference_capture=None,
                association_scope=None, **kwargs):
        # Runtime inventory is authoritative; original RGB remains historical evidence.
        entities = current_inventory_entities(reference, entities)
        entities = {key: value for key, value in entities.items()
                    if value.get('source') != 'configured_demo_region'}
        identity_context = _same_capture_context(capture, reference)
        identity_context['inventory_state'] = visual_inventory_state(reference)
        if kwargs.get('progress'):
            kwargs['progress'] = {key: value for key, value in kwargs['progress'].items() if key in entities}
        release_context = release_verification_context(kwargs.get('progress'), kwargs.get('purpose'))
        if release_context is not None:
            identity_context['release_verification_context'] = release_context
        locations = []
        for index, detection in enumerate(detections):
            points = np.asarray(detection.get('points', []))
            if points.ndim != 2 or points.shape[1] != 3:
                continue
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                locations.append({'candidate': 'candidate-' + str(index), 'camera': detection['camera_id'],
                    'measured_centroid_world_m': points.mean(axis=0).tolist(),
                    'measured_bounds_world_m': [points.min(axis=0).tolist(), points.max(axis=0).tolist()]})
        identity_context.update(candidate_geometric_location_hints=locations,
            geometric_location_rule=(
                'These are calibrated measured RGBD surface locations, not identities. '
                'Robot world +Y is physical robot LEFT, -Y is RIGHT, and +X is forward. '
                'Match each declared entity across all observed regions using RGB and geometry. '
                'Color, task role, camera number or location alone does not establish identity.'))
        all_detections = detections
        containers = (container_correspondence(capture, detections, reference, entities)
                      if kwargs.get('purpose') in {'associate_inventory', 'verify_transfer'} else None)
        from .observation_identity import covered_container_candidates

        covered = (covered_container_candidates(capture, detections, association_scope, containers)
                   if kwargs.get('purpose') == 'associate_inventory' else [])
        covered_indices = {row['candidate_index'] for row in covered}
        if kwargs.get('purpose') == 'associate_inventory':
            # Current contents already covered by a published occupied envelope
            # can retain their dated identity until an operation needs them.
            kwargs['progress_question_ids'] = ()
        if containers and containers['candidate_mapping']:
            indices = kwargs.get('candidate_indices', list(range(len(detections))))
            retained = [(index, detection) for index, detection in zip(indices, detections, strict=True)
                        if str(index) not in containers['candidate_mapping'] and index not in covered_indices]
            kwargs['candidate_indices'] = [index for index, _ in retained]
            detections = [detection for _, detection in retained]
            identity_context['container_correspondence'] = containers
            # Omitting a resolved question must not erase its CURRENT visual
            # anchor when that permanent ID remains an alternative elsewhere.
            geometry = {row['candidate']: row for row in locations}
            identity_context['current_confirmed_container_anchors'] = [{
                'entity_id': identifier, 'candidate': 'candidate-' + index,
                'camera': all_detections[int(index)]['camera_id'],
                'box': mask_box(all_detections[int(index)]['silhouette_mask']),
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'source': containers['source'], 'binding_hash': containers['binding_hash'],
                'initial_request_hash': containers['initial_request_hash'],
                'correspondence': deepcopy(containers['matches'][index]),
                'measured_surface_source': 'current_registered_rgbd_candidate_surface',
                **geometry.get('candidate-' + index, {})}
                for index, identifier in containers['candidate_mapping'].items()]
        from .identity_geometry import (
            cad_identity_exclusions,
            projected_reference_surfaces,
            projected_surface_pairs,
        )

        identity_context['projected_candidate_surfaces'] = projected_surface_pairs(capture, all_detections)
        identity_context['projected_reference_surfaces'] = projected_reference_surfaces(
            capture, all_detections, (reference or {}).get('identity_references', {}))

        exclusions = cad_identity_exclusions(
            capture, all_detections, identity_context.get('current_confirmed_container_anchors', []),
            capture.get('confirmed_cad_models', {}))
        if exclusions:
            identity_context['identity_geometry_exclusions'] = exclusions
        kwargs['identity_context'] = identity_context
        result = super().resolve(capture, detections, reference=reference, entities=entities,
                                 association_scope=association_scope, **kwargs)
        result['association_scope'] = {key: list((association_scope or {}).get(key, ()))
                                       for key in ('focus_entity_ids', 'deferred_entity_ids')}
        result['deferred_candidates'] = covered
        if containers and containers['candidate_mapping']:
            result = merge_container_correspondence(self, all_detections, entities, containers, result)
        return result

    def request(self, payload, capture, purpose, parse, *, identity_subjects=None):
        payload = improve_identity_presentation(payload)
        content = payload['input'][0]['content']
        context = json.loads(content[0]['text'])
        def parse_associations(body, evidence):
            return parse(body, {**evidence, 'generation': capture['generation'],
                **({'container_correspondence': context['container_correspondence']}
                   if context.get('container_correspondence') else {})})
        from .identity_correspondence import request_identity_correspondences

        return request_identity_correspondences(self, payload, capture, purpose, parse_associations,
                                                subjects=identity_subjects)

    def _groups(self, detections, names, entities, judgments, evidence, *, purpose=None, reference=None,
                progress=None):
        accepted = {judgments['identity_' + str(index)] for index in range(len(detections))
                    if judgments['identity_' + str(index)] in entities}
        decisions_accepted = set(accepted)
        container_tracking = evidence.get('container_correspondence', {})
        measured_containers = set(container_tracking.get('candidate_mapping', {}).values())
        accepted |= measured_containers
        required = {identifier for identifier, entity in entities.items()
                    if entity.get('kind') not in {'table', 'marker'}
                    and entity.get('status') not in {'removed', 'consumed'}}
        pending_verification = {identifier for identifier, record in (progress or {}).items()
                                if record.get('release_feedback_id') and record.get('status') != 'completed'}
        if purpose == 'verify_transfer' and pending_verification:
            required = pending_verification
        coverage = {'source': evidence['source'], 'observation_id': evidence['observation_id'],
            'request_hash': evidence['request_hash'], 'purpose': purpose,
            'minimum_confidence': self.minimum_confidence,
            'required_entity_ids': sorted(required), 'accepted_entity_ids': sorted(accepted),
            'complete': required <= accepted}
        if measured_containers:
            coverage.update(source='mixed_decisions_measured_container',
                decisions_accepted_entity_ids=sorted(decisions_accepted),
                measured_container_entity_ids=sorted(measured_containers),
                container_correspondence_binding_hash=container_tracking['binding_hash'])
        unassociated = self._unresolved_identity_candidates(detections, names, judgments, evidence)
        retained = [index for index in range(len(detections))
                    if judgments['identity_' + str(index)] != 'unknown']
        if purpose == 'verify_transfer' and pending_verification - accepted:
            raise VisualInventoryError('visual_verification_target_unconfirmed', evidence={
                **evidence, 'missing_target_ids': sorted(pending_verification - accepted)})
        if not unassociated:
            result = super()._groups(detections, names, entities, judgments, evidence)
            result['identity_coverage'] = coverage
            return result
        # Only trusted candidates enter semantic geometry. Unassociated mask pixels
        # remain in the audit evidence but do not enter the inventory-only map.
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
            reason = ('model_refusal' if raw.get('type') == 'refusal' else
                      'proposed_identity_rejected' if raw.get('type') == 'predicate'
                      and raw['probability'] <= round(1 - self.minimum_confidence, 12) else
                      'proposed_identity_unconfirmed' if raw.get('type') == 'predicate' else
                      'model_selected_unknown' if raw.get('choice') == 'unknown' else
                      'choice_confidence_below_threshold')
            unresolved.append({'candidate_id': names[index], 'camera_id': detection['camera_id'],
                'source': self.source, 'observation_id': evidence['observation_id'],
                'generation': evidence.get('generation'), 'request_hash': evidence['request_hash'],
                'status': 'unknown', 'reason': reason, 'minimum_confidence': self.minimum_confidence,
                'answer': deepcopy(answer), 'original_decisions_answers': [deepcopy(raw)]})
        return unresolved


class DemoPerception(Perception):

    def __init__(self, detector=None, *, robot_surface_classifier=None, visual_inventory=None, cad_resolver=None):
        self.demo_tables = None
        super().__init__(detector, robot_surface_classifier=robot_surface_classifier,
            visual_inventory=visual_inventory if visual_inventory is not None else DemoVisualInventory.from_environment(),
            cad_resolver=cad_resolver)

    @classmethod
    def from_environment(cls):
        import os

        from .cad_perception import CadGeometryResolver

        return cls(cad_resolver=CadGeometryResolver.from_environment()
                   if os.getenv('LUNA_GEOMETRY_MODE') == 'cad_obb' else None)

    def verify_release(self, capture, binding, reference, instruction=""):
        from .demo_transfer_verification import verify_release_result
        # The operator requests qualitative RGB verification. Motion feedback
        # still validates the release endpoint; no post-release metric pose is
        # claimed and no segmentation/registration/map update is performed here.
        return verify_release_result(self.visual_inventory, capture, binding, reference, instruction)

    def bootstrap(self, capture):
        raise PerceptionError('demo_initial_observation_required')

    def _observe_entities(self, capture, instruction, checkpoint, hint, entities, reference, discovery_evidence):
        current_entities = current_inventory_entities(reference, entities)
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
                generic = detection['prompt'] == 'movable object' or previous['prompt'] == 'movable object'
                if not generic and any(previous.get(key) != detection.get(key) for key in ('kind', 'requested_color')):
                    continue
                other = np.asarray(previous['silhouette_mask'], bool)
                if other.shape == mask.shape and np.count_nonzero(other & mask) / max(
                        1, np.count_nonzero(other | mask)) >= .95:
                    duplicate = previous
                    break
            if duplicate is None:
                unique.append(detection | {'prompt_proposals': [detection['prompt']]})
            else:
                duplicate['prompt_proposals'].append(detection['prompt'])
        return unique

    def _prepare_identity_detections(self, detections, views, capture, reference):
        detections, robot_ownership = super()._prepare_identity_detections(detections, views, capture, reference)
        if not (reference or {}).get('inventory_state', {}).get('transformations'):
            return detections, robot_ownership
        cleaned = []
        for entry in detections:
            points = np.asarray(entry['points'])
            keep = supported_surface_points(points)
            if np.count_nonzero(keep) < 30:
                continue
            mask = np.zeros_like(entry['mask'])
            pixels = np.asarray(entry['pixels_rc'])[keep]
            mask[pixels[:, 0], pixels[:, 1]] = True
            candidate = {**entry, 'mask': mask, 'silhouette_mask': mask.copy(),
                'points': points[keep], 'pixels_rc': pixels, 'pixels': np.asarray(entry['pixels'])[keep],
                'identity_proposal_filter': {'source': 'measured_robot_and_surface_support',
                    'raw_point_count': entry['identity_proposal_filter']['raw_point_count'],
                    'retained_point_count': int(keep.sum()),
                    'identity_assignment': False}}
            # Near-identical masks in the same camera are alternative SAM
            # descriptions of the same pixels, not new physical identities.
            duplicate = None
            for previous in cleaned:
                if (previous['camera_id'], previous['kind'], previous.get('requested_color')) != (
                        candidate['camera_id'], candidate['kind'], candidate.get('requested_color')):
                    continue
                intersection = np.count_nonzero(previous['mask'] & mask)
                union = np.count_nonzero(previous['mask'] | mask)
                if intersection / max(union, 1) >= .95:
                    duplicate = previous
                    break
            if duplicate is not None:
                duplicate.setdefault('prompt_proposals', [duplicate['prompt']]).extend(
                    candidate.get('prompt_proposals', [candidate['prompt']]))
            else:
                cleaned.append(candidate)
        return cleaned, robot_ownership

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
                tables[identifier] = fit_inventory_table(view, box, ~clean_seed,
                    supplemental_points=lambda seed, normal, offset, seed_camera=camera:
                        supplemental_points(seed, normal, offset, seed_camera))
                tables[identifier].update(measurement_observation_id=measurement['observation_id'],
                    foreground_exclusion_source='reference_rgb_boxes' if historical else 'current_sam3_masks')
            primary = next(iter(tables), None)
            if primary not in tables:
                raise PerceptionError('demo_destination_table_unobservable')
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
        reference = observation['visual_reference']
        observation.update(inventory_state=deepcopy((reference or {}).get('inventory_state', {})),
                           demo_mapping=True, demo_tables=deepcopy(self.demo_tables))
        entities = {**current_inventory_entities(reference, observation['scene_entities']),
                    **observation['visual_inventory'].get('registered_entities', {})}
        observation['scene_entities'] = entities
        observation['visual_reference'] = reference_with_container_tracks(observation, capture)
        retain_container_placement_regions(observation)
        if capture['observation_id'] == reference.get('observation_id'):
            observation['visual_reference']['initial_instance_boxes'] = {
                camera: {identifier: mask_box(mask) for identifier, mask in view['instance_masks'].items()
                         if np.any(mask)} for camera, view in observation['views'].items()}
        for view in observation['views'].values():
            # Sensor rays support coverage checks separately. Occupied-map
            # integration below still receives only confirmed inventory masks.
            view['coverage_depth_m'] = _readonly(np.array(view['depth_m'], copy=True))
            view['coverage_valid_depth'] = _readonly(np.array(view['valid_depth'], copy=True))
            masks = list(view['instance_masks'].values())
            retained = np.logical_or.reduce(masks) if masks else np.zeros(view['depth_m'].shape, bool)
            valid = retained & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
            view['depth_m'] = _readonly(np.where(valid, view['depth_m'], 0))
            view['valid_depth'] = _readonly(valid)
        observation['depth_policy'] = 'inventory_only_no_background_or_tabletop_voxels'
        return observation
