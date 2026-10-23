"""Demo sensing: raw three-camera bootstrap and a permanent RGB scene inventory."""
import hashlib
import json
from copy import deepcopy

import numpy as np

from .cameras import deproject_view
from .demo_completed_tracking import (
    completed_candidate_scopes,
    completed_identity_tracks,
    merge_completed_correspondence,
    verify_completed_correspondence,
)
from .demo_container_tracking import (
    container_correspondence,
    merge_container_correspondence,
    reference_with_container_tracks,
)
from .demo_geometry import complete_source_height, fit_inventory_table, supported_surface_points
from .demo_identity_presentation import improve_identity_presentation, release_verification_context
from .demo_lifecycle_inventory import resolve_lifecycle_inventory
from .demo_lifecycle_perception import lifecycle_entities, remaining_wrapped_source_ids
from .demo_lifecycle_tracking import lifecycle_candidate_scopes
from .perception import (
    Perception,
    PerceptionError,
    _readonly,
    _remove_own_robot_surface,
)
from .visual_api import VisualInventoryError
from .visual_discovery import DemoVisualDiscovery
from .visual_inventory import VisualInventory, choice, mask_box


def _box_iou(first, second):
    intersection = max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(
        0., min(first[3], second[3]) - max(first[1], second[1]))
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.


def _same_capture_context(capture, detections, reference):
    if (not reference or not capture.get('observation_id')
            or reference.get('observation_id') != capture['observation_id']):
        return None
    verified = {}
    for camera, view in capture['views'].items():
        initial = reference.get('views', {}).get(camera, {})
        if 'rgb' not in initial:
            continue
        current_rgb, initial_rgb = np.asarray(view['rgb']), np.asarray(initial['rgb'])
        if current_rgb.dtype == initial_rgb.dtype and np.array_equal(current_rgb, initial_rgb):
            verified[camera] = hashlib.sha256(np.ascontiguousarray(current_rgb).tobytes()).hexdigest()
    if not verified:
        return None
    overlaps = []
    for index, detection in enumerate(detections):
        camera = detection['camera_id']
        if camera not in verified:
            continue
        box = mask_box(detection['silhouette_mask'])
        ranked = [{'reference_entity_id': identifier, 'reference_box': entity['bbox'],
                   'box_iou': _box_iou(box, entity['bbox'])}
                  for identifier, entity in reference.get('entities', {}).items()
                  if entity.get('reference_camera_id') == camera and entity.get('bbox')]
        if not ranked:
            continue
        ranked.sort(key=lambda row: row['box_iou'], reverse=True)
        overlaps.append({'camera': camera, 'candidate': 'candidate-' + str(index), 'current_box': box,
                         'same_camera_reference_box_overlaps': ranked})
    return {
        'same_capture_reference': {
            'initial_reference_observation_id': reference['observation_id'],
            'current_observation_id': capture['observation_id'], 'generation': capture.get('generation'),
            'verified_camera_ids': sorted(verified), 'raw_rgb_sha256_by_camera': verified,
            'hash_encoding': 'C-contiguous RGB array bytes; shape, dtype and every pixel compared before hashing',
            'meaning': 'For the verified cameras only, initial and current RGB are the same frozen capture '
                       'with identical pixels. Only annotations differ; no intervening object exchange occurred.'},
        'same_capture_box_correspondence_proposals': overlaps,
        'box_correspondence_rule': 'These are same-camera 2D rectangle overlaps, not established identity or mask '
            'ownership. Check each proposed subject in RGB against its permanent reference box on identical pixels. '
            'A box can contain several subjects, especially containers. Retain reject/unknown for unrelated '
            'or ambiguous subjects.'}


def _demo_progress_instructions(instructions):
    return instructions.replace(
        'Slot XY specifies the TCP anchor, not object-centre or footprint containment.',
        'Slot XY specifies the calibrated finger front. Local geometry checks '
        'front XY within the region; RGB does not certify millimetre pose accuracy.').replace(
        'Check visible support/destination consistency; do not infer millimetre TCP accuracy.',
        'Region outlines are support-plane projections onto the tabletop, not a 3D enclosure. '
        'Assess the object base/contact, not its entire silhouette or SAM3 box. Elevated object pixels '
        'may extend outside the same 2D polygon while its supporting base is inside.').replace(
        'at_destination requires visible support on that region or inside that container, not held.',
        'at_destination requires visible base/support on the assigned region or inside the destination '
        'container, and no visible holding by the gripper. Use the union of current views to assess '
        'support and separation from the gripper; action history is not proof.')


class DemoVisualInventory(VisualInventory):
    """Decisions adds an explicit visible-unlisted-object check to fixed identity."""

    def resolve(self, capture, detections, *, reference=None, entities=None, **kwargs):
        # A configured destination is intention, not a physical marker to detect.
        entities = {key: value for key, value in (entities or {}).items()
                    if value.get('source') != 'configured_demo_region'}
        task = (reference or {}).get('demo_task') or {}
        if task.get('workflow') == 'transfer' and task.get('item_lifecycle'):
            raise VisualInventoryError('unpack_workflow_lifecycle_conflict')
        if task.get('workflow') == 'unpack_transfer':
            lifecycle = task.get('item_lifecycle')
            sources = task.get('source_object_ids', [])
            if (not isinstance(lifecycle, dict) or not lifecycle
                    or not set(sources) <= set(lifecycle)):
                raise VisualInventoryError('unpack_lifecycle_required')
        entities = lifecycle_entities(entities, task)
        source_bin = task.get('source_bin_id')
        if source_bin is None:
            bins = [key for key, entity in entities.items()
                    if entity.get('kind') == 'bin' and entity.get('color', '').lower() == 'blue']
            source_bin = bins[0] if len(bins) == 1 else None
        source_ids = task.get('source_object_ids')
        if source_ids is None:
            source_ids = [key for key, entity in entities.items()
                          if entity.get('color', '').lower() == 'pink'
                          and entity.get('kind') not in {'table', 'bin', 'marker'}
                          and (entity.get('initial_parent_id') == source_bin if source_bin is not None
                               else entity.get('visual_membership') == 'required')]
        kwargs['source_inventory'] = {'source_bin_id': source_bin, 'source_object_ids': source_ids}
        identity_context = _same_capture_context(capture, detections, reference) or {}
        if task.get('item_lifecycle'):
            # Execution receipts remain archived locally. In particular mock
            # simulator truth and commanded poses must never become visual input.
            identity_context['item_lifecycle'] = {
                identifier: {key: record[key] for key in ('phase', 'appearance', 'appearance_revision',
                                                          'to_table_arm', 'to_right_bin_arm')}
                for identifier, record in task['item_lifecycle'].items()}
            identity_context['right_bin_id'] = task.get('right_bin_id')
            # Dedicated lifecycle verifiers own transition judgments. Mapping
            # must not ask old table-slot progress questions a second time.
            kwargs['progress_question_ids'] = set()
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
            geometric_location_rule='These are calibrated measured RGBD surface locations, not identities. '
            'Robot world +Y is physical robot LEFT and -Y is robot RIGHT; +X is in front of the robot. '
            'Use them to distinguish physically separate, similarly colored boxes across views. '
            'The source is the robot-left box containing the pink objects; the robot-right box is '
            'initially empty but may now contain completed black inner workpieces, and '
            'has its own fixed ID. A camera number alone does not identify a subject. Do not merge '
            'opposite-side boxes merely because they have the same color; check RGB and geometry together.')
        tracks, scopes, tracking = {}, {}, None
        all_entities, all_detections = entities, detections
        containers = (container_correspondence(capture, detections, reference, entities)
                      if kwargs.get('purpose') == 'associate_inventory' else None)
        if containers and task.get('item_lifecycle'):
            lifecycle_scopes = lifecycle_candidate_scopes(capture, detections, reference, entities,
                kwargs.get('table_scope'), containers, purpose=kwargs.get('purpose'))
            if lifecycle_scopes is not None:
                return resolve_lifecycle_inventory(self, capture, detections, entities, lifecycle_scopes,
                    containers, instruction=kwargs.get('instruction', ''), reference=reference)
        if kwargs.get('purpose') == 'associate_inventory' and not task.get('item_lifecycle'):
            tracks = completed_identity_tracks(capture, reference, kwargs.get('progress'), entities)
            scopes = completed_candidate_scopes(capture, detections, tracks)
            if tracks:
                tracking = verify_completed_correspondence(self, capture, detections, tracks, scopes)
                indices = [index for index in range(len(detections)) if index not in scopes]
                detections = [detections[index] for index in indices]
                entities = {key: value for key, value in entities.items() if key not in tracks}
                reference = {**reference, 'entities': {
                    key: value for key, value in reference.get('entities', {}).items() if key not in tracks}}
                kwargs['candidate_indices'] = indices
                kwargs['progress_question_ids'] = set(kwargs.get('progress', {})) - set(tracks)
                kwargs['placement_regions'] = ()
                identity_context['candidate_geometric_location_hints'] = [
                    row for row in locations if int(row['candidate'].removeprefix('candidate-')) in indices]
                identity_context['completed_correspondence_request_hash'] = tracking['request_hash']
                identity_context['fixed_reference_labels'] = {
                    identifier: f'R{index + 1}' for index, identifier in enumerate(sorted(all_entities))
                    if identifier in entities}
                identity_context['completed_source_correspondences'] = [{
                    'entity_id': identifier, 'display_id': all_entities[identifier].get('display_id'),
                    'description': all_entities[identifier].get('description', 'pink workpiece'),
                    'slot_id': track['binding']['slot_id'], 'observation_id': capture['observation_id'],
                    'generation': capture['generation'], 'request_hash': tracking['request_hash'],
                    'status': 'confirmed' if identifier in tracking['candidate_mapping'].values() else 'unknown',
                    'current_candidates': [
                        {'candidate': 'C' + index, 'camera': all_detections[int(index)]['camera_id'],
                         'box': mask_box(all_detections[int(index)]['silhouette_mask'])}
                        for index, owner in tracking['candidate_mapping'].items() if owner == identifier]}
                    for identifier, track in tracks.items()]
        if containers and containers['candidate_mapping']:
            indices = kwargs.get('candidate_indices', list(range(len(detections))))
            retained = [(index, detection) for index, detection in zip(indices, detections, strict=True)
                        if str(index) not in containers['candidate_mapping']]
            kwargs['candidate_indices'] = [index for index, _ in retained]
            detections = [detection for _, detection in retained]
            identity_context['container_correspondence'] = containers
        kwargs['identity_context'] = identity_context
        try:
            result = super().resolve(capture, detections, reference=reference, entities=entities, **kwargs)
        except VisualInventoryError as error:
            if tracks:
                error.evidence['completed_tracking'] = tracking
            raise
        if containers and containers['candidate_mapping']:
            result = merge_container_correspondence(self, all_detections, entities, containers, result)
        if tracks:
            result = merge_completed_correspondence(
                self, capture, all_detections, all_entities, tracking, result)
        return result

    def request(self, payload, capture, purpose, parse):
        payload = deepcopy(payload)
        content = payload['input'][0]['content']
        context = json.loads(content[0]['text'])
        context['rules'] = context['rules'].replace(
            'Only inventory entities on the designated working table or its containers are in scope, including ',
            'Inventory spans the large table and both bins, including ').replace(
            'no segmentation contours or depth measurements are supplied.',
            'No segmentation contours or raw depth images are supplied. Calibrated measured surface '
            'location summaries are supplied as geometric hints, never automatic identity.')
        context['rules'] += (
            ' Combine all three CURRENT views of the same scene. Camera numbers and C labels locate '
            'subjects; they do not restrict evidence to a single view. A clear view may resolve another '
            'partial view, but do not invent correspondence. INITIAL images are historical references, '
            'and annotated copies or detail images are not additional physical objects. '
            ' The configured unmarked destination demo-destination is a region on the large table. '
            'Its overlay describes intended placement, not a physical marker or proof of release. '
            'Demo slot XY refers to the calibrated gripper finger front, not the held cloud centroid. '
            'Check the union of all three full images for visible unlisted pink source task objects, '
            'even if SAM3 misses them. This visible-subject check does not establish hidden contents or source emptiness.')
        if context.get('item_lifecycle'):
            context['rules'] += (
                ' Original pink input IDs and display numbers are immutable. Verified unpacking creates '
                'a distinct black product entity ID; a process record links input and output. '
                'Do not visually equate black products with original pink identities or reuse the pink ID. '
                'Use INITIAL pink references only for still-wrapped input identity comparisons. '
                'Match black product candidates to the CURRENT black appearance and their recorded current role. '
                'Containers retain their INITIAL visual references: compare each current box with the '
                'same-camera initial reference and calibrated surface-location hints. Being empty or occupied '
                'does not change the container ID. Task roles and camera numbers do not assign identity; '
                'visually confirm the physical box and keep unknown if ambiguous. '
                ' The user declared that pink foam wraps a black inner workpiece. A validated unpacking '
                'phase creates a separately identified black output after current visual confirmation. '
                'Current black items on the main working table are active work; completed black items '
                'belong inside the fixed right bin. Do not confuse these support regions. '
                'The initial source location and pink color must not override current phase evidence. '
                'Action history is contextual, never permission to accept a visually uncertain candidate. '
                'Completed right-bin contents are represented as one collection with one or more visible '
                'black workpieces or parts. Do not assign visual identities to its individual members after random drops.')
        if context.get('completed_source_correspondences'):
            context['rules'] += (
                ' Completed source objects have a separate CURRENT correspondence check in this same capture. '
                'Its confirmed rows locate those known objects by current C label, camera, box and slot; '
                'unknown rows are NOT confirmed. They are omitted from this request\'s identity choices, '
                'visibility and progress questions. Reject a proposal whose subject is an independently confirmed '
                'completed object; do not assign it to an unmoved source ID. For the global unlisted check, '
                'confirmed completed objects are listed known objects, not extra source objects. '
                'Initial RGB images are historical and may still show those objects at their old source positions. '
                'Use initial references only for the remaining IDs being matched in this request.')
        if context.get('container_correspondence'):
            context['rules'] += (
                ' container_correspondence records independent current registered-depth and mask matching '
                'to initially Decisions-confirmed, unmanipulated containers. It is measured tracking evidence, '
                'not a model answer. Only those matched candidate identity questions are omitted. All container '
                'IDs remain choices for other candidates; use current RGB for those and for visibility. '
                'This tracking does not identify pink or black workpieces or prove there are no unlisted objects.')
        content[0]['text'] = json.dumps(context, ensure_ascii=False)
        same_capture = context.get('same_capture_reference')
        grounded_questions = {'identity_' + item['candidate'].removeprefix('candidate-'): item['camera']
                              for item in context.get('same_capture_box_correspondence_proposals', [])}
        if same_capture:
            for part in content:
                if part.get('type') == 'input_text' and part['text'].startswith('INITIAL RGB REFERENCE.'):
                    part['text'] = (
                        'INITIAL RGB REFERENCE. For cameras ' + json.dumps(same_capture['verified_camera_ids'])
                        + ', initial and current are the SAME frozen capture with identical pixels; only annotations '
                        'differ. Permanent IDs locate the same physical subjects in those views. '
                        'Other cameras or later captures may differ; positions may change there.')
        for question in payload['questions']:
            if context.get('item_lifecycle'):
                if question['name'].startswith('identity_'):
                    question['instructions'] += (
                        ' Black products have independent current IDs; never match them to a pink INITIAL ID. '
                        'Still-wrapped pink inputs keep their original IDs. Containers still use their INITIAL references '
                        'and CURRENT RGB; contents do not change container identity.')
                identifier = question['name'].removeprefix('visibility_')
                if (question['name'].startswith('visibility_')
                        and context['item_lifecycle'].get(identifier, {}).get('appearance') == 'unpacked'):
                    question['instructions'] = question['instructions'].replace(
                        'Is initial entity ', 'Is the current lifecycle appearance of entity ')
            if question['name'] in grounded_questions:
                question['instructions'] += (
                    f" In camera {grounded_questions[question['name']]}, initial and current RGB are identical frozen images. "
                    'Use same-camera reference boxes and box-overlap proposals to locate the same visible subject; '
                    'do not infer an intervening exchange between these identical frozen images. '
                    'Overlap is a locating hint only, never automatic identity; check RGB and retain unknown when ambiguous.')
            if question['name'].startswith('progress_'):
                question['instructions'] = _demo_progress_instructions(question['instructions'])
        source = context['source_inventory']
        payload['questions'].append(choice('demo_unlisted_pink',
            f"Source bin ID: {json.dumps(source['source_bin_id'])}; permanent source-object IDs: "
            f"{json.dumps(source['source_object_ids'])}. "
            'Across the union of CURRENT RGB images, are any VISIBLE pink movable source task objects '
            'unmatched to these permanent source IDs? Ignore robot parts, reflections and annotations. '
            'none means every visible pink task subject matches a permanent source ID; it does not claim '
            'the source bin is empty or hidden contents are absent. extra means a clearly visible unmatched '
            'pink source task object. unknown means a visible subject has ambiguous identity or current RGB '
            'is insufficient to assess visible task subjects. Occluded interior alone does not mean extra or unknown. '
            'A known object already moved to its destination is not extra. Missing SAM3 candidates do not prove absence.',
            ['none', 'extra', 'unknown']))
        if context.get('item_lifecycle'):
            payload['questions'][-1]['instructions'] += (
                ' For this unpacking task, also check visible unmatched black inner workpieces. '
                'Black products listed in fixed_entities have their own independent IDs and are not extra. '
                'A listed completed collection may contain multiple visible black pieces without individual visual IDs. '
                'Current table work and completed right-bin contents are distinct; unknown if correspondence '
                'is ambiguous. Initial reference pink objects are historical appearances, not extra current objects.')
        if len(payload['questions']) > 200:
            raise VisualInventoryError('visual_inventory_input_limit')

        def with_demo_check(body, evidence):
            answers = body.get('answers')
            if not isinstance(answers, list) or len(answers) != len(payload['questions']):
                raise VisualInventoryError('visual_decisions_invalid_response')
            answer = answers[-1]
            if not isinstance(answer, dict) or answer.get('name') != 'demo_unlisted_pink':
                raise VisualInventoryError('visual_decisions_invalid_response')
            confidence = answer.get('confidence', 0.)
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise VisualInventoryError('visual_decisions_invalid_response')
            status = answer.get('choice') if answer.get('type') == 'choice' and confidence >= self.minimum_confidence else 'unknown'
            if status not in {'none', 'extra', 'unknown'}:
                raise VisualInventoryError('visual_decisions_invalid_response')
            check = {'status': status, 'source': self.source,
                     'observation_id': capture['observation_id'], 'confidence': confidence,
                     'request_hash': evidence['request_hash']}
            if status == 'extra':
                raise VisualInventoryError('demo_unlisted_pink_object', evidence={
                    **evidence, 'answers': answers, 'demo_inventory_check': check})
            result = parse({**body, 'answers': answers[:-1]}, {
                **evidence, 'demo_inventory_check': check, 'generation': capture['generation'],
                **({'container_correspondence': context['container_correspondence']}
                   if context.get('container_correspondence') else {}),
                **({'completed_correspondence_request_hash': context['completed_correspondence_request_hash']}
                   if context.get('completed_correspondence_request_hash') else {})})
            result['demo_inventory_check'] = check
            result['evidence']['answers'] = answers
            return result
        return super().request(improve_identity_presentation(payload), capture, purpose, with_demo_check)

    def _groups(self, detections, names, entities, judgments, evidence, *, purpose=None, reference=None,
                progress=None):
        accepted = {judgments['identity_' + str(index)] for index in range(len(detections))
                    if judgments['identity_' + str(index)] in entities}
        decisions_accepted = set(accepted)
        container_tracking = evidence.get('container_correspondence', {})
        measured_containers = set(container_tracking.get('candidate_mapping', {}).values())
        accepted |= measured_containers
        required = {identifier for identifier, entity in entities.items()
                    if entity.get('kind') not in {'table', 'marker'}}
        pending_verification = {identifier for identifier, record in (progress or {}).items()
                                if record.get('release_feedback_id') and record.get('status') != 'completed'}
        if purpose == 'verify_transfer' and pending_verification:
            required = pending_verification
        accepted_left = {judgments['identity_' + str(index)] for index, detection in enumerate(detections)
                         if detection['camera_id'] == '2' and judgments['identity_' + str(index)] in entities}
        accepted_left |= {row['entity_id'] for row in container_tracking.get('matches', {}).values()
                          if row['camera_id'] == '2'}
        task = (reference or {}).get('demo_task', {})
        source_ids = set(task.get('source_object_ids', []))
        completed = {identifier for identifier, record in (progress or {}).items()
                     if record.get('status') == 'completed'}
        remaining_source_ids = remaining_wrapped_source_ids(task) - completed
        left_source_covered = remaining_source_ids <= accepted_left
        if any(row.get('phase') == 'unpacked_confirmed' for row in task.get('item_lifecycle', {}).values()):
            left_source_covered = remaining_source_ids <= accepted
        complete_union = bool(required) and required <= accepted
        unassociated, retained = [], []
        for index, detection in enumerate(detections):
            identity = judgments['identity_' + str(index)]
            confirmed_visible_inventory = evidence.get('demo_inventory_check', {}).get('status') == 'none'
            supplementary = (purpose == 'verify_transfer' or (
                purpose in {'initialize_inventory', 'associate_inventory'}
                and left_source_covered
                and (detection['camera_id'] != '2' or confirmed_visible_inventory)))
            if identity == 'unknown':
                answer = evidence['answers'][index]
                if not complete_union or not supplementary:
                    raise VisualInventoryError('visual_identity_unknown', evidence={
                        **evidence, 'candidate': names[index], 'camera_id': detection['camera_id'],
                        'choice': answer.get('choice'), 'confidence': answer.get('confidence'),
                        'minimum_confidence': self.minimum_confidence})
                unassociated.append({'candidate': names[index], 'camera_id': detection['camera_id'],
                    'status': 'unknown', 'choice': answer.get('choice'), 'confidence': answer.get('confidence'),
                    'minimum_confidence': self.minimum_confidence, 'geometry_policy': 'discard_from_inventory_map',
                    'reason': 'supplementary_identity_uncertain'})
            else:
                retained.append(index)
        if purpose == 'verify_transfer' and pending_verification - accepted:
            raise VisualInventoryError('visual_verification_target_unconfirmed', evidence={
                **evidence, 'missing_target_ids': sorted(pending_verification - accepted)})
        # These sets contain only identities already accepted by the shared
        # confidence parser. Supplementary unknown masks never enter this proof.
        coverage = {'source': self.source, 'observation_id': evidence['observation_id'],
            'request_hash': evidence['request_hash'], 'purpose': purpose,
            'minimum_confidence': self.minimum_confidence,
            'required_entity_ids': sorted(required), 'accepted_entity_ids': sorted(accepted),
            'accepted_left_entity_ids': sorted(accepted_left), 'source_object_ids': sorted(source_ids),
            'remaining_source_ids': sorted(remaining_source_ids)}
        if measured_containers:
            coverage.update(source='mixed_decisions_measured_container',
                decisions_accepted_entity_ids=sorted(decisions_accepted),
                measured_container_entity_ids=sorted(measured_containers),
                container_correspondence_binding_hash=container_tracking['binding_hash'])
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


class DemoPerception(Perception):
    visual_plan_source = 'responses_program'

    def __init__(self, detector=None, *, robot_surface_classifier=None, visual_inventory=None, visual_discovery=None):
        self.demo_tables = None
        super().__init__(detector, robot_surface_classifier=robot_surface_classifier,
            visual_inventory=visual_inventory if visual_inventory is not None else DemoVisualInventory.from_environment(),
            visual_discovery=visual_discovery if visual_discovery is not None else DemoVisualDiscovery.from_environment())

    @classmethod
    def from_environment(cls):
        return cls()

    def verify_release(self, capture, binding, reference, instruction=""):
        from .demo_transfer_verification import verify_release_result
        return verify_release_result(self.visual_inventory, capture, binding, reference, instruction)

    def bootstrap(self, capture):
        raise PerceptionError('demo_initial_observation_required')

    def _observe_entities(self, capture, instruction, checkpoint, hint, entities, reference, discovery_evidence):
        task = (reference or {}).get('demo_task', {})
        if any(record['phase'] == 'unpack_pending' for record in task.get('item_lifecycle', {}).values()):
            raise PerceptionError('unpacking_visual_confirmation_required_before_geometry')
        try:
            current_entities = lifecycle_entities(entities, task)
        except VisualInventoryError as error:
            raise PerceptionError(str(error), details=error.evidence) from error
        return super()._observe_entities(capture, instruction, checkpoint, hint, current_entities,
                                         reference, discovery_evidence)

    @staticmethod
    def _segmentation_queries(entities):
        queries = {}
        for entity in entities.values():
            query = entity.get('segmentation_query')
            if query is None:
                continue
            for prompt in entity.get('segmentation_prompts', [query['prompt']]):
                # Shape-neutral source prompts are shared even when Responses
                # describes separate sources with different shape nouns.
                queries[prompt] = {**query, 'prompt': prompt}
        return list(queries.values())

    def _record_segmentation_attempt(self, evidence, view, proposals):
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
                if any(previous.get(key) != detection.get(key)
                       for key in ('camera_id', 'kind', 'requested_color')):
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
        task = (reference or {}).get('demo_task', {})
        if not any(row.get('phase') == 'unpacked_confirmed' for row in task.get('item_lifecycle', {}).values()):
            return detections
        # VLA hands occupy much of the image. Remove measured self returns
        # before showing proposals to Decisions, not only after identity binding.
        groups, _ = _remove_own_robot_surface([[entry] for entry in detections], views, capture,
                                             self.robot_surface_classifier)
        from .cad_assets import workpiece_height_prior

        table = self.demo_tables['tables'][self.demo_tables['primary_table_id']]
        floor = table['surface_height_m']
        center, half = np.asarray(table['model']['pose'][:2]), np.asarray(table['model']['dims'][:2]) / 2
        cleaned = []
        for group in groups:
            entry = group[0]
            points = np.asarray(entry['points'])
            keep = supported_surface_points(points)
            completed = any(row.get('phase') == 'completed' for row in task.get('item_lifecycle', {}).values())
            if (entry.get('requested_color') == 'black' and entry['kind'] != 'bin'
                    and not completed):
                keep &= (np.all(np.abs(points[:, :2] - center) <= half + .006, axis=1)
                         & (points[:, 2] > floor + .006)
                         & (points[:, 2] <= floor + workpiece_height_prior('black_core') + .015))
            if np.count_nonzero(keep) < 30:
                continue
            mask = np.zeros_like(entry['mask'])
            pixels = np.asarray(entry['pixels_rc'])[keep]
            mask[pixels[:, 0], pixels[:, 1]] = True
            candidate = {**entry, 'mask': mask, 'silhouette_mask': mask.copy(),
                'points': points[keep], 'pixels_rc': pixels, 'pixels': np.asarray(entry['pixels'])[keep],
                'identity_proposal_filter': {'source': 'measured_robot_and_confirmed_table_support',
                    'raw_point_count': len(points), 'retained_point_count': int(keep.sum()),
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
        return cleaned

    def _filter_geometry_groups(self, groups, views, *, entities=None):
        for group in groups:
            for entry in group:
                points = np.asarray(entry['points'])
                keep = supported_surface_points(points)
                entity = (entities or {}).get(entry.get('entity_id'), {})
                if (entity.get('current_appearance') == 'unpacked'
                        and entity.get('lifecycle_phase') == 'unpacked_confirmed'):
                    # This phase has a fresh visual table confirmation. A broad
                    # SAM proposal must not import floor/wall returns into that
                    # object's measured cloud, even if its identity was accepted.
                    identifier = self.demo_tables['primary_table_id']
                    support = self.demo_tables['tables'][identifier]
                    model = support['model']
                    center = np.asarray(model['pose'][:2])
                    half = np.asarray(model['dims'][:2]) / 2
                    inside = (np.all(np.abs(points[:, :2] - center) <= half + .006, axis=1)
                              & (points[:, 2] >= support['surface_height_m'] - .006))
                    keep &= inside
                    entry['geometry_filter_evidence'] = {
                        'source': 'cached_measured_table_and_confirmed_lifecycle',
                        'support_entity_id': identifier, 'camera_id': entry['camera_id'],
                        'lifecycle_phase': entity['lifecycle_phase'],
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
            if reference_id != capture['observation_id']:
                raise PerceptionError('demo_initial_table_capture_required')
            masks = {camera: np.asarray(view['valid_depth'], dtype=bool)
                     for camera, view in capture['views'].items()}
            ownership = self.robot_surface_classifier(capture, masks)
            if ownership.get('status') != 'measured':
                raise PerceptionError('demo_table_robot_feedback_required')
            excluded_by_camera, table_points_by_camera = {}, {}
            for camera, view in capture['views'].items():
                excluded = np.asarray(ownership.get('masks', {}).get(camera,
                    np.zeros(view['depth_m'].shape, bool)), dtype=bool).copy()
                for detection in detections:
                    if detection['camera_id'] == camera:
                        excluded |= detection['mask']
                excluded_by_camera[camera] = excluded
                table_points_by_camera[camera] = deproject_view(view, ~excluded)
            tables = {}
            for identifier, entity in entities.items():
                if entity.get('kind') != 'table':
                    continue
                camera = entity['reference_camera_id']
                view = capture['views'][camera]
                tables[identifier] = fit_inventory_table(view, entity['bbox'], excluded_by_camera[camera],
                    supplemental_points={key: points for key, points in table_points_by_camera.items()
                                         if key != camera})
            primary = (reference or {}).get('demo_task', {}).get('destination_table_id')
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
        if not reference or 'demo_task' not in reference or 'demo_program' not in reference:
            raise PerceptionError('demo_discovery_task_and_program_required')
        observation.update(demo_task=deepcopy(reference['demo_task']), demo_program=deepcopy(reference['demo_program']),
                           demo_mapping=True, demo_tables=deepcopy(self.demo_tables))
        observation['demo_inventory_check'] = observation['visual_inventory'].get('demo_inventory_check', {
            'status': 'unknown', 'source': 'missing_decisions_check', 'observation_id': capture['observation_id']})
        source_ids = set(reference['demo_task']['source_object_ids']) | set(reference['demo_task'].get('product_entities', {}))
        by_id = {obj['id']: obj for obj in observation['objects']}
        source_bin_id = reference['demo_task']['source_bin_id']
        source_bin = by_id.get(source_bin_id, {})
        measured_floor = source_bin.get('placement_reference', {}).get('floor_z_m')
        for instance in observation['objects']:
            if instance['id'] not in source_ids:
                continue
            instance['measured_color_class'] = instance['color']
            lifecycle = reference['demo_task'].get('item_lifecycle', {}).get(instance['id'])
            unpacked = lifecycle is not None and lifecycle['appearance'] == 'unpacked'
            from .cad_assets import cad_asset, workpiece_height_prior

            asset_id = 'black_core' if unpacked else 'pink_wrapped'
            height = workpiece_height_prior(asset_id)
            instance['dimension_prior'] = {'source': 'user_cad_asset', 'asset_id': asset_id,
                                           'size_m': cad_asset(asset_id)['size_m']}
            instance['color'] = 'black' if unpacked else reference['entities'][instance['id']]['color']
            if lifecycle is not None:
                instance.update(current_appearance=lifecycle['appearance'],
                                appearance_revision=lifecycle['appearance_revision'],
                                lifecycle_phase=lifecycle['phase'])
            if lifecycle is not None and lifecycle['phase'] in {'right_bin_released', 'completed'}:
                # Dropping can change orientation. An upright body height is not
                # a world-Z measurement after release; retain all current samples.
                instance['geometry_completion'] = {'status': 'not_applied_released_bin_pose',
                    'height_prior_m': height, 'reason': 'upright_pose_not_established_after_drop',
                    'observation_id': capture['observation_id'], 'synthetic_point_count': 0}
                continue
            record = hint.get('visual_progress', {}).get(instance['id'], {})
            released = (bool(record.get('release_feedback_id')) or record.get('status') == 'completed'
                        or lifecycle is not None and lifecycle['phase'] != 'wrapped_source')
            if measured_floor is not None and not released:
                support, support_source = measured_floor, 'measured_bin_floor'
            else:
                parent = (self.demo_tables['primary_table_id'] if released else
                          observation['scene_entities'].get(source_bin_id, {}).get('initial_parent_id'))
                plane = self.demo_tables['tables'].get(parent)
                if plane is None:
                    instance['geometry_completion'] = {'status': 'unavailable',
                        'reason': 'measured_source_support_required', 'height_prior_m': height}
                    continue
                plane_z = plane['surface_height_m']
                inferred = float(np.max(instance['points_world_m'][:, 2])) - height
                if not plane_z - .015 <= inferred <= plane_z + .15:
                    raise PerceptionError('demo_height_prior_support_inconsistent')
                support = plane_z if unpacked else max(plane_z, inferred)
                support_source = ('cached_measured_table_plane' if unpacked else
                                  'measured_top_minus_user_height_bounded_by_cached_table_plane')
            complete_source_height(instance, support_z_m=support,
                support_source=support_source, observation_id=capture['observation_id'], height_prior_m=height)
        observation['visual_reference'] = reference_with_container_tracks(observation, capture)
        if capture['observation_id'] == reference.get('observation_id'):
            observation['visual_reference']['initial_instance_boxes'] = {
                camera: {identifier: mask_box(mask) for identifier, mask in view['instance_masks'].items()
                         if np.any(mask)} for camera, view in observation['views'].items()}
        for view in observation['views'].values():
            masks = list(view['instance_masks'].values())
            retained = np.logical_or.reduce(masks) if masks else np.zeros(view['depth_m'].shape, bool)
            valid = retained & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)
            view['depth_m'] = _readonly(np.where(valid, view['depth_m'], 0))
            view['valid_depth'] = _readonly(valid)
        observation['depth_policy'] = 'inventory_only_no_background_or_tabletop_voxels'
        return observation
