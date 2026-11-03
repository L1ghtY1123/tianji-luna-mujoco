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
    retain_container_placement_regions,
)
from .demo_geometry import fit_inventory_table, supported_surface_points
from .demo_identity_presentation import improve_identity_presentation, release_verification_context
from .identity_pixel_evidence import IDENTICAL_REGION_RULE, identical_reference_regions
from .inventory_reference import current_inventory_entities, source_inventory, visual_inventory_state
from .multiview_identity import (
    expand_identity_answers,
    measured_view_groups,
    reduce_identity_questions,
    validate_group_proof,
)
from .perception import (
    Perception,
    PerceptionError,
    _readonly,
    _remove_own_robot_surface,
)
from .visual_api import VisualInventoryError
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
        references = [{'reference_entity_id': identifier, 'reference_box': entity['bbox'],
                   'box_iou': _box_iou(box, entity['bbox'])}
                  for identifier, entity in reference.get('entities', {}).items()
                  if entity.get('reference_camera_id') == camera and entity.get('bbox')]
        if not references:
            continue
        overlaps.append({'camera': camera, 'candidate': 'candidate-' + str(index), 'current_box': box,
                         'same_camera_reference_box_overlaps': references})
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
            'ownership. Reference boxes are fallible proposals: they can be displaced, truncated or cover only an edge. '
            'Identical pixels prove capture equality, not box accuracy. Zero overlap does not rule out identity; '
            'high overlap does not establish it. Rows are not ranked identity candidates and IoU is not an identity '
            'probability. Match the full RGB subject, appearance description and visible neighbors across views. '
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

    def __init__(self, *, legacy_identity_protocol=False, **kwargs):
        super().__init__(**kwargs)
        # Explicit archive/test compatibility only; production never enables it.
        self.legacy_identity_protocol = legacy_identity_protocol

    def resolve(self, capture, detections, *, reference=None, entities=None, reference_capture=None, **kwargs):
        # Runtime inventory is authoritative; original RGB remains historical evidence.
        entities = current_inventory_entities(reference, entities)
        entities = {key: value for key, value in entities.items()
                    if value.get('source') != 'configured_demo_region'}
        kwargs['source_inventory'] = source_inventory(reference)
        identity_context = _same_capture_context(capture, detections, reference) or {}
        identical_regions = identical_reference_regions(capture, detections, reference, reference_capture)
        if identical_regions:
            identity_context.update(identical_reference_regions=identical_regions,
                                    identical_reference_region_rule=IDENTICAL_REGION_RULE)
        identity_context['task_spec'] = deepcopy((reference or {}).get('task_spec', {}))
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
        tracks, scopes, tracking = {}, {}, None
        all_entities, all_detections = entities, detections
        view_groups = measured_view_groups(capture, detections)
        if view_groups:
            identity_context['current_measured_view_groups'] = view_groups
        containers = (container_correspondence(capture, detections, reference, entities)
                      if kwargs.get('purpose') in {'associate_inventory', 'verify_transfer'} else None)
        if kwargs.get('purpose') == 'associate_inventory':
            tracks = completed_identity_tracks(capture, reference, kwargs.get('progress'), entities)
            scopes = completed_candidate_scopes(capture, detections, tracks)
            if scopes:
                tracking = verify_completed_correspondence(self, capture, detections, tracks, scopes)
                scoped_ids = {scope['owner_id'] for scope in scopes.values()}
                confirmed_ids = set(tracking['candidate_mapping'].values())
                indices = [index for index in range(len(detections)) if index not in scopes]
                detections = [detections[index] for index in indices]
                # A failed auxiliary view cannot remove an identity from the
                # independent association questions for other current views.
                entities = {key: value for key, value in entities.items() if key not in confirmed_ids}
                reference = {**reference, 'entities': {
                    key: value for key, value in reference.get('entities', {}).items() if key not in confirmed_ids}}
                kwargs['candidate_indices'] = indices
                kwargs['progress_question_ids'] = set(kwargs.get('progress', {})) - scoped_ids
                kwargs['placement_regions'] = ()
                identity_context['candidate_geometric_location_hints'] = [
                    row for row in locations if int(row['candidate'].removeprefix('candidate-')) in indices]
                identity_context['completed_correspondence_request_hash'] = tracking['request_hash']
                identity_context['fixed_reference_labels'] = {
                    identifier: f'Reference {index + 1}' for index, identifier in enumerate(sorted(all_entities))
                    if identifier in entities}
                identity_context['completed_source_correspondences'] = [{
                    'entity_id': identifier, 'display_id': all_entities[identifier].get('display_id'),
                    'description': all_entities[identifier].get('description', 'task object'),
                    'slot_id': track['binding']['slot_id'], 'observation_id': capture['observation_id'],
                    'generation': capture['generation'], 'request_hash': tracking['request_hash'],
                    'status': 'confirmed' if identifier in tracking['candidate_mapping'].values() else 'unknown',
                    'current_candidates': [
                        {'candidate': 'Candidate ' + index, 'camera': all_detections[int(index)]['camera_id'],
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
        from .identity_geometry import cad_identity_exclusions

        exclusions = cad_identity_exclusions(
            capture, all_detections, identity_context.get('current_confirmed_container_anchors', []),
            capture.get('confirmed_cad_models', {}))
        if exclusions:
            identity_context['identity_geometry_exclusions'] = exclusions
        kwargs['identity_context'] = identity_context
        try:
            result = super().resolve(capture, detections, reference=reference, entities=entities, **kwargs)
        except VisualInventoryError as error:
            if tracking is not None:
                error.evidence['completed_tracking'] = tracking
            raise
        if containers and containers['candidate_mapping']:
            result = merge_container_correspondence(self, all_detections, entities, containers, result)
        if tracking is not None:
            frozen = (reference or {}).get('inventory_state', {}).get('inventory_finalized') is True
            partial_completed = {identifier for identifier, record in (kwargs.get('progress') or {}).items()
                                 if frozen and record.get('status') == 'completed'}
            result = merge_completed_correspondence(
                self, capture, all_detections, all_entities, tracking, result,
                partial_completed_ids=partial_completed)
        return result

    def request(self, payload, capture, purpose, parse, *, identity_subjects=None):
        payload = deepcopy(payload)
        content = payload['input'][0]['content']
        context = json.loads(content[0]['text'])
        # Completed/container correspondence can remove candidates before this
        # request. Keep the full geometric proof, but only expand subject views
        # that still carry candidate provenance in this request.
        context['identity_request_candidate_ids'] = [
            row['candidate'] for part in content if part.get('type') == 'input_text'
            and part['text'].startswith('{')
            and isinstance(row := json.loads(part['text']), dict)
            and 'candidate_provenance' in row and 'candidate' in row]
        context['rules'] = (
            'Use the union of all CURRENT full RGB views to associate the declared scene entities. '
            'All observed regions are in scope, including supports, containers, task objects and obstacles. '
            'Robot arms, grippers and attached robot hardware without a corresponding declared scene entity '
            'are outside this inventory; reject those proposals. '
            'Match physical subjects using RGB appearance and frozen INITIAL or verified PRODUCT references; distinguish '
            'individual objects and retain unknown for ambiguous correspondence. SAM3 prompts and boxes '
            'are proposals, not identity or segmentation ownership. Multiple views can show one object. '
            'Measured geometric locations are hints, not identity. Task roles, colors and camera numbers '
            'do not assign identity. INITIAL images show historical appearances, not additional current objects. '
            'Regions are intended destinations, not physical markers or proof of release. '
            'Judge visibility from all current images even when SAM3 misses an entity. Action history is '
            'context, never success evidence. Check for visible unlisted task candidates matching '
            'task_spec.target_selector and its source constraints across all observed regions. '
            'Unlisted non-task objects are intentionally outside this inventory map; geometric '
            'visibility and clearance are checked separately, never certified by this judgment. '
            'Do not infer hidden contents or empty space.')
        context['rules'] += (
            ' Retired entity IDs describe consumed inputs and cannot identify current objects. '
            'Verified transformation outputs have independent IDs and their own appearance; use their '
            'PRODUCT reference images when supplied, not the consumed input in INITIAL images. '
            'Lineage records link inputs and outputs without equating their visual identities. '
            'Historical reference pixels are not additional current objects; containers keep identity '
            'when their contents change. Execution receipts never prove current visual success.')
        if context.get('completed_source_correspondences'):
            context['rules'] += (
                ' Completed source objects have a separate CURRENT correspondence check in this same capture. '
                'Its confirmed rows locate those known objects by current C label, camera, box and slot; '
                'unknown rows are NOT confirmed. Only confirmed IDs are omitted from this request\'s identity '
                'choices and visibility questions; unknown IDs and their historical reference images remain '
                'available for independent matching in other views. Reject a proposal whose subject is an independently confirmed '
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
                'current_confirmed_container_anchors remain labeled in the CURRENT annotated views with '
                'their permanent IDs, camera, box, measured world bounds and tracking provenance. '
                'Compare unresolved subjects with these same-capture anchors before historical references. '
                'Multiple cameras can observe the same container, but simultaneously distinct containers '
                'in this same capture cannot share one permanent ID. Movement between captures does not '
                'explain two distinct current subjects; changed contents do not change container identity. '
                'Use RGB and calibrated geometry together; a camera number or location alone never assigns an ID. '
                'Keep unknown if the current correspondence remains ambiguous. '
                'This tracking does not identify task objects or prove there are no unlisted objects.')
        original_questions = deepcopy(payload['questions'])
        view_groups = context.get('current_measured_view_groups', {})
        validate_group_proof(view_groups, capture)
        from .identity_proposals import resolve_identity_proposals, scope_same_capture_identity

        content[0]['text'] = json.dumps(context, ensure_ascii=False)
        _, same_view_questions = scope_same_capture_identity(payload)
        # An exact same-capture reference takes priority; otherwise use the
        # largest observed subject, not the first camera's tiny occluded fragment.
        # This selects a question, never an identity or confidence.
        visible_pixels = {'identity_' + candidate.removeprefix('candidate-'):
                          int(np.count_nonzero(subject['silhouette_mask']))
                          for candidate, subject in (identity_subjects or {}).items()}
        payload['questions'], identity_aliases = reduce_identity_questions(
            original_questions, view_groups, preferred_questions=same_view_questions,
            visible_pixels=visible_pixels)
        if identity_aliases:
            context['measured_view_identity_aliases'] = identity_aliases
            context['rules'] += (
                ' current_measured_view_groups contains mutual registered RGBD surface correspondences '
                'within this CURRENT capture, with depth, calibration and mask hashes. Each listed group '
                'is one measured physical surface observed by different cameras. Judge its permanent ID '
                'once using all member RGB views and the historical references. The omitted member questions '
                'inherit that same answer and confidence through the recorded measured correspondence. '
                'This geometric grouping does not choose a permanent ID; keep unknown when reference '
                'identity is ambiguous. Visibility and scene closure remain separate judgments.')
        content[0]['text'] = json.dumps(context, ensure_ascii=False)
        same_capture = context.get('same_capture_reference')
        if same_capture:
            for part in content:
                if part.get('type') == 'input_text' and part['text'].startswith('INITIAL RGB REFERENCE.'):
                    part['text'] = (
                        'INITIAL RGB REFERENCE. For cameras ' + json.dumps(same_capture['verified_camera_ids'])
                        + ', initial and current are the SAME frozen capture with identical pixels; only annotations '
                        'differ. Permanent IDs locate the same physical subjects in those views. '
                        'Other cameras or later captures may differ; positions may change there.')
        for question in payload['questions']:
            if question['name'].startswith('progress_'):
                question['instructions'] = _demo_progress_instructions(question['instructions'])
        closure_name, extra_status = 'scene_inventory_closure', 'unlisted'
        payload['questions'].append(choice(closure_name,
            'Across all observed regions in the CURRENT full RGB views, are there visible task '
            'candidates matching task_spec.target_selector and its source constraints that are not '
            'represented by the declared permanent source_object_ids? For all/count scope, '
            'a matching subject listed only in fixed_entities but absent from source_object_ids '
            'is still a missing task candidate. For ids scope, only the explicitly requested subset '
            'is relevant; other matching subjects are not omissions. Independently confirmed '
            'completed objects and their verified transformation products are not new source candidates. '
            'Ignore non-task supports, containers, obstacles, robot parts, reflections and annotations. '
            'Unlisted non-task objects are intentionally filtered from the inventory map. '
            'none means each visible task candidate is represented; it does not certify hidden space '
            'or source emptiness. unlisted means a clearly visible relevant subject lacks a declared '
            'entity. unknown means correspondence or current visibility is insufficient. '
            'Missing segmentation candidates never prove absence.', ['none', 'unlisted', 'unknown']))
        if len(payload['questions']) > 200:
            raise VisualInventoryError('visual_inventory_input_limit')

        def with_demo_check(body, evidence):
            answers = body.get('answers')
            if not isinstance(answers, list) or len(answers) != len(payload['questions']):
                raise VisualInventoryError('visual_decisions_invalid_response')
            answer = answers[-1]
            if not isinstance(answer, dict) or answer.get('name') != closure_name:
                raise VisualInventoryError('visual_decisions_invalid_response')
            confidence = answer.get('confidence', 0.)
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise VisualInventoryError('visual_decisions_invalid_response')
            status = answer.get('choice') if answer.get('type') == 'choice' and confidence >= self.minimum_confidence else 'unknown'
            if status not in {'none', extra_status, 'unknown'}:
                raise VisualInventoryError('visual_decisions_invalid_response')
            check = {'status': status, 'source': self.source,
                     'observation_id': capture['observation_id'], 'confidence': confidence,
                     'request_hash': evidence['request_hash'], 'component': 'decisions',
                     'choice': answer.get('choice'), 'minimum_confidence': self.minimum_confidence,
                     'diagnostic_checks': [deepcopy(item) for item in evidence.get('diagnostic_checks', [])
                                           if item['name'] == closure_name]}
            if status == extra_status:
                raise VisualInventoryError('scene_unlisted_object', evidence={
                    **evidence, 'answers': answers, 'demo_inventory_check': check})
            base_body = {**body, 'answers': answers[:-1]}
            group_evidence = {}
            if identity_aliases:
                base_body, derivations = expand_identity_answers(
                    base_body, original_questions, identity_aliases, view_groups)
                group_evidence = {
                    'measured_view_identity_derivations': derivations,
                    'current_measured_view_groups': deepcopy(view_groups),
                    'original_decisions_answers': deepcopy(evidence.get('original_decisions_answers', answers))}
            result = parse(base_body, {
                **group_evidence,
                **evidence, 'demo_inventory_check': check, 'generation': capture['generation'],
                **({'container_correspondence': context['container_correspondence']}
                   if context.get('container_correspondence') else {}),
                **({'completed_correspondence_request_hash': context['completed_correspondence_request_hash']}
                   if context.get('completed_correspondence_request_hash') else {})})
            result['demo_inventory_check'] = check
            result['evidence']['answers'] = base_body['answers'] + [answer]
            return result
        from .demo_identity_presentation import ground_identity_locations

        presented, proposals = scope_same_capture_identity(improve_identity_presentation(payload))
        presented = ground_identity_locations(presented)
        from .identity_comparison_images import same_capture_comparison_images

        presented = same_capture_comparison_images(presented, proposals)

        def with_identity_proposals(body, evidence):
            if not proposals:
                return with_demo_check(body, evidence)
            resolved = resolve_identity_proposals(body, proposals, self.minimum_confidence)
            return with_demo_check(resolved, {**evidence,
                'original_decisions_answers': evidence.get('original_decisions_answers', deepcopy(body['answers'])),
                'same_capture_identity_proposals': proposals,
                'identity_derivation': 'positive_visual_predicate_required_not_numeric_identity_assignment'})

        from .identity_correspondence import (
            has_product_reference,
            request_identity_correspondences,
            scope_identity_clarification,
        )

        # Later observations use direct predicates. Initial same-capture binding
        # retains scoped clarification: accepted CURRENT identities become new
        # anchors for the remaining ambiguous subject, without new photography.
        single_pass = not self.legacy_identity_protocol and not context.get('same_capture_reference')
        if purpose in {'associate_inventory', 'verify_transfer'} and (
                single_pass or has_product_reference(presented)):
            return request_identity_correspondences(self, presented, capture, purpose, with_identity_proposals,
                identity_subjects=identity_subjects, include_proposal_subjects=self.legacy_identity_protocol)
        initial_error = None
        try:
            initial_result = super().request(presented, capture, purpose, with_identity_proposals)
        except VisualInventoryError as error:
            if str(error) != 'visual_identity_unknown':
                raise
            initial_error, initial_result = error, None
        has_identity = any(q['name'].startswith('identity_') and q['type'] in {'choice', 'predicate'}
                           for q in presented['questions'])
        partial_unknown = (purpose in {'associate_inventory', 'verify_transfer'} and initial_result is not None
                           and initial_result.get('missing_geometry_ids')
                           and initial_result.get('unassociated_candidates'))
        if not has_identity or (initial_error is None and not partial_unknown):
            if initial_error is not None:
                raise initial_error
            return initial_result
        # Partial success must not bypass the same bounded clarification used
        # for a wholly unknown identity. No new capture or physical action.
        initial_evidence = initial_error.evidence if initial_error is not None else initial_result['evidence']
        clarification, retained = scope_identity_clarification(presented, initial_evidence, self.minimum_confidence)

        def clarified(body, evidence):
            answers = {answer['name']: answer for answer in body['answers']}
            answers.update({name: record['answer'] for name, record in retained.items()})
            merged = {**body, 'answers': [deepcopy(answers[question['name']]) for question in presented['questions']]}
            return with_identity_proposals(merged, {**evidence,
                'retained_identity_answers': deepcopy(list(retained.values())),
                'clarifies_request_hash': initial_evidence.get('request_hash'),
                'clarification_reason': 'categorical_identity_unknown'})
        try:
            return request_identity_correspondences(self, clarification, capture, purpose, clarified,
                                                    identity_subjects=identity_subjects)
        except VisualInventoryError as clarification_error:
            if initial_error is not None:
                clarification_error.evidence['initial_identity_failure'] = {
                    'error': str(initial_error), 'evidence': deepcopy(initial_error.evidence)}
                raise clarification_error from initial_error
            if str(clarification_error) != 'visual_identity_unknown':
                raise
            initial_result['evidence']['identity_clarification'] = {
                'error': str(clarification_error), 'evidence': clarification_error.evidence}
            return initial_result

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
        source_ids = set(source_inventory(reference)['source_object_ids'])
        completed = {identifier for identifier, record in (progress or {}).items()
                     if record.get('status') == 'completed'}
        remaining_source_ids = source_ids & set(entities) - completed
        source_covered = remaining_source_ids <= accepted
        complete_union = bool(required) and required <= accepted
        # These sets contain only identities already accepted by the shared
        # confidence parser. They remain available when an unresolved view fails.
        coverage = {'source': self.source, 'observation_id': evidence['observation_id'],
            'request_hash': evidence['request_hash'], 'purpose': purpose,
            'minimum_confidence': self.minimum_confidence,
            'required_entity_ids': sorted(required), 'accepted_entity_ids': sorted(accepted),
            'complete': required <= accepted,
            'source_object_ids': sorted(source_ids),
            'remaining_source_ids': sorted(remaining_source_ids)}
        if measured_containers:
            coverage.update(source='mixed_decisions_measured_container',
                decisions_accepted_entity_ids=sorted(decisions_accepted),
                measured_container_entity_ids=sorted(measured_containers),
                container_correspondence_binding_hash=container_tracking['binding_hash'])
        unassociated, retained = [], []
        for index, detection in enumerate(detections):
            identity = judgments['identity_' + str(index)]
            supplementary = (purpose == 'verify_transfer' or (
                purpose in {'initialize_inventory', 'associate_inventory'}
                and source_covered))
            if identity == 'unknown':
                answer = evidence['answers'][index]
                # Later captures may cover only part of the established inventory.
                # Return its measured subset and missing IDs; the map ledger keeps
                # unseen layers and rejects stale/missing geometry before motion.
                visible_unconfirmed = {identifier for identifier in required - accepted
                                       if judgments.get('visibility_' + identifier) == 'visible'}
                frozen_inventory = (reference or {}).get('inventory_state', {}).get('inventory_finalized') is True
                partial_association = (purpose == 'associate_inventory' and (
                    frozen_inventory or source_covered and not visible_unconfirmed))
                if not partial_association and (not complete_union or not supplementary):
                    details = self._identity_failure_evidence(evidence, detection, names[index], answer)
                    details.update(identity_coverage=coverage, missing_geometry_ids=sorted(required - accepted),
                        unresolved_identity_candidates=self._unresolved_identity_candidates(
                            detections, names, judgments, evidence))
                    raise VisualInventoryError('visual_identity_unknown', evidence=details)
                unassociated.append({'candidate': names[index], 'camera_id': detection['camera_id'],
                    'status': 'unknown', 'choice': answer.get('choice'), 'confidence': answer.get('confidence'),
                    'minimum_confidence': self.minimum_confidence, 'geometry_policy': 'discard_from_inventory_map',
                    'reason': 'supplementary_identity_uncertain'})
            else:
                retained.append(index)
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
        correspondences = {row['name']: row for row in evidence.get('identity_correspondence', [])}
        original_answers = evidence.get('original_decisions_answers', evidence['answers'])
        unresolved = []
        for index, detection in enumerate(detections):
            if judgments['identity_' + str(index)] != 'unknown':
                continue
            answer = evidence['answers'][index]
            name = answer['name']
            probabilities = correspondences.get(name, {}).get('probabilities', {})
            positive = sorted(identifier for identifier, probability in probabilities.items()
                              if probability >= self.minimum_confidence)
            prefix = 'match_' + name.removeprefix('identity_') + '_'
            unresolved.append({'candidate_id': names[index], 'camera_id': detection['camera_id'],
                'source': self.source, 'observation_id': evidence['observation_id'],
                'generation': evidence.get('generation'), 'request_hash': evidence['request_hash'],
                'status': 'conflicting' if len(positive) > 1 else 'unknown',
                'positive_entity_ids': positive, 'minimum_confidence': self.minimum_confidence,
                'answer': deepcopy(answer), 'probabilities': deepcopy(probabilities),
                'original_decisions_answers': [deepcopy(row) for row in original_answers
                    if row['name'] == name or row['name'].startswith(prefix)]})
        return unresolved


class DemoPerception(Perception):
    visual_plan_source = 'responses_program'

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
        if not (reference or {}).get('inventory_state', {}).get('transformations'):
            return detections
        # Processing hands can occupy the image. Remove measured self returns
        # before showing proposals to Decisions, not only after identity binding.
        groups, _ = _remove_own_robot_surface([[entry] for entry in detections], views, capture,
                                             self.robot_surface_classifier)
        cleaned = []
        for group in groups:
            entry = group[0]
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
            task = (reference or {}).get('task_spec', {})
            preferred = [task.get('work_area_ref'), task.get('destination_ref')]
            primary = next((identifier for identifier in preferred if identifier in tables), None)
            if primary is None and tables:
                primary = next(iter(tables))
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
        if 'task_spec' in hint.get('visual_reference', {}) and not hint.get('scene_entities'):
            raise PerceptionError('controller_scene_inventory_required')
        hint['camera_ids'] = ['1', '2', '3']
        observation = super().observe(capture, instruction=instruction, checkpoint=checkpoint, source_hint=hint)
        reference = observation['visual_reference']
        if not reference or 'task_spec' not in reference:
            raise PerceptionError('task_spec_required')
        observation.update(task_spec=deepcopy(reference['task_spec']),
                           policy_graph=deepcopy(reference.get('policy_graph', {})),
                           inventory_state=deepcopy(reference.get('inventory_state', {})),
                           demo_mapping=True, demo_tables=deepcopy(self.demo_tables))
        observation['demo_inventory_check'] = observation['visual_inventory'].get('demo_inventory_check', {
            'status': 'unknown', 'source': 'missing_decisions_check', 'observation_id': capture['observation_id']})
        entities = current_inventory_entities(reference, observation['scene_entities'])
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
