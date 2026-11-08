"""Instance-scoped evidence inputs, independent of graph and task control."""
import json
import math
from copy import deepcopy

from .decision_conditions import accepted_identity
from .demo_identity_presentation import box_overlap, decode_image_part
from .observation_labels import outline_masks, raster_visual_annotations
from .visual_inventory import image_part


def metadata(part):
    if part.get('type') != 'input_text':
        return {}
    try:
        value = json.loads(part['text'])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def reference_labels(context):
    return {row['entity_id']: row for row in context['identity_display_legend'].values()}


def candidate_labels(payload):
    content = payload['input'][0]['content']
    context = metadata(content[0])
    labels = {row['candidate']: {'label': 'C' + row['candidate'].removeprefix('candidate-'),
                                'camera': row['camera']}
              for part in content if 'candidate_provenance' in (row := metadata(part))}
    for anchor in context.get('current_confirmed_container_anchors', []):
        labels[anchor['candidate']] = {'label': anchor['image_label'], 'camera': anchor['camera']}
    return labels


def current_bindings(context, answers, threshold):
    """Expose only accepted same-capture answers and independently tracked anchors."""
    by_name = {answer['name']: answer for answer in answers}
    bindings = {}
    for name, answer in by_name.items():
        identifier = accepted_identity(answer, threshold)
        if name.startswith('identity_') and identifier in context['memory_entities']:
            bindings['candidate-' + name.removeprefix('identity_')] = identifier
    for anchor in context.get('current_confirmed_container_anchors', []):
        bindings[anchor['candidate']] = anchor['entity_id']
    return bindings


def inventory_payload(payload, family, questions, *, bindings=None, candidate_ids=None, entity_ids=None, identity_subjects=None):
    """Build one semantic request with its own state and one image per camera/time."""
    source = payload['input'][0]['content']
    context = metadata(source[0])
    labels = reference_labels(context)
    subjects = candidate_labels(payload)
    current_candidates = set(candidate_ids or ()) | set(bindings or ())
    focus_candidate = ('candidate-' + questions[0]['name'].removeprefix('identity_')
                       if family == 'identity' else None)
    entities = [{'id': identifier, 'label': labels[identifier]['image_label'],
                 'description': entity['description'], 'kind': entity.get('kind'),
                 'color': entity.get('color')}
                for identifier, entity in context['memory_entities'].items()
                if entity_ids is None or identifier in entity_ids]
    candidates = [row for part in source if 'candidate_provenance' in (row := metadata(part))
                  and (candidate_ids is None or row['candidate'] in current_candidates)]
    state = {'purpose': family, 'observation_id': context['observation_id'],
        'capture_time_s': context.get('capture_time_s'), 'entities': entities,
        'candidates': [{'label': 'C' + row['candidate'].removeprefix('candidate-'),
                       'camera': row['camera'], 'box': row['box'],
                       'proposed_subject': row['sam3_prompt_proposal']} for row in candidates]}
    raw_images = {}
    for index, part in enumerate(source[:-1]):
        for prefix, role in (('CURRENT RGB camera ', 'current_annotations'),
                             ('INITIAL REFERENCE RGB camera ', 'initial_reference_annotations')):
            if part.get('type') == 'input_text' and part['text'].startswith(prefix):
                raw_images[role, part['text'].removeprefix(prefix)] = source[index + 1]
    images, reference_overlaps = [], []
    product_references, creation_references, creation_before_images = [], {}, []
    combined_cameras = {row['camera'] for part in source
                        if (row := metadata(part)).get('includes_initial_references')}
    for index, part in enumerate(source[:-1]):
        row = metadata(part)
        role = row.get('image_role')
        if role not in {'initial_reference_annotations', 'current_annotations', 'verified_product_reference',
                        'instance_reference', 'product_creation_before'}:
            continue
        if family != 'identity' and role != 'current_annotations':
            continue
        if role == 'initial_reference_annotations' and not row.get('items'):
            continue
        if role == 'initial_reference_annotations' and row['camera'] in combined_cameras:
            continue
        following = source[index + 1]
        if following.get('type') != 'input_image':
            continue
        if role == 'product_creation_before':
            if entity_ids is None or row['entity_id'] in entity_ids:
                creation_references[row['entity_id']] = row['creation_reference']
                view = {key: value for key, value in row.items() if key != 'creation_reference'}
                creation_before_images.extend([{'type': 'input_text', 'text': json.dumps(view)}, following])
            continue
        if role == 'verified_product_reference':
            if entity_ids is not None and row['entity_id'] not in entity_ids:
                continue
            same_capture = (row.get('observation_id') == context['observation_id']
                            and row.get('pixels_identical_to_current_same_camera') is True)
            product_references.append({key: row[key] for key in (
                'entity_id', 'description', 'camera_id', 'observation_id', 'capture_time_s') if key in row}
                | {'image_scope': 'current_capture' if same_capture else 'historical_capture'})
            if same_capture:
                continue
        items = row.get('items', [])
        if (role == 'instance_reference' and entity_ids is not None
                and not any(item.get('id') in entity_ids for item in items)):
            continue
        rgb = None
        current_masks = []
        if family == 'identity' and candidate_ids is not None and role not in {'verified_product_reference', 'instance_reference'}:
            items = [item for item in items if item.get('id') in current_candidates
                     or item.get('id') in (entity_ids or ())]
            if not items and role != 'current_annotations':
                continue
            raw = raw_images.get((role, row.get('camera')))
            if raw is not None:
                rgb, _ = decode_image_part(raw)
                current_masks = [identity_subjects[item['id']]['silhouette_mask'] for item in items
                                 if identity_subjects and item['id'] in identity_subjects] if role == 'current_annotations' else []
                following = image_part(raster_visual_annotations(
                    outline_masks(rgb, current_masks), items, label_layout='near_box'))
        view = {key: row[key] for key in (
            'image_role', 'camera', 'camera_id', 'entity_id', 'description', 'observation_id', 'capture_time_s',
            'pixels_identical_to_current_same_camera', 'pixel_comparison_meaning', 'mask_contours_supplied', 'meaning')
                if key in row}
        if 'items' in row:
            view['items'] = [{key: item[key] for key in ('image_label', 'box') if key in item}
                             for item in items]
        if role == 'current_annotations':
            view['mask_contours_supplied'] = bool(current_masks)
            view['meaning'] = (
                'CURRENT RGB. C labels locate the candidate subjects; R labels, when present, are reference '
                'annotations. Rectangles locate subjects and can include occluders or container contents. '
                + ('The visible contours show the supplied SAM3 candidate masks; they are segmentation proposals, '
                   'not accepted identity or complete physical boundaries.' if current_masks else
                   'No segmentation contour is supplied in this view; the rectangle alone does not assign pixel ownership.'))
        if row.get('includes_initial_references'):
            view['reference_and_current_pixels_identical'] = True
            if family == 'identity':
                current_items = [item for item in items if item.get('id') in (candidate_ids or ())]
                reference_items = [item for item in items if item.get('id') in (entity_ids or ())]
                for current in current_items:
                    for reference in reference_items:
                        overlap = box_overlap(current['box'], reference['box'])
                        if overlap > 0:
                            reference_overlaps.append({'candidate': current['image_label'],
                                'reference': reference['image_label'], 'camera': row['camera'],
                                'box_iou': round(overlap, 3)})
        if row.get('regions') and family != 'identity':
            view['regions'] = row['regions']
        images.extend([{'type': 'input_text', 'text': json.dumps(view, ensure_ascii=False)}, following])
        if role == 'current_annotations' and rgb is not None:
            focus = next((item for item in items if item.get('id') == focus_candidate), None)
            if focus is not None:
                from .task_binding_images import current_subject_detail_content

                images.extend(current_subject_detail_content(rgb, focus['box'], focus['image_label'], row['camera']))
    state['current_identity_bindings'] = {subjects[candidate]['label']: identifier
        for candidate, identifier in (bindings or {}).items() if candidate in subjects}
    if family == 'identity':
        state['historical_surface_projection'] = [
            {**row, 'target_candidate': subjects[row['target_candidate']]['label']}
            for row in context.get('projected_reference_surfaces', [])
            if row['target_candidate'] == focus_candidate
            and (entity_ids is None or row['entity_id'] in entity_ids)]
        state['historical_projection_scope'] = (
            'Measured 3D surfaces from dated masks/depth/calibration projected into the CURRENT camera. '
            'mask_iou compares the projected measured surface pixels with the current candidate mask. '
            'Large overlap and agreeing depth support an unchanged instance at that location. '
            'Low overlap does NOT prove a different instance: movement, occlusion and previously unseen '
            'faces can reduce overlap. Positive depth residual means the historical surface is behind '
            'the current surface. Judge identity using RGB, masks and the dated evidence together. '
            'Robot action receipts describe actions, not visually proven object states.')
        state['projected_candidate_surfaces'] = [
            {**row, 'source_candidate': subjects[row['source_candidate']]['label'],
             'target_candidate': subjects[row['target_candidate']]['label'],
             'source_instance_id': bindings[row['source_candidate']]}
            for row in context.get('projected_candidate_surfaces', [])
            if row['target_candidate'] == focus_candidate and row['source_candidate'] in (bindings or {})
            and row['source_candidate'] in subjects and row['target_candidate'] in subjects]
        state['product_creation_references'] = product_references
        state['creation_references'] = creation_references
        images.extend(creation_before_images)
        state['dated_measured_positions'] = [
            {'reference': labels[identifier]['image_label'], **entity['last_position_measurement']}
            for identifier, entity in context['memory_entities'].items()
            if (entity_ids is None or identifier in entity_ids) and entity.get('last_position_measurement')]
        state['current_identity_proposals'] = [
            {'reference': labels[identifier]['image_label'], **proposal,
             'same_camera_candidate_overlaps': [
                 {'candidate': 'C' + row['candidate'].removeprefix('candidate-'),
                  'box_iou': round(box_overlap(row['box'], proposal['bbox']), 3)}
                 for row in candidates if row['camera'] == proposal['camera_id']]}
            for identifier, entity in context['memory_entities'].items()
            if (entity_ids is None or identifier in entity_ids)
            and (proposal := entity.get('current_identity_proposal', {})).get('observation_id')
                == context['observation_id']]
        state.update(
            same_image_reference_overlaps=reference_overlaps,
            current_anchors=[{'entity_id': anchor['entity_id'], 'label': anchor['image_label'],
                              'camera': anchor['camera']} for anchor in context.get(
                                  'current_confirmed_container_anchors', [])],
            candidate_locations=[{'candidate': subjects[row['candidate']]['label'], 'camera': row['camera'],
                'visible_surface_centroid_world_m': [round(value, 4) for value in row['measured_centroid_world_m']],
                **({'visible_surface_bounds_world_m': row['measured_bounds_world_m']}
                   if 'measured_bounds_world_m' in row else {})}
                for row in context.get('candidate_geometric_location_hints', []) if row['candidate'] in subjects
                and (candidate_ids is None or row['candidate'] in current_candidates)],
            retired_inputs=context.get('inventory_state', {}).get('retired_entity_ids', []),
            identical_reference_image_cameras=context.get('identical_reference_image_cameras', []),
            same_capture_reference_cameras=context.get('same_capture_reference', {}).get('verified_camera_ids', []))
        history = context.get('inventory_state', {})
        state['object_history'] = {identifier: deepcopy(row) for identifier, row in history.get('object_history', {}).items()
                                   if entity_ids is None or identifier in entity_ids}
        state['robot_action_receipts'] = []
        for identifier, record in state['object_history'].items():
            receipt = record.pop('last_motion_receipt', None)
            if receipt is None:
                continue
            state['robot_action_receipts'].append({
                'entity_id': identifier, **{key: receipt[key] for key in
                    ('kind', 'arm', 'generation', 'feedback_time_s') if key in receipt},
                'scope': 'Robot action/feedback only, not visual proof of object holding, release or location.'})
            if record.get('transport_status') == 'held' and receipt.get('source') == 'robot_feedback':
                record.pop('transport_status')  # Do not promote an older telemetry-only record to a visual fact.
        for identifier, record in state['object_history'].items():
            if (record.get('input_entity_id') and not record.get('confirmed_observation_id')
                    and not record.get('container_entry')):
                # Older records inherited the input's processing region. The
                # output has its own observed support and later placement history.
                record = deepcopy(record)
                support = record.get('observed_support') or context['memory_entities'][identifier].get('observed_support', {})
                record['current_region'] = support.get('support_entity_id')
                state['object_history'][identifier] = record
        state['verified_release_locations'] = []
        for entity in entities:
            release = state['object_history'].get(entity['id'], {}).get('visual_release', {})
            if release.get('confirmed_observation_id'):
                state['verified_release_locations'].append({
                    'reference': entity['label'], 'destination_id': release['destination_id'],
                    'region_id': release.get('region_id'),
                    **({'placement_region': deepcopy(release['placement_region'])}
                       if release.get('placement_region') else {}),
                    'observation_id': release['confirmed_observation_id'],
                    'same_capture_as_current': release['confirmed_observation_id'] == context['observation_id']})
        state['transformations'] = {key: {field: row[field] for field in (
            'id', 'skill', 'input_entity_id', 'output_entity_id') if field in row}
            for key, row in history.get('transformations', {}).items()
            if entity_ids is None or row.get('input_entity_id') in entity_ids or row.get('output_entity_id') in entity_ids}
        state['last_failure'] = history.get('last_failure', {})
        rule = (
            'Identify the outlined CURRENT subject C using permanent references R. Containers and their '
            'contents have separate identities. Use the clearest confirmed current view. In '
            'identical_reference_image_cameras, R and C are annotations of IDENTICAL source pixels, '
            'with measured same_image_reference_overlaps. Compare those outlined visible subjects directly. '
            'Other reference images and descriptions are dated. Current coordinates are measured in metres, '
            '+Y robot left. current_identity_bindings are accepted same-capture identities. '
            'verified_release_locations are dated verified placements. Processing creates a distinct '
            'product. A container_entry with source=operator_container_release_rule records an accepted '
            'above-container release. Its expected containment is an operator assumption and its '
            'individual settled pose is unmeasured. Compare current evidence independently, including '
            'contradictions or indistinguishable stacked contents. Processing retires its input. The '
            'permanent references identify distinct simultaneous instances, including identical '
            'objects. product_creation_references with image_scope=current_capture were verified in '
            'the CURRENT images shown here. Match their described subject in those same pixels. '
            'For the stated identity predicate, a robot part or background is not the claimed external '
            'object: report a low probability. Report uncertainty when the subject cannot be distinguished. '
            'SAM names and intended movements are proposals. '
            'current_identity_proposals locate proposed instances on CURRENT pixels. Independently verify '
            'those proposals using appearance, references and verified movement history. '
            'dated_measured_positions are historical RGBD measurements in the same calibrated world frame '
            'as candidate_locations. Compare them with dated RGB and recorded actions. They may be stale '
            'after motion and serve only as identity evidence. Visible-surface centroids and bounds can '
            'change with viewpoint or occlusion while the physical object stays still. '
            'Keep the probability uncertain when the evidence cannot distinguish the physical instance.')

    else:
        rule = ('All images show the same CURRENT capture. C labels locate candidate subjects. '
                'current_identity_bindings contains accepted identities from this capture. '
                'Use the full images, including areas without SAM proposals. '
                'Judge each requested fact independently. An identity binding specifies the subject only.')
        if family == 'progress':
            state['assigned_destinations'] = context.get('transport_progress', {})
    if sum(part['type'] == 'input_image' for part in images) > 128:
        from .visual_api import VisualInventoryError

        raise VisualInventoryError('visual_inventory_input_limit')
    return {'model': payload['model'], 'input': [{'role': 'user', 'content': [
        {'type': 'input_text', 'text': rule},
        {'type': 'input_text', 'text': json.dumps(state, ensure_ascii=False)}, *images]}],
        'questions': deepcopy(questions)}


def scene_questions(payload, family):
    return [deepcopy(question) for question in payload['questions']
            if family == 'progress' and question['name'].startswith('progress_')]


def add_correspondence_geometry(payload, candidate_label, camera, reference_label, *, current_matches=()):
    """Present the queried measurement pair; no distance gate or automatic identity acceptance."""
    content = payload['input'][0]['content']
    state = json.loads(content[1]['text'])
    state.pop('correspondence_geometry', None)
    entity = next(row for row in state['entities'] if row['label'] == reference_label)
    creation = state.pop('creation_references', {}).get(entity['id'])
    if creation:
        state['creation_reference'] = creation
    subject = next(row for row in state['candidates'] if row['label'] == candidate_label)
    state['correspondence_subjects'] = {
        'current': {'label': candidate_label, 'camera': camera,
                    'segmentation_proposal': subject['proposed_subject']},
        'reference': {'label': reference_label, 'kind': entity['kind'], 'color': entity['color'],
                      'dated_description': entity['description'], 'accepted_current_views': list(current_matches)}}
    current_proposal = any(row['reference'] == reference_label for row in state['current_identity_proposals'])
    state['correspondence_subjects']['reference']['reference_type'] = (
        'same_capture_confirmed_instance' if current_matches else
        'created_instance_first_localization' if creation else
        'initial_same_capture_proposal' if current_proposal else 'dated_instance_reference')
    current = next((row for row in state['candidate_locations'] if row['candidate'] == candidate_label), None)
    historical = next((row for row in state['dated_measured_positions'] if row['reference'] == reference_label), None)
    latest_release = next((row for row in state['verified_release_locations']
                           if row['reference'] == reference_label), None)
    current_release = latest_release if latest_release and latest_release['same_capture_as_current'] else None
    release = state['object_history'].get(entity['id'], {}).get('visual_release', {})
    entry = state['object_history'].get(entity['id'], {}).get('container_entry', {})
    measurement_before_release = bool(historical and historical['observation_id'] in {
        release.get('pre_release_measurement_observation_id'), entry.get('pre_release_measurement_observation_id')})
    assessment = release.get('assessment', {})
    assessed_release = (release.get('action_id') and assessment.get('request_hash')
                        and assessment.get('observation_id') == state['observation_id'])
    if assessed_release:
        state['correspondence_subjects']['reference']['post_release_assessment'] = deepcopy(assessment)
        state['correspondence_subjects']['reference']['movement_scope'] = (
            'This instance was physically transported and released before the CURRENT capture. '
            'The assessment preserves separate visual facts: uncertain placement precision does not '
            'undo confirmed release or support. Its old position is a pre-move reference, not its '
            'expected current position. These receipts support identity comparison; they do not '
            'automatically assign the queried mask or confirm task completion.')
    relocated = (historical and not current_matches and (assessed_release or current_release or measurement_before_release)
                 and historical['observation_id'] != state['observation_id'])
    if relocated:
        if latest_release and not current_release:
            state['correspondence_subjects']['reference']['dated_verified_release'] = latest_release
            payload['questions'][0]['instructions'] += (
                ' The stored reference measurement explicitly predates this instance transport and verified '
                'release into the supplied dated placement_region. Its former source location no longer '
                'locates it. Compare the CURRENT candidate RGB/mask and measured location with the released '
                'appearance and destination region, while checking for current contrary evidence.')
        if current_release:
            state['correspondence_subjects']['reference']['verified_release_in_current_capture'] = current_release
            if current_release.get('placement_region'):
                payload['questions'][0]['instructions'] += (
                    ' This same capture already has a verified placement of the reference instance in '
                    'the supplied placement_region. Compare its world XY region with the current candidate '
                    'surface location and RGB. A similar object remaining at a different location conflicts '
                    'with that verified placement. The region is a location constraint, and RGB/mask '
                    'correspondence must still establish which subject is the released instance.')
        state['correspondence_subjects']['reference']['historical_position_scope'] = (
            'This historical position predates the bound physical release. '
            'It is not the expected current location. Identify the released instance in the current images; '
            'the verified release does not by itself accept this candidate correspondence.')
    counterparts = [row for row in state['candidate_locations'] if row['candidate'] != candidate_label
                    and state['current_identity_bindings'].get(row['candidate']) == entity['id']]
    if current and counterparts:
        state['same_capture_measurement_pairs'] = [{
            'current_subject': candidate_label, 'accepted_reference_subject': row['candidate'],
            'reference_camera': row['camera'], 'frame': 'calibrated_world',
            'current_surface_centroid_world_m': current['visible_surface_centroid_world_m'],
            'reference_surface_centroid_world_m': row['visible_surface_centroid_world_m'],
            'surface_centroid_distance_m': round(math.dist(
                current['visible_surface_centroid_world_m'], row['visible_surface_centroid_world_m']), 4),
            'meaning': 'Both measurements are from this capture. Surface sampling varies by view. '
                'Spatial proximity alone does not establish identity.'} for row in counterparts]
    state['projected_surface_correspondences'] = [row for row in state.pop('projected_candidate_surfaces', [])
                                                 if row['source_instance_id'] == entity['id']]
    if state['projected_surface_correspondences']:
        state['projection_evidence_scope'] = (
            'The source surface is measured in an already accepted SAME-capture instance view. '
            'Its calibrated 3D points are projected into the queried CURRENT camera and compared '
            'with the exact supplied candidate mask. High overlap supports correspondence; it is not '
            'an automatic identity assignment. These are partial visible surfaces, not a complete '
            'object silhouette or full-mask IoU. Different faces, occlusion and segmentation boundaries '
            'can reduce coverage. Positive projected-minus-current depth means the source surface is '
            'behind the currently visible surface; negative means in front. Use these measurements '
            'with RGB appearance and separate-instance context for the single identity judgment.')
    if current and historical and not current_matches and not relocated:
        now, before = current['visible_surface_centroid_world_m'], historical['position_world_m']
        state['correspondence_geometry'] = {
            'current_subject': candidate_label, 'historical_reference': reference_label,
            'frame': 'calibrated_world', 'reference_observation_id': historical['observation_id'],
            'current_surface_centroid_world_m': now, 'historical_surface_centroid_world_m': before,
            'centroid_displacement_world_m': [round(a-b, 4) for a, b in zip(now, before, strict=True)],
            'centroid_distance_m': round(math.dist(now, before), 4),
            'meaning': 'Measured surface locations, not complete object centers or an accepted identity. '
                'Viewpoint and occlusion change visible surfaces. Combine with RGB and dated actions.'}
        bounds = current.get('visible_surface_bounds_world_m')
        if bounds:
            state['correspondence_geometry']['historical_centroid_inside_current_surface_bounds'] = all(
                bounds[0][axis] <= before[axis] <= bounds[1][axis] for axis in range(3))
    content[1]['text'] = json.dumps(state, ensure_ascii=False)
