"""Fresh visual evidence for one task item whose wrapper has been removed."""
import base64
import hashlib
import json
import math
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cad_assets import workpiece_height_prior
from .decision_conditions import condition, parse_answers, result_answers
from .demo_bin_geometry import validate_published_bin_components
from .demo_process import completed_product_collection
from .observation_labels import project_placement_regions
from .visual_api import VisualAPI, VisualInventoryError
from .visual_inventory import image_part

WRAPPED_PHASES = {'wrapped_source', 'table_released', 'unpack_pending', 'consumed'}
UNPACKED_PHASES = {'unpacked_confirmed', 'black_held', 'right_bin_released', 'completed'}


def _require(condition, reason):
    if not condition:
        raise VisualInventoryError('lifecycle_' + reason)


def validate_lifecycle(record):
    phase = record.get('phase')
    _require(phase in WRAPPED_PHASES | UNPACKED_PHASES, 'phase_invalid')
    expected = 'wrapped' if phase in WRAPPED_PHASES else 'unpacked'
    _require(record.get('appearance') == expected, 'appearance_phase_mismatch')
    _require(type(record.get('appearance_revision')) is int and record['appearance_revision'] >= 0,
             'appearance_revision_invalid')
    _require(all(record.get(key) in {'A', 'B'} for key in ('to_table_arm', 'to_right_bin_arm')),
             'arm_binding_invalid')
    return record


def lifecycle_entities(entities, task):
    """Overlay current descriptions; the initial RGB inventory remains immutable."""
    current = deepcopy(entities)
    current.update(deepcopy(task.get('product_entities', {})))
    for identifier, record in task.get('item_lifecycle', {}).items():
        validate_lifecycle(record)
        if record['phase'] == 'consumed' or (identifier in task.get('product_entities', {}) and record['phase'] == 'completed'):
            current.pop(identifier, None)
            continue
        _require(identifier in current, 'entity_missing')
        entity = current[identifier]
        unpacked = record['appearance'] == 'unpacked'
        color = 'black' if unpacked else 'pink'
        description = ('black unpacked inner workpiece' if unpacked else 'pink foam wrapped workpiece')
        if record['phase'] == 'wrapped_source' and entity.get('description'):
            initial_description = entity.setdefault('initial_visual_description', entity['description'])
            description += '; initial visual description: ' + initial_description
        location = ('inside the right blue destination bin' if record['phase'] in
                    {'right_bin_released', 'completed'} else
                    'on the large working table in the right workspace' if unpacked else
                    'inside the left source bin' if record['phase'] == 'wrapped_source' else
                    'on the central working-table region')
        entity.update(color=color, kind='object', description=f'{description}, {location}',
                      current_appearance=record['appearance'], appearance_revision=record['appearance_revision'],
                      lifecycle_phase=record['phase'],
                      height_prior_m=workpiece_height_prior('black_core' if unpacked else 'pink_wrapped'),
                      segmentation_query={'kind': 'object', 'color': color, 'prompt': color + ' object'},
                      segmentation_prompts=([ 'black object', 'black unpacked object', 'black rectangular object']
                          if unpacked else ['pink object', 'pink foam wrapped object', 'elliptical pink object']))
    collection = completed_product_collection(task)
    if collection:
        current[collection['id']] = collection
    return current


def remaining_wrapped_source_ids(task):
    lifecycle = task.get('item_lifecycle', {})
    return {identifier for identifier in task.get('source_object_ids', [])
            if identifier not in lifecycle or lifecycle[identifier]['phase'] == 'wrapped_source'}


def _validate_action(capture, target_id, lifecycle, action, owner_id, *, source, phase):
    validate_lifecycle(lifecycle)
    _require(lifecycle['phase'] == phase, 'action_phase_invalid')
    _require(bool(owner_id) and action.get('owner_id') == owner_id
             and action.get('target_id') == target_id and bool(target_id)
             and action.get('generation') == capture.get('generation')
             and type(capture.get('generation')) is int
             and action.get('source') == source and action.get('completed') is True
             and bool(action.get('action_id')), 'action_binding_invalid')
    _require(set(capture.get('views', {})) == {'1', '2', '3'}
             and bool(capture.get('observation_id')), 'three_current_views_required')
    current, completed = capture.get('time'), action.get('completed_time')
    _require(all(type(value) in (int, float) and math.isfinite(value) for value in (current, completed))
             and current >= completed, 'capture_before_action')
    acquired = capture.get('captured_perf_counter_ns')
    finished = action.get('completed_perf_counter_ns')
    if finished is not None:
        _require(type(finished) is int and type(acquired) is int and acquired > finished,
                 'capture_before_action')
    else:
        _require(current > completed, 'capture_before_action')


def _current_content(capture, context):
    content = [{'type': 'input_text', 'text': json.dumps(context, ensure_ascii=False)}]
    for camera in ('2', '1', '3'):
        content.extend([{'type': 'input_text', 'text': 'CURRENT RGB camera ' + camera},
                        image_part(capture['views'][camera]['rgb'])])
    return content


def unpack_table_support(observation, table_id):
    """Reuse the initially measured main tabletop for result-image presentation."""
    cached = observation.get('demo_tables', {})
    identifier = cached.get('primary_table_id')
    table = cached.get('tables', {}).get(identifier, {})
    model = table.get('model', {})
    pose, size = np.asarray(model.get('pose'), float), np.asarray(model.get('dims'), float)
    floor = table.get('surface_height_m')
    _require(bool(identifier) and identifier == table_id and bool(cached.get('source_observation_id'))
             and table.get('source') == 'initial_inventory_ransac_table'
             and pose.shape == (7,) and size.shape == (3,)
             and np.isfinite(pose).all() and np.isfinite(size).all() and np.all(size > 0)
             and np.allclose(np.abs(pose[3:]), [1, 0, 0, 0], atol=1e-9)
             and type(floor) in (int, float) and math.isfinite(floor), 'measured_table_support_required')
    lower, upper = pose[:2] - size[:2] / 2, pose[:2] + size[:2] / 2
    return {'center_world_xy_m': pose[:2].tolist(),
        'region_polygon_world_xy_m': [[lower[0], lower[1]], [upper[0], lower[1]],
                                      [upper[0], upper[1]], [lower[0], upper[1]]],
        'floor_z_m': floor, 'source': 'measured_cached_main_table_support',
        'table_id': identifier, 'source_observation_id': cached['source_observation_id']}


def _work_area_zoom(capture, work_area):
    """Magnify a configured support region, without selecting any object pixels."""
    center = np.asarray(work_area.get('center_world_xy_m'), float)
    polygon = np.asarray(work_area.get('region_polygon_world_xy_m'), float)
    floor = work_area.get('floor_z_m')
    _require(work_area.get('source') in {'configured_xy_measured_support', 'measured_cached_main_table_support'}
             and center.shape == (2,) and polygon.shape == (4, 2)
             and np.isfinite(center).all() and np.isfinite(polygon).all()
             and type(floor) in (int, float) and math.isfinite(floor), 'work_area_invalid')
    regions = [{'region_id': 'WORK AREA', 'layout': {'floor_z_m': floor, 'cells': [{
        'slot_id': 1, 'center_world_xy_m': center.tolist(), 'polygon_world_xy_m': polygon.tolist()}]}}]
    view = capture['views']['3']
    projections = project_placement_regions({'views': {'3': view}}, regions, require_visible_support=False)
    if not projections:
        return None, None
    rgb = view['rgb']
    height, width = rgb.shape[:2]
    pixels = np.round(np.asarray(projections[0]['polygon']) * [width, height], 8)
    if not np.isfinite(pixels).all():
        return None, None
    lower, upper = pixels.min(axis=0), pixels.max(axis=0)
    if not np.all(np.minimum(upper, [width, height]) > np.maximum(lower, 0)):
        return None, None
    # The support-plane crop includes a fixed image margin, not an object mask.
    padding = max(12, int(np.ceil(.15 * max(upper - lower))))
    lower = np.maximum(np.floor(lower).astype(int) - padding, 0)
    upper = np.minimum(np.ceil(upper).astype(int) + padding, [width, height])
    size = upper - lower
    scale = min(4., 768. / max(size))
    display_size = np.maximum(np.rint(size * scale).astype(int), 1)
    crop = rgb[lower[1]:upper[1], lower[0]:upper[0]]
    enlarged = np.asarray(Image.fromarray(crop).resize(tuple(display_size), Image.Resampling.NEAREST))
    annotation = None
    if work_area['source'] == 'measured_cached_main_table_support':
        picture = Image.fromarray(enlarged)
        draw = ImageDraw.Draw(picture)
        outline = (pixels - lower) * (display_size / size)
        vertices = [tuple(point) for point in outline]
        draw.line(vertices + [vertices[0]], fill=(255, 220, 0), width=3)
        # Label outside the RGB, so it cannot obscure a workpiece or a bin edge.
        labeled = Image.new('RGB', (picture.width, picture.height + 24), (24, 24, 24))
        labeled.paste(picture, (0, 0))
        ImageDraw.Draw(labeled).text((8, picture.height + 5),
            'YELLOW: measured MAIN TABLE boundary; bins are outside', fill=(255, 220, 0))
        enlarged = np.asarray(labeled)
        annotation = 'yellow outline of cached measured main-table support; label below original RGB'
    encoded = image_part(rgb)['image_url'].split(',', 1)[1]
    metadata = {'image_role': 'current_work_area_zoom', 'source_camera_id': '3',
        'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'source': work_area['source'], 'table_id': work_area.get('table_id'),
        'source_observation_id': work_area.get('source_observation_id'),
        'source_image_sha256': hashlib.sha256(base64.b64decode(encoded)).hexdigest(),
        'source_rgb_size_px': [width, height], 'crop_bounds_xyxy_px': [*lower.tolist(), *upper.tolist()],
        'projected_support_polygon_px': pixels.tolist(), 'display_size_px': display_size.tolist(),
        'scale_xy': (display_size / size).tolist(), 'resampling': 'nearest; pixel magnification only',
        'annotation': annotation, 'output_size_px': [enlarged.shape[1], enlarged.shape[0]],
        'meaning': 'CURRENT camera 3 pixels enlarged around the supplied measured tabletop support. '
            'This is not an object mask, not a new observation, and not proof that an object is present. '
            'Use the three full CURRENT images as well; a crop cannot rule out competitors outside it.'}
    return metadata, image_part(enlarged)


def table_release_reference(capture, evidence, owner_id):
    """Small immutable binding; RGBD stays in the existing private capture."""
    _require(isinstance(capture, dict) and isinstance(evidence, dict)
             and set(capture.get('views', {})) == {'1', '2', '3'}
             and bool(owner_id) and bool(capture.get('observation_id'))
             and capture['observation_id'] == evidence.get('observation_id')
             and type(capture.get('generation')) is int
             and evidence.get('status') == 'at_destination'
             and evidence.get('source') == 'openai_decisions'
             and evidence.get('identity_source') == 'execution_chain'
             and type(evidence.get('confidence')) in (int, float) and .8 <= evidence['confidence'] <= 1.
             and all(evidence.get(key) for key in ('target_id', 'release_feedback_id', 'binding_id', 'request_hash')),
             'table_release_reference_invalid')
    images = {}
    for camera in ('2', '1', '3'):
        rgb = capture['views'][camera].get('rgb')
        _require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
                 and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.size > 0,
                 'table_release_reference_invalid')
        encoded = image_part(rgb)['image_url'].split(',', 1)[1]
        images[camera] = {'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'image_sha256': hashlib.sha256(base64.b64decode(encoded)).hexdigest(), 'shape': list(rgb.shape)}
    return {'source': 'confirmed_table_release_rgb', 'owner_id': owner_id,
        **{key: capture.get(key) for key in ('observation_id', 'generation', 'time', 'captured_perf_counter_ns')},
        **{key: evidence[key] for key in ('target_id', 'release_feedback_id', 'binding_id', 'request_hash')},
        'images': images}


def validate_table_release_reference(before, reference, capture, table, action, owner_id):
    _require(isinstance(reference, dict) and isinstance(before, dict), 'table_release_reference_required')
    expected = table_release_reference(before, table, owner_id)
    _require(reference == expected and before['generation'] == capture.get('generation')
             and before['observation_id'] != capture.get('observation_id')
             and table['target_id'] == action.get('target_id')
             and table['release_feedback_id'] == action.get('table_release_feedback_id'),
             'table_release_reference_mismatch')
    before_time, completed = before.get('time'), action.get('completed_time')
    _require(all(type(value) in (int, float) and math.isfinite(value) for value in (before_time, completed))
             and before_time < completed, 'table_release_reference_time_invalid')
    finished = action.get('completed_perf_counter_ns')
    if finished is not None:
        _require(type(before.get('captured_perf_counter_ns')) is int
                 and before['captured_perf_counter_ns'] < finished, 'table_release_reference_time_invalid')


def unpack_verification_capture(capture):
    """Bind an action-result RGB frame independently from a later mapping frame."""
    _require(isinstance(capture, dict) and bool(capture.get('observation_id'))
             and type(capture.get('generation')) is int
             and type(capture.get('time')) in (int, float) and math.isfinite(capture['time'])
             and (capture.get('captured_perf_counter_ns') is None
                  or type(capture['captured_perf_counter_ns']) is int)
             and set(capture.get('views', {})) == {'1', '2', '3'}, 'unpack_capture_reference_invalid')
    images = {}
    for camera in ('2', '1', '3'):
        rgb = capture['views'][camera].get('rgb')
        _require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
                 and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.size > 0,
                 'unpack_capture_reference_invalid')
        encoded = image_part(rgb)['image_url'].split(',', 1)[1]
        images[camera] = {'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'image_sha256': hashlib.sha256(base64.b64decode(encoded)).hexdigest(), 'shape': list(rgb.shape)}
    return {'source': 'post_unpack_current_rgb',
        **{key: capture.get(key) for key in ('observation_id', 'generation', 'time', 'captured_perf_counter_ns')},
        'images': images}


def unpack_result_questions():
    return [condition('unpack_result', 'In the CURRENT views, a black workpiece is visible on the main '
        'tabletop, outside both blue bins. It may be partly occluded by an open gripper; judge its exposed '
        'body using the views together. Black robot components and workpieces inside a blue bin do not count.')]


def unpack_result_evidence(body, minimum_confidence):
    return result_answers(body, 'unpack_result', minimum_confidence)


def unpack_result_context():
    return {
        'task': 'Unwrap one pink foam-wrapped workpiece at a time, then move its black inner part into the right blue bin.',
        'stage': 'BEFORE images show the verified pink workpiece placed on the main table. '
                 'CURRENT images were taken after the unpacking action. Its result must be checked visually; '
                 'the action record alone does not establish success.',
        'views': 'Use the three CURRENT camera views together. They show one scene; repeated views of the same '
                 'object count once. A clear view in one camera is enough; the others may be occluded. '
                 'BEFORE images are historical references and are not current objects.',
        'scope': 'Judge workpieces on the main tabletop. Black robot or gripper parts and black workpieces '
                 'already delivered into the right blue bin do not count. The head detail repeats the current '
                 'head image; its yellow line marks the previously measured table boundary.',
        'appearance': 'The supplied black CAD workpiece is an irregular solid part with protrusions, not '
                      'necessarily a rectangular block. A workpiece near or partly behind an open gripper '
                      'still counts if its exposed body is distinguishable from the robot. Do not require '
                      'a visible air gap between the workpiece and fingers.',
    }


def verify_unpacking(client, capture, *, target_id, lifecycle, action_evidence, owner_id, work_area,
                     table_release_capture=None, table_release_reference=None, active_target_ids=None):
    """Confirm the expected appearance transition; do not mutate state or build geometry."""
    _validate_action(capture, target_id, lifecycle, action_evidence, owner_id,
                     source='mock_vla', phase='unpack_pending')
    _require(active_target_ids == [target_id], 'single_active_target_required')
    table = lifecycle.get('table_visual_evidence', {})
    confidence = table.get('confidence')
    _require(bool(action_evidence.get('table_release_feedback_id'))
             and lifecycle.get('table_release_feedback_id') == action_evidence['table_release_feedback_id']
             and table.get('release_feedback_id') == action_evidence['table_release_feedback_id']
             and table.get('target_id') == target_id and table.get('status') == 'at_destination'
             and table.get('identity_source') == 'execution_chain'
             and type(confidence) in (int, float) and client.minimum_confidence <= confidence <= 1.
             and bool(table.get('request_hash')) and bool(table.get('binding_id')), 'table_release_required')
    validate_table_release_reference(table_release_capture, table_release_reference, capture,
                                     table, action_evidence, owner_id)
    verification_capture = unpack_verification_capture(capture)
    zoom, enlarged = _work_area_zoom(capture, work_area)
    # Cache isolation is an audit marker, not a demand for RGB to verify IDs/hashes.
    scope = {'target_id': target_id, 'owner_id': owner_id, 'generation': capture['generation'],
        'active_target_ids': list(active_target_ids), 'action_id': action_evidence['action_id'],
        'table_release_reference': table_release_reference}
    marker = hashlib.sha256(json.dumps({**scope, 'verification_capture': verification_capture},
                                      sort_keys=True).encode()).hexdigest()
    context = dict(unpack_result_context(), observation_id=capture['observation_id'], audit_marker=marker)
    questions = unpack_result_questions()
    content = _current_content(capture, context)
    before_parts = []
    for camera in ('2', '1', '3'):
        before_parts.extend([{'type': 'input_text',
                              'text': 'BEFORE UNPACKING, AFTER VERIFIED TABLE RELEASE RGB camera ' + camera},
                             image_part(table_release_capture['views'][camera]['rgb'])])
    content[1:1] = before_parts
    if zoom is not None:
        content.extend([{'type': 'input_text', 'text': json.dumps(zoom)}, enlarged])
    payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}],
               'questions': questions}

    def parse(body, evidence):
        parse_answers(body, questions, client.minimum_confidence)
        views = unpack_result_evidence(body, client.minimum_confidence)
        success = views['judgments'] == {'unpack_result': 'yes'}
        visual = {**evidence, **views, 'status': 'confirmed' if success else 'unknown',
                  'target_id': target_id, 'action_id': action_evidence['action_id'],
                  'identity_source': 'verified_single_active_execution_chain', 'visibility_source': 'openai_decisions',
                  'identity_scope': 'process_input_only_not_product_identity',
                  'execution_chain': {key: deepcopy(value) for key, value in scope.items()
                                      if key != 'table_release_reference'},
                  'table_release_request_hash': table['request_hash'], 'table_release_binding_id': table['binding_id'],
                  'table_release_reference': deepcopy(table_release_reference),
                  'verification_capture': deepcopy(verification_capture),
                  'work_area_zoom': deepcopy(zoom)}
        return {**evidence, 'success': success, 'status': visual['status'], 'target_id': target_id,
                'generation': capture['generation'], 'appearance': 'unpacked' if success else lifecycle['appearance'],
                'appearance_revision': lifecycle['appearance_revision'] + int(success),
                'action_id': action_evidence['action_id'], 'visual_evidence': visual,
                'verification_capture': deepcopy(verification_capture),
                'response': deepcopy(body), 'geometry_updated': False}

    return VisualAPI.request(client, payload, capture, 'verify_unpacking', parse)


def _drop_bin_zooms(capture, reference, destination):
    """Locate the same measured bin in two calibrated views; never crop from a goal pose."""
    region = reference.get('drop_bin_region')
    if region is None:
        return [], []
    low, high = (np.asarray(region.get(key), float) for key in ('minimum_world_m', 'maximum_world_m'))
    _require(region.get('source') == 'published_inventory_container_occupancy_bounds'
             and region.get('entity_id') == destination
             and region.get('observation_id') == reference['observation_id']
             and region.get('generation') == reference['generation'] == capture['generation']
             and isinstance(region.get('scene_token'), str) and bool(region['scene_token'])
             and region.get('layer_name') == 'semantic_instance::' + destination
             and region.get('bounds_source') in {'all_native_surface_cells_before_preview_sampling',
                                                'exact_inventory_native_cells_and_derived_shell_union'}
             and type(region.get('cell_count')) is int and region['cell_count'] > 0
             and type(region.get('voxel_size_m')) in (int, float)
             and math.isfinite(region['voxel_size_m']) and region['voxel_size_m'] > 0
             and low.shape == high.shape == (3,) and np.isfinite([low, high]).all()
             and np.all(high > low), 'drop_bin_region_invalid')
    components = region.get('derived_shell_components', {})
    names = region.get('layer_names')
    native_name = 'semantic_instance::' + destination
    if region['bounds_source'] == 'exact_inventory_native_cells_and_derived_shell_union':
        try:
            validate_published_bin_components(components, destination, region['observation_id'], region['generation'])
        except (TypeError, ValueError) as exc:
            raise VisualInventoryError('lifecycle_drop_bin_region_invalid') from exc
        _require(isinstance(names, list) and all(isinstance(name, str) for name in names)
                 and len(names) == len(components)+1 and set(names) == set(components) | {native_name},
                 'drop_bin_region_invalid')
        # These are the same published component fields checked by the bounds
        # producer; the shared scene token belongs to the containing sidecar.
        for layer in components.values():
            _require(isinstance(layer, dict), 'drop_bin_region_invalid')
            try:
                lower, upper = (np.asarray(layer.get(key), float)
                                for key in ('minimum_world_m', 'maximum_world_m'))
            except (TypeError, ValueError) as exc:
                raise VisualInventoryError('lifecycle_drop_bin_region_invalid') from exc
            _require(lower.shape == upper.shape == (3,) and np.isfinite([lower, upper]).all()
                     and np.all(upper > lower) and np.all(lower >= low-1e-9)
                     and np.all(upper <= high+1e-9), 'drop_bin_region_invalid')
    else:
        _require(not components and (names is None or names == [native_name]), 'drop_bin_region_invalid')
    polygon = [[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]
    regions = [{'region_id': 'BIN REGION', 'layout': {'floor_z_m': float(z), 'cells': [{
        'slot_id': 1, 'center_world_xy_m': ((low[:2] + high[:2]) / 2).tolist(),
        'polygon_world_xy_m': polygon}]}} for z in (low[2], high[2])]
    parts, evidence = [], []
    frames = ((role, frame, camera) for camera in ('1', '3')
              for role, frame in (('BEFORE', reference), ('CURRENT', capture)))
    for role, frame, camera in frames:
        view = frame['views'][camera]
        projected = project_placement_regions({'views': {camera: view}}, regions, require_visible_support=False)
        if len(projected) != 2:
            continue
        rgb = view['rgb']
        height, width = rgb.shape[:2]
        pixels = np.asarray([p['polygon'] for p in projected]).reshape(-1, 2) * [width, height]
        if not np.isfinite(pixels).all():
            continue
        lower, upper = pixels.min(0), pixels.max(0)
        if not np.all(np.minimum(upper, [width, height]) > np.maximum(lower, 0)):
            continue
        padding = max(12, int(np.ceil(.08 * max(upper - lower))))
        lower = np.maximum(np.floor(lower).astype(int) - padding, 0)
        upper = np.minimum(np.ceil(upper).astype(int) + padding, [width, height])
        size = upper - lower
        scale = min(4., 740. / max(size))
        display_size = np.maximum(np.rint(size * scale).astype(int), 1)
        crop = Image.fromarray(rgb[lower[1]:upper[1], lower[0]:upper[0]])
        enlarged = crop.resize(tuple(display_size), Image.Resampling.NEAREST)
        labelled = Image.new('RGB', (max(int(display_size[0]), 256), int(display_size[1]) + 28), '#10212d')
        labelled.paste(enlarged, (0, 28))
        ImageDraw.Draw(labelled).text((6, 4), role + ' | BIN REGION | camera ' + camera,
                                     font=ImageFont.load_default(size=14), fill='white')
        encoded = image_part(rgb)['image_url'].split(',', 1)[1]
        metadata = {'image_role': 'drop_bin_region_zoom', 'frame_role': role, 'label': 'BIN REGION',
            'source_camera_id': camera, 'observation_id': frame['observation_id'], 'generation': frame['generation'],
            'measured_region': {key: deepcopy(value) for key, value in region.items()
                                if key not in {'derived_shell_components', 'layer_names'}},
            'source_rgb_size_px': [width, height],
            'source_image_sha256': hashlib.sha256(base64.b64decode(encoded)).hexdigest(),
            'crop_bounds_xyxy_px': [*lower.tolist(), *upper.tolist()], 'display_size_px': display_size.tolist(),
            'label_margin_px': 28, 'resampling': 'nearest; pixel magnification only',
            'meaning': 'Existing RGB pixels around the same previously measured robot-right bin, projected '
                'using each frame camera calibration. This is a locating aid, not an object mask or new measurement, '
                'and not proof of an object being inside. BEFORE and CURRENT viewpoints can differ. '
                'Use all original views too; an unavailable crop does not mean the bin is empty.'}
        parts.extend([{'type': 'input_text', 'text': json.dumps(metadata)}, image_part(np.asarray(labelled))])
        evidence.append(metadata)
    return parts, evidence


def drop_result_evidence(body, minimum_confidence):
    """One fused visual addition result covers presence and delivery progress."""
    result = result_answers(body, 'current_item_added', minimum_confidence)
    judgments = result['judgments']
    confirmed = judgments['current_item_added'] == 'yes'
    return {**result,
            'status': 'at_destination' if confirmed else 'unknown',
            'primary_required_checks': ['current_item_added'],
            'view_scope': 'fused_current_three_rgb'}


def verify_right_bin_drop(client, capture, *, target_id, lifecycle, action_evidence, reference, owner_id):
    """Judge this release against the last measured bin contents, never an old success."""
    _validate_action(capture, target_id, lifecycle, action_evidence, owner_id,
                     source='validated_held_attachment_pose_drop', phase='right_bin_released')
    task = reference.get('demo_task', {})
    destination = task.get('right_bin_id')
    _require(bool(destination) and action_evidence.get('destination_id') == destination
             and bool(action_evidence.get('release_feedback_id'))
             and all(action_evidence.get(key) is True for key in
                     ('holding_verified', 'attachment_verified', 'open_confirmed')), 'drop_chain_invalid')
    _require(bool(reference.get('observation_id'))
             and reference['observation_id'] == action_evidence.get('before_observation_id')
             and reference['observation_id'] != capture['observation_id']
             and reference.get('generation') == capture['generation']
             and type(reference.get('time')) in (int, float) and math.isfinite(reference['time'])
             and reference['time'] <= action_evidence['completed_time']
             and set(reference.get('views', {})) == {'1', '2', '3'}, 'drop_before_reference_invalid')
    prior_ids = [identifier for identifier, item in task.get('item_lifecycle', {}).items()
                 if identifier != target_id and item.get('phase') == 'completed']
    delivery_count = {'initial_right_bin_black_count': 0, 'source': 'user_scene_prior',
                      'already_completed_count': len(prior_ids), 'expected_current_count': len(prior_ids) + 1}
    detail_parts, bin_zooms = _drop_bin_zooms(capture, reference, destination)
    context = {'purpose': 'verify_right_bin_drop', 'observation_id': capture['observation_id'],
        'user_prior': 'Pink foam wraps this same black workpiece; VLA removes packaging; finish one item before the next.',
        'generation': capture['generation'], 'task_id': target_id,
        'appearance_revision': lifecycle['appearance_revision'], 'action_id': action_evidence['action_id'],
        'right_bin_id': destination, 'before_observation_id': reference['observation_id'],
        'already_completed_task_ids': prior_ids, 'delivery_count': delivery_count,
        'recorded_action': 'The current black object was held, transported with an attachment '
            'above the right blue bin, and released by opening the gripper. This is execution context, '
            'not proof of its final location.',
        'rules': 'Judge visible black workpieces, without requiring hidden core or packaging properties. '
            'A clear view in any one CURRENT image is sufficient; other cameras need not see it. '
            'Robot-right identifies the designated bin, not the right side of every image. '
            'BIN REGION detail views locate that same measured bin in BEFORE and CURRENT; camera poses can differ. '
            'The user specifies that this bin initially contained zero black objects. The task record gives '
            'the number of completed deliveries; this is context, not proof of the CURRENT visible count. '
            'Compare BEFORE and CURRENT contents of that same bin. Free-fall delivery can stack or overlap '
            'workpieces, so separately counting every completed item is not required. Look for an additional '
            'visible workpiece body or additional protrusions/body extent consistent with another workpiece '
            'in the pile. Use camera 3 for a consistent head-view comparison when available. '
            'Account for camera perspective: viewpoint change or displacement/rotation of the old object '
            'alone does not prove an addition. Earlier objects alone and action records alone are insufficient. '
            'Robot parts, shadows, and objects outside the bin do not count. If the new body is completely '
            'hidden and the comparison gives no positive visual evidence, answer uncertain. '
            'No individual dropped-object identity, precise pose, or physical-contact measurement is requested.'}
    content = _current_content(capture, context)
    for camera in ('2', '1', '3'):
        content.extend([{'type': 'input_text', 'text': 'BEFORE this transport, RGB camera ' + camera},
                        image_part(reference['views'][camera]['rgb'])])
    content.extend(detail_parts)
    questions = [condition('current_item_added',
                        'Using all CURRENT views together and comparison with BEFORE, an additional black workpiece '
                        'has visibly arrived inside the same designated BIN REGION, the robot-right destination box. '
                        'Workpieces may overlap or stack: an added visible body or added workpiece features in the '
                        'pile can establish delivery without separately counting all previous pieces. '
                        'An unchanged old object, its movement/rotation alone, or viewpoint change alone is insufficient. '
                        'Exclude robot parts, shadows, and objects on the table or outside this bin. '
                        'If occlusion leaves no positive visual evidence of addition, remain uncertain. '
                        'Historical counts and action records alone do not prove arrival. '
                        'No exact pose or identity of individual dropped pieces is required.')]
    payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': questions}

    def parse(body, evidence):
        parse_answers(body, questions, client.minimum_confidence)
        result = drop_result_evidence(body, client.minimum_confidence)
        success = result['status'] == 'at_destination'
        visual = {**evidence, **result, 'target_id': target_id,
                  'destination_id': destination, 'release_feedback_id': action_evidence['release_feedback_id'],
                  'action_id': action_evidence['action_id'], 'before_observation_id': reference['observation_id'],
                  'identity_source': 'execution_chain_and_fresh_drop_comparison',
                  'completion_semantics': 'released_in_bin_and_visible_addition',
                  'delivery_count': deepcopy(delivery_count), 'already_completed_task_ids': list(prior_ids),
                  'bin_region_geometry': deepcopy(reference.get('drop_bin_region')),
                  'bin_region_zooms': bin_zooms}
        return {**evidence, 'success': success, 'status': 'confirmed' if success else 'unknown',
                'target_id': target_id, 'generation': capture['generation'], 'visual_evidence': visual,
                'response': deepcopy(body), 'geometry_updated': False}

    return VisualAPI.request(client, payload, capture, 'verify_right_bin_drop', parse)
