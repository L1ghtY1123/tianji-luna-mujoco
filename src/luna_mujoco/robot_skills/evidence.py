"""Immutable image references and calibrated result presentation helpers."""
import base64
import hashlib
import json
import math
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..decision_conditions import result_answers
from ..observation_labels import project_placement_regions
from ..visual_api import VisualInventoryError
from ..visual_inventory import image_part


def _require(condition, reason):
    if not condition:
        raise VisualInventoryError('result_evidence_' + reason)


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


def validate_work_area(work_area):
    """Require measured support metadata without constructing image overlays."""
    center = np.asarray(work_area.get('center_world_xy_m'), float)
    polygon = np.asarray(work_area.get('region_polygon_world_xy_m'), float)
    floor = work_area.get('floor_z_m')
    _require(work_area.get('source') in {'configured_xy_measured_support', 'measured_cached_main_table_support'}
             and center.shape == (2,) and polygon.shape == (4, 2)
             and np.isfinite(center).all() and np.isfinite(polygon).all()
             and type(floor) in (int, float) and math.isfinite(floor), 'work_area_invalid')


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
             and region.get('bounds_source') == 'all_native_surface_cells_before_preview_sampling'
             and type(region.get('cell_count')) is int and region['cell_count'] > 0
             and type(region.get('voxel_size_m')) in (int, float)
             and math.isfinite(region['voxel_size_m']) and region['voxel_size_m'] > 0
             and low.shape == high.shape == (3,) and np.isfinite([low, high]).all()
             and np.all(high > low), 'drop_bin_region_invalid')
    components = region.get('derived_shell_components', {})
    names = region.get('layer_names')
    native_name = 'semantic_instance::' + destination
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
            'meaning': 'Existing RGB pixels around the same previously measured bound container, projected '
                'using each frame camera calibration. This is a locating aid, not an object mask or new measurement, '
                'and not proof of an object being inside. BEFORE and CURRENT viewpoints can differ. '
                'Use all original views too; an unavailable crop does not mean the bin is empty.'}
        parts.extend([{'type': 'input_text', 'text': json.dumps(metadata)}, image_part(np.asarray(labelled))])
        evidence.append(metadata)
    return parts, evidence


def drop_result_evidence(body, minimum_confidence):
    """Parse the camera's visible target-location verdict, preserving its raw answer."""
    result = result_answers({'answers': [row for row in body.get('answers', [])
        if row.get('name') == 'current_item_added']}, 'current_item_added', minimum_confidence)
    judgments = result['judgments']
    confirmed = judgments['current_item_added'] == 'yes'
    return {**result,
            'status': 'at_destination' if confirmed else 'unknown',
            'primary_required_checks': ['current_item_added'],
            'view_scope': 'fused_current_three_rgb'}
