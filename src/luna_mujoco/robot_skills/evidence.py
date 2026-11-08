"""Immutable image references and calibrated result presentation helpers."""
import base64
import hashlib
import math

import numpy as np

from ..visual_api import VisualInventoryError
from ..visual_inventory import image_part


def _require(condition, reason):
    if not condition:
        raise VisualInventoryError('result_evidence_' + reason)


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
