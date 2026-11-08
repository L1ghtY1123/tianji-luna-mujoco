"""Present current measured instance locations without deciding semantic roles."""
import hashlib
import json
import math

import numpy as np

from .observation_labels import raster_visual_annotations
from .visual_api import VisualInventoryError
from .visual_inventory import image_part, mask_box


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def current_instance_locations(observation):
    labels = {obj['id']: 'ID' + obj['id'] for obj in observation.get('objects', [])}
    views = {}
    for camera, view in observation.get('views', {}).items():
        items = []
        for identifier, mask in view.get('instance_masks', {}).items():
            if identifier not in labels:
                continue
            rgb = view.get('rgb')
            if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                    or rgb.shape[2] != 3 or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
                    or mask.shape != rgb.shape[:2] or not mask.any()):
                raise VisualInventoryError('task_binding_mask_invalid')
            items.append({'id': identifier, 'image_label': labels[identifier],
                'box': mask_box(mask), 'mask_sha256': _digest(mask)})
        if items:
            if (view.get('frame_id') != observation['observation_id']
                    or view.get('simulation_generation') != observation['generation']
                    or view.get('camera_id') != camera or not view.get('calibration_id')):
                raise VisualInventoryError('task_binding_locations_stale')
            views[camera] = {'items': items, 'rgb_sha256': _digest(view['rgb']),
                'frame_id': view['frame_id'], 'calibration_id': view['calibration_id']}
    return ({'observation_id': observation['observation_id'], 'generation': observation['generation'],
             'views': views} if views else {})


def current_instance_location_content(capture, locations, *, compact=False, table_regions=()):
    if not locations:
        return []
    if any(locations.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
        raise VisualInventoryError('task_binding_locations_stale')
    content = []
    cameras = list(dict.fromkeys([*locations['views'], *(row['camera_id'] for row in table_regions)]))
    for camera in cameras:
        measured = locations['views'].get(camera)
        view = capture.get('views', {}).get(camera, {})
        if (not isinstance(view.get('rgb'), np.ndarray)
                or view.get('simulation_generation') != capture['generation'] or view.get('camera_id') != camera
                or measured is not None and (
                    view.get('frame_id') != measured['frame_id']
                    or view.get('calibration_id') != measured['calibration_id']
                    or _digest(view['rgb']) != measured['rgb_sha256'])):
            raise VisualInventoryError('task_binding_locations_stale')
        items = measured['items'] if measured is not None else []
        regions = [row for row in table_regions if row['camera_id'] == camera]
        metadata = {'image_role': 'current_measured_instance_locations', 'camera_id': camera,
            'observation_id': capture['observation_id'], 'generation': capture['generation'],
            'items': [{key: item[key] for key in ('id', 'image_label', 'box')} for item in items],
            'meaning': 'Locating boxes from CURRENT identity-associated measured masks. ID labels map to permanent '
                       'IDs in items. They identify which current visible subject a predicate names; '
                       'they do not certify task membership, requested destination, free space or success. '
                       'Use these complete current views and the original instruction to judge those predicates.'}
        if compact:
            metadata = {'camera_id': camera, 'items': [
                {key: item[key] for key in ('image_label', 'box')} for item in items]}
        if regions:
            metadata['table_reference_polygons'] = regions
            metadata['table_reference_meaning'] = (
                'Dated measured table bounds projected with current camera calibration locate the named support. '
                'The polygons retain their measurement date; current RGB supplies the visual relation evidence.')
        content.extend([{'type': 'input_text', 'text': json.dumps(metadata)},
            image_part(raster_visual_annotations(view['rgb'], items, regions=regions,
                frame_color=(225, 225, 225), label_layout='near_box'))])
    return content


def current_subject_detail_content(rgb, box, label, camera):
    """A same-capture close-up retains the subject's immediate support context."""
    height, width = rgb.shape[:2]
    left, top, right, bottom = box
    pad_x, pad_y = (right - left) / 2, (bottom - top) / 2
    context_box = [max(0., left - pad_x), max(0., top - pad_y),
                   min(1., right + pad_x), min(1., bottom + pad_y)]
    left, top, right, bottom = context_box
    detail = rgb[math.floor(top * height):math.ceil(bottom * height),
                 math.floor(left * width):math.ceil(right * width)]
    return [{'type': 'input_text', 'text': json.dumps({
        'image_role': 'current_subject_detail', 'candidate': label,
        'camera': camera, 'box': box, 'context_box': context_box, 'mask_contours_supplied': False,
        'meaning': 'Unannotated CURRENT RGB crop around this candidate, including nearby context. '
            'The full view supplies the exact outlined mask when available. Compare the physical '
            'object and its surroundings; crop boundaries do not represent the object boundary.'})}, image_part(detail)]
