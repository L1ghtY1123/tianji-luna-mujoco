"""Present current measured instance locations without deciding semantic roles."""
import hashlib
import json

import numpy as np

from .observation_labels import raster_visual_annotations
from .visual_api import VisualInventoryError
from .visual_inventory import image_part, mask_box


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def current_instance_locations(observation):
    labels = {identifier: 'ID' + str(index + 1) for index, identifier in enumerate(
        sorted(obj['id'] for obj in observation.get('objects', [])))}
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


def current_instance_location_content(capture, locations):
    if not locations:
        return []
    if any(locations.get(key) != capture.get(key) for key in ('observation_id', 'generation')):
        raise VisualInventoryError('task_binding_locations_stale')
    content = []
    for camera, measured in locations['views'].items():
        view = capture.get('views', {}).get(camera, {})
        if (not isinstance(view.get('rgb'), np.ndarray)
                or view.get('frame_id') != measured['frame_id']
                or view.get('calibration_id') != measured['calibration_id']
                or view.get('simulation_generation') != capture['generation']
                or view.get('camera_id') != camera or _digest(view['rgb']) != measured['rgb_sha256']):
            raise VisualInventoryError('task_binding_locations_stale')
        metadata = {'image_role': 'current_measured_instance_locations', 'camera_id': camera,
            'observation_id': capture['observation_id'], 'generation': capture['generation'],
            'rgb_sha256': measured['rgb_sha256'], 'items': measured['items'],
            'meaning': 'Locating boxes from CURRENT identity-associated measured masks. ID labels map to permanent '
                       'IDs in items. They identify which current visible subject a predicate names; '
                       'they do not certify task membership, requested destination, free space or success. '
                       'Use the unchanged original RGB and original instruction to judge those predicates.'}
        content.extend([{'type': 'input_text', 'text': json.dumps(metadata)},
            image_part(raster_visual_annotations(view['rgb'], measured['items'],
                frame_color=(225, 225, 225), label_layout='near_box'))])
    return content
