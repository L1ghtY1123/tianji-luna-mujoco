"""Consume the official cuRobo mask bound to this exact RGBD observation."""
import numpy as np


def own_robot_surface_pixels(capture, masks, *, mask_only=False):
    result = {'status': 'measured', 'source': 'curobo.perception.RobotSegmenter.get_robot_mask',
              'masks': {}, 'matches': {}, 'unresolved_near_robot_masks': {}}
    for camera, selected in masks.items():
        view = capture['views'][camera]
        if view.get('robot_mask_source') != result['source'] or 'robot_mask' not in view:
            raise ValueError('capture_curobo_robot_mask_missing:' + camera)
        mask = np.asarray(view['robot_mask'], dtype=bool)
        if mask.shape != selected.shape:
            raise ValueError('capture_curobo_robot_mask_shape_mismatch')
        result['masks'][camera] = mask & selected
    return result
