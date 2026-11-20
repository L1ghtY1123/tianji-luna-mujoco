"""Same-image overlap measurements. These numbers never assign an identity."""
import numpy as np


def _mask(value):
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.ndim != 2 or not np.any(result):
        raise ValueError('nonempty_2d_boolean_mask_required')
    return result


def box_mask_metrics(box_xyxy_normalized, mask):
    mask = _mask(mask)
    box = np.asarray(box_xyxy_normalized, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0)
            or np.any(box > 1) or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError('invalid_normalized_box')
    height, width = mask.shape
    box = box * [width, height, width, height]
    rows, cols = np.nonzero(mask)
    tight = np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], dtype=float)
    extent = np.maximum(0., np.minimum(box[2:], tight[2:]) - np.maximum(box[:2], tight[:2]))
    intersection = float(np.prod(extent))
    box_area = float(np.prod(box[2:] - box[:2]))
    mask_box_area = float(np.prod(tight[2:] - tight[:2]))
    inside_x = (np.arange(width) + .5 >= box[0]) & (np.arange(width) + .5 < box[2])
    inside_y = (np.arange(height) + .5 >= box[1]) & (np.arange(height) + .5 < box[3])
    inside = inside_y[:, None] & inside_x[None, :]
    pixel_area = int(np.count_nonzero(inside))
    overlap = int(np.count_nonzero(mask & inside))
    return {'box_iou': intersection / (box_area + mask_box_area - intersection),
            'mask_coverage': overlap / len(rows),
            'box_fill': overlap / pixel_area if pixel_area else None,
            'box_fill_status': 'valid' if pixel_area else 'no_pixel_centers',
            'mask_box_xyxy_px': tight.tolist()}


def mask_pair_metrics(left_mask, right_mask):
    left, right = _mask(left_mask), _mask(right_mask)
    if left.shape != right.shape:
        raise ValueError('same_image_dimensions_required')
    intersection = int(np.count_nonzero(left & right))
    left_area, right_area = int(np.count_nonzero(left)), int(np.count_nonzero(right))
    return {'mask_iou': intersection / (left_area + right_area - intersection),
            'left_coverage': intersection / left_area, 'right_coverage': intersection / right_area,
            'exact_alias': bool(np.array_equal(left, right))}
