"""Keep image labels separate from the measured robot-relative pickup order."""

import numpy as np


def robot_distance_order(objects):
    """Rank measured centers by horizontal distance to the fixed base at world XY=0.

    Input order breaks exact ties. Color, shape, camera pixels and current TCP
    position do not affect proximity. Missing measurements cannot be guessed.
    """
    rows = []
    for obj in objects:
        position = np.asarray(obj.get('visible_centroid_world', obj.get('position')), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('source_distance_unmeasured:' + obj['id'])
        rows.append({'id': obj['id'], 'position_world_m': position.tolist(),
                     'distance_xy_m': float(np.linalg.norm(position[:2]))})
    return sorted(rows, key=lambda row: row['distance_xy_m'])


def head_image_order(objects, head_view):
    entries = []
    heights = []
    for obj in objects:
        identifier = obj['id']
        mask = np.asarray(head_view.get('instance_masks', {}).get(identifier))
        if mask.ndim != 2 or not mask.any():
            raise ValueError('inventory_head_mask_missing:' + identifier)
        rows, columns = np.nonzero(mask)
        entries.append((float(rows.mean()), float(columns.mean()), identifier))
        heights.append(int(rows.max() - rows.min() + 1))
    return _row_order(entries, heights)


def head_rgb_box_order(entities):
    """Initial approximate RGB boxes order the ledger, never metric geometry."""
    entries, heights = [], []
    for entity in entities:
        left, top, right, bottom = entity['bbox']
        entries.append(((top + bottom) / 2, (left + right) / 2, entity['id']))
        heights.append(bottom - top)
    return _row_order(entries, heights)


def _row_order(entries, heights):
    if not entries:
        return []
    tolerance = float(np.median(heights)) / 2
    result = []
    row = []
    anchor = None
    for y, x, identifier in sorted(entries):
        if anchor is not None and y - anchor > tolerance:
            result.extend(identifier for _, identifier in sorted(row))
            row = []
            anchor = None
        if anchor is None:
            anchor = y
        row.append((x, identifier))
    result.extend(identifier for _, identifier in sorted(row))
    return result
