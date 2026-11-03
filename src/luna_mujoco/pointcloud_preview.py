"""Bounded display samples from measured map input, never planning geometry."""
from copy import deepcopy

import numpy as np


def observed_obb_preview(observation):
    """Export existing fitted boxes, never fit geometry from a display sample."""
    from .object_obb import observed_object_obb

    rows = []
    for obj in observation.get('objects', []):
        if obj.get('obb') is None:
            continue
        obb = observed_object_obb(obj)
        provenance = obb['observation_provenance']
        if any(provenance.get(key) != observation[key] for key in ('observation_id', 'generation')):
            raise ValueError('obb_preview_capture_binding_changed')
        rows.append({'instance_id': obj['id'], 'kind': obj.get('kind'), 'obb': obb,
                     'observation_id': observation['observation_id'], 'generation': observation['generation'],
                     'status': 'current'})
    return rows


def reconcile_obb_preview(previous, current, identity):
    """Follow map ownership/freshness; this is not a fused object shape model."""
    older = {row['instance_id']: row for row in previous if row['generation'] == identity['generation']}
    incoming = {row['instance_id']: row for row in current if row['generation'] == identity['generation']}
    result = []
    for identifier, record in identity.get('inventory_map', {}).get('entities', {}).items():
        refreshed = (record.get('status') == 'observed'
                     and record.get('last_observation_id') == identity['observation_id'])
        row = incoming.get(identifier) if refreshed else older.get(identifier)
        # Missing exports must not relabel an older box as a current measurement.
        if row is None or row['observation_id'] != record.get('last_observation_id'):
            continue
        result.append(deepcopy(row) | {'status': 'current' if refreshed else
                                      'stale' if record.get('status') == 'stale' else 'retained'})
    return result


def pointcloud_preview(points, keep, labels=None, *, maximum_points=30000):
    points = np.asarray(points, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    if points.shape != (len(keep), 3) or not np.isfinite(points).all():
        raise ValueError("preview_requires_aligned_measured_points")
    if labels is not None and np.asarray(labels).shape != keep.shape:
        raise ValueError("preview_cluster_labels_not_aligned")
    selected = np.arange(len(points))
    if len(selected) > maximum_points:
        rejected = np.flatnonzero(~keep)
        retained = np.flatnonzero(keep)
        rejected_count = min(len(rejected), maximum_points // 2)
        kept_count = min(len(retained), maximum_points - rejected_count)
        rejected_count = min(len(rejected), maximum_points - kept_count)
        def sample(indices, count):
            return indices[np.linspace(0, len(indices) - 1, count, dtype=int)] if count else indices[:0]
        selected = np.sort(np.concatenate((sample(rejected, rejected_count), sample(retained, kept_count))))
    return {"source": "registered_rgbd", "counts": {"input": len(points), "kept": int(keep.sum()),
             "rejected": int((~keep).sum())}, "total_point_count": len(points),
            "points_world_m": points[selected].tolist(), "kept_mask": keep[selected].tolist(),
            "cluster_labels": None if labels is None else np.asarray(labels)[selected].tolist(),
            "original_point_indices": selected.tolist(), "preview_sampling": "stratified_original_indices"}
