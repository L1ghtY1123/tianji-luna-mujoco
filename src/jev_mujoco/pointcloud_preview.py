"""Bounded display samples from measured map input, never planning geometry."""
import numpy as np


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
