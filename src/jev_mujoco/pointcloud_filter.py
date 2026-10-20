"""Optional Open3D filtering of measured points, preserving original indices.

The adapter never edits coordinates, sources or depth. The mapper must suppress
rejected depth rays as well as their occupied points. DBSCAN labels are diagnostic
and never determine occupancy. Dense unsupported islands may survive filtering.
"""

import numpy as np

DEFAULT_SETTINGS = {
    "enabled": False, "method": "statistical", "nb_neighbors": 20, "std_ratio": 2.0,
    "radius_m": .01, "min_neighbors": 4, "dbscan_preview": False,
    "dbscan_eps_m": .02, "dbscan_min_points": 10,
}


class PointCloudFilterError(RuntimeError):
    """Invalid measurement/settings or an unavailable optional dependency."""


def filter_observed_points(points_world, point_sources, *, settings):
    """Return original-index keep/rejected masks, optional labels and counts.

    ``points_world`` is finite Nx3 metres. ``point_sources`` is an aligned
    sequence, or a dict of aligned camera_ids and pixels_rc arrays. Settings
    override DEFAULT_SETTINGS. Cluster labels refer to the original input;
    preview does not remove noise (-1) or any other cluster.
    """
    points = np.asarray(points_world)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PointCloudFilterError("pointcloud_filter_invalid_points")
    count = len(points)
    if isinstance(point_sources, dict):
        if ("camera_ids" not in point_sources or "pixels_rc" not in point_sources
                or len(point_sources["camera_ids"]) != count
                or np.asarray(point_sources["pixels_rc"]).shape != (count, 2)):
            raise PointCloudFilterError("pointcloud_filter_source_alignment_mismatch")
    elif len(point_sources) != count:
        raise PointCloudFilterError("pointcloud_filter_source_alignment_mismatch")
    if not isinstance(settings, dict) or set(settings) - DEFAULT_SETTINGS.keys():
        raise PointCloudFilterError("pointcloud_filter_invalid_settings")
    config = DEFAULT_SETTINGS | settings
    if (type(config["enabled"]) is not bool or type(config["dbscan_preview"]) is not bool
            or config["method"] not in {"statistical", "radius"}):
        raise PointCloudFilterError("pointcloud_filter_invalid_settings")
    for key in ("nb_neighbors", "min_neighbors", "dbscan_min_points"):
        if type(config[key]) is not int or config[key] < 1:
            raise PointCloudFilterError("pointcloud_filter_invalid_settings")
    for key in ("std_ratio", "radius_m", "dbscan_eps_m"):
        if (isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                or not np.isfinite(config[key]) or config[key] <= 0):
            raise PointCloudFilterError("pointcloud_filter_invalid_settings")
    keep = np.ones(count, dtype=bool)
    labels = None
    if config["enabled"] or config["dbscan_preview"]:
        try:
            import open3d as o3d
        except ImportError as error:
            raise PointCloudFilterError("open3d_unavailable: install the pointcloud extra") from error
        if count:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
            if config["enabled"]:
                if config["method"] == "statistical":
                    _, indices = cloud.remove_statistical_outlier(
                        nb_neighbors=config["nb_neighbors"], std_ratio=config["std_ratio"])
                    keep[:] = False
                    keep[np.asarray(indices, dtype=np.int64)] = True
                else:
                    # Open3D 0.19's legacy radius API loses valid indices intermittently
                    # with parallel execution on Windows. Use the public tensor API's
                    # original-index mask without changing process-wide threading.
                    tensor_cloud = o3d.t.geometry.PointCloud(
                        o3d.core.Tensor(points.astype(np.float64), dtype=o3d.core.Dtype.Float64))
                    _, mask = tensor_cloud.remove_radius_outliers(
                        nb_points=config["min_neighbors"], search_radius=config["radius_m"])
                    keep[:] = mask.numpy()
            if config["dbscan_preview"]:
                labels = np.asarray(cloud.cluster_dbscan(eps=config["dbscan_eps_m"],
                    min_points=config["dbscan_min_points"], print_progress=False), dtype=np.int64)
        elif config["dbscan_preview"]:
            labels = np.empty(0, dtype=np.int64)
    rejected = ~keep
    return {"keep_mask": keep, "rejected_mask": rejected, "cluster_labels": labels,
            "counts": {"input": count, "kept": int(keep.sum()), "rejected": int(rejected.sum())}}
