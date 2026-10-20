"""Original-index filtering and lazy optional dependency contracts."""

import importlib
import sys

import numpy as np
import pytest


def filter_module():
    assert importlib.util.find_spec("jev_mujoco.pointcloud_filter") is not None, "optional filter is missing"
    return importlib.import_module("jev_mujoco.pointcloud_filter")


def samples():
    x, y = np.meshgrid(np.linspace(0, .02, 11), np.linspace(0, .02, 11))
    points = np.vstack((np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))), [3, 3, 3]))
    return points, [{"camera_id": "3", "pixel_rc": [i, 0]} for i in range(len(points))]


def test_default_off_retains_all_original_points_and_sources_without_open3d(monkeypatch):
    module = filter_module()
    monkeypatch.setitem(sys.modules, "open3d", None)
    points, sources = samples()
    original = points.copy()
    result = module.filter_observed_points(points, sources, settings={})
    assert result["keep_mask"].all() and not result["rejected_mask"].any()
    assert result["counts"] == {"input": 122, "kept": 122, "rejected": 0}
    assert result["cluster_labels"] is None
    np.testing.assert_array_equal(points, original)
    assert sources[-1]["pixel_rc"] == [121, 0]


def test_enabled_filter_explicitly_fails_without_open3d(monkeypatch):
    module = filter_module()
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(module.PointCloudFilterError, match="open3d_unavailable"):
        module.filter_observed_points(*samples(), settings={"enabled": True})


@pytest.mark.parametrize("method", ["statistical", "radius"])
def test_real_open3d_outlier_mask_indexes_original_unchanged_points(method):
    module = filter_module()
    pytest.importorskip("open3d")
    points, sources = samples()
    original = points.copy()
    result = module.filter_observed_points(points, sources, settings={"enabled": True, "method": method,
        "nb_neighbors": 12, "std_ratio": 2., "radius_m": .006, "min_neighbors": 4})
    assert result["keep_mask"][:121].all()
    assert result["rejected_mask"].tolist() == [False] * 121 + [True]
    np.testing.assert_array_equal(points, original)


def test_radius_filter_repeatedly_preserves_dense_surface_original_indices():
    module = filter_module()
    pytest.importorskip("open3d")
    points, sources = samples()
    # Put the rejected point in the middle so the mask must address original indices.
    order = np.r_[np.arange(60), 121, np.arange(60, 121)]
    points = points[order]
    sources = [sources[index] for index in order]
    expected = order != 121
    for _ in range(10):
        result = module.filter_observed_points(points, sources, settings={
            "enabled": True, "method": "radius", "radius_m": .006, "min_neighbors": 4})
        np.testing.assert_array_equal(result["keep_mask"], expected)
        np.testing.assert_array_equal(result["rejected_mask"], ~expected)
        assert result["counts"] == {"input": 122, "kept": 121, "rejected": 1}


def test_dbscan_preview_does_not_remove_noise_labels_when_filter_disabled():
    module = filter_module()
    pytest.importorskip("open3d")
    result = module.filter_observed_points(*samples(), settings={"dbscan_preview": True,
        "dbscan_eps_m": .006, "dbscan_min_points": 4})
    assert result["keep_mask"].all()
    assert result["cluster_labels"][-1] == -1
    assert (result["cluster_labels"][:121] == 0).all()


def test_filter_rejects_source_alignment_mismatch():
    module = filter_module()
    points, sources = samples()
    with pytest.raises(module.PointCloudFilterError, match="source"):
        module.filter_observed_points(points, sources[:-1], settings={})
