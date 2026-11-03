import numpy as np
import pytest

from luna_mujoco.demo_geometry import fuse_table_plane, supported_surface_points


def test_surface_density_filter_removes_isolated_depth_without_altering_supported_samples():
    grid = plane_points(np.arange(.2, .23, .003), np.arange(.4, .43, .003))
    points = np.vstack((grid, [[.1, .1, .56], [.21, .41, 1.2], [np.nan, .4, .56]]))
    before = points.copy()
    keep = supported_surface_points(points)
    assert keep[:len(grid)].all()
    assert not keep[len(grid):].any()
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(points[keep], grid)


def plane_points(x, y, z=.56):
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))


def test_table_plane_fuses_connected_near_edge_returns_without_bin_wall_or_remote_patch():
    seed = plane_points(np.arange(.171, .761, .005), np.arange(-.4, .401, .005))
    wrist = plane_points(np.arange(.1605, .301, .004), np.arange(-.4, .401, .004))
    wall_y, wall_z = np.meshgrid(np.arange(-.4, .401, .004), np.arange(.52, .601, .004))
    wall = np.column_stack((np.full(wall_y.size, .152), wall_y.ravel(), wall_z.ravel()))
    remote = plane_points(np.arange(.1, .13, .004), np.arange(-.02, .021, .004))
    fused, evidence = fuse_table_plane(seed, {"2": wrist, "1": np.vstack((wall, remote))},
                                      normal=np.array([0., 0., 1.]), offset=-.56)
    assert fused[:, 0].min() == pytest.approx(.1605)
    assert fused[:, 0].max() == pytest.approx(seed[:, 0].max())
    assert evidence["accepted_point_count_by_camera"]["2"] > 100
    assert evidence["accepted_point_count_by_camera"]["1"] == 0
    assert evidence["maximum_footprint_growth_m"] == .02
    measured = {tuple(point) for point in np.vstack((seed, wrist, wall, remote))}
    assert all(tuple(point) in measured for point in fused)


def test_table_fusion_rejects_sparse_same_height_noise():
    seed = plane_points(np.arange(.171, .761, .005), np.arange(-.4, .401, .005))
    noise = np.array([[.155, .38, .56], [.156, -.38, .56]])
    fused, evidence = fuse_table_plane(seed, {"2": noise}, normal=np.array([0., 0., 1.]), offset=-.56)
    np.testing.assert_array_equal(fused, seed)
    assert evidence["accepted_point_count_by_camera"]["2"] == 0
