
import numpy as np

from luna_mujoco.measured_regions import supported_floor_rectangle
from luna_mujoco.placement import region_contains_xy


def test_complementary_cameras_do_not_shrink_floor_to_the_densest_strip():
    coarse_x, coarse_y = np.meshgrid(np.arange(.40, .561, .002), np.arange(.13, .331, .002))
    dense_x, dense_y = np.meshgrid(np.arange(.36, .401, .001), np.arange(.13, .331, .001))
    points = np.vstack((np.c_[coarse_x.ravel(), coarse_y.ravel()],
                        np.c_[dense_x.ravel(), dense_y.ravel()]))
    region = supported_floor_rectangle(points)
    polygon = np.asarray(region['polygon_world_xy_m'])
    assert np.ptp(polygon[:, 0]) >= .19
    assert np.ptp(polygon[:, 1]) >= .19
    assert region_contains_xy([[.46, .23]], polygon)[0]
