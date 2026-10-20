"""SAM silhouette/background depth must not become carried object geometry."""
from copy import deepcopy

import numpy as np
from geometry_perception_fixture import GeometryPerception
from test_perception import FrozenSegmenter, capture_fixture


def test_background_depth_at_source_silhouette_is_not_attached_or_erased():
    capture, masks = capture_fixture()
    # One accepted silhouette pixel sees a distant support, not the red body.
    pixel = (36, 11)
    for view in capture['views'].values():
        view['depth_m'][pixel] = .5
        view['rgb'][pixel] = [100, 60, 120]
    before = deepcopy(capture)
    result = GeometryPerception(FrozenSegmenter(masks)).observe(capture)
    assert len(result['objects']) == 2
    for item in result['objects']:
        np.testing.assert_allclose(item['points_world_m'][:, 2], .7, atol=1e-6)
    for camera, view in result['views'].items():
        assert all(not mask[pixel] for mask in view['instance_masks'].values())
        assert view['depth_m'][pixel] == .5  # Still available to background collision mapping.
        np.testing.assert_array_equal(view['depth_m'], before['views'][camera]['depth_m'])
        np.testing.assert_array_equal(capture['views'][camera]['depth_m'], before['views'][camera]['depth_m'])


def test_silhouette_filter_does_not_expand_interior_depth_holes():
    capture, masks = capture_fixture()
    for view in capture['views'].values():
        for column in (15, 65):
            for row, offset in ((38, -2), (38, 2), (42, -2), (42, 2)):
                view['valid_depth'][row, column + offset] = False
    result = GeometryPerception(FrozenSegmenter(masks)).observe(capture)
    assert len(result['objects']) == 2
    for view in result['views'].values():
        for mask in view['instance_masks'].values():
            assert np.count_nonzero(mask) == 45  # 7x7 interior minus only four invalid returns.
            assert not np.any(mask & ~view['valid_depth'])
