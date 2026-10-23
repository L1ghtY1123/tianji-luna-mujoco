import numpy as np
import pytest
from test_perception import bin_cloud

from luna_mujoco.perception import measured_bin_reference
from luna_mujoco.placement import measured_source_relationships, region_contains_xy


@pytest.mark.parametrize('partial', [False, True])
def test_floor_plane_extent_survives_occlusion_but_does_not_expand_its_outer_boundary(partial):
    x, y = np.meshgrid(np.arange(-.07, .071, .002), np.arange(-.06, .061, .002))
    valid = (x > .005) if partial else (np.hypot(x, y) >= .018)
    floor = np.column_stack((x[valid], y[valid], np.full(valid.sum(), .61)))
    reference = measured_bin_reference(np.vstack((floor, bin_cloud()[121:])),
        {'surface_height_m': .6, 'uncertainty_m': .003})
    polygon = reference['region_polygon_world_xy_m']
    assert len(polygon) == 4
    assert bool(region_contains_xy([[0., 0.]], polygon)[0]) is not partial
    assert reference['region_measurement']['method'] == 'plane_hull_inscribed_rectangle'
    if partial:
        assert np.asarray(polygon)[:, 0].min() >= .005


def test_source_membership_uses_observed_floor_extent_not_free_placement_patch():
    x, y = np.meshgrid(np.arange(-.07, .071, .002), np.arange(-.06, .061, .002))
    visible = np.hypot(x, y) >= .018
    floor = np.column_stack((x[visible], y[visible], np.full(visible.sum(), .61)))
    reference = measured_bin_reference(np.vstack((floor, bin_cloud()[121:])),
        {'surface_height_m': .6, 'uncertainty_m': .003})
    source = {'points_world_m': [[0., 0., .64], [.005, 0., .64]]}
    relationships = measured_source_relationships(source, [{'id': 'bin', 'kind': 'bin', 'placement_reference': reference}])
    assert relationships and relationships[0]['visible_xy_fraction'] == 1.
    assert relationships[0]['complete_containment_verified'] is False
    assert region_contains_xy([[0., 0.]], reference['region_polygon_world_xy_m'])[0]


def test_complementary_planar_patches_use_their_shared_measured_extent():
    floor = []
    for dx, dy in ((-.04, -.04), (-.04, .04), (.04, -.04), (.04, .04)):
        floor.extend([[x + dx, y + dy, .61] for x in np.arange(-.008, .009, .002)
                      for y in np.arange(-.008, .009, .002)])
    points = np.vstack((floor, bin_cloud()[121:]))
    reference = measured_bin_reference(points, {'surface_height_m': .6, 'uncertainty_m': .003})
    assert reference is not None
    polygon = np.asarray(reference['region_polygon_world_xy_m'])
    assert region_contains_xy([[0., 0.]], polygon)[0]
    assert np.max(np.abs(polygon)) <= .048 + 1e-9
