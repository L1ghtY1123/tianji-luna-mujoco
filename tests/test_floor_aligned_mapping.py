import numpy as np
import pytest

from luna_mujoco.instance_mapping import instance_grid_layout
from luna_mujoco.release_clearance import release_clearance_evidence


@pytest.mark.parametrize('floor', [.60999888, .6163, .6327])
def test_box_floor_is_on_a_cell_center_without_moving_measured_points(floor):
    points = np.array([[.33, .12, .60], [.57, .34, .70]])
    original = points.copy()
    layout = instance_grid_layout(points, .19, voxel_size_m=.02, floor_z_m=floor)
    assert layout['esdf_shape'][2] % 2 == 0
    phase = (floor - layout['center'][2]) / .02 - .5
    assert phase == pytest.approx(round(phase), abs=1e-10)
    np.testing.assert_array_equal(points, original)
    half = np.asarray(layout['esdf_shape']) * .02 / 2
    assert np.all(np.abs(points - layout['center']) + .19 <= half + 1e-8)


def test_release_uses_the_same_numerical_cost_epsilon_as_trajectory_validation():
    from test_release_clearance import request
    spheres = [[0., 0., .1, .01]]
    near_zero = release_clearance_evidence(spheres, request(), lambda _: np.array([[[5e-11]]]), scene_token='scene')
    real_conflict = release_clearance_evidence(spheres, request(), lambda _: np.array([[[5e-5]]]), scene_token='scene')
    assert not near_zero['conflict']
    assert real_conflict['conflict']


def test_measured_instance_floor_is_bound_to_scene_identity():
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload
    request = wire()
    decoded, first = decode_mapping_payload(request | {'instance_floor_heights_m': {'source': .61}})
    _, second = decode_mapping_payload(request | {'instance_floor_heights_m': {'source': .62}})
    assert decoded['instance_floor_heights_m'] == {'source': .61}
    assert first['scene_token'] != second['scene_token']
    np.testing.assert_array_equal(decoded['views']['3']['depth_left_m'], np.ones((2, 3)))


@pytest.mark.parametrize('floors', [{'absent': .61}, {'source': np.nan}, {'source': '0.61'}])
def test_unmeasured_or_invalid_floor_metadata_is_rejected(floors):
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload
    with pytest.raises(ValueError, match='instance_floor_heights'):
        decode_mapping_payload(wire() | {'instance_floor_heights_m': floors})
