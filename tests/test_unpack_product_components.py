import numpy as np
import pytest

from luna_mujoco.local_object_obstacle import LocalObjectDepthError


def cloud(center):
    offsets = np.array([[x, y, z] for x in (-.002, 0, .002) for y in (-.002, 0, .002) for z in (-.002, 0, .002)])
    return np.repeat(np.array(center) + offsets, 2, axis=0)


def test_whole_table_black_components_never_merge_two_candidates():
    from luna_mujoco.local_object_obstacle import select_unpack_component
    points = np.concatenate([cloud([.45, -.2, .60]), cloud([.55, -.2, .60])])
    with pytest.raises(LocalObjectDepthError) as caught:
        select_unpack_component(points, {})
    assert caught.value.details['reason'] == 'multiple_unbound_black_candidates'


def test_prior_measured_black_identity_is_excluded_before_binding_new_candidate():
    from luna_mujoco.local_object_obstacle import select_unpack_component
    old, new = cloud([.45, -.2, .6]), cloud([.55, -.2, .6])
    points, proof = select_unpack_component(np.concatenate([old, new]), {
        'known_black_objects': [{'id': 'old-product', 'minimum_world_m': old.min(0).tolist(),
                                'maximum_world_m': old.max(0).tolist()}]})
    assert np.mean(points, axis=0) == pytest.approx([.55, -.2, .6])
    assert proof['excluded_known_entity_ids'] == ['old-product']


def test_old_visible_product_cannot_stand_in_for_occluded_new_product():
    from luna_mujoco.local_object_obstacle import select_unpack_component
    old = cloud([.45, -.2, .6])
    with pytest.raises(LocalObjectDepthError) as caught:
        select_unpack_component(old, {'known_black_objects': [{'id': 'old-product',
            'minimum_world_m': old.min(0).tolist(), 'maximum_world_m': old.max(0).tolist()}]})
    assert caught.value.details['reason'] == 'new_black_candidate_missing'


def test_black_candidate_outside_right_workspace_cannot_be_selected():
    from luna_mujoco.local_object_obstacle import select_unpack_component
    with pytest.raises(LocalObjectDepthError):
        select_unpack_component(cloud([8., 8., .6]), {})

@pytest.mark.parametrize('center', [[.4, -.15, .60], [.6, -.25, .60]])
def test_whole_table_pipeline_measures_product_outside_input_square(center, monkeypatch):
    from types import SimpleNamespace

    from luna_mujoco.local_object_obstacle import measure_object_obstacle
    shape = (10, 10)
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1., -1., -1.])
    transform[:3, 3] = [center[0], center[1], 1.6]
    view = {'rgb': np.zeros((*shape, 3), np.uint8), 'depth_m': np.ones(shape),
        'valid_depth': np.ones(shape, bool), 'intrinsics': {'fx': 1000., 'fy': 1000., 'cx': 4.5, 'cy': 4.5},
        'T_world_camera': transform.tolist()}
    detector = SimpleNamespace(segment=lambda *args: [{'mask': np.ones(shape, bool), 'score': .9}])
    monkeypatch.setattr('luna_mujoco.local_object_obstacle.own_robot_surface_pixels',
        lambda capture, masks: {'status': 'measured', 'masks': {'3': np.zeros(shape, bool)}})
    measured = measure_object_obstacle({'views': {'3': view}, 'time': 1., 'generation': 1,
        'observation_id': 'current'}, detector, {'source': 'measured_cached_main_table_support',
        'region_polygon_world_xy_m': [[.272, -.426], [.872, -.426], [.872, .374], [.272, .374]],
        'floor_z_m': .572, 'known_black_objects': []}, 'unpacked')
    assert measured['points_world_m'].mean(axis=0) == pytest.approx(center, abs=1e-6)
    assert measured['component_evidence']['new_candidate_count'] == 1
    assert measured['observation_id'] == 'current'
