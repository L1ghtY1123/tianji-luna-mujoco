import numpy as np

from luna_mujoco.release_region_reference import release_region_reference


def inputs():
    capture = {'observation_id': 'after', 'generation': 2, 'views': {'3': {
        'rgb': np.zeros((100, 100, 3), dtype=np.uint8), 'T_world_camera': np.eye(4),
        'intrinsics': {'fx': 100, 'fy': 100, 'cx': 50, 'cy': 50}}}}
    binding = {'placement_zone': {'geometry_source': 'configured_xy_measured_support',
        'support_entity_id': 'table', 'observation_id': 'before', 'floor_z_m': 1.,
        'polygon_world_xy_m': [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]],
        'intent': {'relation': 'on', 'relative': 'none', 'zone': 'whole'}}}
    return capture, binding


def test_reference_projects_known_region_without_depth_mask_or_object_pose():
    capture, binding = inputs()
    reference = release_region_reference(capture, binding)
    assert reference['status'] == 'projected'
    assert reference['observation_id'] == 'after'
    assert reference['region_source_observation_id'] == 'before'
    assert reference['not_a_landing_measurement'] is True
    assert reference['projections'][0]['camera_id'] == '3'
    assert np.allclose(reference['projections'][0]['polygon'][0], [.4, .4])


def test_offscreen_region_is_not_claimed_visible():
    capture, binding = inputs()
    capture['views']['3']['T_world_camera'][0, 3] = 10
    reference = release_region_reference(capture, binding)
    assert reference['status'] == 'outside_current_views'
    assert reference['projections'] == []


def test_whole_container_needs_only_containment_not_an_arbitrary_cell_center():
    capture, binding = inputs()
    binding['placement_zone'].update(geometry_source='cad_floor',
        intent={'relation': 'inside', 'relative': 'none', 'zone': 'whole'})
    assert release_region_reference(capture, binding) is None


def test_reference_outline_respects_language_halfplanes_not_the_entire_support():
    capture, binding = inputs()
    binding['placement_zone']['halfplanes_world_xy'] = [[1., 0., 0.]]
    reference = release_region_reference(capture, binding)
    polygon = np.asarray(reference['projections'][0]['polygon'])
    assert polygon[:, 0].max() == .5
    assert polygon[:, 0].min() == .4
