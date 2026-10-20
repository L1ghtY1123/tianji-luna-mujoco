from copy import deepcopy

import numpy as np
import pytest
from test_region_placement import region_inputs

from jev_mujoco import placement
from jev_mujoco.placement_layout import partition_region
from jev_mujoco.task_progress import TaskProgress


def inventory(count=3):
    objects, masks = [], {}
    for i in range(count):
        identifier = f'part-{i + 1}'
        objects.append({'id': identifier, 'kind': 'cube'})
        mask = np.zeros((40, 80), bool)
        mask[4:10, 4 + 12 * i:10 + 12 * i] = True
        masks[identifier] = mask
    camera = np.diag([1., -1., -1., 1.])
    camera[2, 3] = 1.
    observation = {'observation_id': 'initial-head', 'objects': objects, 'views': {'3': {
        'instance_masks': masks, 'T_world_camera': camera,
        'intrinsics': {'fx': 100., 'fy': 100., 'cx': 40., 'cy': 20.}}}}
    progress = TaskProgress('搬运所有指定物体', 'collection', route='table_box')
    progress.observe(objects, 'initial-head')
    progress.assess_membership(dict.fromkeys(masks, 'required'))
    progress.freeze_inventory(observation)
    return progress, observation


@pytest.mark.parametrize('count,sections', [(1, 2), (2, 2), (3, 4), (4, 4), (5, 6)])
def test_even_partitions_use_paired_center_lines(count, sections):
    layout = partition_region([[0., 0.], [.4, 0.], [.4, .2], [0., .2]], count)
    assert layout['section_count'] == sections
    center = np.asarray(layout['center_world_xy_m'])
    rays = [np.asarray(cell['polygon_world_xy_m'])[1] - center for cell in layout['cells']]
    for index in range(sections // 2):
        np.testing.assert_allclose(rays[index], -rays[index + sections // 2], atol=1e-12)


def test_asymmetric_measured_boundary_uses_only_an_inner_symmetric_region():
    polygon = [[0., 0.], [.4, 0.], [.35, .18], [0., .2]]
    layout = partition_region(polygon, 5)
    center = np.asarray(layout['center_world_xy_m'])
    rays = [np.asarray(cell['polygon_world_xy_m'])[1] - center for cell in layout['cells']]
    for index in range(3):
        np.testing.assert_allclose(rays[index], -rays[index + 3], atol=1e-12)
    assert placement.region_contains_xy(layout['region_polygon_world_xy_m'], polygon).all()


def test_four_slots_follow_initial_head_rows_and_object_two_keeps_slot_two():
    progress, observation = inventory()
    _, destination, _, _ = region_inputs('marker')
    progress.select(observation['objects'][1], destination['id'])
    progress.bind_placement_region(destination, observation)
    request = progress.placement_request(destination['id'])
    assert request['object_id'] == 'part-2'
    assert request['object_display_id'] == request['slot_id'] == 2
    assert request['target_count'] == 3 and request['layout']['section_count'] == 4
    centers = [cell['center_world_xy_m'] for cell in request['layout']['cells']]
    np.testing.assert_allclose(centers, [[.64, -.14], [.76, -.14], [.64, -.26], [.76, -.26]])
    before = deepcopy(request)
    destination['placement_reference']['region_polygon_world_xy_m'] = [[.58, -.32], [.70, -.32], [.70, -.08], [.58, -.08]]
    progress.bind_placement_region(destination, observation | {'observation_id': 'occluded'})
    assert progress.placement_request(destination['id']) == before
    progress.complete('part-2', feedback_id='released')
    progress.select(observation['objects'][2], destination['id'])
    after = progress.placement_request(destination['id'])
    assert after['slot_id'] == 3 and after['target_count'] == 3
    assert after['layout_revision'] == before['layout_revision']


def test_occupied_assigned_slot_keeps_tcp_candidates_for_three_dimensional_check():
    target, destination, holding, spheres = region_inputs('bin')
    progress, observation = inventory()
    progress.select(observation['objects'][1], destination['id'])
    progress.bind_placement_region(destination, observation)
    request = progress.placement_request(destination['id'])
    options = {'spheres_tool_m': spheres, 'clearance_m': .01, 'placement_request': request}
    candidates = placement.release_candidates(target, destination, holding, **options)
    assert len(candidates) == 18
    assert {goal['release_reference']['slot_id'] for goal in candidates} == {2}
    center = request['layout']['cells'][1]['center_world_xy_m']
    obstacle = [[x, y, .7] for x in np.linspace(center[0] - .06, center[0] + .06, 25)
                for y in np.linspace(center[1] - .06, center[1] + .06, 25)]
    goals = placement.release_candidates(target, destination, holding, **options,
        observation={'objects': [{'id': 'obstacle', 'points_world_m': obstacle}]})
    assert len(goals) == 18
    for goal in goals:
        assert goal['release_reference']['slot_id'] == 2
        np.testing.assert_allclose(np.asarray(goal['T_world_tcp'])[:2, 3], center)


@pytest.mark.parametrize('slot_floor_visible', [True, False])
def test_frozen_slot_checks_current_floor_samples_not_a_new_inner_rectangle(slot_floor_visible):
    target, destination, holding, spheres = region_inputs('bin')
    progress, observation = inventory()
    progress.select(observation['objects'][1], destination['id'])
    progress.bind_placement_region(destination, observation)
    request = progress.placement_request(destination['id'])
    # A new view chooses a different supported rectangle, while the assigned
    # slot and the actual measured floor samples remain independent of it.
    destination['placement_reference']['region_polygon_world_xy_m'] = [
        [.58, -.32], [.68, -.32], [.68, -.08], [.58, -.08]]
    if not slot_floor_visible:
        points = destination['points_world_m']
        destination['points_world_m'] = points[points[:, 0] < .68]
    options = {'spheres_tool_m': spheres, 'clearance_m': .01, 'placement_request': request}
    if not slot_floor_visible:
        with pytest.raises(placement.PlacementError, match='placement_slot_support_unobserved'):
            placement.release_candidates(target, destination, holding, **options)
        return
    goals = placement.release_candidates(target, destination, holding, **options)
    assert goals
    for goal in goals:
        assert goal['release_reference']['slot_id'] == 2
        np.testing.assert_allclose(goal['release_reference']['target_reference_world_xy_m'],
                                   request['layout']['cells'][1]['center_world_xy_m'])


def test_measured_support_translation_preserves_slot_labels_and_invalidates_layout_revision():
    progress, observation = inventory()
    _, destination, _, _ = region_inputs('bin')
    walls = [[x, y, z] for x in (.58, .82) for y in np.linspace(-.32, -.08, 50)
             for z in np.linspace(.63, .70, 20)]
    walls += [[x, -.32, z] for x in np.linspace(.58, .82, 50) for z in np.linspace(.63, .70, 20)]
    destination['points_world_m'] = np.vstack((destination['points_world_m'], walls))
    progress.select(observation['objects'][1], destination['id'])
    first = progress.bind_placement_region(destination, observation)
    shift = np.array([.04, -.02])
    reference = destination['placement_reference']
    reference['region_polygon_world_xy_m'] = (np.asarray(reference['region_polygon_world_xy_m']) + shift).tolist()
    destination['points_world_m'] += np.r_[shift, 0.]
    second = progress.bind_placement_region(destination, observation | {'observation_id': 'moved-bin'})
    assert second['slot_id'] == first['slot_id'] == 2
    assert second['layout_revision'] != first['layout_revision']
    for before, after in zip(first['layout']['cells'], second['layout']['cells']):
        assert before['slot_id'] == after['slot_id']
        np.testing.assert_allclose(np.asarray(after['center_world_xy_m']) - before['center_world_xy_m'], shift)


def test_congruent_visible_floor_patches_do_not_move_a_stationary_support_layout():
    from jev_mujoco.measured_regions import supported_floor_rectangle

    progress, observation = inventory()
    _, destination, _, _ = region_inputs('bin')
    x, y = np.meshgrid(np.linspace(.58, .82, 121), np.linspace(-.32, -.08, 121))
    left = supported_floor_rectangle(np.c_[x[x < .69], y[x < .69]])
    right = supported_floor_rectangle(np.c_[x[x > .71], y[x > .71]])
    reference = destination['placement_reference']
    reference['region_polygon_world_xy_m'] = left['polygon_world_xy_m']
    first = progress.bind_placement_region(destination, observation)
    reference['region_polygon_world_xy_m'] = right['polygon_world_xy_m']
    second = progress.bind_placement_region(destination, observation | {'observation_id': 'occluded'})
    assert second == first


def test_unchanged_parallel_walls_cannot_prove_tangential_patch_translation():
    progress, observation = inventory()
    _, destination, _, _ = region_inputs('bin')
    walls = [[x, y, z] for x in (.58, .82) for y in np.linspace(-.32, -.08, 50)
             for z in np.linspace(.63, .70, 20)]
    destination['points_world_m'] = np.vstack((destination['points_world_m'], walls))
    first = progress.bind_placement_region(destination, observation)
    reference = destination['placement_reference']
    reference['region_polygon_world_xy_m'] = (np.asarray(reference['region_polygon_world_xy_m'])
                                               + [0., .24 / 49 * 2]).tolist()
    second = progress.bind_placement_region(destination, observation | {'observation_id': 'new-patch'})
    assert second == first


def test_parallel_wall_crops_do_not_prove_motion_even_when_both_clouds_shift():
    progress, observation = inventory()
    _, destination, _, _ = region_inputs('bin')
    walls = [[x, y, z] for x in (.58, .82) for y in np.linspace(-.32, -.08, 50)
             for z in np.linspace(.63, .70, 20)]
    destination['points_world_m'] = np.vstack((destination['points_world_m'], walls))
    first = progress.bind_placement_region(destination, observation)
    reference = destination['placement_reference']
    reference['region_polygon_world_xy_m'] = (np.asarray(reference['region_polygon_world_xy_m']) + [0., .04]).tolist()
    destination['points_world_m'] += [0., .04, 0.]
    second = progress.bind_placement_region(destination, observation | {'observation_id': 'wall-crops'})
    assert second == first


def test_partition_lines_follow_initial_head_direction_not_world_axes():
    from scipy.spatial.transform import Rotation

    progress, observation = inventory()
    head = progress.initial_head_reference
    head['T_world_camera'][:3, :3] = Rotation.from_euler('z', 35, degrees=True).as_matrix() @ head['T_world_camera'][:3, :3]
    _, destination, _, _ = region_inputs('marker')
    progress.select(observation['objects'][0], destination['id'])
    result = progress.bind_placement_region(destination, observation)
    np.testing.assert_allclose(result['layout']['partition_direction_world_xy'], [np.cos(np.deg2rad(35)), np.sin(np.deg2rad(35))])
    pixels = np.array([cell['initial_head_center_pixel_xy'] for cell in result['layout']['cells']])
    assert pixels[0, 0] < pixels[1, 0] and pixels[2, 0] < pixels[3, 0]
    assert max(pixels[:2, 1]) < min(pixels[2:, 1])
