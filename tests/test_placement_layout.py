"""Task-scoped placement partitions, including already occupied destinations."""
from copy import deepcopy

import numpy as np
import pytest
from test_region_placement import region_inputs

from jev_mujoco import placement
from jev_mujoco.task_progress import TaskProgress


@pytest.mark.parametrize('count,sections', [(1, 2), (2, 2), (3, 4), (4, 4), (5, 6)])
def test_partition_uses_area_centers_and_even_capacity(count, sections):
    from jev_mujoco.placement_layout import partition_region

    polygon = [[0., 0.], [.4, 0.], [.4, .2], [0., .2]]
    layout = partition_region(polygon, count)
    assert layout['section_count'] == sections
    assert len(layout['cells']) == sections
    np.testing.assert_allclose(layout['center_world_xy_m'], [.2, .1])
    assert sum(cell['area_m2'] for cell in layout['cells']) == pytest.approx(.08)
    for cell in layout['cells']:
        assert cell['area_m2'] == pytest.approx(.08 / sections)
        assert placement.region_contains_xy(cell['center_world_xy_m'], polygon)[0]
    if count == 1:
        np.testing.assert_allclose(layout['cells'][0]['center_world_xy_m'], [.2, .15])
    if count == 3:
        np.testing.assert_allclose([cell['center_world_xy_m'] for cell in layout['cells']],
                                  [[.3, .15], [.1, .15], [.1, .05], [.3, .05]])


def test_polygonal_circle_and_rotated_rectangle_have_equal_area_cells():
    from jev_mujoco.placement_layout import partition_region

    angles = np.linspace(0, 2 * np.pi, 129)[:-1]
    circle = np.column_stack((np.cos(angles), np.sin(angles))) * .2 + [.6, -.1]
    rectangle = np.array([[-.2, -.1], [.2, -.1], [.2, .1], [-.2, .1]])
    rotation = np.array([[.8, -.6], [.6, .8]])
    for polygon in (circle, rectangle @ rotation.T + [.6, -.1]):
        layout = partition_region(polygon, 5)
        areas = [cell['area_m2'] for cell in layout['cells']]
        np.testing.assert_allclose(areas, np.full(6, sum(areas) / 6), atol=1e-12)
        assert all(placement.region_contains_xy(cell['center_world_xy_m'], polygon)[0]
                   for cell in layout['cells'])


def test_layout_count_is_task_scoped_and_does_not_shrink_after_completion():
    from test_task_progress import inventory_observation
    progress = TaskProgress('move selected objects', 'collection', route='table_box')
    objects = [{'id': str(i), 'kind': 'cube'} for i in range(4)]
    progress.observe(objects, 'head')
    progress.assess_membership({'0': 'required', '1': 'required', '2': 'required', '3': 'excluded'})
    progress.freeze_inventory(inventory_observation(objects))
    progress.select(objects[0], 'box')
    first = progress.placement_request('box')
    assert first['target_count'] == 3 and first['target_ids'] == ['0', '1', '2']
    progress.complete('0', observation_id='after')
    progress.observe(objects + [{'id': 'new', 'kind': 'cube'}], 'new-head')
    progress.select(objects[1], 'box')
    after = progress.placement_request('box')
    assert after['target_count'] == first['target_count']
    assert after['target_ids'] == first['target_ids']
    assert after['slot_id'] == 2 and after['object_id'] == '1'
    first['target_ids'].clear()
    assert len(progress.placement_request('box')['target_ids']) == 3


@pytest.mark.parametrize('kind', ['marker', 'bin'])
def test_two_targets_start_in_separate_regions_and_keep_placed_obstacle(kind):
    target, destination, holding, spheres = region_inputs(kind)
    request = {'target_count': 2, 'target_ids': ['source', 'next'], 'slot_id': 1}
    options = {'spheres_tool_m': spheres, 'clearance_m': .01, 'placement_request': request}
    events = []
    before = deepcopy(destination)
    first = placement.release_candidates(target, destination, holding, **options,
        progress=lambda kind, title, data: events.append((kind, data)))
    locations = {tuple(item['release_reference']['position_world_m'][:2]) for item in first}
    assert len(locations) == 1
    assert 0 < len(first) <= 18
    chosen = np.asarray(first[0]['T_world_tcp'])
    occupied_center = np.asarray(spheres)[0, :3] @ chosen[:3, :3].T + chosen[:3, 3]
    occupied_center[2] = .70
    observation = {'objects': [destination, {'id': 'placed', 'points_world_m': [occupied_center]}]}
    second = placement.release_candidates(target, destination, holding,
        **(options | {'placement_request': request | {'slot_id': 2}}), observation=observation)
    assert second and len(second) <= 18
    assert {item['release_reference']['slot_id'] for item in second} == {2}
    for item in second:
        tcp = np.asarray(item['T_world_tcp'])
        center = np.asarray(spheres)[0, :3] @ tcp[:3, :3].T + tcp[:3, 3]
        assert np.linalg.norm(center[:2] - occupied_center[:2]) > spheres[0][3] + .01
    assert any(kind == 'placement_layout' for kind, _ in events)
    assert sum(data['accepted_yaw_count'] for kind, data in events if kind == 'placement_position_checked') == len(first)
    assert destination['placement_reference'] == before['placement_reference']


def test_obstacle_projection_does_not_remove_yaws_before_three_dimensional_planning():
    target, destination, holding, spheres = region_inputs('marker')
    obstacle = [[x, y, .7] for x in np.linspace(.58, .82, 40) for y in np.linspace(-.32, -.08, 40)]
    observation = {'objects': [destination, {'id': 'other', 'points_world_m': obstacle}]}
    events = []
    goals = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01,
        placement_request={'target_count': 2}, observation=observation,
        progress=lambda kind, title, data: events.append((kind, data)))
    assert len(goals) == 18
    checked = [data for kind, data in events if kind == 'placement_position_checked']
    assert len(checked) == 1
    assert checked[0]['accepted_yaw_count'] == 18
    assert checked[0]['generated_yaw_count'] == 18


def test_multiple_objects_require_region_geometry_instead_of_reusing_one_reference():
    target, destination, holding, spheres = region_inputs()
    destination['placement_reference'].pop('region_polygon_world_xy_m')
    with pytest.raises(ValueError, match='placement_requires_measured_region_polygon'):
        placement.release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01,
            placement_request={'target_count': 2})


def test_each_route_stage_keeps_global_capacity_across_multiple_destinations():
    progress = TaskProgress('transfer objects through table', 'collection', route='box_table_box')
    objects = [{'id': str(i), 'kind': 'cube', 'measured_relationships': [{
        'relation': 'above_observed_bin_floor', 'object_id': 'source-box', 'visible_xy_fraction': 1.}]
    } for i in range(3)]
    progress.observe(objects, 'head')
    progress.assess_membership({str(i): 'required' for i in range(3)})
    progress.select(objects[0], 'table')
    progress.records['2']['destination_id'] = 'other-table'
    assert progress.placement_request('table')['target_count'] == 3
    progress.complete('0', observation_id='after')
    progress.select(objects[0], 'box2')
    second = progress.placement_request('box2')
    assert second['stage_index'] == 1
    assert second['target_count'] == 3
