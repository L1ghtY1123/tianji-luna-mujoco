"""Physical pickup order is independent of permanent image labels."""
from copy import deepcopy

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.inventory_order import robot_distance_order


class DistanceVision(DemoVision):
    def observe(self, capture, **kwargs):
        result = super().observe(capture, **kwargs)
        positions = {'pink-1': [.8, .1, .65], 'pink-2': [.3, .1, 1.2],
                     'pink-3': [.5, .1, .65]}
        for obj in result['objects']:
            if obj['id'] in positions:
                obj['visible_centroid_world'] = positions[obj['id']]
        return result


class ReferenceCheckingRunner(DemoRunner):
    def _bind_inventory(self, context):
        original = deepcopy(context['visual_inventory_reference']['entities'])
        result = super()._bind_inventory(context)
        assert context['visual_inventory_reference']['entities'] == original
        return result


@pytest.mark.parametrize('count, expected', [(None, ['pink-2', 'pink-3', 'pink-1']),
                                           (2, ['pink-2', 'pink-3'])])
async def test_actual_runner_picks_nearest_sources_without_renumbering(tmp_path, count, expected):
    algorithms = DistanceVision(count=3, requested=count)
    runner = ReferenceCheckingRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move objects from nearest to farthest')
    await runner.worker
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert algorithms.target_order == expected
    assert runner.demo_task['transport_ids'] == expected
    assert runner.demo_task['source_object_ids'] == ['pink-1', 'pink-2', 'pink-3']
    assert {key: runner.task_progress.records[key]['display_id']
            for key in runner.demo_task['source_object_ids']} == {'pink-1': 1, 'pink-2': 2, 'pink-3': 3}
    assert algorithms.discoveries == 1
    evidence = runner.demo_task['execution_order']
    assert evidence['selected_ids'] == expected
    assert evidence['source'] == 'measured_rgbd_centroids'
    assert next(event for event in runner.events if event['kind'] == 'target_execution_order')['data'] == evidence


async def test_missing_current_distance_stops_before_any_grasp(tmp_path):
    class MissingDepth(DistanceVision):
        def observe(self, capture, **kwargs):
            result = super().observe(capture, **kwargs)
            result['objects'][0]['visible_centroid_world'] = [float('nan'), 0., .6]
            return result

    algorithms = MissingDepth(count=3)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move objects from nearest to farthest')
    await runner.worker
    assert runner.state['status'] == 'needs_attention'
    assert runner.state['error'] == 'source_distance_unmeasured:pink-1'
    assert algorithms.target_order == []


def test_distance_uses_geometry_for_any_color_shape_and_stable_ties():
    objects = [{'id': 'a', 'kind': 'cube', 'color': 'black', 'position': [.6, 0., .2]},
               {'id': 'b', 'kind': 'object', 'color': 'green', 'position': [0., -.3, 2.]},
               {'id': 'c', 'kind': 'sphere', 'color': 'pink', 'position': [0., .3, .1]}]
    assert [row['id'] for row in robot_distance_order(objects)] == ['b', 'c', 'a']


async def test_unpack_process_order_preserves_source_identity_and_arm_bindings(tmp_path):
    class UnpackVision(DistanceVision):
        def observe(self, capture, **kwargs):
            result = super().observe(capture, **kwargs)
            result['demo_task'].update(workflow='unpack_transfer', right_bin_id='right-bin',
                item_lifecycle={identifier: {'phase': 'wrapped_source', 'appearance': 'wrapped',
                    'to_table_arm': 'A', 'to_right_bin_arm': 'B'}
                    for identifier in result['demo_task']['source_object_ids']})
            result['demo_program'] = {'version': 1, 'steps': [{'type': 'stop', 'reason': 'test_after_binding'}]}
            return result

    algorithms = UnpackVision(count=3, requested=2)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('unpack nearest objects')
    await runner.worker
    assert runner.state['error'] == 'program_stopped:test_after_binding'
    task = runner.demo_task
    rows = [task['process_records'][key] for key in task['process_order']]
    assert [row['input_entity_id'] for row in rows] == ['pink-2', 'pink-3']
    assert [row['input_display_id'] for row in rows] == [2, 3]
    assert runner.task_progress.snapshot()['not_started_input_ids'] == ['pink-2', 'pink-3']
    assert task['item_lifecycle']['pink-2']['to_table_arm'] == 'A'
    with pytest.raises(ValueError, match='target_violates_inventory_order'):
        runner.task_progress.select({'id': 'pink-3'})
