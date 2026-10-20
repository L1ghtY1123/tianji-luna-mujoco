import copy

import numpy as np
import pytest
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from jev_mujoco.runner import Runner
from jev_mujoco.task_progress import TaskProgress


def identity_runner(tmp_path, algorithms=None, *, route='direct'):
    runner = Runner(SensorRobot(), algorithms or VisionAlgorithms(), lambda record: Choices(), tmp_path)
    runner.state.update(task='grasp_place', generation=1, attributes={'kind': 'cube', 'color': 'red'},
                        destination_attributes={'kind': 'bin', 'color': 'blue'})
    runner.state['facts']['collection_task'] = True
    runner.task_progress = TaskProgress('move the selected objects', 'collection', route=route)
    return runner


async def observe_identity(runner, context, *, checkpoint='initial', **kwargs):
    purpose = ('initialize_inventory' if not runner.task_progress.scene_inventory_observation_id else
               'verify_transfer' if checkpoint == 'post_place' else 'associate_inventory')
    context['visual_plan'] = {checkpoint: purpose}
    return await runner.observe(context, checkpoint=checkpoint, purpose='scene', recovery=True, **kwargs)


@pytest.mark.parametrize('planned_release_height', [None, .12])
async def test_partial_place_capture_keeps_visual_ids_and_initial_measured_reference(tmp_path, planned_release_height):
    runner = identity_runner(tmp_path)
    context = {'planning_settings': runner.planning_settings.model_dump()}
    first = await observe_identity(runner, context)
    source, destination = first['objects']
    runner.task_progress.assess_membership({source['id']: 'required'})
    runner.task_progress.select(source, destination['id'])
    runner.state.update(target_id=source['id'], destination_id=destination['id'], arm='A')
    reference = np.eye(4)
    height = planned_release_height or context['planning_settings']['place_release_height_m']
    reference[:3, 3] = [.6, -.2, .64 + height]
    context['pending_transfer'] = {'target_id': source['id'], 'destination_id': destination['id'],
                                   'release_height_m': height}
    context['holding'] = {'state': 'release_pending', 'instance_id': source['id'], 'arm': 'A',
                          'points_tool_m': (source['points_world_m'] - source['points_world_m'].mean(0)).tolist(),
                          'release_tcp': reference.tolist()}
    runner.simulation.stage = 5
    current = await observe_identity(runner, context, checkpoint='post_place', camera_ids=['3'])
    initial_reference = runner.algorithms.source_hints[-1]['visual_reference']
    assert np.array_equal(initial_reference['objects'][0]['points_world_m'], source['points_world_m'])
    assert current['objects'][0]['id'] == source['id']
    assert current['objects'][0]['points_world_m'].mean(0) == pytest.approx([.6, -.2, .64])
    assert current['visual_inventory']['progress'][source['id']]['status'] == 'at_destination'
    assert current['association_evidence']['source'] == 'openai_decisions'
    context.pop('holding')
    runner.state.update(target_id=None, destination_id=None)
    after = await observe_identity(runner, context)
    assert after['objects'][0]['id'] == source['id']
    assert len(after['objects']) == 2


async def test_held_visual_identity_keeps_current_measurements_and_original_reference(tmp_path):
    runner = identity_runner(tmp_path)
    context = {}
    first = await observe_identity(runner, context)
    source = first['objects'][0]
    context['holding'] = {'state': 'provisional', 'instance_id': source['id'], 'arm': 'A',
                          'points_tool_m': (source['points_world_m'] - runner.simulation.tcp).tolist()}
    runner.simulation.tcp = [.5, .1, .95]
    runner.simulation.closed = True
    current = await observe_identity(runner, context, checkpoint='fine')
    assert current['objects'][0]['id'] == source['id']
    assert current['objects'][0]['points_world_m'].mean(0) == pytest.approx([.5, .1, .79])
    assert np.array_equal(runner.algorithms.source_hints[-1]['visual_reference']['objects'][0]['points_world_m'],
                          source['points_world_m'])


async def test_pending_release_visual_request_survives_cleared_holding_and_target(tmp_path):
    runner = identity_runner(tmp_path)
    context = {}
    first = await observe_identity(runner, context)
    source, destination = first['objects']
    runner.task_progress.assess_membership({source['id']: 'required'})
    runner.task_progress.select(source, destination['id'])
    context['pending_transfer'] = {'target_id': source['id'], 'destination_id': destination['id']}
    runner.simulation.stage = 5
    current = await observe_identity(runner, context, checkpoint='post_place')
    assert current['objects'][0]['id'] == source['id']
    evidence = current['visual_inventory']['progress'][source['id']]
    assert evidence['destination_id'] == destination['id']
    assert evidence['observation_id'] == current['observation_id']
    assert evidence['status'] == 'at_destination'


@pytest.mark.parametrize('next_destination_selected', [False, True])
async def test_two_stage_route_keeps_visual_id_before_next_grasp(tmp_path, next_destination_selected):
    class IntermediateIds(VisionAlgorithms):
        def observe(self, capture, **kwargs):
            result = super().observe(capture, **kwargs)
            source, destination = result['objects']
            source['measured_relationships'] = [{'relation': 'above_observed_bin_floor',
                'object_id': 'source-box', 'visible_xy_fraction': 1.}]
            marker = copy.deepcopy(destination)
            marker.update(id='marker-yellow', kind='marker', color='yellow')
            result['objects'].append(marker)
            result['visual_inventory']['membership'][marker['id']] = 'excluded'
            if capture['views']['3']['rgb'][0, 0, 0] >= 4:
                source['measured_relationships'] = [{'relation': 'above_observed_marker',
                    'object_id': marker['id'], 'visible_xy_fraction': .5}]
            return result

    runner = identity_runner(tmp_path, IntermediateIds(), route='box_table_box')
    context = {}
    first = await observe_identity(runner, context)
    source, destination, marker = first['objects']
    runner.task_progress.assess_membership({source['id']: 'required'})
    runner.task_progress.select(source, marker['id'])
    runner.task_progress.record_release(source['id'], feedback_id='release-on-marker')
    assert runner.task_progress.confirm_visual_transfer(source['id'], {
        'status': 'at_destination', 'destination_id': marker['id'],
        'observation_id': 'released-on-marker', 'source': 'openai_decisions'})
    if next_destination_selected:
        runner.task_progress.select(source, destination['id'])
        runner.state.update(target_id=source['id'], destination_id=destination['id'])
    runner.simulation.stage = 5
    current = await observe_identity(runner, context)
    assert current['objects'][0]['id'] == source['id']
    assert current['objects'][0]['measured_relationships'][0]['object_id'] == marker['id']
    assert runner.task_progress.records[source['id']]['stage_index'] == 1
    assert runner.task_progress.completed_ids == []


async def test_temporarily_occluded_object_keeps_visual_reference_without_stale_geometry(tmp_path):
    algorithms = VisionAlgorithms()
    runner = identity_runner(tmp_path, algorithms)
    context = {}
    first = await observe_identity(runner, context)
    original = copy.deepcopy(first['objects'][0])
    observe = algorithms.observe

    def occluded(*args, **kwargs):
        result = observe(*args, **kwargs)
        result['objects'] = result['objects'][1:]
        for view in result['views'].values():
            view['instance_masks'].pop(original['id'])
        return result

    algorithms.observe = occluded
    missing = await observe_identity(runner, context)
    assert original['id'] not in {obj['id'] for obj in missing['objects']}
    assert missing['mapping_unresolved'][0]['previous_id'] == original['id']
    assert context['visual_inventory_reference']['objects'][0]['id'] == original['id']
    algorithms.observe = observe
    restored = await observe_identity(runner, context)
    assert restored['objects'][0]['id'] == original['id']
    assert restored['mapping_unresolved'] == []
