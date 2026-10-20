import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from test_decisions import install_agent
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from jev_mujoco.decisions import Decisions
from jev_mujoco.runner import Runner
from jev_mujoco.skills import SkillSelectionError
from jev_mujoco.task_progress import TaskProgress


def visual_runner(tmp_path):
    algorithms = VisionAlgorithms()
    algorithms.perception = SimpleNamespace(visual_inventory=SimpleNamespace(audit=None))
    runner = Runner(SensorRobot(), algorithms, lambda record: Choices(), tmp_path)
    runner.state.update(task='grasp_place', generation=1, instruction='搬运红色物体')
    runner.task_progress = TaskProgress('搬运红色物体')
    return runner


async def test_jev_selects_purpose_before_skill_execution(tmp_path):
    runner = visual_runner(tmp_path)
    class Selection:
        async def choose_visual_plan(self, state, skill, requests):
            assert skill == 'observe_scene'
            assert requests == {'initial': 'initialize_inventory'}
            return {'initial': 'initialize_inventory'}
    arguments = await runner.select_arguments('observe_scene', {}, Selection())
    assert arguments['visual_plan'] == {'initial': 'initialize_inventory'}


async def test_jev_can_decline_call_without_silently_running_it(tmp_path):
    runner = visual_runner(tmp_path)
    class Selection:
        async def choose_visual_plan(self, *args):
            return {'initial': 'skip'}
    with pytest.raises(SkillSelectionError, match='visual_decisions_required_for_observation'):
        await runner.select_arguments('observe_scene', {}, Selection())
    assert not runner.algorithms.source_hints


async def test_missing_jev_selection_blocks_analysis(tmp_path):
    runner = visual_runner(tmp_path)
    with pytest.raises(ValueError, match='visual_decisions_selection_required'):
        await runner.observe({}, purpose='scene', recovery=True)
    assert not runner.algorithms.source_hints


async def test_selection_is_consumed_by_one_observation_and_reuse_does_not_call_again(tmp_path):
    runner = visual_runner(tmp_path)
    context = {'visual_plan': {'initial': 'initialize_inventory'}}
    first = await runner.observe(context, purpose='scene', recovery=True)
    assert runner.algorithms.source_hints[-1]['visual_decision'] == {
        'source': 'jev', 'purpose': 'initialize_inventory'}
    assert context['visual_plan'] == {}
    with pytest.raises(ValueError, match='visual_decisions_selection_required'):
        await runner.observe(context, purpose='scene', recovery=True)
    assert len(runner.algorithms.source_hints) == 1
    assert first['observation_id']


def test_perception_cannot_bypass_outer_jev_selection():
    from test_perception import capture_fixture

    from jev_mujoco.perception import Perception, PerceptionError
    capture, _ = capture_fixture()
    class NeverCalled:
        def segment(self, *args):
            raise AssertionError('Segmentation should not run without the required selection')
    perception = Perception(NeverCalled(), visual_inventory=object())
    with pytest.raises(PerceptionError, match='visual_decisions_selection_required'):
        perception.observe(capture)


def test_atomic_build_scene_declares_at_most_two_concrete_observations(tmp_path):
    runner = visual_runner(tmp_path)
    context = {'visual_inventory_reference': {}, 'observation': {'observation_id': 'current'},
               'scene': {'observation_id': 'old', 'exclusion_state': 'selected'}}
    runner.task_progress.scene_inventory_observation_id = 'initial'
    assert runner.visual_requests('build_scene', context, {}) == {
        'initial': 'associate_inventory', 'fine': 'associate_inventory'}
    context['scene'] = {'observation_id': 'current', 'exclusion_state': 'observation'}
    assert runner.visual_requests('build_scene', context, {}) == {'fine': 'associate_inventory'}
    context['pending_transfer'] = {'verified': True}
    assert runner.visual_requests('verify_place', context, {}) == {}


async def test_jev_visual_choice_has_explicit_skip_and_no_robot_commands(monkeypatch):
    records = []
    install_agent(monkeypatch, {'initial': 'initialize_inventory'}, {'initial': 1.})
    result = await Decisions(None, lambda *args: records.append(args[2])).choose_visual_plan(
        {'instruction': 'move the objects'}, 'observe_scene', {'initial': 'initialize_inventory'})
    assert result == {'initial': 'initialize_inventory'}
    assert set(records[0]['questions']['initial']['criteria']) == {'initialize_inventory', 'skip'}
    assert records[0]['state']['visual_requests'] == {'initial': 'initialize_inventory'}


def test_visual_progress_contains_actual_assigned_region_and_slot():
    progress = TaskProgress('move red')
    progress.records = {'red': {'id': 'red', 'status': 'selected', 'display_id': 2,
                                'destination_id': 'box'}}
    progress.membership = {'red': 'required'}
    progress.placement_requests[(0, 'box')] = {'region_id': 'region:box', 'layout_revision': 'v1',
        'layout': {'floor_z_m': .6, 'cells': [{'slot_id': 2, 'center_world_xy_m': [.4, .2],
            'polygon_world_xy_m': [[.3, .1], [.5, .1], [.5, .3], [.3, .3]]}]}}
    record = progress.visual_progress()['red']
    assert record['region_id'] == 'region:box' and record['slot_id'] == 2
    assert record['assigned_slot']['center_world_xy_m'] == [.4, .2]
    progress.current_id = 'red'
    progress.record_release('red', feedback_id='release')
    evidence = {'source': 'openai_decisions', 'status': 'at_destination', 'observation_id': 'after',
                'destination_id': 'box', 'region_id': 'region:box', 'slot_id': 1}
    assert not progress.confirm_visual_transfer('red', evidence)
    evidence['slot_id'] = 2
    assert progress.confirm_visual_transfer('red', evidence)


def test_all_inventory_numbers_share_transport_numbers_and_do_not_change():
    progress = TaskProgress('move red')
    observation = {'observation_id': 'initial', 'objects': [
        {'id': 'red', 'kind': 'cube', 'color': 'red'},
        {'id': 'box', 'kind': 'bin', 'color': 'blue'},
        {'id': 'black', 'kind': 'cube', 'color': 'black'},
        {'id': 'yellow', 'kind': 'marker', 'color': 'yellow'}], 'table': {'surface_height_m': .6},
        'views': {'3': {'instance_masks': {'red': np.ones((4, 4), bool)}}}}
    progress.observe(observation['objects'], 'initial')
    progress.assess_membership({'red': 'required', 'black': 'excluded'})
    progress.freeze_inventory(observation)
    numbers = {key: entity['display_id'] for key, entity in progress.scene_entities.items()}
    assert numbers['red'] == progress.records['red']['display_id'] == 1
    assert sorted(numbers.values()) == [1, 2, 3, 4, 5]
    later = {'objects': list(reversed(observation['objects']))}
    progress.assign_display_ids(later)
    assert all(obj['display_id'] == numbers[obj['id']] for obj in later['objects'])


async def test_complete_mocked_transfer_chooses_three_visual_calls_outside_skills(tmp_path):
    from test_visual_inventory_lifecycle import VisualIds

    from jev_mujoco.experiments import ExperimentStore

    plans = []
    class GatedVision(VisualIds):
        def __init__(self):
            super().__init__()
            self.perception = SimpleNamespace(visual_inventory=SimpleNamespace(audit=None))

        def observe(self, *args, **kwargs):
            assert kwargs['source_hint']['visual_decision']['source'] == 'jev'
            observation = super().observe(*args, **kwargs)
            if kwargs.get('checkpoint') == 'post_place':
                assignment = kwargs['source_hint']['visual_progress']['source']
                assert assignment['assigned_slot']['slot_id'] == 1
                observation['visual_inventory']['progress']['source'] = {
                    'source': 'openai_decisions', 'status': 'at_destination',
                    'observation_id': observation['observation_id'],
                    **{key: assignment[key] for key in ('destination_id', 'region_id', 'slot_id')}}
            return observation

    class Selection(Choices):
        async def choose_visual_plan(self, state, skill, requests):
            assert runner.state['current_step']['skill'] == 'select_arguments'
            plans.append((skill, requests.copy()))
            return requests

    class RecordingRobot(SensorRobot):
        def snapshot(self):
            return self.robot_snapshot()

    store = ExperimentStore(tmp_path / 'experiments')
    runner = Runner(RecordingRobot(), GatedVision(), lambda record: Selection('grasp_place'),
                    tmp_path, experiments=store)
    await runner.start('把红色物体放进蓝盒')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert plans == [('observe_scene', {'initial': 'initialize_inventory'}),
                     ('build_scene', {'fine': 'associate_inventory'}),
                     ('verify_place', {'post_place': 'verify_transfer'})]
    assert runner.task_progress.completed_ids == ['source']
    first = next(event for event in runner.events if event['kind'] == 'observation_annotated')
    assert set(first['data']['annotated_rgb']) == {'3'}
    store.close()


async def test_missing_collision_entity_in_view_union_is_retained_as_unresolved(tmp_path):
    from test_visual_inventory_lifecycle import VisualIds

    runner = Runner(SensorRobot(), VisualIds(), lambda record: Choices(), tmp_path)
    runner.state.update(task='grasp_place', generation=1, instruction='move red')
    runner.task_progress = TaskProgress('move red')
    context = {'visual_plan': {'initial': 'initialize_inventory'}}
    await runner.observe(context, purpose='scene', recovery=True)
    runner.task_progress.scene_entities['obstacle'] = {'id': 'obstacle', 'kind': 'cube',
        'roles': ['collision'], 'segmentation_query': {'kind': 'cube', 'color': 'black', 'prompt': 'black block'}}
    context['visual_plan'] = {'initial': 'associate_inventory'}
    after = await runner.observe(context, purpose='scene', recovery=True)
    # Visual location evidence may still be read, but this incomplete current
    # geometry must never publish a map permitting the next arm motion.
    assert [item['previous_id'] for item in after['mapping_unresolved']] == ['obstacle']
    assert runner.task_progress.scene_entities['obstacle']['visibility'] == 'unseen'
    assert 'table' not in [item['previous_id'] for item in after['mapping_unresolved']]
    from jev_mujoco.algorithms import AlgorithmError, Algorithms
    calls = []
    algorithms = Algorithms(curobo_request=lambda *args: calls.append(args) or {})
    for scope in ('observation', 'selected'):
        with pytest.raises(AlgorithmError, match='semantic_scene_identity_unresolved'):
            algorithms.build_scene(after, 'source', exclusion_state=scope)
    assert not calls


def test_standalone_place_retains_held_number_and_initial_transport_capacity():
    progress = TaskProgress('place the held object')
    progress.scene_entities = {f'part-{i}': {'id': f'part-{i}', 'kind': 'cube',
        'roles': ['transport', 'collision'], 'display_id': i} for i in range(1, 4)}
    progress.scene_entities['box'] = {'id': 'box', 'kind': 'bin', 'roles': ['container', 'collision'],
                                     'display_id': 4}
    progress.scene_inventory_observation_id = 'saved-head'
    progress.select({'id': 'part-3', 'kind': 'cube'}, destination_id='box')
    planned = progress.placement_request('box')
    assert planned['target_count'] == 3
    assert planned['slot_id'] == 3
    assert progress.visual_progress()['part-3']['slot_id'] == planned['slot_id']
