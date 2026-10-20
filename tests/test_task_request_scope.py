import pytest
from geometry_perception_fixture import GeometryPerception

from jev_mujoco.decisions import Decisions
from jev_mujoco.task_progress import TaskProgress


@pytest.mark.parametrize('instruction', ['抓取盒子中的绿色物体放到黄色区域', '把盒子中多个物体搬到桌面',
                                       'Move the red and green objects to the yellow area'])
def test_no_literal_quantity_does_not_create_a_quantity_decision(instruction):
    from jev_mujoco.task_request import normalize_request_scope, request_scope_questions
    assert 'quantity' not in request_scope_questions(instruction)
    assert normalize_request_scope({})['requested_count'] is None


@pytest.mark.parametrize('instruction', ['拿一个物体', '拿单个物体', 'Move a red object', 'Move two objects'])
def test_explicit_quantity_still_requires_jev_to_interpret_its_task_role(instruction):
    from jev_mujoco.task_request import request_scope_questions
    assert 'quantity' in request_scope_questions(instruction)


@pytest.mark.parametrize('instruction', [
    'Move a red object and a green object into a blue box',
    'Move an orange object and a green object into the box',
    'Move a single red object and a single green object into the box',
    '将单个红色物体和单个绿色物体放进盒子',
])
def test_repeated_explicit_singulars_offer_the_combined_target_quantity(instruction):
    from jev_mujoco.task_request import request_scope_questions

    # Candidate coverage must retain two target mentions even when a destination
    # also has an article. JEV, not the candidate extractor, decides their roles.
    assert '2' in request_scope_questions(instruction)['quantity']['criteria']


def test_exact_quantity_limits_inventory_and_prevents_extra_pick():
    from test_task_progress import inventory_observation
    progress = TaskProgress('从盒子拿一个', route='table_box', requested_count=1,
                            selection_order='inventory')
    objects = [{'id': name, 'kind': 'cube', 'position': [distance, 0, .6]}
               for name, distance in [('far', .7), ('near', .3)]]
    progress.observe(objects, 'head')
    progress.assess_membership({'far': 'required', 'near': 'required'})
    observation = inventory_observation(objects)
    with pytest.raises(ValueError, match='inventory_quantity_mismatch'):
        progress.freeze_inventory(observation)
    progress.assess_membership({'far': 'excluded', 'near': 'required'})
    progress.freeze_inventory(observation)
    assert [obj['id'] for obj in progress.snapshot()['objects']] == ['near']
    assert progress.placement_request('bin')['target_count'] == 1
    assert progress.terminal_status('complete') == 'needs_attention'
    progress.select(objects[1], 'bin')
    progress.complete('near', observation_id='released')
    assert progress.terminal_status('complete') == 'succeeded'
    with pytest.raises(ValueError, match='quantity|inventory'):
        progress.select(objects[0], 'bin')


def test_default_inventory_order_is_not_current_tcp_distance():
    from test_task_progress import inventory_observation
    progress = TaskProgress('全部搬走', route='table_box', selection_order='inventory')
    objects = [{'id': name, 'kind': 'cube', 'position': [distance, 0, .6]}
               for name, distance in [('one', .3), ('two', .7)]]
    progress.observe(objects, 'head')
    progress.assess_membership({'one': 'required', 'two': 'required'})
    observation = inventory_observation(objects)
    observation['arms']['A']['tcp'][0][3] = .8
    progress.freeze_inventory(observation)
    assert [row['id'] for row in progress.rank_candidates(observation, arm='A')] == ['one', 'two']
    with pytest.raises(ValueError, match='inventory_order'):
        progress.select(objects[1], 'bin')


async def test_jev_selects_target_quantity_separately_from_environment_counts(monkeypatch):
    calls = []

    async def ask(self, kind, state, questions, **kwargs):
        calls.append(questions)
        return {'task': 'grasp_place', 'route': 'box_table', 'arm': 'auto', 'scope': 'single',
                'visual_semantics': 'optional', 'kind': 'any', 'color': 'any', 'position': 'any',
                'destination_kind': 'marker', 'destination_color': 'yellow', 'destination_position': 'any',
                'quantity': '1', 'selection_order': 'inventory',
                'environment_red': 'yes', 'environment_black': 'yes', 'environment_yellow': 'no'}

    monkeypatch.setattr(Decisions, 'ask', ask)
    result = await Decisions(None, lambda *args: None).bind_request(
        '桌面有两个红色和三个黑色物体，把盒子里一个物体放到黄色区域')
    assert 'quantity' in calls[0]
    assert set(calls[0]['quantity']['criteria']) >= {'1', '2', '3', 'unbounded'}
    assert result['requested_count'] == 1
    assert result['environment_queries'] == [
        {'kind': 'object', 'color': 'black', 'prompt': 'black object'},
        {'kind': 'object', 'color': 'red', 'prompt': 'red object'}]


def test_declared_environment_is_observed_without_changing_source_query():
    from test_broad_perception import PromptSegmenter
    from test_perception import capture_fixture


    capture, masks = capture_fixture()
    for view in capture['views'].values():
        view['rgb'][masks[1]] = [10, 10, 10]
    detector = PromptSegmenter({'red cube': [masks[0]], 'black object': [masks[1]]})
    result = GeometryPerception(detector, kinds={'second': 'object'},
        prompts={'first': 'red cube', 'second': 'black object'}).observe(capture, checkpoint='fine', source_hint={
        'kind': 'cube', 'color': 'red', 'camera_ids': ['3'],
        'environment_queries': [{'kind': 'object', 'color': 'black', 'prompt': 'black object'}]})
    assert detector.calls == ['red cube', 'black object']
    assert {obj['color'] for obj in result['objects']} == {'red', 'black'}
    assert 'second' in result['scene_entities']


def test_missing_declared_obstacle_is_not_silently_deleted_in_full_observation():
    from test_broad_perception import PromptSegmenter
    from test_perception import capture_fixture


    capture, masks = capture_fixture()
    result = GeometryPerception(PromptSegmenter({'red cube': [masks[0]]}), identities=('first',)).observe(
        capture, checkpoint='fine', source_hint={
            'kind': 'cube', 'color': 'red', 'camera_ids': ['3'],
            'scene_queries': [
                {'kind': 'cube', 'color': 'red', 'prompt': 'red cube', 'entity_ids': ['first']},
                {'kind': 'object', 'color': 'black', 'prompt': 'black object', 'entity_ids': ['obstacle']} ]})
    assert 'obstacle' in result['scene_entities']
    assert result['mapping_unresolved'][0]['previous_id'] == 'obstacle'



def test_over_quota_membership_is_rejected_without_losing_jev_reassessment_choices():
    progress = TaskProgress('搬一个红色物体', scope='collection', route='table_box',
                            requested_count=1, selection_order='explicit')
    objects = [{'id': identifier, 'kind': 'cube', 'color': 'red'} for identifier in ('red-a', 'red-b')]
    progress.observe(objects, 'head')
    before = progress.snapshot()
    with pytest.raises(ValueError, match='requested_quantity_membership_ambiguous'):
        progress.assess_membership({'red-a': 'required', 'red-b': 'required'})
    assert progress.snapshot() == before
    assert progress.snapshot()['membership_pending_ids'] == ['red-a', 'red-b']
    progress.assess_membership({'red-a': 'required', 'red-b': 'excluded'})
    progress.assign_display_ids({'objects': objects})
    progress.select(objects[0], 'bin')
    assert progress.current_id == 'red-a'


def test_each_destination_retains_global_task_capacity():
    progress = TaskProgress('把一个红色放进蓝盒，一个绿色放进黄盒', scope='collection', route='table_box',
                            requested_count=2, selection_order='explicit')
    objects = [{'id': 'red', 'kind': 'cube', 'color': 'red'},
               {'id': 'green', 'kind': 'cube', 'color': 'green'}]
    progress.observe(objects, 'head')
    progress.assess_membership({'red': 'required', 'green': 'required'})
    progress.select(objects[0], 'blue-bin')
    progress.complete('red', observation_id='released-red')
    progress.select(objects[1], 'yellow-bin')
    placement = progress.placement_request('yellow-bin')
    assert placement['target_ids'] == ['green', 'red']
    assert placement['target_count'] == 2


async def test_direct_single_inventory_uses_membership_step_and_keeps_grasp_completion(tmp_path):
    from test_runner import Algorithms, Sim, wait
    from test_runner import Decisions as RunnerDecisions

    from jev_mujoco.runner import Runner

    class InventoryChoices(RunnerDecisions):
        async def bind_request(self, instruction):
            return {**await super().bind_request(instruction),
                    'requested_count': 1, 'selection_order': 'inventory'}

        async def choose_skill(self, state, registry):
            if state['facts'].get('observed') and not state['facts'].get('task_assessed'):
                return 'assess_task'
            return await super().choose_skill(state, registry)

        async def assess_task(self, state):
            return {'judgment': 'continue', 'membership': {'black': 'required'}}

    runner = Runner(Sim(), Algorithms(), lambda record: InventoryChoices(), tmp_path)
    await runner.start('抓取一个物体')
    await wait(runner)
    completed = [event['title'] for event in runner.events if event['kind'] == 'skill_complete']
    assert 'assess_task' in completed
    assert completed.index('assess_task') < completed.index('identify_target')
    assert runner.state['status'] == 'succeeded'
    assert runner.task_progress.scope == 'single'
    assert runner.task_progress.inventory_initialized is True
    assert runner.task_progress.records['black']['display_id'] == 1
    runner.reset_transfer({})
    assert runner.state['facts']['collection_task'] is True
    assert runner.state['facts']['task_assessed'] is False


async def test_direct_inventory_cannot_confirm_unknown_target_during_identification(tmp_path):
    from test_rgbd_algorithms import observed_scene
    from test_runner import Sim

    from jev_mujoco.runner import Runner

    runner = Runner(Sim(), None, None, tmp_path)
    runner.state.update(task='grasp', generation=1)
    runner.task_progress = TaskProgress('抓取一个物体', scope='single',
                                        requested_count=1, selection_order='inventory')
    observation = observed_scene()
    runner.task_progress.observe(observation['objects'], observation['observation_id'])
    with pytest.raises(ValueError, match='target_not_in_task_inventory'):
        await runner.execute('identify_target', {'observation': observation}, {'target_id': 'seen-red'})
    assert runner.task_progress.membership == {'seen-red': 'unknown'}
    assert runner.task_progress.current_id is None
