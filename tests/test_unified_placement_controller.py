import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision
from placement_intent_fixture import Answers
from test_placement_zones import observation

from luna_mujoco.runtime.controller import PolicyController


def controller(tmp_path):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.state.update(target_id='held', destination_id='area', arm='A')
    task = {'instruction': '放在杯子左边的桌面上，以机器人视角为准', 'destination_ref': 'area'}
    runner.task_spec = SimpleNamespace(model_dump=lambda **kwargs: task)
    runner.active_object_task = None
    runner.task_progress = SimpleNamespace(records={'held': {}})
    return runner


def confirmed_relation_case(tmp_path, *, product=False):
    from luna_mujoco.robot_skills.placement_resolution import digest

    runner = controller(tmp_path)
    runner.run_id = 'run-a'
    task = runner.task_spec.model_dump()
    task['processing'] = ['unpack'] if product else []
    runner.task_spec = SimpleNamespace(model_dump=lambda **kwargs: task)
    root = 'input' if product else 'held'
    runner.object_tasks = {root: runner.task_spec}
    runner.task_progress.records['held'].update(destination_ref='area', root_input_id=root,
                                               processing_history=['unpack'] if product else [])
    if product:
        runner.task_progress.records['input'] = {'status': 'consumed', 'output_entity_id': 'held'}
        runner.task_progress.records['held'].update(input_entity_id='input', creation_evidence={
            'observation_id': 'creation', 'request_hash': 'creation-hash', 'action_id': 'unpack-action'})
    proof = {'judgments': {'destination_requirement_matches': {'status': 'yes'}},
        'evidence': {'source': 'openai_decisions', 'observation_id': 'initial', 'request_hash': 'binding-hash'},
        'confirmed_task_scope': {'owner_id': runner.run_id, 'generation': 1,
            'task_sha256': digest(task), 'input_ids': [root], 'destination_ref': 'area',
            'request_hash': 'binding-hash'}}
    runner.state['object_task_binding_evidence'] = {root: proof}
    obs = observation()
    obs['visual_inventory'] = {'identity_coverage': {'observation_id': 'obs', 'accepted_entity_ids': ['area']}}
    return runner, task, obs, proof


@pytest.mark.parametrize('product', [False, True])
@pytest.mark.parametrize('additional', ['yes', 'no', 'unknown'])
async def test_confirmed_final_destination_is_reused_without_fabricating_a_model_answer(tmp_path, product, additional):
    runner, _, obs, _ = confirmed_relation_case(tmp_path, product=product)
    runner._judge = AsyncMock(return_value={'judgments': {
        'placement_additional_spatial_constraint': {'status': additional}}})
    result = await runner.execute('perceive_placement_zone', {'observation': obs, 'latest_capture': {'views': {}}})
    assert result['status'] == ('unknown' if additional == 'unknown' else 'success')
    if additional != 'unknown':
        assert result['semantic_scope'] == ('entire_region' if additional == 'no' else 'candidate_selection')
    requests = runner._judge.call_args.args[1]
    assert [request.name for request in requests] == ['placement_additional_spatial_constraint']
    assert 'placement_destination_matches' not in result['evidence']['judgments']
    proof = result['evidence']['reused_destination_binding']
    assert proof['original_evidence']['observation_id'] == 'initial'
    assert proof['current_identity_observation_id'] == 'obs'
    assert proof['current_pose_or_completion_claimed'] is False


@pytest.mark.parametrize('damage', ['owner', 'generation', 'task', 'destination', 'hash', 'verdict',
    'current_identity', 'old_identity', 'scope_missing', 'processing', 'lineage', 'creation'])
def test_confirmed_relation_reuse_requires_unchanged_scope_and_verified_lineage(tmp_path, damage):
    from luna_mujoco.runtime.placement import confirmed_destination_binding

    runner, task, obs, proof = confirmed_relation_case(tmp_path, product=True)
    if damage == 'owner': runner.run_id = 'other-run'
    elif damage == 'generation': obs['generation'] += 1
    elif damage == 'task': task['instruction'] = 'different request'
    elif damage == 'destination': task['destination_ref'] = 'other'
    elif damage == 'hash': proof['evidence']['request_hash'] = 'other'
    elif damage == 'verdict': proof['judgments']['destination_requirement_matches']['status'] = 'unknown'
    elif damage == 'current_identity': obs['visual_inventory']['identity_coverage']['accepted_entity_ids'] = []
    elif damage == 'old_identity': obs['visual_inventory']['identity_coverage']['observation_id'] = 'initial'
    elif damage == 'scope_missing': proof.pop('confirmed_task_scope')
    elif damage == 'processing': runner.task_progress.records['held']['processing_history'] = []
    elif damage == 'lineage': runner.task_progress.records['input']['output_entity_id'] = 'other'
    elif damage == 'creation': runner.task_progress.records['held']['creation_evidence'].pop('action_id')
    assert confirmed_destination_binding(runner, obs, 'held', 'area', task) is None


def test_production_catalog_has_unified_place_and_no_transport_drop_branch(tmp_path):
    catalog = controller(tmp_path).registry.describe()
    assert {'perceive_placement_zone', 'resolve_placement', 'plan_place', 'execute_place', 'verify_place'} <= catalog.keys()
    assert not {'plan_transport', 'execute_transport', 'drop', 'verify_drop'} & catalog.keys()
    assert catalog['plan_place']['requires']['placement_resolved']


async def test_perception_records_decisions_intent_and_geometry_without_motion(tmp_path):
    runner = controller(tmp_path)
    answers = Answers(placement_destination_matches=True, placement_additional_spatial_constraint=True)
    async def judge(context, requests, *, extra=None, **kwargs):
        if requests[0].name == 'placement_xy':
            return {'judgments': {'placement_xy': {'status': 'proposed', 'value': requests[0].choices[0]}}}
        return await answers(requests, extra)
    runner._judge = judge
    obs = observation()
    context = {'observation': obs, 'latest_capture': {'views': {'1': {}, '2': {}, '3': {}}},
               'visual_inventory_reference': {'entities': {item['id']: item for item in obs['objects']}}}
    before = deepcopy(obs)
    result = await runner.execute('perceive_placement_zone', context)
    assert result['status'] == 'success'
    assert result['semantic_scope'] == 'candidate_selection'
    assert context['placement_zone']['semantic_scope'] == 'candidate_selection'
    assert '杯子左边' in context['placement_zone']['original_requirement']['instruction']
    assert context['placement_zone']['observation_id'] == 'obs'
    assert obs == before
    assert len(answers.calls) == 1


async def test_region_freezes_current_transfer_scope_separately_from_final_destination(tmp_path):
    runner = controller(tmp_path)
    active = {'instruction': '先在工作区拆包，再放入盒子', 'destination_ref': 'box', 'work_area_ref': 'area'}
    runner.active_object_task = SimpleNamespace(model_dump=lambda **kwargs: active)
    runner._judge = AsyncMock(return_value={'judgments': {'placement_destination_matches': {'status': 'yes'}, 'placement_additional_spatial_constraint': {'status': 'no'}}})
    result = await runner.execute('perceive_placement_zone', {'observation': observation(), 'latest_capture': {'views': {}}})
    assert result['placement_zone']['current_transfer'] == {
        'bound_destination_ref': 'area', 'final_destination_ref': 'box',
        'work_area_ref': 'area', 'scope': 'declared_work_area'}


async def test_plain_container_or_surface_needs_no_direction_questionnaire(tmp_path):
    runner = controller(tmp_path)
    answers = Answers(placement_destination_matches=True, placement_additional_spatial_constraint=False)
    async def judge(context, requests, *, extra=None, **kwargs):
        if requests[0].name == 'placement_xy':
            return {'judgments': {'placement_xy': {'status': 'proposed', 'value': requests[0].choices[0]}}}
        return await answers(requests, extra)
    runner._judge = judge
    context = {'observation': observation(), 'latest_capture': {'views': {}}}
    result = await runner.execute('perceive_placement_zone', context)
    assert result['placement_zone']['semantic_scope'] == 'entire_region'
    assert [r.name for requests, _ in answers.calls for r in requests] == ['placement_destination_matches', 'placement_additional_spatial_constraint']


async def test_uncertain_region_permission_cannot_establish_a_ready_zone(tmp_path):
    runner = controller(tmp_path)
    runner._judge = AsyncMock(return_value={'judgments': {
        'placement_destination_matches': {'status': 'yes'},
        'placement_additional_spatial_constraint': {'status': 'unknown', 'probability': .77}}})
    context = {'observation': observation(), 'latest_capture': {'views': {}}}
    result = await runner.execute('perceive_placement_zone', context)
    assert result['status'] == 'unknown'
    assert result['reason'] == 'placement_region_semantics_unknown'
    assert 'placement_zone' not in context
    assert runner.state['facts']['placement_zone_ready'] is False


@pytest.mark.parametrize('destination', ['yes', 'no', 'unknown'])
@pytest.mark.parametrize('additional', ['yes', 'no', 'unknown'])
async def test_destination_and_internal_constraints_have_independent_gates(tmp_path, destination, additional):
    runner = controller(tmp_path)
    runner._judge = AsyncMock(return_value={'judgments': {
        'placement_destination_matches': {'status': destination},
        'placement_additional_spatial_constraint': {'status': additional}}})
    context = {'observation': observation(), 'latest_capture': {'views': {}}}
    result = await runner.execute('perceive_placement_zone', context)
    if destination == 'yes' and additional in ('yes', 'no'):
        assert result['status'] == 'success'
        assert result['semantic_scope'] == ('candidate_selection' if additional == 'yes' else 'entire_region')
    else:
        assert result['status'] == 'unknown'
        assert 'placement_zone' not in context
    runner._judge.assert_awaited_once()


@pytest.mark.parametrize('damage', ['epoch', 'capture', 'generation'])
async def test_candidate_capture_preflight_runs_before_endpoint_search(tmp_path, damage):
    runner = controller(tmp_path)
    runner.call_algorithm = AsyncMock()
    runner.verify_scene = AsyncMock()
    runner.prepare_attachment = AsyncMock()
    zone = {'semantic_scope': 'candidate_selection', 'observation_id': 'current', 'generation': 2}
    capture = {'observation_id': 'current', 'generation': 2}
    context = {'placement_zone': zone, 'latest_capture': capture,
               'capture_epochs': {'current': runner.observation_epoch}}
    if damage == 'epoch': context['capture_epochs']['current'] -= 1
    if damage == 'capture': zone['observation_id'] = 'old'
    if damage == 'generation': zone['generation'] = 1
    result = await runner.execute('resolve_placement', context, {'preference': 'clearance'})
    assert result['status'] == 'unknown'
    assert result['reason'] == 'placement_candidate_capture_stale'
    runner.call_algorithm.assert_not_called()
    runner.verify_scene.assert_not_called()
    runner.prepare_attachment.assert_not_called()


async def test_region_judgment_does_not_serialize_raw_depth_provenance_arrays(tmp_path):
    runner = controller(tmp_path)
    obs = observation()
    obs['objects'][0]['placement_reference']['measurement_sources'] = {'pixels_rc': np.zeros((30, 2))}
    async def judge(context, requests, *, extra=None, **kwargs):
        json.dumps(extra)
        assert 'measurement_sources' not in extra['configured_region_constraints']
        return {'judgments': {request.name: {'status': 'yes'} for request in requests}}
    runner._judge = judge
    result = await runner.execute('perceive_placement_zone', {'observation': obs, 'latest_capture': {'views': {}}})
    assert result['status'] == 'success'


@pytest.mark.parametrize('source, configured', [('configured_xy_measured_support', True), ('measured_depth', False)])
async def test_region_question_uses_actual_configuration_provenance(tmp_path, source, configured):
    runner = controller(tmp_path)
    obs = observation()
    obs['objects'][0]['placement_reference']['source'] = source
    async def judge(context, requests, *, extra=None, **kwargs):
        assert ('need not have a visible marking' in requests[0].instructions) is configured
        return {'judgments': {'placement_destination_matches': {'status': 'yes'},
                             'placement_additional_spatial_constraint': {'status': 'no'}}}
    runner._judge = judge
    result = await runner.execute('perceive_placement_zone', {'observation': obs, 'latest_capture': {'views': {}}})
    assert result['status'] == 'success'


async def test_failed_reinterpretation_removes_old_resolution(tmp_path):
    runner = controller(tmp_path)
    runner._judge = AsyncMock(side_effect=ValueError('fresh_observation_required_before_judgment'))
    runner.state['facts'].update(placement_zone_ready=True, placement_resolved=True, place_plan_ready=True)
    context = {'observation': observation(), 'latest_capture': {'views': {}},
        'visual_inventory_reference': {'entities': {}}, 'placement_zone': {'old': True},
        'placement_resolution': {'old': True}, 'place_plan': {'old': True}}
    with pytest.raises(ValueError, match='fresh_observation_required'):
        await runner.execute('perceive_placement_zone', context)
    assert not any(key in context for key in ('placement_zone', 'placement_resolution', 'place_plan'))
    assert not runner.state['facts']['placement_zone_ready']


async def test_failed_surface_refresh_revokes_old_plan_before_grounding(tmp_path):
    runner = controller(tmp_path)
    runner.state['destination_id'] = 'surface-region:support'
    runner.state['facts'].update(placement_zone_ready=True, placement_resolved=True, place_plan_ready=True)
    runner._bind_region = Mock(side_effect=ValueError('placement_horizontal_surface_unobserved'))
    context = {'observation': observation(), 'placement_zone': {'old': True},
               'placement_resolution': {'old': True}, 'place_plan': {'old': True}}
    with pytest.raises(ValueError, match='placement_horizontal_surface_unobserved'):
        await runner.execute('perceive_placement_zone', context)
    assert not any(key in context for key in ('placement_zone', 'placement_resolution', 'place_plan'))
    assert not runner.state['facts']['placement_resolved']
    assert not runner.state['facts']['place_plan_ready']


async def test_plan_place_requires_explicit_resolution_without_calling_responses(tmp_path):
    runner = controller(tmp_path)
    runner.compiler = Mock()
    context = {'observation': observation()}
    with pytest.raises(ValueError, match='placement_resolution_required'):
        await runner.execute('plan_place', context)
    assert runner.compiler.mock_calls == []


def test_binding_refreshes_container_polygon_and_support_identity(tmp_path):
    from luna_mujoco.task_progress import TaskProgress

    runner = controller(tmp_path)
    runner.task_progress = TaskProgress('put into box')
    runner.task_progress.current_id = 'held'
    runner.task_progress.records['held'] = {'id': 'held', 'display_id': 1}
    obs = observation()
    destination = obs['objects'][0]
    destination['kind'] = 'bin'
    destination['placement_reference'].pop('support_entity_id', None)
    first = runner._bind_region(obs)
    obs['observation_id'] = 'new'
    polygon = [[.1, .1], [.4, .1], [.5, .2], [.4, .4], [.1, .4]]
    destination['placement_reference'].update(region_polygon_world_xy_m=polygon, observation_id='new')
    second = runner._bind_region(obs)
    assert second['layout']['region_polygon_world_xy_m'] == polygon
    assert second['layout']['observation_id'] == 'new'
    assert second['layout']['support_entity_id'] == 'area'
    assert first['layout_revision'] != second['layout_revision']


async def test_failed_plan_does_not_leave_previous_plan_executable(tmp_path):
    runner = controller(tmp_path)
    runner.verify_scene = AsyncMock(side_effect=ValueError('stale scene'))
    context = {'placement_resolution': {}, 'place_plan': {'old': True},
               'place_plan_binding': {'old': True}}
    runner.state['facts']['place_plan_ready'] = True
    with pytest.raises(ValueError, match='stale scene'):
        await runner.execute('plan_place', context)
    assert 'place_plan' not in context and 'place_plan_binding' not in context
    assert not runner.state['facts']['place_plan_ready']


@pytest.mark.parametrize('container', [False, True])
async def test_native_graph_runs_decisions_region_solver_and_three_stage_planner(tmp_path, monkeypatch, container):
    from test_resolve_placement import setup

    from luna_mujoco.runtime.native_graph import NativeGraphExecutor
    from luna_mujoco.task_progress import TaskProgress

    algorithm, service, obs, scene, held, _, _ = setup(monkeypatch, container=container)
    runner = controller(tmp_path)
    runner.algorithms = algorithm
    runner.state.update(target_id='seen-red', destination_id='demo-destination', arm='A',
                        generation=runner.simulation.robot_snapshot()['generation'])
    runner.state['facts'].update(target_selected=True, destination_selected=True, arm_selected=True,
        observed=True, grasp_held=True, scene_ready=True, needs_place=True)
    runner.task_progress = TaskProgress('put into box' if container else 'put onto support')
    runner.task_progress.current_id = 'seen-red'
    runner.task_progress.records['seen-red'] = {'id': 'seen-red', 'display_id': 1, 'status': 'selected'}
    answers = Answers(placement_destination_matches=True, placement_additional_spatial_constraint=False)
    async def judge(context, requests, *, extra=None, **kwargs):
        if requests[0].name == 'placement_xy':
            return {'judgments': {'placement_xy': {'status': 'proposed', 'value': requests[0].choices[0]}}}
        return await answers(requests, extra)
    runner._judge = judge
    runner.verify_scene = AsyncMock()
    runner.measured_state = lambda ctx: ctx['observation']
    runner.prepare_attachment = AsyncMock()
    context = {'observation': obs, 'scene': scene, 'holding': held,
        'planning_settings': TEST_SETTINGS.model_dump(), 'latest_capture': {'views': {'1': {}, '2': {}, '3': {}}}}
    names = ['perceive_placement_zone', 'resolve_placement', 'plan_place']
    graph = {'version': 2, 'id': 'placement', 'entry': names[0], 'subgraphs': [],
        'nodes': [{'id': name, 'type': 'skill', 'skill': name, 'arguments': {}} for name in names]
            + [{'id': 'done', 'type': 'end', 'status': 'success', 'reason': 'checked plan'}],
        'edges': [{'id': 'edge_' + name, 'source': name, 'target': (names + ['done'])[index + 1],
                   'on': 'success'} for index, name in enumerate(names)]}
    invoked = []
    async def invoke(name, arguments):
        invoked.append(name)
        runner.begin_step(name)
        result = await runner.registry.execute(name, runner.state['facts'], context, arguments)
        for fact in runner.registry.contracts[name].provides:
            runner.state['facts'][fact] = True
        return result
    await NativeGraphExecutor(invoke, target_ids=[], bindings={}).run(graph)
    assert invoked == names
    assert set(context['place_plan']['phases']) == {'approach', 'place', 'retreat'}
    assert context['place_plan']['placement_zone']['intent']['relation'] == ('inside' if container else 'on')
    assert service.calls[-1][0] == '/carrying/validate'
