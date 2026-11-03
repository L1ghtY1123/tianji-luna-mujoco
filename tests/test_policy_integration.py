"""Exercise real model adapters, inventory binding and the runtime with physical-service fixtures."""
import asyncio
import json
from copy import deepcopy

import pytest
from test_agent_model_layer import Client, plan_data
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.agent.decisions import DecisionsJudgments
from luna_mujoco.agent.responses import ResponsesCompiler
from luna_mujoco.runtime.controller import PolicyController


class FixedInventoryVision(DemoVision):
    """Use the real compiler's permanent IDs with the existing measured-service fixture."""
    def observe(self, capture, **kwargs):
        output = super().observe(capture, **kwargs)
        scene = kwargs['source_hint']['scene_entities']
        targets = [identifier for identifier, item in scene.items() if item.get('visual_membership') == 'required']
        table = next(identifier for identifier, item in scene.items() if item['kind'] == 'table')
        aliases = {f'pink-{index}': identifier for index, identifier in enumerate(targets, 1)}
        aliases.update(table=table, **{'source-bin': table})

        def remap(value, field=None):
            if isinstance(value, str):
                return aliases.get(value, value) if field not in {'kind', 'color', 'source'} else value
            if isinstance(value, list):
                return [remap(item, field) for item in value]
            if isinstance(value, dict):
                return {(aliases.get(key, key) if key != 'table' or field in {'scene_entities', 'instance_masks'}
                         else key): remap(item, key) for key, item in value.items()}
            return value
        output = remap(output)
        for item in output['objects']:
            item['color'] = 'red'
        return output


class ModelHTTP(Client):
    def __init__(self, plan, *, probabilities=None):
        super().__init__([plan])
        # Ordinary transfer fixtures have no restriction within the destination.
        self.probabilities = {'placement_additional_spatial_constraint': .01, **(probabilities or {})}

    def post(self, url, **kwargs):
        if url.endswith('/responses'):
            return super().post(url, **kwargs)
        import httpx
        self.requests.append((url, kwargs['json']))
        answers = []
        for question in kwargs['json']['questions']:
            if question['type'] == 'choice':
                choices = [item['value'] for item in question['choices']]
                choice = ('global' if question['name'] == 'mapping_scope' else
                          'stop' if 'stop' in choices else choices[0])
                answers.append({'type': 'choice', 'name': question['name'], 'choice': choice, 'confidence': .99})
            else:
                answers.append({'type': 'predicate', 'name': question['name'],
                                'probability': self.probabilities.get(question['name'], .99)})
        return httpx.Response(200, json={'answers': answers})


@pytest.mark.parametrize('final_complete', [False, True])
@pytest.mark.parametrize('order', ['inventory', 'explicit', 'nearest'])
async def test_uncertain_initial_closure_does_not_erase_known_targets_or_prove_completion(tmp_path, final_complete, order):
    class UncertainClosureVision(FixedInventoryVision):
        def observe(self, capture, **kwargs):
            observation = super().observe(capture, **kwargs)
            observation['demo_inventory_check'].update(status='unknown', confidence=.69)
            return observation

    class ClosureHTTP(ModelHTTP):
        def post(self, url, **kwargs):
            if url.endswith('/decisions'):
                final = any(q['name'] == 'goal_satisfied' for q in kwargs['json']['questions'])
                self.probabilities['source_complete'] = .99 if final and final_complete else .5
            return super().post(url, **kwargs)

    plan = transfer_plan()
    plan['task']['selection_order'] = order
    http = ClosureHTTP(plan)
    compiler = ResponsesCompiler(registry={}, configured_regions={'demo-destination': {}},
                                 api_key='test', client=http)
    vision = UncertainClosureVision(count=1)
    controller = PolicyController(DemoRobot(), vision, None, tmp_path, planning_settings=TEST_SETTINGS,
                                  compiler=compiler, judgments=DecisionsJudgments(api_key='test', client=http))
    compiler.registry = controller.registry
    await controller.start(plan['task']['instruction'])
    await asyncio.wait_for(controller.worker, 10)
    if order == 'nearest':
        assert vision.target_order == []
        assert controller.state['error'].endswith('inventory_order_unconfirmed')
        assert controller.state['failure_report']['checks'][0]['actual'] == 'nearest'
        return
    assert len(vision.target_order) == 1
    assert controller.state['visible_inventory_resolution']['status'] == 'unknown'
    assert controller.state['status'] == ('succeeded' if final_complete else 'needs_attention')
    if not final_complete:
        assert controller.state['error'].endswith('task_completion_unconfirmed')
    assert controller.compiler.budget.used == 1
    assert controller.state['task_inventory']['execution_order']['source'] == 'confirmed_inventory_order'


def transfer_plan(count=1, mapping_scope='global'):
    from graph_contract_fixtures import call, linear_graph
    plan = plan_data()
    plan['entities'] = [item for item in plan['entities'] if item['ref'] != 'tray']
    for index in range(2, count + 1):
        entity = deepcopy(plan['entities'][-1])
        entity['ref'] = f'item-{index}'
        plan['entities'].append(entity)
        plan['task']['target_refs'].append(entity['ref'])
    plan['task'].update(destination_ref='demo-destination', completion_conditions=['items visibly placed'])
    root_nodes = [call('initial_observe', 'observe_scene', capture_mode='reuse_current'),
                  call('confirm_inventory', 'confirm_inventory')]
    scopes = []
    for index, identifier in enumerate(plan['task']['target_refs']):
        prefix = f'item{index}_'
        skills = [('identify_target', {'target_id': '$target'}),
            ('identify_destination', {'destination_id': '$destination', 'destination_kind': 'support_region', 'orientation_mode': 'preserve'}),
            ('perceive_placement_zone', {}), ('choose_arm', {'arm': 'auto'}),
            ('build_scene', {'scope': mapping_scope, 'reason': 'current task collision evidence', 'entity_ids': [], 'exclusion_state': 'selected'}),
            ('generate_grasps', {}), ('plan_grasp', {}), ('execute_grasp', {}), ('verify_grasp', {}),
            ('resolve_placement', {}), ('plan_place', {}), ('execute_place', {}), ('verify_place', {}),
            ('observe_scene', {}), ('build_scene', {'exclusion_state': 'observation'}),
            ('move_to_observe', {'viewpoint': 'configured'}), ('observe_scene', {})]
        nodes = [call(prefix + str(i) + '_' + name, name, **args) for i, (name, args) in enumerate(skills)]
        scope = linear_graph(nodes, identifier=prefix + 'task', finish=prefix+'done')
        scope.pop('version')
        scope.pop('subgraphs')
        item_task = deepcopy(plan['task'])
        item_task.update(scope='ids', target_refs=[identifier], quantity=1, selection_order='explicit', completion_pose=None)
        scope.update(input_id=identifier, task=item_task)
        scopes.append(scope)
        root_nodes.append({'type': 'subgraph', 'id': prefix+'call', 'subgraph_id': scope['id']})
    root_nodes.append(call('final_verification','verify_task_completion'))
    plan['graph'] = linear_graph(root_nodes)
    plan['graph']['subgraphs'] = scopes
    return plan


@pytest.mark.parametrize('order', ['inventory', 'explicit'])
async def test_model_can_choose_target_order_without_user_order_instruction(tmp_path, order):
    plan = transfer_plan(2)
    plan['task'].update(selection_order=order, target_refs=['item-2', 'item'])
    calls = [node for node in plan['graph']['nodes'] if node['type'] == 'subgraph']
    calls[0]['subgraph_id'], calls[1]['subgraph_id'] = calls[1]['subgraph_id'], calls[0]['subgraph_id']
    http = ModelHTTP(plan)
    compiler = ResponsesCompiler(registry={}, configured_regions={'demo-destination': {}},
                                 api_key='test', client=http)
    vision = FixedInventoryVision(count=2)
    controller = PolicyController(DemoRobot(), vision, None, tmp_path, planning_settings=TEST_SETTINGS,
                                  compiler=compiler, judgments=DecisionsJudgments(api_key='test', client=http))
    compiler.registry = controller.registry
    await controller.start(plan['task']['instruction'])
    await asyncio.wait_for(controller.worker, 10)
    assert controller.state['status'] == 'succeeded', controller.state.get('error')
    assert vision.target_order == list(controller.task_spec.target_refs)
    assert vision.target_order == controller.task_inventory['selected_ids']
    assert controller.compiler.budget.used == 1


@pytest.mark.parametrize('count,recording', [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize('mapping_scope', ['global', 'decide'])
async def test_compiler_inventory_graph_and_decisions_complete_transfers(tmp_path, count, recording, mapping_scope):
    from luna_mujoco.experiments import ExperimentOptions, ExperimentStore

    class ArchiveRobot(DemoRobot):
        def snapshot(self):
            return self.robot_snapshot() | {'objects': []}

    plan = transfer_plan(count, mapping_scope)
    http = ModelHTTP(plan)
    compiler = ResponsesCompiler(registry={}, configured_regions={'demo-destination': {}},
                                 api_key='test', client=http)
    judgments = DecisionsJudgments(api_key='test', client=http)
    vision = FixedInventoryVision(count=count)
    store = ExperimentStore(tmp_path / 'experiments') if recording else None
    controller = PolicyController(ArchiveRobot(), vision, None, tmp_path, planning_settings=TEST_SETTINGS,
                                  compiler=compiler, judgments=judgments, experiments=store)
    compiler.registry = controller.registry
    await controller.start(plan['task']['instruction'], experiment=ExperimentOptions(record_video=False))
    await asyncio.wait_for(controller.worker, 10)
    if store:
        store.close()
        assert store.get(controller.run_id)['errors'] == []
    assert controller.state['status'] == 'succeeded', (
        controller.state.get('error'), controller.graph_executor.results.get('verify_place')
        if controller.graph_executor else None)
    assert len(vision.target_order) == count
    assert vision.target_order == controller.task_progress.completed_ids
    for request in controller.task_progress.placement_requests.values():
        assert request['free_placement'] is True
        assert request['layout']['method'] == 'shared_work_area'
        assert len(request['layout']['cells']) == 1
    assert controller.compiler.budget.used == 1
    endpoints = [url.rsplit('/', 1)[-1] for url, _ in http.requests]
    assert endpoints[0] == 'responses' and endpoints.count('responses') == 1
    assert set(endpoints[1:]) == {'decisions'}
    assert controller.state['task_binding_evidence']['judgments']['target_0']['status'] == 'yes'
    for identifier in controller.task_progress.completed_ids:
        source = controller.task_progress.records[identifier]['initial_source']['task_source_evidence']
        assert source['target_id'] == identifier
        assert source['source_id'] == controller.task_spec.source_ref
        assert source['source_kind'] == 'table'
        assert source['source'] == 'openai_decisions'
    planning_input = json.loads(http.requests[0][1]['input'][0]['content'][0]['text'])
    assert set(planning_input['skill_instructions']) == set(planning_input['skill_catalog'])
    assert vision.discoveries == 0
    assert controller.task_spec.target_selector == 'red items'


@pytest.mark.parametrize('probability', [.5, .1])
async def test_unconfirmed_target_membership_stops_before_robot_actions(tmp_path, probability):
    plan = transfer_plan()
    http = ModelHTTP(plan, probabilities={'target_0': probability})
    compiler = ResponsesCompiler(registry={}, configured_regions={'demo-destination': {}},
                                 api_key='test', client=http)
    vision = FixedInventoryVision(count=1)
    controller = PolicyController(DemoRobot(), vision, None, tmp_path, planning_settings=TEST_SETTINGS,
                                  compiler=compiler, judgments=DecisionsJudgments(api_key='test', client=http))
    compiler.registry = controller.registry
    await controller.start(plan['task']['instruction'])
    await asyncio.wait_for(controller.worker, 10)
    assert controller.state['status'] == 'needs_attention'
    assert controller.state['error'].endswith('task_binding_unconfirmed:target_0')
    report = controller.state['failure_report']
    check = next(row for row in report['checks'] if row['name'] == 'target_0')
    assert check['actual'] == probability
    assert check['threshold'] == .8
    assert check['question'] and check['required'] is True
    assert report['request_hash']
    assert controller.state['failed_step']['skill'] == 'confirm_inventory'
    assert vision.target_order == []
    assert controller.compiler.budget.used == 1
    assert [url.rsplit('/', 1)[-1] for url, _ in http.requests] == ['responses', 'decisions', 'decisions']
    assert 'recovery' in http.requests[-1][1]['questions'][0]['name']
    assert all('task_source_evidence' not in row.get('initial_source', {})
               for row in controller.task_progress.records.values())


async def test_vla_effect_failure_survives_retry_and_transfer_reset(tmp_path):
    from math import radians
    from types import SimpleNamespace
    from unittest.mock import Mock

    from policy_processing_fixture import runner_and_context

    from luna_mujoco.algorithms import AlgorithmError
    from luna_mujoco.robot_skills.unpack import VLAUnpack
    from luna_mujoco.vla_configuration import vla_unpack_configuration

    fixture, context, _ = runner_and_context(tmp_path / 'fixture')
    controller = PolicyController(fixture.simulation, fixture.algorithms, None, tmp_path / 'controller',
                                  planning_settings=TEST_SETTINGS)
    controller.run_id, controller.state = fixture.run_id, fixture.state
    controller.task_inventory, controller.task_progress = fixture.task_inventory, fixture.task_progress
    feedback = controller.simulation.robot_snapshot()
    for arm, degrees in vla_unpack_configuration(context['planning_settings'])['preparation_joints_deg'].items():
        feedback['arms'][arm]['joints_rad'] = [radians(value) for value in degrees]
    controller.simulation.robot_snapshot = lambda: deepcopy(feedback)
    controller.state['facts'].update(vla_pose_ready=True, table_release_verified=True, grasp_held=False)
    context.pop('processing_action')
    preparation = context.pop('vla_pose_execution')
    effect = Mock(side_effect=AlgorithmError('effect receipt unavailable', code='effect_unknown'))
    controller.unpack_effect = VLAUnpack(SimpleNamespace(unpack_at_pose=effect))
    # A rejected precondition must not be confused with an effect dispatch.
    with pytest.raises(ValueError, match='completed_vla_pose'):
        await controller._step('vla_unpack', context)
    assert not controller.state.get('partial_execution_failure')
    context['vla_pose_execution'] = preparation
    with pytest.raises(AlgorithmError, match='effect receipt unavailable'):
        await controller._step('vla_unpack', context)
    controller.reset_transfer(context)
    with pytest.raises(ValueError, match='partial_execution_requires_explicit_recovery'):
        await controller._step('vla_unpack', context)
    assert effect.call_count == 1
    assert controller.state['partial_execution_failure']['partial_execution']['effect_commands'] == ['vla_unpack']


