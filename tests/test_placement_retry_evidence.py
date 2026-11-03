import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_unified_placement_controller import controller

from luna_mujoco.algorithms import AlgorithmError


def recorded_inputs(index=0):
    data = json.loads((Path(__file__).parent / 'fixtures/black_b3_placement_retry.json').read_text())
    attempt = data['attempts'][index]
    context = {key: data[key] for key in ('scene', 'holding', 'planning_settings')}
    context['observation'] = {'arms': attempt.pop('arms')}
    context['placement_resolution'] = attempt
    for candidate in attempt['candidates']:
        candidate['binding'] = deepcopy(data['holding'])
        candidate['payload']['carrying_binding'] = deepcopy(data['holding'])
    return context, data['state']


def setup_runner(tmp_path):
    runner = controller(tmp_path)
    context, state = recorded_inputs()
    runner.state.update(state)
    runner.state['generation'] = runner.simulation.robot_snapshot()['generation']
    runner.state['facts'].update(runner.registry.contracts['plan_place'].requires)
    runner.begin_step('plan_place')
    runner.algorithms = Mock()
    runner.emit = Mock()
    runner.verify_scene = AsyncMock()
    runner.measured_state = lambda ctx: ctx['observation']
    runner.prepare_attachment = AsyncMock()
    runner._bind_region = Mock(return_value={})
    runner.call_algorithm = AsyncMock(side_effect=AlgorithmError('No checked path',
        code='place_no_valid_candidate', details={'candidate_failures': []}))
    return runner, context


async def test_recorded_b3_initial_failure_blocks_same_goals_after_resolution_rebuild(tmp_path):
    runner, context = setup_runner(tmp_path)
    with pytest.raises(AlgorithmError, match='No checked path'):
        await runner.execute('plan_place', context)
    later, _ = recorded_inputs(1)
    assert context['placement_resolution']['resolution_id'] != later['placement_resolution']['resolution_id']
    context.update(later)
    with pytest.raises(AlgorithmError) as failure:
        await runner.execute('plan_place', context)
    assert runner.call_algorithm.await_count == 1
    assert failure.value.details['reason'] == 'identical_placement_inputs_no_progress'


@pytest.mark.parametrize('change', ['goal', 'scene', 'holding', 'joint', 'peer_joint', 'gripper', 'settings'])
async def test_new_numerical_inputs_allow_another_plan(tmp_path, change):
    runner, context = setup_runner(tmp_path)
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    if change == 'goal':
        context['placement_resolution']['candidates'][0]['goal']['T_world_tcp'][0][3] += .01
    elif change == 'scene':
        context['scene']['scene_token'] = 'new-published-scene'
    elif change == 'holding':
        context['holding']['cad_tool']['T_tool_object'][0][3] += .001
    elif change in ('joint', 'peer_joint'):
        arm = 'B' if change == 'joint' else 'A'
        context['observation']['arms'][arm]['joints_rad'][0] += .001
    elif change == 'gripper':
        context['observation']['arms']['B']['gripper_position_rad'] += .001
    else:
        context['planning_settings']['ik_num_seeds'] += 1
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    assert runner.call_algorithm.await_count == 2


async def test_ids_timing_pixels_and_candidate_order_do_not_create_progress(tmp_path):
    runner, context = setup_runner(tmp_path)
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    resolution = context['placement_resolution']
    resolution.update(resolution_id='new-result', request_timings={'wall_s': 123})
    resolution['candidates'].reverse()
    for i, candidate in enumerate(resolution['candidates']):
        candidate['candidate_id'] = f'renamed-{i}'
    resolution['candidates'].append(deepcopy(resolution['candidates'][0]))
    context['latest_capture'] = {'observation_id': 'new-raw-pixels', 'views': {}}
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    assert runner.call_algorithm.await_count == 1


@pytest.mark.parametrize('delta, calls', [(5e-7, 1), (2e-6, 2)])
async def test_feedback_consistency_tolerance_only_controls_retry_eligibility(tmp_path, delta, calls):
    runner, context = setup_runner(tmp_path)
    settings = deepcopy(context['planning_settings'])
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    context['observation']['arms']['B']['joints_rad'][0] += delta
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    assert runner.call_algorithm.await_count == calls
    assert context['planning_settings'] == settings


async def test_service_error_without_exhausted_candidates_does_not_block_retry(tmp_path):
    runner, context = setup_runner(tmp_path)
    runner.call_algorithm.side_effect = AlgorithmError('unavailable', code='service_unavailable')
    for _ in range(2):
        with pytest.raises(AlgorithmError):
            await runner.execute('plan_place', context)
    assert runner.call_algorithm.await_count == 2
    assert not context.get('failed_placement_attempts')


async def test_removing_previously_failed_candidates_does_not_add_new_input(tmp_path):
    runner, context = setup_runner(tmp_path)
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    context['placement_resolution']['candidates'] = context['placement_resolution']['candidates'][:1]
    with pytest.raises(AlgorithmError):
        await runner.execute('plan_place', context)
    assert runner.call_algorithm.await_count == 1


@pytest.mark.parametrize('mode', ['unchanged', 'new_goal', 'changed_back_during_selection'])
async def test_native_recovery_checks_inputs_before_choice_and_after_selection(tmp_path, mode):
    from luna_mujoco.agent.schema import Graph
    from luna_mujoco.runtime.execution_state import PolicyStopped

    runner, context = setup_runner(tmp_path)
    runner.task_progress = SimpleNamespace(completed_input_ids=set, snapshot=dict,
        inventory_finalized=True)
    runner.sync_inventory = Mock()
    runner.task_inventory = {}
    runner.state['facts'].update(runner.registry.contracts['resolve_placement'].requires)
    context['latest_capture'] = {'observation_id': 'current', 'views': {}}
    context['observation']['objects'] = []
    context['capture_epochs'] = {'current': runner.observation_epoch}
    runner.policy_graph = Graph.model_validate({'version': 2, 'id': 'retry-test', 'entry': 'plan',
        'subgraphs': [], 'nodes': [
            {'id': 'plan', 'type': 'skill', 'skill': 'plan_place', 'arguments': {}},
            {'id': 'done', 'type': 'end', 'status': 'success', 'reason': 'planned'}],
        'edges': [{'id': 'planned', 'source': 'plan', 'target': 'done', 'on': 'success'}]})
    original_pose = deepcopy(context['placement_resolution']['candidates'][0]['goal']['T_world_tcp'])

    async def resolve(ctx, arguments):
        later, _ = recorded_inputs(1)
        ctx.update(later)
        ctx['observation']['objects'] = []
        if mode != 'unchanged':
            ctx['placement_resolution']['candidates'][0]['goal']['T_world_tcp'][0][3] += .01
        return {'status': 'success', **ctx['placement_resolution']}

    runner.registry.handlers['resolve_placement'] = resolve
    calls = []

    async def judge(ctx, requests, *, extra=None, **kwargs):
        request = requests[0]
        calls.append(request.name)
        if len(calls) == 1:
            assert 'retry_failed_node' not in request.choices
            assert 'resolve_placement' in request.choices
            assert extra['rejected_proposals'][0]['placement_evidence']['reason'] == (
                'identical_placement_inputs_no_progress')
            choice = 'resolve_placement'
        elif request.name == 'recovery_arguments':
            choice = next(key for key, values in extra['argument_options'].items()
                          if values == {'preference': 'decide', 'capture_mode': 'reuse_current'})
        elif len(calls) == 3 and mode != 'unchanged':
            assert request.name == 'recovery_continuation'
            assert 'retry_failed_node' in request.choices
            if mode == 'changed_back_during_selection':
                ctx['placement_resolution']['candidates'][0]['goal']['T_world_tcp'] = original_pose
            choice = 'retry_failed_node'
        else:
            assert request.name == 'recovery_capability'
            assert 'retry_failed_node' not in request.choices
            choice = 'stop'
        return {'judgments': {request.name: {'status': 'proposed', 'value': choice}}}

    runner._judge = judge
    with pytest.raises(PolicyStopped):
        await runner._execute_graph(context)
    assert runner.call_algorithm.await_count == (2 if mode == 'new_goal' else 1), (
        calls, runner.graph_executor.last_failure)
