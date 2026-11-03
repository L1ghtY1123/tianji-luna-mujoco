"""Cross-layer regressions for deferred arm selection in processing policies."""
from copy import deepcopy

import pytest
from test_agent_model_layer import CATALOG, Client, capture, plan_data
from test_unpack_perception_integration import runner_and_context

from luna_mujoco.agent.responses import ResponsesCompiler
from luna_mujoco.demo_lifecycle_perception import lifecycle_entities, validate_lifecycle
from luna_mujoco.demo_perception import DemoPerception
from luna_mujoco.perception import Perception
from luna_mujoco.task_progress import TaskProgress, original_source_bin, source_region_confirmed
from luna_mujoco.visual_api import VisualInventoryError


def lifecycle(phase, table='A', delivery='auto'):
    return {'phase': phase, 'appearance': 'wrapped' if phase in {
        'wrapped_source', 'table_released', 'unpack_pending', 'consumed'} else 'unpacked',
        'appearance_revision': 0, 'to_table_arm': table, 'to_right_bin_arm': delivery}


def test_compiled_unpack_inventory_reaches_perception_before_any_arm_selection(monkeypatch):
    output = plan_data()
    output['task']['processing'] = ['vla_unpack']
    compiler = ResponsesCompiler(registry={**CATALOG, 'vla_unpack': CATALOG['inspect']},
                                 api_key='test', client=Client([output]))
    proposal = compiler.discover(capture(), instruction=output['task']['instruction'])
    received = []

    def observe(self, frame, instruction, checkpoint, hint, entities, reference, evidence):
        received.append(entities)
        return {'observed_ids': list(entities)}

    monkeypatch.setattr(Perception, '_observe_entities', observe)
    engine = DemoPerception(visual_inventory=object())
    engine._observe_entities(capture(), '', 'initial', {}, proposal['entities'], proposal['reference'], None)
    assert len(received) == 1
    assert set(proposal['task_spec']['target_refs']) <= set(received[0])
    assert all(row['to_table_arm'] == row['to_right_bin_arm'] == 'auto'
               for row in proposal['demo_task']['item_lifecycle'].values())


@pytest.mark.parametrize('phase,table,delivery', [
    ('wrapped_source', 'auto', 'auto'), ('wrapped_source', 'B', 'auto'),
    ('table_released', 'B', 'auto'), ('unpack_pending', 'B', 'auto'),
    ('consumed', 'B', 'auto'), ('unpacked_confirmed', 'B', 'auto'),
    ('black_held', 'B', 'A'), ('right_bin_released', 'B', 'A'), ('completed', 'B', 'A'),
])
def test_lifecycle_allows_unselected_future_leg_without_inventing_an_arm(phase, table, delivery):
    row = lifecycle(phase, table, delivery)
    original = deepcopy(row)
    assert validate_lifecycle(row) == original
    assert row == original


@pytest.mark.parametrize('phase,table,delivery', [
    ('wrapped_source', None, 'auto'), ('wrapped_source', 'auto', 'left'),
    ('table_released', 'auto', 'auto'), ('unpack_pending', 'auto', 'B'),
    ('consumed', 'auto', 'auto'), ('unpacked_confirmed', 'auto', 'B'),
    ('black_held', 'A', 'auto'), ('right_bin_released', 'A', 'auto'), ('completed', 'A', 'auto'),
])
def test_executed_leg_requires_concrete_arm_and_invalid_values_remain_rejected(phase, table, delivery):
    with pytest.raises(VisualInventoryError, match='lifecycle_arm_binding_invalid'):
        validate_lifecycle(lifecycle(phase, table, delivery))


@pytest.mark.parametrize('reachable_arms', [('A',), ('B',), ('A', 'B'), ()])
async def test_unpack_verification_checks_both_workspaces_without_selecting_future_arm(tmp_path, reachable_arms):
    runner, context, _ = runner_and_context(tmp_path)
    runner.current_lifecycle()['to_right_bin_arm'] = 'auto'
    runner.state['facts']['observation_pose_ready'] = True
    checked = []

    def workspace(measured, arm):
        assert arm in {'A', 'B'}
        checked.append(arm)
        return {'reachable': arm in reachable_arms, 'arm': arm, 'observation_id': measured['observation_id']}

    runner.algorithms.measured_object_workspace = workspace
    result = await runner._verify_unpack(context)
    assert checked == ['A', 'B']
    assert result['success'] == bool(reachable_arms)
    assert result['workspace_evidence']['arm_selection'] == 'pending_choose_arm'
    assert result['workspace_evidence']['feasible_arms'] == list(reachable_arms)
    if reachable_arms:
        product = result['output_entity_id']
        assert runner.demo_task['item_lifecycle'][product]['to_right_bin_arm'] == 'auto'
        assert runner.demo_task['item_lifecycle']['pink']['phase'] == 'consumed'
        current = lifecycle_entities(context['visual_inventory_reference']['entities'], runner.demo_task)
        assert product in current and 'pink' not in current
    else:
        assert result['reason'] == 'unpacked_object_outside_execution_workspace'
        assert not runner.demo_task['product_entities']


@pytest.mark.parametrize('appearance,field', [('wrapped', 'to_table_arm'), ('unpacked', 'to_right_bin_arm')])
async def test_choose_arm_synchronizes_binding_to_inventory_and_observation_reference(tmp_path, appearance, field):
    runner, context, _ = runner_and_context(tmp_path)
    row = runner.current_lifecycle()
    row.update(lifecycle('wrapped_source' if appearance == 'wrapped' else 'unpacked_confirmed'))
    row[field] = 'auto'
    context['visual_inventory_reference']['demo_task'] = deepcopy(runner.demo_task)
    runner.state['destination_id'] = 'destination'
    runner.algorithms.evaluate_reachability = lambda *args, **kwargs: {'A': {'reachable': True}, 'B': {'reachable': False}}
    result = await runner.execute('choose_arm', context, {'arm': 'auto'})
    assert result['arm'] == 'A'
    assert row[field] == 'A'
    assert context['visual_inventory_reference']['demo_task']['item_lifecycle']['pink'][field] == 'A'
    assert runner.task_progress.records['pink']['lifecycle'][field] == 'A'


@pytest.mark.parametrize('fraction', [.49, .7346])
def test_proposed_source_parent_must_acquire_measured_source_evidence_before_binding(fraction):
    from test_task_progress import inventory_observation
    progress = TaskProgress('move package', route='box_table_box')
    progress.initialize_scene_inventory({'observation_id': 'initial', 'objects': [], 'scene_entities': {
        'bin': {'id': 'bin', 'kind': 'bin'}, 'item': {'id': 'item', 'kind': 'object',
            'identity_source': 'openai_responses_proposal', 'initial_parent_id': 'bin',
            'visual_membership': 'required'}}})
    item = {'id': 'item', 'kind': 'object', 'color': 'pink', 'position': [.4, .2, .6],
        'measured_relationships': [{'relation': 'above_observed_bin_floor', 'object_id': 'bin',
                                    'visible_xy_fraction': fraction}]}
    assert original_source_bin(progress.records['item']) is None
    progress.observe([item], 'measured-initial')
    if fraction < .5:
        with pytest.raises(ValueError, match='task_source_region_unconfirmed'):
            progress.assess_membership({'item': 'required'})
        return
    progress.assess_membership({'item': 'required'})
    progress.freeze_inventory(inventory_observation([item]))
    assert original_source_bin(progress.records['item']) == 'bin'
    initial = deepcopy(progress.records['item']['initial_source'])
    assert initial['measurement_observation_id'] == 'measured-initial'
    progress.select(item, 'table')
    progress.observe([{**item, 'measured_relationships': []}], 'after-movement')
    assert progress.records['item']['initial_source'] == initial


@pytest.mark.parametrize('kind', ['bin', 'table'])
@pytest.mark.parametrize('fault', [None, 'proposal', 'unknown', 'wrong_target', 'stale', 'no_hash'])
def test_task_source_identity_uses_bound_decisions_without_claiming_geometric_coverage(kind, fault):
    evidence = {'source': 'openai_decisions', 'status': 'yes', 'target_id': 'item',
                'source_id': 'origin', 'source_kind': kind, 'observation_id': 'obs', 'request_hash': 'hash'}
    if fault == 'proposal':
        evidence['source'] = 'openai_responses_proposal'
    elif fault == 'unknown':
        evidence['status'] = 'unknown'
    elif fault == 'wrong_target':
        evidence['target_id'] = 'other'
    elif fault == 'stale':
        evidence['observation_id'] = 'old'
    elif fault == 'no_hash':
        evidence.pop('request_hash')
    record = {'id': 'item', 'initial_source': {'measurement_observation_id': 'obs',
        'task_source_evidence': evidence, 'measured_relationships': [
            {'relation': 'above_observed_bin_floor', 'object_id': 'origin', 'visible_xy_fraction': .4988}]}}
    assert source_region_confirmed('box_table_box', record) == (fault is None)
    assert original_source_bin(record) == ('origin' if fault is None and kind == 'bin' else None)
    assert record['initial_source']['measured_relationships'][0]['visible_xy_fraction'] == .4988
