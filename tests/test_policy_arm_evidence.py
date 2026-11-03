"""Deferred arm choice and independent evidence across policy boundaries."""
import pytest
from policy_processing_fixture import runner_and_context
from test_agent_model_layer import CATALOG, Client, capture, plan_data

from luna_mujoco.agent.responses import ResponsesCompiler
from luna_mujoco.demo_perception import DemoPerception
from luna_mujoco.perception import Perception
from luna_mujoco.task_progress import confirmed_task_source, original_source_bin


def test_compiled_unpack_inventory_reaches_perception_before_any_arm_selection(monkeypatch):
    output = plan_data()
    output['task']['processing'] = ['vla_unpack']
    compiler = ResponsesCompiler(registry={**CATALOG, 'vla_unpack': {**CATALOG['inspect'], 'processing_effect': {'fixture': True}}},
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
    assert 'demo_task' not in proposal
    assert 'item_lifecycle' not in proposal['reference']
    assert not any('arm' in key for key in proposal['task_inventory'])


@pytest.mark.parametrize('reachable_arms', [('A',), ('B',), ('A', 'B'), ()])
async def test_unpack_verification_checks_both_workspaces_without_selecting_future_arm(tmp_path, reachable_arms):
    runner, context, _ = runner_and_context(tmp_path)
    runner.state['facts']['observation_pose_ready'] = True
    checked = []

    def workspace(measured, arm):
        assert arm in {'A', 'B'}
        checked.append(arm)
        return {'reachable': arm in reachable_arms, 'arm': arm, 'observation_id': measured['observation_id']}

    runner.algorithms.measured_object_workspace = workspace
    result = await runner._verify_unpack(context)
    assert checked == ['A', 'B']
    assert result['success'] is True
    assert result['workspace_evidence']['arm_selection'] == 'pending_choose_arm'
    assert result['workspace_evidence']['feasible_arms'] == list(reachable_arms)
    product = result['output_entity_id']
    assert runner.task_progress.records[product].get('execution_arm') is None
    assert runner.task_progress.records['pink']['status'] == 'consumed'
    current = context['visual_inventory_reference']['entities']
    assert product in current and 'pink' not in current


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
    assert bool(confirmed_task_source(record)) == (fault is None)
    assert original_source_bin(record) == ('origin' if fault is None and kind == 'bin' else None)
    assert record['initial_source']['measured_relationships'][0]['visible_xy_fraction'] == .4988

