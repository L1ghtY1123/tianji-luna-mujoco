"""A partial observation confirms current identities without deleting unseen known entities."""
from copy import deepcopy

import pytest
from test_agent_model_layer import task_data
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.runtime.controller import PolicyController
from luna_mujoco.runtime.inventory import PolicyTaskProgress


def observation(identifiers, frame='initial'):
    return {'source': 'rgbd', 'observation_id': frame, 'generation': 1,
        'objects': [{'id': identifier, 'kind': 'object', 'position': [.4, index * .1, .6],
                     'points_world_m': [[.4 + offset * .001, index * .1, .6] for offset in range(16)],
                     'visual_membership': 'required'}
                    for index, identifier in enumerate(identifiers)],
        'visual_inventory': {'identity_coverage': {'source': 'openai_decisions', 'observation_id': frame,
            'request_hash': frame, 'minimum_confidence': .8, 'accepted_entity_ids': list(identifiers)},
            'evidence': {'request_hash': frame}},
        'demo_inventory_check': {'source': 'openai_decisions', 'observation_id': frame,
            'request_hash': frame, 'confidence': .5, 'status': 'unknown'}}


def controller(tmp_path):
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.task_spec = TaskSpec.model_validate(task_data() | {'target_refs': ['a', 'b']})
    runner.task_inventory = {'source_object_ids': ['a', 'b'], 'selected_ids': ['a', 'b']}
    runner.task_progress = PolicyTaskProgress('move both')
    initial = observation(['a', 'b'])
    runner.task_progress.initialize_scene_inventory(initial)
    runner.task_progress.observe(initial['objects'], initial['observation_id'])
    runner.task_progress.membership = {'a': 'required', 'b': 'required'}
    runner._check_source_view(initial)
    runner.task_progress.inventory_finalized = True
    runner.task_progress.configure_task(runner.task_spec, ['a', 'b'])
    return runner


def test_later_partial_inventory_keeps_unseen_identity_and_measurement_history(tmp_path):
    runner = controller(tmp_path)
    previous = deepcopy(runner.task_progress.records['b'])
    partial = observation(['a'], 'partial')
    runner.task_progress.observe(partial['objects'], partial['observation_id'])
    runner._check_source_view(partial)
    assert runner.task_progress.records['b']['visibility'] == 'unseen'
    assert runner.task_progress.records['b']['last_seen_observation_id'] == previous['last_seen_observation_id']
    assert runner.task_progress.records['b']['last_observed_position'] == previous['last_observed_position']
    assert runner.task_progress.scene_entities['b']['geometry_status'] == 'missing'
    assert runner.task_progress.inventory_state()['retired_entity_ids'] == []
    assert runner.state['visible_inventory_resolution']['status'] == 'unknown'


def test_current_visible_entity_still_requires_fresh_identity_proof(tmp_path):
    runner = controller(tmp_path)
    partial = observation(['a'], 'partial')
    partial['visual_inventory']['identity_coverage']['accepted_entity_ids'] = []
    with pytest.raises(ValueError, match='source_inventory_unknown'):
        runner._check_source_view(partial)


def test_initial_missing_target_cannot_be_promoted_to_confirmed_inventory(tmp_path):
    runner = controller(tmp_path)
    runner.task_progress.inventory_finalized = False
    with pytest.raises(ValueError, match='task_target_geometry_unknown:b'):
        runner._check_source_view(observation(['a']))


async def test_local_mapping_does_not_require_other_known_pending_targets_to_reappear(tmp_path, monkeypatch):
    from test_inventory_map_lifecycle import scene

    from luna_mujoco.inventory_map import inventory_admission, reconcile_inventory

    runner = controller(tmp_path)
    runner.state.update(target_id='a', arm=None, destination_id=None)
    runner.state['facts']['target_selected'] = True
    partial = observation(['a'], 'partial')
    runner._check_source_view(partial)
    context = {'observation': partial, 'mapping_lifecycle': {'invalidated_entity_ids': ['b']}}
    seen = []
    previous_map = reconcile_inventory(None, *scene('initial', ['a', 'b']), required_entity_ids=['a', 'b'])

    async def build(active_context, exclusion_state):
        seen.append(deepcopy(active_context['mapping_lifecycle']))
        return reconcile_inventory(previous_map, *scene('partial', ['a']), required_entity_ids=['a', 'b'],
                                   **active_context['mapping_lifecycle'])[0]

    monkeypatch.setattr(runner, 'build_scene', build)
    partial['objects'][0]['coverage'] = {'camera_ids': ['1', '2', '3']}
    partial['objects'][0]['measurement_sources'] = {'camera_ids': ['1', '2', '3']}
    result = await runner.execute('build_scene', context, {
        'scope': 'local', 'reason': 'refresh current target measurement', 'entity_ids': ['a']})
    assert result['coverage']['stale_entity_ids'] == ['b']
    assert seen[0]['invalidated_entity_ids'] == ['b']
    assert seen[0]['mapping_entity_ids'] == ['a']
    assert inventory_admission(result)['accepted'] is False
    assert inventory_admission(result)['reason'] == 'inventory_geometry_stale'


async def test_missing_current_target_still_cannot_build_selected_geometry(tmp_path):
    runner = controller(tmp_path)
    runner.state.update(target_id='b', arm=None, destination_id=None)
    runner.state['facts']['target_selected'] = True
    partial = observation(['a'], 'partial')
    runner._check_source_view(partial)
    with pytest.raises(ValueError, match='observed_instance_missing_or_ambiguous:b'):
        await runner.execute('build_scene', {'observation': partial}, {'scope': 'local', 'entity_ids': ['a']})
