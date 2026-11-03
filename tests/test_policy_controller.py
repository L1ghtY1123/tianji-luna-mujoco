from types import SimpleNamespace

import pytest
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision


def test_policy_catalog_exposes_generic_arguments_and_no_legacy_mock(tmp_path):
    from luna_mujoco.runtime.controller import PolicyController
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    catalog = runner.registry.describe()
    assert 'vla_unpack' in catalog
    assert 'mock_unpack' not in catalog
    assert 'target_id' in catalog['identify_target']['arguments']['properties']
    assert 'destination_id' in catalog['identify_destination']['arguments']['properties']


@pytest.mark.parametrize('name', ['recover_task', 'assess_task', 'inspect_scene', 'mock_unpack'])
async def test_current_controller_rejects_retired_names_even_with_injected_contract(tmp_path, name):
    from luna_mujoco.runtime.controller import PolicyController
    from luna_mujoco.skills import Contract
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.registry.contracts[name] = Contract(id=name, description='Obsolete contract', requires={}, provides=[])
    with pytest.raises(ValueError, match='unregistered_skill:' + name):
        await runner.execute(name, {})


def test_generic_visible_inventory_does_not_require_left_camera_or_all_scene_objects(tmp_path):
    from luna_mujoco.runtime.controller import PolicyController
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.task_inventory = {'source_object_ids': ['red-1'], 'transport_ids': ['red-1']}
    runner.task_progress = SimpleNamespace(inventory_finalized=False)
    observation = {'observation_id': 'obs', 'objects': [
        {'id': 'red-1', 'coverage': {'camera_ids': ['1']}}],
        'visual_inventory': {'visibility': {'red-1': 'visible', 'background-box': 'unseen'},
            'identity_coverage': {'source': 'openai_decisions', 'observation_id': 'obs',
                'request_hash': 'current', 'minimum_confidence': .8, 'accepted_entity_ids': ['red-1']},
            'evidence': {'request_hash': 'current'}},
        'demo_inventory_check': {'source': 'openai_decisions', 'observation_id': 'obs',
            'request_hash': 'current', 'confidence': .99, 'status': 'none'}}
    runner._check_source_view(observation)
    runner._require_visible_inventory(observation)

    # Enumeration may remain unknown, but its evidence and all target identities
    # must still be current and independently accepted.
    observation['demo_inventory_check'].update(status='unknown', confidence=.69)
    runner._check_source_view(observation)
    assert runner.state['visible_inventory_resolution']['status'] == 'unknown'
    observation['visual_inventory']['identity_coverage']['accepted_entity_ids'] = []
    with pytest.raises(ValueError, match='source_inventory_unknown'):
        runner._check_source_view(observation)
    observation['visual_inventory']['identity_coverage']['accepted_entity_ids'] = ['red-1']
    observation['demo_inventory_check']['request_hash'] = 'stale'
    with pytest.raises(ValueError, match='source_inventory_unknown'):
        runner._check_source_view(observation)
    observation['demo_inventory_check'].update(request_hash='current', status='unlisted', confidence=.99)
    with pytest.raises(ValueError, match='unlisted_source_object'):
        runner._check_source_view(observation)


@pytest.mark.asyncio
async def test_registry_dispatch_can_add_skill_without_interpreter_change(tmp_path):
    from luna_mujoco.runtime.controller import PolicyController
    from luna_mujoco.skills import Contract, NoArguments
    runner = PolicyController(DemoRobot(), DemoVision(), None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.registry.contracts['weigh'] = Contract(id='weigh',description='Measure weight',requires={},provides=[])

    async def weigh(context, arguments):
        return {'mass_g': 42, 'success': True}

    runner.registry.register('weigh', NoArguments, weigh)
    assert await runner.registry.execute('weigh', {}, {}, {}) == {'mass_g': 42, 'success': True}
