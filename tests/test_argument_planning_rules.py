"""Argument-dependent contract facts are generic and never rewrite a graph."""
import json
from copy import deepcopy

import pytest
from graph_contract_fixtures import call, check, robot_catalog

from luna_mujoco.agent.responses import BOOTSTRAP_FACTS as RAW_ENTRY_FACTS
from luna_mujoco.runtime.controller import BuildSceneArguments
from luna_mujoco.skills import Contract

BOOTSTRAP_FACTS = {**RAW_ENTRY_FACTS, 'observed': True, 'inventory_ready': True,
    'inventory_finalized': True, 'task_assessed': True, 'grasp_view_ready': True}


def catalog():
    value = robot_catalog()
    value['build_scene']['arguments'] = BuildSceneArguments.model_json_schema()
    return value




def test_all_motion_contracts_expose_inventory_invalidation_to_responses():
    for name, contract in catalog().items():
        if contract.get('invalidates_observation'):
            assert contract['planning_effects']['observed'] is False, name
    assert 'observed' not in catalog()['verify_place']['provides']
    assert 'observed' not in catalog()['verify_place']['planning_effects']


@pytest.mark.parametrize('observation_first', [False, True])
def test_model_can_choose_selected_map_or_observation_map_before_binding(observation_first):
    steps = [call('measure', 'observe_scene')]
    if observation_first:
        steps.append(call('all_objects', 'build_scene', exclusion_state='observation'))
    steps += [call('select', 'identify_target', target_id='observed-product'),
              call('destination', 'identify_destination', destination_id='observed-bin', destination_kind='container'),
              call('arm', 'choose_arm', arm='auto'),
              call('target_map', 'build_scene', exclusion_state='selected'),
              call('grasps', 'generate_grasps')]
    graph = check(steps, catalog(), facts=BOOTSTRAP_FACTS)
    assert graph.model_dump()['nodes'][:-2] == steps


@pytest.mark.parametrize('mode', ['observation', None])
def test_map_mode_and_registered_default_cannot_bypass_target_requirements(mode):
    arguments = {} if mode is None else {'exclusion_state': mode}
    steps = [call('map', 'build_scene', **arguments), call('grasps', 'generate_grasps')]
    message = 'map:target_selected' if mode is None else 'grasps:.*scene_ready'
    with pytest.raises(ValueError, match='planning_precondition_not_satisfied:' + message):
        check(steps, catalog(), facts={**BOOTSTRAP_FACTS, 'arm_selected': True,
            'destination_ready': True, 'target_selected': mode is not None})


def test_target_rebinding_revokes_previous_target_map_before_new_grasps():
    steps = [call('first', 'identify_target', target_id='first'), call('map', 'build_scene'),
             call('second', 'identify_target', target_id='second'),
             call('destination', 'identify_destination', destination_id='region', destination_kind='support_region'),
             call('arm', 'choose_arm', arm='auto'), call('stale_grasps', 'generate_grasps')]
    with pytest.raises(ValueError, match='planning_precondition_not_satisfied:stale_grasps:.*scene_ready'):
        check(steps, catalog(), facts=BOOTSTRAP_FACTS)


def dummy_catalog():
    return {'custom_capture': {'id': 'custom_capture', 'description': 'generic adapter',
        'requires': {}, 'provides': [],
        'arguments': {'type': 'object', 'additionalProperties': False, 'properties': {
            'mode': {'type': 'string', 'enum': ['inspect', 'operate'], 'default': 'operate'},
            'fast': {'type': 'boolean', 'default': False}}},
        'argument_rules': [
            {'when_arguments': {'mode': 'operate'}, 'requires': {'bound': True}, 'effects': {'ready': True}},
            {'when_arguments': {'mode': 'inspect'}, 'requires': {}, 'effects': {'ready': False}}]},
        'custom_consume': {'requires': {'ready': True}, 'provides': []}}


def test_extension_rules_apply_defaults_and_success_effects_without_skill_name_branches():
    values = dummy_catalog()
    steps = [call('capture', 'custom_capture'), call('use', 'custom_consume')]
    assert check(steps, values, facts={'bound': True}).model_dump()['nodes'][:-2] == steps
    with pytest.raises(ValueError, match='capture:bound'):
        check(steps, values, facts={'bound': False})
    with pytest.raises(ValueError, match='use:ready'):
        check([call('capture', 'custom_capture', mode='inspect'), call('use', 'custom_consume')], values)


@pytest.mark.parametrize('rule', [
    {'when_arguments': {}, 'requires': {}, 'effects': {'ready': True}},
    {'when_arguments': {'mode': 'operate'}, 'requires': {'bound': 'true'}, 'effects': {}},
    {'when_arguments': {'mode': ['operate']}, 'requires': {}, 'effects': {'ready': True}},
    {'when_arguments': {'mode': 'operate'}, 'requires': {}, 'effects': {'ready': 'true'}},
])
def test_argument_rule_structure_is_strict(rule):
    with pytest.raises(ValueError):
        Contract(id='custom', description='custom', requires={}, provides=[], argument_rules=[rule])


@pytest.mark.parametrize('condition', [{'missing': 'operate'}, {'mode': 'invalid'}, {'fast': 1}])
def test_rule_conditions_must_match_registered_argument_schema(condition):
    values = dummy_catalog()
    values['custom_capture']['argument_rules'][0]['when_arguments'] = condition
    with pytest.raises(ValueError, match='invalid_argument_rule'):
        check([call('capture', 'custom_capture')], values, facts={'bound': True})


def test_overlapping_conflicting_rules_are_rejected_independently_of_list_order():
    values = dummy_catalog()
    rules = values['custom_capture']['argument_rules']
    rules.append({'when_arguments': {'fast': False}, 'requires': {}, 'effects': {'ready': False}})
    for order in (rules, list(reversed(deepcopy(rules)))):
        values['custom_capture']['argument_rules'] = order
        with pytest.raises(ValueError, match='conflicting_argument_rules'):
            check([call('capture', 'custom_capture')], values, facts={'bound': True})


async def test_registered_extension_enforces_argument_requirements_before_dispatch(tmp_path):
    from typing import Literal

    from pydantic import BaseModel, ConfigDict

    from luna_mujoco.skills import Registry, SkillSelectionError

    class Arguments(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        mode: Literal['inspect', 'operate'] = 'operate'
        fast: bool = False

    contract = dummy_catalog()['custom_capture']
    contract.pop('arguments')
    path = tmp_path / 'custom_capture'
    path.mkdir()
    (path / 'contract.json').write_text(json.dumps(contract), encoding='utf-8')
    (path / 'SKILL.md').write_text('Use an explicitly selected mode.', encoding='utf-8')
    registry = Registry(tmp_path)
    calls = []

    async def handler(context, arguments):
        calls.append(arguments.mode)
        return {'status': 'success'}

    registry.register('custom_capture', Arguments, handler)
    with pytest.raises(SkillSelectionError, match='skill_contract_not_satisfied'):
        await registry.execute('custom_capture', {'bound': False}, {}, {})
    assert not calls
    await registry.execute('custom_capture', {'bound': False}, {}, {'mode': 'inspect'})
    assert calls == ['inspect']
