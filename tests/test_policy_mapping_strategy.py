"""Model-selected mapping scope is separate from numerical coverage authorization."""
import json
from copy import deepcopy

import httpx
import pytest
from mapping_fixtures import setup

from luna_mujoco.agent.decisions import DecisionsJudgments
from luna_mujoco.robot_skills.operations import RobotOperations


class MappingHTTP:
    def __init__(self, scope='local', confidence=.99, updates=None):
        self.scope, self.confidence = scope, confidence
        self.updates = updates or {'update_entity_0': .99, 'update_entity_1': .01}
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append(kwargs['json'])
        answers = []
        for question in kwargs['json']['questions']:
            if question['type'] == 'choice':
                answers.append({'type': 'choice', 'name': question['name'],
                                'choice': self.scope, 'confidence': self.confidence})
            else:
                answers.append({'type': 'predicate', 'name': question['name'],
                                'probability': self.updates[question['name']]})
        return httpx.Response(200, json={'answers': answers})


def mapping_runner(tmp_path, monkeypatch, http=None):
    runner, context, _, _ = setup(tmp_path, [])
    runner.state['facts']['target_selected'] = True
    runner.task_progress.scene_entities = {
        'pending': {'id': 'pending', 'kind': 'object', 'geometry_status': 'measured'},
        'unseen': {'id': 'unseen', 'kind': 'object', 'geometry_status': 'missing'}}
    context['observation']['objects'] = [{'id': 'pending', 'kind': 'object'}]
    context['scene_observation'] = {'source': 'rgbd', 'observation_id': 'previous-frame',
        'objects': [{'id': 'pending', 'kind': 'object', 'position': [.3, .1, .6]}]}
    context['scene'] = {'scene_token': 'old-scene', 'coverage': {'complete': False, 'unknown_cells': 40}}
    context['mapping_lifecycle'] = {'invalidated_entity_ids': ['pending']}
    http = http or MappingHTTP()
    runner.judgments = DecisionsJudgments(api_key='test', client=http)
    builds = []

    async def build(self, skill, active_context, arguments):
        assert skill == 'build_scene'
        builds.append(deepcopy(active_context['mapping_lifecycle']))
        return {'scene_token': 'new-scene', 'coverage': {'complete': False}}

    monkeypatch.setattr(RobotOperations, 'execute', build)
    return runner, context, http, builds


@pytest.mark.parametrize('scope,entities', [('global', []), ('local', ['pending'])])
async def test_explicit_responses_mapping_scope_passes_exact_selection_without_new_model_call(
        tmp_path, monkeypatch, scope, entities):
    runner, context, http, builds = mapping_runner(tmp_path, monkeypatch)
    result = await runner.execute('build_scene', context, {
        'scope': scope, 'reason': 'task and changed entity evidence', 'entity_ids': entities})
    assert result['scene_token'] == 'new-scene' and result['coverage']['complete'] is False
    assert not http.requests
    assert builds == [{'invalidated_entity_ids': ['pending'], 'requested_scope': scope,
                       'scope_reason': 'task and changed entity evidence', 'mapping_entity_ids': entities,
                       'strategy_source': 'responses'}]


@pytest.mark.parametrize('scope', ['global', 'local'])
async def test_decisions_selects_scope_and_local_entities_from_current_evidence(tmp_path, monkeypatch, scope):
    runner, context, http, builds = mapping_runner(tmp_path, monkeypatch, MappingHTTP(scope))
    await runner.execute('build_scene', context, {})
    assert builds[0]['requested_scope'] == scope
    assert builds[0]['mapping_entity_ids'] == (['pending'] if scope == 'local' else [])
    assert builds[0]['strategy_source'] == 'decisions'
    payload = json.loads(http.requests[0]['input'][0]['content'][0]['text'])
    assert payload['context']['mapping_evidence']['previous_scene']['coverage']['complete'] is False
    assert payload['context']['mapping_evidence']['lifecycle']['invalidated_entity_ids'] == ['pending']
    assert payload['context']['mapping_evidence']['local_update_candidates'] == ['pending']
    assert 'unseen' in payload['context']['mapping_evidence']['inventory_entities']
    assert payload['context']['mapping_evidence']['previous_observation']['objects'][0]['position'] == [.3, .1, .6]


async def test_decisions_authored_explicit_mapping_does_not_require_a_second_decision(tmp_path, monkeypatch):
    runner, context, http, _builds = mapping_runner(tmp_path, monkeypatch)
    runner.state['active_policy_node'] = {'author': 'decisions', 'node_id': 'recovery_map'}
    result = await runner._mapping_strategy(context, {'scope': 'global', 'reason': '', 'entity_ids': []})
    assert result['strategy_source'] == 'decisions'
    assert result['requested_scope'] == 'global'
    assert not http.requests


@pytest.mark.parametrize('scope,confidence', [('unknown', .99), ('unknown', .5)])
async def test_uncertain_mapping_strategy_does_not_build_or_choose_a_fallback(tmp_path, monkeypatch, scope, confidence):
    runner, context, _, builds = mapping_runner(tmp_path, monkeypatch, MappingHTTP(scope, confidence))
    result = await runner.execute('build_scene', context, {})
    assert result['status'] == 'unknown' and result['reason'] == 'mapping_strategy_unknown'
    assert not builds
    assert context['scene']['scene_token'] == 'old-scene'


@pytest.mark.parametrize('scope', ['global', 'local'])
async def test_mapping_proposal_preserves_model_choice_without_claiming_certainty(tmp_path, monkeypatch, scope):
    runner, context, _, builds = mapping_runner(tmp_path, monkeypatch, MappingHTTP(scope, .72))
    result = await runner.execute('build_scene', context, {})
    assert builds[0]['requested_scope'] == scope
    assert result['coverage']['complete'] is False
    choice = context['mapping_strategy_evidence']['judgments']['mapping_scope']
    assert choice == {'status': 'proposed', 'value': scope, 'confidence': .72, 'selection_mode': 'proposal'}


async def test_local_mapping_unknown_entity_is_rejected_without_default_global(tmp_path, monkeypatch):
    runner, context, http, builds = mapping_runner(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='mapping_entity_not_in_inventory'):
        await runner.execute('build_scene', context, {'scope': 'local', 'entity_ids': ['invented']})
    assert not http.requests and not builds


async def test_missing_decisions_result_does_not_build(tmp_path, monkeypatch):
    runner, context, _, builds = mapping_runner(tmp_path, monkeypatch)

    def unavailable(*args, **kwargs):
        raise ValueError('decisions_unavailable')

    monkeypatch.setattr(runner.judgments, 'judge', unavailable)
    with pytest.raises(ValueError, match='decisions_unavailable'):
        await runner.execute('build_scene', context, {})
    assert not builds


@pytest.mark.parametrize('scope', ['global', 'local'])
async def test_first_map_explains_missing_layers_without_forcing_global(tmp_path, monkeypatch, scope):
    runner, context, http, builds = mapping_runner(tmp_path, monkeypatch, MappingHTTP(scope))
    context.pop('scene')
    context.pop('scene_observation')
    context['mapping_lifecycle'] = {}
    await runner.execute('build_scene', context, {})
    evidence = json.loads(http.requests[0]['input'][0]['content'][0]['text'])['context']['mapping_evidence']
    assert evidence['retained_map_available'] is False
    assert evidence['required_collision_entity_ids'] == ['pending', 'unseen']
    assert evidence['entity_layer_state']['pending'] == {
        'currently_observed': True, 'has_retained_layer': False, 'layer_status': 'missing',
        'layer_names': [], 'last_observation_id': None, 'invalidated': False}
    assert evidence['entity_layer_state']['unseen']['currently_observed'] is False
    assert evidence['scope_effects']['global']['update_candidate_ids'] == ['pending']
    assert evidence['scope_effects']['local']['update_candidate_ids'] == ['pending']
    assert evidence['scope_effects']['global']['proves_workspace_coverage'] is False
    assert builds[0]['requested_scope'] == scope
    questions = {question['name']: question for question in http.requests[0]['questions']}
    assert 'create a missing collision layer' in questions['update_entity_0']['instructions']
    assert 'No movement does not imply no mapping work' in questions['update_entity_0']['instructions']


async def test_mapping_layer_context_comes_from_actual_scene_not_observation_labels(tmp_path, monkeypatch):
    runner, context, http, _ = mapping_runner(tmp_path, monkeypatch)
    context['scene']['inventory_map'] = {'required_entity_ids': ['pending', 'unseen', 'not-in-current-view'],
        'entities': {'pending': {'status': 'stale', 'layer_names': ['semantic_instance::pending'],
                                 'last_observation_id': 'older-measured-frame'},
                     'unseen': {'status': 'retained', 'layer_names': ['semantic_instance::unseen'],
                                'last_observation_id': 'initial-frame'}}}
    await runner.execute('build_scene', context, {})
    evidence = json.loads(http.requests[0]['input'][0]['content'][0]['text'])['context']['mapping_evidence']
    assert evidence['retained_map_available'] is True
    assert evidence['required_collision_entity_ids'] == ['not-in-current-view', 'pending', 'unseen']
    assert evidence['entity_layer_state']['pending'] == {
        'currently_observed': True, 'has_retained_layer': True, 'layer_status': 'stale',
        'layer_names': ['semantic_instance::pending'], 'last_observation_id': 'older-measured-frame',
        'invalidated': True}
    assert evidence['entity_layer_state']['unseen']['has_retained_layer'] is True
    assert evidence['entity_layer_state']['unseen']['layer_status'] == 'retained'
    assert evidence['entity_layer_state']['unseen']['currently_observed'] is False
    assert evidence['entity_layer_state']['not-in-current-view']['has_retained_layer'] is False
    assert evidence['local_update_candidates'] == ['pending']


async def test_scene_without_layer_metadata_is_unknown_not_invented_empty_map(tmp_path, monkeypatch):
    runner, context, http, _ = mapping_runner(tmp_path, monkeypatch)
    await runner.execute('build_scene', context, {})
    evidence = json.loads(http.requests[0]['input'][0]['content'][0]['text'])['context']['mapping_evidence']
    assert evidence['retained_map_available'] is True
    assert evidence['entity_layer_state']['pending']['has_retained_layer'] is None
    assert evidence['entity_layer_state']['pending']['layer_status'] == 'unknown'


async def test_recorded_low_confidence_scope_still_stops_without_map_or_fallback(tmp_path, monkeypatch):
    runner, context, _, builds = mapping_runner(tmp_path, monkeypatch,
        MappingHTTP('local', .67, updates={'update_entity_0': .79}))
    context.pop('scene')
    result = await runner.execute('build_scene', context, {})
    assert result['status'] == 'unknown'
    assert result['reason'] == 'mapping_strategy_unknown'
    assert not builds
    assert 'scene' not in context


async def test_observation_mode_success_does_not_publish_target_bound_facts(tmp_path, monkeypatch):
    runner, context, _, _ = mapping_runner(tmp_path, monkeypatch)
    runner.state.update(target_id=None, arm=None, destination_id=None)
    runner.state['facts'].update(observed=True, inventory_finalized=True, grasp_held=False,
                                recovery_needed=False, target_selected=False, scene_ready=True, views_ready=True)

    async def observation_map(active_context, mode):
        assert mode == 'observation'
        return {'scene_token': 'observation-map', 'exclusion_state': mode}

    monkeypatch.setattr(runner, 'build_scene', observation_map)
    await runner._step('build_scene', context, {'scope': 'global', 'exclusion_state': 'observation'})
    assert runner.state['facts']['scene_available'] is True
    assert runner.state['facts']['scene_ready'] is False
    assert runner.state['facts']['views_ready'] is False


async def test_tables_cannot_be_selected_as_local_object_layers(tmp_path, monkeypatch):
    runner, context, http, builds = mapping_runner(tmp_path, monkeypatch)
    runner.task_progress.scene_entities['table'] = {'id': 'table', 'kind': 'table'}
    context['observation']['objects'].append({'id': 'table', 'kind': 'table'})
    with pytest.raises(ValueError, match='local_mapping_requires_object_entities'):
        await runner.execute('build_scene', context, {'scope': 'local', 'entity_ids': ['table']})
    assert not http.requests and not builds


async def test_declared_table_alongside_local_objects_uses_separate_cuboid_boundary(tmp_path, monkeypatch):
    runner, context, http, builds = mapping_runner(tmp_path, monkeypatch)
    runner.task_progress.scene_entities['table'] = {'id': 'table', 'kind': 'table'}
    await runner.execute('build_scene', context, {
        'scope': 'local', 'entity_ids': ['pending', 'table'], 'reason': 'Refresh object and support'})
    assert builds[0]['mapping_entity_ids'] == ['pending']
    assert builds[0]['requested_scope'] == 'local'
    assert 'table' in builds[0]['scope_reason']
    assert not http.requests


@pytest.mark.parametrize('scope', ['decide', 'global', 'local'])
async def test_default_reason_uses_selection_provenance_and_passes_actual_mapping_boundaries(
        tmp_path, monkeypatch, scope):
    from test_inventory_map_lifecycle import scene
    from test_perception_mapping import wire

    from luna_mujoco.inventory_map import reconcile_inventory
    from luna_mujoco.perception_mapping import decode_mapping_payload

    runner, context, _, _ = mapping_runner(tmp_path, monkeypatch)
    runner.task_progress.scene_entities['source'] = runner.task_progress.scene_entities.pop('pending') | {'id': 'source'}
    context['observation']['objects'][0]['id'] = 'source'
    arguments = {'scope': scope, 'entity_ids': ['source'] if scope == 'local' else []}
    strategy = await runner._mapping_strategy(context, arguments)
    if scope == 'decide':
        assert strategy['scope_reason'] == ('Decisions mapping choice; request_hash=' +
            context['mapping_strategy_evidence']['evidence']['request_hash'])
    else:
        assert strategy['scope_reason'] == 'Explicit scope in responses-authored graph node'
    decoded, _ = decode_mapping_payload(wire() | {'mapping_lifecycle': strategy})
    previous = reconcile_inventory(None, *scene('previous', ['source', 'unseen']),
                                   required_entity_ids=['source', 'unseen'])
    current = reconcile_inventory(previous, *scene('current', ['source']),
        required_entity_ids=['source', 'unseen'], **decoded['mapping_lifecycle'])
    assert current[0]['inventory_map']['entities']['unseen']['status'] == 'retained'
