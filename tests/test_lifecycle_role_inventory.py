"""Current role checks bind IDs only with unique measured and recorded lifecycle evidence."""
import importlib
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from decision_fixtures import decision_answer

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError


def inputs():
    rgb = np.full((80, 100, 3), 150, np.uint8)
    capture = {'observation_id': 'current', 'generation': 2, 'views': {
        camera: {'rgb': rgb.copy()} for camera in ('1', '2', '3')}}
    entities = {key: {'kind': kind, 'description': key, 'roles': ['collision'],
                      'visual_membership': 'required'} for key, kind in
                [('left', 'bin'), ('right', 'bin'), ('pink', 'object'), ('black', 'object')]}
    detections = []
    for index, (camera, kind) in enumerate([('2', 'bin'), ('1', 'bin'), ('2', 'object'), ('3', 'object')]):
        mask = np.zeros(rgb.shape[:2], bool)
        mask[10 + 15*index:20 + 15*index, 20:30] = True
        pixels = np.column_stack(np.nonzero(mask))
        detections.append({'camera_id': camera, 'kind': kind, 'prompt': kind, 'score': .9,
            'mask': mask.copy(), 'silhouette_mask': mask.copy(), 'pixels_rc': pixels,
            'pixels': rgb[mask], 'points': np.column_stack((pixels / 100, np.ones(len(pixels))))})
    tracking = {'source': 'unique_current_lifecycle_role', 'observation_id': 'current', 'generation': 2,
        'bindings': {'pink': {'role': 'wrapped_source', 'current_description': 'a separate pink object inside the left blue box'},
                     'black': {'role': 'unpacked_confirmed', 'current_description': 'a separate black object on the grey working table'}},
        'candidate_scopes': {2: {'entity_id': 'pink', 'role': 'wrapped_source', 'camera_id': '2'},
                             3: {'entity_id': 'black', 'role': 'unpacked_confirmed', 'camera_id': '3'}}}
    containers = {'observation_id': 'current', 'generation': 2, 'candidate_mapping': {'0': 'left', '1': 'right'},
                  'matches': {'0': {'entity_id': 'left', 'camera_id': '2'}, '1': {'entity_id': 'right', 'camera_id': '1'}},
                  'binding_hash': 'container-binding', 'scope_id': 'initial-scope', 'initial_request_hash': 'initial-request'}
    return capture, detections, entities, tracking, containers


def reference_for(capture, entities):
    initial = deepcopy(entities)
    for identifier, number, box in [('black', 1, [.1, .1, .3, .4]), ('pink', 2, [.6, .1, .8, .4])]:
        initial[identifier].update(display_id=number, color='pink', description='original pink ' + identifier,
                                   bbox=box, reference_camera_id='2')
        entities[identifier]['display_id'] = number
    return {'observation_id': 'initial', 'entities': initial,
            'demo_task': {'source_object_ids': ['black', 'pink']},
            'views': {camera: {'rgb': np.full_like(view['rgb'], 100 + int(camera))}
                      for camera, view in capture['views'].items()}}


def resolve(overrides=None, mutate=None):
    module = importlib.import_module('luna_mujoco.demo_lifecycle_inventory')
    capture, detections, entities, tracking, containers = inputs()
    if mutate:
        mutate(capture, detections, entities, tracking, containers)
    reference = reference_for(capture, entities)
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for question in payload['questions']:
            default = 'none' if question['name'] == 'demo_unlisted_pink' else 'yes'
            value, confidence = (overrides or {}).get(question['name'], (default, .95))
            answers.append(decision_answer(question, value, confidence))
        return httpx.Response(200, json={'answers': answers})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    result = module.resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
        instruction='sequentially unpack each workpiece', reference=reference)
    return result, calls, (capture, detections, entities, tracking, containers)


def test_one_call_confirms_current_roles_and_original_pink_identity():
    result, calls, _ = resolve()
    assert len(calls) == 1
    questions = calls[0]['questions']
    assert [q['name'] for q in questions] == ['identity_2', 'identity_3', 'wrapped_identity_2']
    assert [q['type'] for q in questions] == ['predicate', 'predicate', 'choice']
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    assert 'Pink foam wraps the black inner workpiece' in context['user_prior']
    assert 'identity_N role questions judge workpiece appearance' in context['rules']
    assert 'The initial inventory is frozen. These questions associate its known current roles' not in context['rules']
    assert any('INITIAL' in p.get('text', '') for p in calls[0]['input'][0]['content'])
    assert 'same original physical' in calls[0]['questions'][-1]['instructions']
    assert {group[0]['entity_id'] for group in result['groups']} == {'left', 'right', 'pink', 'black'}
    assert result['missing_geometry_ids'] == []
    assert result['demo_inventory_check']['status'] == 'not_reassessed'
    assert result['demo_inventory_check']['source'] == 'frozen_initial_inventory'
    assert 'confidence' not in result['demo_inventory_check']
    assert result['identity_coverage']['accepted_left_entity_ids'] == ['left', 'pink']
    assert result['lifecycle_correspondence']['accepted_candidate_ids'] == ['2', '3']
    black = next(group[0] for group in result['groups'] if group[0]['entity_id'] == 'black')
    assert black['identity_evidence'][0]['method'] == 'current_visual_role_and_verified_lifecycle'


@pytest.mark.parametrize('answer', [('yes', .79), ('unknown', .99), ('no', .99)])
def test_missing_current_role_is_never_promoted_by_lifecycle(answer):
    with pytest.raises(VisualInventoryError, match='visual_lifecycle_role_unconfirmed'):
        resolve({'identity_3': answer})


def collection_role(capture, detections, entities, tracking, containers):
    from luna_mujoco.demo_lifecycle_tracking import DESCRIPTIONS
    tracking['bindings']['black'].update(role='completed_collection',
        current_description=DESCRIPTIONS['completed_collection'])
    tracking['candidate_scopes'][3]['role'] = 'completed_collection'


def test_completed_collection_question_accepts_single_visible_workpiece():
    result, calls, _ = resolve({'identity_3': ('yes', .8)}, collection_role)
    question = calls[0]['questions'][1]
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    description = context['current_roles']['black']['description']
    assert description == 'one or more visible black workpieces (or their visible parts) inside the right destination bin'
    assert 'one or more black workpieces or their visible parts' in question['instructions']
    assert 'inside the right' not in question['instructions']
    assert question['type'] == 'predicate'
    assert 'black' in result['identity_coverage']['accepted_entity_ids']


@pytest.mark.parametrize('answer', [('yes', .79), ('unknown', .99), ('no', .99)])
def test_collection_description_never_bypasses_visual_role_confirmation(answer):
    with pytest.raises(VisualInventoryError, match='visual_lifecycle_role_unconfirmed'):
        resolve({'identity_3': answer}, collection_role)


@pytest.mark.parametrize('members', [['product-a'], ['product-a', 'product-b']])
def test_collection_members_remain_data_semantics_for_one_or_more_products(members):
    from luna_mujoco.demo_process import completed_product_collection
    collection = completed_product_collection({'right_bin_id': 'right',
        'product_entities': {identifier: {} for identifier in members},
        'item_lifecycle': {identifier: {'phase': 'completed'} for identifier in members}})
    assert collection['id'] == 'product-collection:right'
    assert collection['kind'] == 'collection'
    assert collection['members'] == members
    assert collection['identity_assignment'] is False
    assert collection['description'] == 'one or more visible black workpieces (or their visible parts) inside the right destination bin'


def test_left_camera_coverage_remains_required_for_wrapped_source():
    def mutate(capture, detections, entities, tracking, containers):
        detections[2]['camera_id'] = '3'
        tracking['candidate_scopes'][2]['camera_id'] = '3'
    with pytest.raises(VisualInventoryError, match='visual_lifecycle_role_unconfirmed'):
        resolve(mutate=mutate)


def test_supplementary_unknown_never_enters_geometry_when_role_has_confirmed_current_candidate():
    def mutate(capture, detections, entities, tracking, containers):
        detections.append(deepcopy(detections[3]))
        detections[-1]['camera_id'] = '1'
        tracking['candidate_scopes'][4] = {**tracking['candidate_scopes'][3], 'camera_id': '1'}
    result, _, _ = resolve({'identity_4': ('unknown', .99)}, mutate)
    group = next(group for group in result['groups'] if group[0]['entity_id'] == 'black')
    assert [entry['camera_id'] for entry in group] == ['3']
    assert result['unassociated_candidates'][0]['geometry_policy'] == 'discard_from_inventory_map'


def test_cache_binding_includes_current_role_and_container_evidence():
    module = importlib.import_module('luna_mujoco.demo_lifecycle_inventory')
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [decision_answer(q, 'none' if q['name'] == 'demo_unlisted_pink' else 'yes')
            for q in payload['questions']]})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    first = module.resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
        instruction='unpack', reference=reference)
    tracking['binding_hash'] = 'changed-role-proof'
    containers['binding_hash'] = 'changed-container-proof'
    second = module.resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
        instruction='unpack', reference=reference)
    assert len(calls) == 2
    assert first['evidence']['request_hash'] != second['evidence']['request_hash']
    assert second['container_correspondence']['binding_hash'] == 'changed-container-proof'
    assert second['lifecycle_correspondence']['binding_hash'] == 'changed-role-proof'


@pytest.mark.parametrize('mutation', [None, 'not_frozen', 'stale_capture', 'missing_bin', 'missing_geometry',
                                    'wrong_request', 'missing_role', 'initial_reference', 'owner'])
def test_runner_requires_frozen_inventory_and_all_current_roles(mutation):
    from luna_mujoco.demo_runner import DemoRunner
    result, _, (capture, _, entities, _, _) = resolve()
    runner = object.__new__(DemoRunner)
    runner.demo_task = {'workflow': 'unpack_transfer', 'source_object_ids': ['pink', 'black']}
    runner.run_id = 'run'
    runner.task_progress = SimpleNamespace(inventory_finalized=True, scene_inventory_observation_id='initial')
    role = result['lifecycle_correspondence']
    role['reference_observation_id'] = 'initial'
    role['source'] = 'current_geometry_unique_lifecycle_role_proposal'
    role['owner_id'] = 'run'
    observation = {'observation_id': capture['observation_id'], 'generation': 2, 'scene_entities': entities,
        'visual_inventory': result, 'demo_inventory_check': result['demo_inventory_check']}
    if mutation == 'not_frozen': runner.task_progress.inventory_finalized = False
    elif mutation == 'stale_capture': role['observation_id'] = 'old'
    elif mutation == 'missing_bin': result['identity_coverage']['accepted_entity_ids'].remove('right')
    elif mutation == 'missing_geometry': result['missing_geometry_ids'] = ['black']
    elif mutation == 'wrong_request': result['identity_coverage']['request_hash'] = 'old'
    elif mutation == 'missing_role': role['bindings'].pop('pink')
    elif mutation == 'initial_reference': role['reference_observation_id'] = 'other'
    elif mutation == 'owner': role['owner_id'] = 'other'
    if mutation is None:
        runner._require_visible_inventory(observation)
    else:
        with pytest.raises(ValueError, match='source_inventory_unknown'):
            runner._require_visible_inventory(observation)


@pytest.mark.parametrize('second_identity', ['pink-other', 'unknown', 'pink'])
def test_multiple_remaining_pink_sources_require_every_original_id_in_left_view(second_identity):
    module = importlib.import_module('luna_mujoco.demo_lifecycle_inventory')
    capture, detections, entities, tracking, containers = inputs()
    reference = reference_for(capture, entities)
    entities['pink-other'] = {**deepcopy(entities['pink']), 'display_id': 3}
    reference['entities']['pink-other'] = {**deepcopy(reference['entities']['pink']), 'display_id': 3,
                                         'bbox': [.2, .6, .4, .8]}
    reference['demo_task']['source_object_ids'].append('pink-other')
    tracking['bindings']['pink-other'] = deepcopy(tracking['bindings']['pink'])
    detections.append(deepcopy(detections[2]))
    detections[-1]['points'][:, 0] += .3
    tracking['candidate_scopes'][2].update(entity_id=None, entity_ids=['pink', 'pink-other'])
    tracking['candidate_scopes'][4] = deepcopy(tracking['candidate_scopes'][2])
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = [decision_answer(q, ('pink' if q['name'] == 'wrapped_identity_2' else second_identity) if q['name'].startswith('wrapped_identity_') else 'yes')
                   for q in payload['questions']]
        return httpx.Response(200, json={'answers': answers})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    def run():
        return module.resolve_lifecycle_inventory(client, capture, detections, entities, tracking, containers,
                                                  instruction='unwrap', reference=reference)
    if second_identity != 'pink-other':
        with pytest.raises(VisualInventoryError, match='original_identity_unconfirmed'):
            run()
    else:
        result = run()
        assert set(result['identity_coverage']['remaining_source_ids']) == {'pink', 'pink-other'}
        assert {'pink', 'pink-other'} <= set(result['identity_coverage']['accepted_left_entity_ids'])
    assert len(calls) == 1
    assert all([item['value'] for item in question['choices']] == ['pink', 'pink-other', 'reject', 'unknown']
               for question in calls[0]['questions'] if question['name'].startswith('wrapped_identity_'))
