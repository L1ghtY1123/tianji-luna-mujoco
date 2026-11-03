"""A processing output has independent, immutable RGB reference evidence."""
import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from luna_mujoco.visual_inventory import product_reference_content


def example():
    evidence = {'output_entity_id': 'product', 'owner_id': 'run', 'request_hash': 'verified',
                'generation': 1, 'observation_id': 'product-created'}
    entity = {'id': 'product', 'description': 'new black workpiece', 'creation_evidence': evidence}
    views = {str(i): {'rgb': np.full((10, 12, 3), i, np.uint8)} for i in (1, 2, 3)}
    proof = {'generation': 1, 'observation_id': 'product-created', 'time': 2., 'images': {
        key: {'rgb_sha256': hashlib.sha256(view['rgb'].tobytes()).hexdigest(), 'shape': [10, 12, 3]}
        for key, view in views.items()}}
    reference = {'owner_id': 'run', 'product_references': {'product': {
        'entity': deepcopy(entity), 'evidence': deepcopy(evidence), 'generation': 1,
        'observation_id': 'product-created', 'verification_capture': proof, 'views': views}}}
    return reference, {'generation': 1, 'time': 3., 'observation_id': 'current',
                       'views': deepcopy(views)}, {'product': entity}


def test_product_reference_supplies_each_original_camera_and_independent_id():
    reference, capture, entities = example()
    parts = product_reference_content(reference, capture, entities)
    assert len(parts) == 6
    metadata = [json.loads(part['text']) for part in parts if part['type'] == 'input_text']
    assert {row['camera_id'] for row in metadata} == {'1', '2', '3'}
    assert all(row['entity_id'] == 'product' and row['observation_id'] == 'product-created' for row in metadata)
    assert all('does not prove current location' in row['meaning'] for row in metadata)
    assert all(row['pixels_identical_to_current_same_camera'] is True for row in metadata)


def test_general_judgments_receive_validated_product_references():
    from test_agent_model_layer import Client

    from luna_mujoco.agent.decisions import DecisionsJudgments, JudgmentRequest

    reference, capture, entities = example()
    reference['entities'] = entities
    http = Client([{'answers': [{'name': 'supported', 'type': 'predicate', 'probability': .9}]}])
    judgments = DecisionsJudgments(api_key='test', client=http)
    judgments.judge(capture, requests=[JudgmentRequest(name='supported', instructions='Is the product supported?')],
                    context={'subject_id': 'product'}, reference=reference)
    parts = http.requests[0][1]['input'][0]['content']
    metadata = [json.loads(part['text']) for part in parts
                if part['type'] == 'input_text' and part['text'].startswith('{')]
    products = [row for row in metadata if row.get('image_role') == 'verified_product_reference']
    assert {row['camera_id'] for row in products} == {'1', '2', '3'}


def test_equal_pixels_are_checked_per_camera_and_never_assumed_after_a_change():
    reference, capture, entities = example()
    capture['views']['1']['rgb'][0, 0] = 255
    del capture['views']['2']
    parts = product_reference_content(reference, capture, entities)
    metadata = [json.loads(part['text']) for part in parts if part['type'] == 'input_text']
    assert {row['camera_id']: row['pixels_identical_to_current_same_camera'] for row in metadata} == {
        '1': False, '2': False, '3': True}


@pytest.mark.parametrize('mutation', ['retired', 'owner', 'generation', 'time', 'pixels', 'hash',
                                     'observation', 'entity', 'evidence', 'camera'])
def test_invalid_or_retired_product_reference_never_supplies_images(mutation):
    reference, capture, entities = example()
    stored = reference['product_references']['product']
    if mutation == 'retired':
        entities.clear()
    elif mutation == 'owner':
        reference['owner_id'] = 'other-run'
    elif mutation == 'generation':
        capture['generation'] = 2
    elif mutation == 'time':
        capture['time'] = 1.
    elif mutation == 'pixels':
        stored['views']['2']['rgb'][0, 0] = 255
    elif mutation == 'hash':
        stored['verification_capture']['images']['1']['rgb_sha256'] = 'tampered'
    elif mutation == 'observation':
        stored['verification_capture']['observation_id'] = 'other'
    elif mutation == 'entity':
        stored['entity']['id'] = 'input'
    elif mutation == 'evidence':
        entities['product']['creation_evidence'] = {'output_entity_id': 'input'}
    else:
        del stored['views']['3']
    assert product_reference_content(reference, capture, entities) == []
