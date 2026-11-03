from copy import deepcopy

import pytest


def payload():
    import json
    context = {'observation_id': 'frame', 'generation': 2,
        'same_capture_reference': {'initial_reference_observation_id': 'frame',
            'current_observation_id': 'frame', 'generation': 2, 'verified_camera_ids': ['3']},
        'same_capture_box_correspondence_proposals': [{'camera': '3', 'candidate': 'candidate-4',
            'same_camera_reference_box_overlaps': [{'reference_entity_id': 'box-a', 'box_iou': .96},
                                                  {'reference_entity_id': 'box-b', 'box_iou': .01}]}]}
    return {'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': json.dumps(context)}]}],
        'questions': [{'type': 'choice', 'name': 'identity_4', 'instructions': 'Which subject?',
            'choices': [{'value': 'box-a', 'description': 'R2 blue box'},
                        {'value': 'box-b', 'description': 'R3 blue box'},
                        {'value': 'unknown'}, {'value': 'reject'}]}]}


def test_same_capture_overlap_only_proposes_a_visual_question():
    from luna_mujoco.identity_proposals import scope_same_capture_identity
    original = payload()
    before = deepcopy(original)
    scoped, proposals = scope_same_capture_identity(original)
    assert original == before
    assert proposals == {'identity_4': 'box-a'}
    assert scoped['questions'][0]['type'] == 'predicate'
    assert 'R2 blue box' in scoped['questions'][0]['instructions']


@pytest.mark.parametrize('damage', ['later', 'camera', 'generation', 'overlap', 'ambiguous'])
def test_no_scoping_for_unverified_or_ambiguous_reference(damage):
    import json

    from luna_mujoco.identity_proposals import scope_same_capture_identity
    value = payload()
    context = json.loads(value['input'][0]['content'][0]['text'])
    proof = context['same_capture_reference']
    overlaps = context['same_capture_box_correspondence_proposals'][0]['same_camera_reference_box_overlaps']
    if damage == 'later': proof['initial_reference_observation_id'] = 'old'
    elif damage == 'camera': proof['verified_camera_ids'] = ['1']
    elif damage == 'generation': proof['generation'] = 1
    elif damage == 'overlap': overlaps[0]['box_iou'] = .7
    else: overlaps[1]['box_iou'] = .9
    value['input'][0]['content'][0]['text'] = json.dumps(context)
    assert scope_same_capture_identity(value) == (value, {})


@pytest.mark.parametrize('probability,expected', [(1., 'box-a'), (.8, 'box-a'), (.79, 'unknown'), (0., 'unknown')])
def test_geometry_never_assigns_identity_without_positive_model_evidence(probability, expected):
    from luna_mujoco.identity_proposals import resolve_identity_proposals
    body = {'answers': [{'type': 'predicate', 'name': 'identity_4', 'probability': probability}]}
    converted = resolve_identity_proposals(body, {'identity_4': 'box-a'}, .8)
    assert converted['answers'][0]['choice'] == expected
    assert body['answers'][0]['type'] == 'predicate'


@pytest.mark.parametrize('probability', [float('nan'), True, -1., 2., None])
def test_malformed_model_probability_cannot_confirm_identity(probability):
    from luna_mujoco.identity_proposals import resolve_identity_proposals
    with pytest.raises(ValueError, match='invalid_response'):
        resolve_identity_proposals({'answers': [{'type': 'predicate', 'name': 'identity_4',
            'probability': probability}]}, {'identity_4': 'box-a'}, .8)


def test_initial_container_tracking_accepts_confirmed_predicate_with_its_evidence():
    from test_demo_container_tracking import case, module
    from test_demo_same_capture import inventory

    from luna_mujoco.visual_inventory import mask_box

    observation, capture, _, detections, entities = case()
    entities.pop('pink')
    entities['right'].update(bbox=mask_box(detections[0]['mask']), reference_camera_id='1')
    reference = observation['visual_reference']
    reference.update(entities=entities, task_spec={'target_refs': ['right']})
    result = inventory([], value='right').resolve(capture, detections, entities=entities,
        reference=reference, instruction='observe the box', purpose='associate_inventory')
    assert result['evidence']['original_decisions_answers'][0]['type'] == 'predicate'
    observation.update(scene_entities=entities, visual_inventory=result)
    tracked = module().reference_with_container_tracks(observation, capture)
    answers = tracked['container_identity_tracks']['containers']['right']['initial_identity_answers']
    assert answers[0]['choice'] == 'right'
    assert answers[0]['confidence'] == .99
