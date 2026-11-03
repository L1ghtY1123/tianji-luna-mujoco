"""Exclude impossible same-instance assignments, never choose a new ID by position."""
from copy import deepcopy

import numpy as np
import pytest


def inputs():
    capture = {'observation_id': 'now', 'generation': 2}
    # B6: simultaneous right-head and left-wrist blue box surfaces.
    bounds = [((.4, -.715584, .5), (.6, -.420000, .7)),
              ((.4, .385795, .5), (.6, .694361, .7))]
    detections = [{'camera_id': camera, 'points': np.tile(np.array(box), (20, 1))}
                  for camera, box in zip(('3', '2'), bounds)]
    anchors = [{'entity_id': 'right-box', 'candidate': 'candidate-0', 'camera': '3',
        'observation_id': 'now', 'generation': 2, 'source': 'measured_container_correspondence',
        'binding_hash': 'verified-current-correspondence'}]
    models = {'right-box': {'instance_id': 'right-box', 'generation': 2,
        'asset_id': 'blue-box-cad', 'source_sha256': 'catalog-source', 'mesh_sha256': 'mesh',
        'validated_observation_id': 'earlier', 'maximum_diameter_m': float(np.linalg.norm([.410, .310, .145])),
        'surface_tolerance_m': .00725, 'source': 'validated_cad_model_binding'}}
    return capture, detections, anchors, models


def exclusions(*args):
    from luna_mujoco.identity_geometry import cad_identity_exclusions
    return cad_identity_exclusions(*args)


def test_b6_separate_boxes_cannot_be_combined_just_because_they_share_cad():
    args = inputs()
    result = exclusions(*args)
    assert len(result) == 1
    row = result[0]
    assert row['candidate'] == 'candidate-1' and row['excluded_instance_id'] == 'right-box'
    assert row['minimum_surface_separation_m'] == pytest.approx(.805795)
    assert row['maximum_same_instance_separation_m'] == pytest.approx(np.linalg.norm([.410, .310, .145]) + .0145)
    assert row['assignment'] == 'excluded_not_reassigned'


@pytest.mark.parametrize('change', ['no_model', 'old_generation', 'old_anchor', 'no_binding', 'too_few_points'])
def test_missing_or_historical_evidence_does_not_exclude_identity(change):
    capture, detections, anchors, models = inputs()
    if change == 'no_model':
        models = {}
    elif change == 'old_generation':
        models['right-box']['generation'] = 1
    elif change == 'old_anchor':
        anchors[0]['observation_id'] = 'old'
    elif change == 'no_binding':
        anchors[0]['binding_hash'] = None
    else:
        detections[1]['points'] = detections[1]['points'][:3]
    assert exclusions(capture, detections, anchors, models) == []


def test_overlapping_or_nearby_surfaces_still_require_visual_identity():
    capture, detections, anchors, models = inputs()
    detections[1]['points'][:, 1] -= .9
    assert exclusions(capture, detections, anchors, models) == []


def test_only_impossible_model_id_is_excluded_and_inputs_are_unchanged():
    capture, detections, anchors, models = inputs()
    models['other-box'] = deepcopy(models['right-box']) | {'instance_id': 'other-box'}
    before = [item['points'].copy() for item in detections]
    result = exclusions(capture, detections, anchors, models)
    assert {row['excluded_instance_id'] for row in result} == {'right-box'}
    for actual, expected in zip(detections, before):
        np.testing.assert_array_equal(actual['points'], expected)


@pytest.mark.parametrize('reply', ['other-box', 'right-box'])
def test_wire_removes_only_excluded_id_and_parser_rejects_a_model_override(reply):
    from test_visual_inventory import client_with_answers, fixture

    from luna_mujoco.visual_inventory import VisualInventory, VisualInventoryError

    capture, detection = fixture()
    geometry = exclusions(*inputs())
    row = geometry[0] | {'candidate': 'candidate-0', 'camera': '3',
        'observation_id': capture['observation_id'], 'generation': capture['generation']}
    requests = []
    visual = VisualInventory(api_key='test', client=client_with_answers({'identity_0': reply}, requests))
    entities = {identifier: {'kind': 'bin', 'color': 'blue'} for identifier in ('right-box', 'other-box')}
    arguments = {'instruction': 'inspect boxes', 'entities': entities,
        'identity_context': {'identity_geometry_exclusions': [row]}}
    if reply == 'right-box':
        with pytest.raises(VisualInventoryError, match='visual_decisions_invalid_response'):
            visual.resolve(capture, [detection], **arguments)
    else:
        result = visual.resolve(capture, [detection], **arguments)
        assert result['groups'][0][0]['entity_id'] == 'other-box'
        assert result['evidence']['identity_geometry_exclusions'] == [row]
    question = requests[0]['questions'][0]
    assert {item['value'] for item in question['choices']} == {'other-box', 'reject', 'unknown'}


def test_no_possible_inventory_id_keeps_explicit_reject_unknown_in_clarification():
    from luna_mujoco.identity_correspondence import correspondence_payload, resolve_correspondences

    question = {'type': 'choice', 'name': 'identity_0', 'instructions': 'Identify current candidate.',
        'choices': [{'value': name, 'description': name} for name in ('reject', 'unknown')]}
    payload = {'input': [{'content': [{'type': 'input_text', 'text':
        '{"candidate":"candidate-0","camera":"3"}'}]}], 'questions': [question]}
    request, pairs = correspondence_payload(payload)
    assert request['questions'] == [question] and pairs == {}
    body, _ = resolve_correspondences({'answers': [{'type': 'choice', 'name': 'identity_0',
        'choice': 'unknown', 'confidence': 1.}]}, request, payload, pairs, .8)
    assert body['answers'][0]['choice'] == 'unknown'
