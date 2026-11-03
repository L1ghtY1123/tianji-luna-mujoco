"""Supplementary uncertain proposals never erase a fully identified view union."""
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from test_perception import capture_fixture
from test_visual_inventory import fixture

from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import VisualInventory


def clients(identities, *, extra="unknown", visibility=None):
    def respond(request):
        payload = json.loads(request.content)
        answers = []
        for question in payload["questions"]:
            name = question["name"]
            if question['type'] == 'predicate' and name.startswith('match_'):
                value, confidence = identities[int(name.split('_')[1])]
                answers.append({'name': name, 'type': 'predicate', 'probability': confidence
                                if question['instructions'].partition(' Reference subject: ')[0].endswith(f'permanent reference ID {value}.')
                                else 1 - confidence})
                continue
            if name.startswith("identity_"):
                value, confidence = identities[int(name.split("_")[1])]
            else:
                value = extra if name == "scene_inventory_closure" else (
                    "at_destination" if name.startswith("progress_") else "visible")
                value = (visibility or {}).get(name, value)
                confidence = .99
            answers.append({"name": name, "type": "choice", "choice": value, "confidence": confidence})
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})
    return httpx.Client(transport=httpx.MockTransport(respond))


def inputs(cameras=("2", "1")):
    capture, detection = fixture()
    capture["views"] = {camera: deepcopy(capture["views"]["3"]) for camera in ("1", "2", "3")}
    detections = [{**deepcopy(detection), "camera_id": camera, "prompt": "pink cup",
                   "requested_color": "pink"} for camera in cameras]
    entities = {"pink": {"id": "pink", "kind": "object", "color": "pink", "roles": ["collision"],
                         "visual_membership": "required"}}
    reference = {"task_spec": {"target_refs": ["pink"]}, "views": {}, "entities": entities}
    return capture, detections, entities, reference


def test_primary_identity_survives_uncertain_supplement_without_lowering_threshold():
    capture, detections, entities, reference = inputs()
    before = deepcopy(capture)
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="initialize_inventory")
    assert [entry["camera_id"] for entry in result["groups"][0]] == ["2"]
    assert result["missing_geometry_ids"] == []
    assert result["demo_inventory_check"]["status"] == "unknown"
    assert result["unassociated_candidates"] == [{
        "candidate": "candidate-1", "camera_id": "1", "status": "unknown",
        "choice": "pink", "confidence": .64, "minimum_confidence": .8,
        "geometry_policy": "discard_from_inventory_map", "reason": "supplementary_identity_uncertain"}]
    np.testing.assert_array_equal(capture["views"]["1"]["depth_m"], before["views"]["1"]["depth_m"])
    assert result["evidence"]["answers"][1]["confidence"] == .64


def test_frozen_inventory_can_return_measured_subset_for_recovery_mapping():
    capture, detections, entities, reference = inputs()
    entities['green'] = {'id': 'green', 'kind': 'object', 'color': 'green',
                         'roles': ['collision'], 'visual_membership': 'required'}
    reference['task_spec']['target_refs'] = ['pink', 'green']
    reference['inventory_state'] = {'inventory_finalized': True}
    inventory = DemoVisualInventory(api_key='test',
        client=clients([('pink', .99), ('green', .77)], visibility={'visibility_green': 'visible'}))
    result = inventory.resolve(capture, detections, instruction='move both', entities=entities,
        reference=reference, purpose='associate_inventory')
    assert result['missing_geometry_ids'] == ['green']
    assert result['identity_coverage']['accepted_entity_ids'] == ['pink']
    assert {entry['entity_id'] for group in result['groups'] for entry in group} == {'pink'}
    assert result['unassociated_candidates'][0]['confidence'] < .8
    assert result['identity_coverage']['complete'] is False
    assert result['evidence']['identity_correspondence']
    assert 'clarifies_request_hash' not in result['evidence']


@pytest.mark.parametrize('green_match,other_match,expected_missing', [
    (.99, .0, []), (.65, .0, ['green']), (.99, .99, ['green'])])
def test_partial_identity_uses_one_pairwise_request(green_match, other_match, expected_missing):
    capture, detections, entities, reference = inputs()
    entities['green'] = {'id': 'green', 'kind': 'object', 'color': 'green',
                         'roles': ['collision'], 'visual_membership': 'required'}
    reference['task_spec']['target_refs'] = ['pink', 'green']
    reference['inventory_state'] = {'inventory_finalized': True}
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        answers = []
        for question in payload['questions']:
            name = question['name']
            if question['type'] == 'predicate':
                matching_green = question['instructions'].partition(' Reference subject: ')[0].endswith('permanent reference ID green.')
                probability = ((green_match if matching_green else other_match) if name.startswith('match_1_')
                               else (0. if matching_green else .99))
                answers.append({'name': name, 'type': 'predicate', 'probability': probability})
            else:
                value = 'pink' if name == 'identity_0' else 'green' if name == 'identity_1' else (
                    'none' if name == 'scene_inventory_closure' else 'visible')
                answers.append({'name': name, 'type': 'choice', 'choice': value,
                                'confidence': .59 if name == 'identity_1' else .99})
        return httpx.Response(200, json={'answers': answers})
    inventory = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = inventory.resolve(capture, detections, instruction='move both', entities=entities,
                               reference=reference, purpose='associate_inventory')
    assert result['missing_geometry_ids'] == expected_missing
    assert len(requests) == 1
    assert any(question['name'].startswith('match_0_') for question in requests[0]['questions'])
    assert any(question['name'].startswith('match_1_') for question in requests[0]['questions'])
    assert 'clarifies_request_hash' not in result['evidence']
    assert result['evidence']['identity_correspondence'][1]['probabilities']['green'] == green_match


@pytest.mark.parametrize("case", ["only_unknown", "missing_bin"])
def test_incomplete_or_primary_uncertainty_remains_a_failure(case):
    cameras = ("2", "1")
    identities = [("pink", .99), ("pink", .64)]
    if case == "only_unknown":
        cameras, identities = ("1",), [("pink", .64)]
    capture, detections, entities, reference = inputs(cameras)
    if case == "missing_bin":
        entities["bin"] = {"id": "bin", "kind": "bin", "color": "blue", "roles": ["collision"]}
    inventory = DemoVisualInventory(api_key="test", client=clients(identities))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        inventory.resolve(capture, detections, instruction="move all", entities=entities,
            reference=reference, purpose="initialize_inventory")


@pytest.mark.parametrize('target_confidence', [.99, .64])
@pytest.mark.parametrize('bin_visibility', ['visible', 'unseen', 'unknown'])
def test_later_partial_observation_keeps_confirmed_source_and_reports_missing_bin(
        target_confidence, bin_visibility):
    capture, detections, entities, reference = inputs()
    entities['bin'] = {'id': 'bin', 'kind': 'bin', 'color': 'blue', 'roles': ['collision']}
    inventory = DemoVisualInventory(api_key='test',
        client=clients([('pink', target_confidence), ('bin', .64)],
                       visibility={'visibility_bin': bin_visibility}))
    if target_confidence < .8 or bin_visibility == 'visible':
        with pytest.raises(VisualInventoryError, match='visual_identity_unknown'):
            inventory.resolve(capture, detections, instruction='move one', entities=entities,
                              reference=reference, purpose='associate_inventory')
        return
    result = inventory.resolve(capture, detections, instruction='move one', entities=entities,
                               reference=reference, purpose='associate_inventory')
    assert result['missing_geometry_ids'] == ['bin']
    assert result['identity_coverage']['accepted_entity_ids'] == ['pink']
    assert result['identity_coverage']['required_entity_ids'] == ['bin', 'pink']
    assert {entry['entity_id'] for group in result['groups'] for entry in group} == {'pink'}
    assert result['unassociated_candidates'][0]['confidence'] < .8
    assert result['evidence']['identity_correspondence']
    assert 'clarifies_request_hash' not in result['evidence']
    assert result['unassociated_candidates'][0]['geometry_policy'] == 'discard_from_inventory_map'


@pytest.mark.parametrize("camera", ["1", "2", "3"])
def test_any_camera_can_supply_identity_while_unknown_candidates_keep_their_evidence(camera):
    capture, detections, entities, reference = inputs(("1", camera, "2"))
    inventory = DemoVisualInventory(api_key="test",
        client=clients([("pink", .64), ("pink", .99), ("pink", .61)]))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="associate_inventory")
    assert result["groups"][0][0]["camera_id"] == camera
    assert [(row["candidate"], row["camera_id"], row["confidence"])
            for row in result["unassociated_candidates"]] == [
                ("candidate-0", "1", 0.), ("candidate-2", "2", 0.)]
    checks = result['evidence']['identity_correspondence']
    assert checks[0]['probabilities']['pink'] == .64
    assert checks[2]['probabilities']['pink'] == .61


def test_visible_right_bin_without_confident_identity_still_blocks_complete_union():
    # Frozen run 3c5879: left sources are trusted; the right bin is visible but
    # its head identity is .79 and the right wrist misidentifies it at .63.
    capture, detections, entities, reference = inputs(("3", "2", "2", "1"))
    entities.update({
        "source-bin": {"id": "source-bin", "kind": "bin", "color": "blue", "roles": ["collision"]},
        "right-bin": {"id": "right-bin", "kind": "bin", "color": "blue", "roles": ["collision"]},
    })
    inventory = DemoVisualInventory(api_key="test", client=clients([
        ("right-bin", .79), ("pink", .94), ("source-bin", .97), ("source-bin", .63)],
        extra="none"))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown") as caught:
        inventory.resolve(capture, detections, instruction="move all", entities=entities,
            reference=reference, purpose="initialize_inventory")
    assert caught.value.evidence["candidate"] == "candidate-0"
    assert caught.value.evidence["minimum_confidence"] == .8
    assert any(answer["name"] == "visibility_right-bin" and answer["choice"] == "visible"
               and answer["confidence"] == .99 for answer in caught.value.evidence["answers"])


def test_redundant_primary_proposal_is_discarded_only_with_complete_visible_inventory():
    capture, detections, entities, reference = inputs(("2", "2"))
    inventory = DemoVisualInventory(api_key="test",
        client=clients([("pink", .99), ("pink", .61)], extra="none"))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="initialize_inventory")
    assert len(result["groups"]) == 1
    assert result["unassociated_candidates"][0]["camera_id"] == "2"
    assert result["unassociated_candidates"][0]["confidence"] == .61
    assert result["demo_inventory_check"]["status"] == "none"


def test_post_place_uses_camera_union_and_keeps_unknown_source_check_separate():
    capture, detections, entities, reference = inputs(("3", "2"))
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    result = inventory.resolve(capture, detections, instruction="move one", entities=entities,
        reference=reference, purpose="verify_transfer", progress={"pink": {"destination_id": "table"}})
    assert result["progress"]["pink"]["status"] == "at_destination"
    assert result["demo_inventory_check"]["status"] == "unknown"
    assert result["groups"][0][0]["camera_id"] == "3"


def test_association_after_transfer_does_not_require_completed_objects_in_left_camera():
    capture, detections, entities, reference = inputs(("3", "1"))
    inventory = DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    result = inventory.resolve(capture, detections, instruction="move all", entities=entities,
        reference=reference, purpose="associate_inventory",
        progress={"pink": {"status": "completed", "destination_id": "table"}})
    assert result["groups"][0][0]["camera_id"] == "3"


def test_legacy_inventory_still_rejects_any_uncertain_candidate():
    capture, detections, entities, reference = inputs()
    inventory = VisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)]))
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        inventory.resolve(capture, detections, instruction="move all", entities=entities, reference=reference)


def test_perception_discards_uncertain_secondary_depth_from_inventory_map():
    capture, masks = capture_fixture()
    for camera, view in capture["views"].items():
        view["rgb"][0, 0] = [int(camera), 0, 0]
    entities = {"pink": {"id": "pink", "kind": "cube", "color": "pink", "roles": ["collision"],
        "visual_membership": "required", "segmentation_query": {"kind": "cube", "color": "pink", "prompt": "pink cup"}}}
    entities["table"] = {"kind": "table", "bbox": [0, 0, 1, 1], "reference_camera_id": "3"}
    reference = {"observation_id": capture["observation_id"], "entities": entities, "views": {},
        "task_spec": {"source_ref": "bin", "target_refs": ["pink"], "destination_ref": "table"},
        "policy_graph": {"version": 1, "steps": [{"type": "stop", "reason": "fixture"}]}}
    class Detector:
        def segment(self, rgb, text, timestamp_us):
            return [] if rgb[0, 0, 0] == 3 else [{"mask": masks[0], "score": .98}]
    engine = DemoPerception(Detector(),
        visual_inventory=DemoVisualInventory(api_key="test", client=clients([("pink", .99), ("pink", .64)])),
        robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}})
    result = engine.observe(capture, instruction="move all", source_hint={"scene_entities": entities,
        "visual_reference": reference, "visual_decision": {"source": "responses_program", "purpose": "associate_inventory"}})
    assert result["objects"][0]["coverage"]["camera_ids"] == ["2"]
    assert result["views"]["1"]["instance_masks"] == {}
    assert not np.any(result["views"]["1"]["depth_m"])
    assert not result["views"]["1"]["valid_depth"].any()
    assert result["visual_inventory"]["unassociated_candidates"][0]["camera_id"] == "1"
