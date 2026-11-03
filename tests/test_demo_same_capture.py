"""Same-frame hints require matching capture provenance and exact per-camera pixels."""
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest
from test_visual_inventory import fixture

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_inventory import VisualInventory, mask_box


def same_capture_inputs():
    capture, detection = fixture()
    capture["views"] = {"2": capture["views"]["3"]}
    detection["camera_id"] = "2"
    entities = {"pink": {"id": "pink", "kind": "cylinder", "color": "pink",
                         "bbox": mask_box(detection["silhouette_mask"]), "reference_camera_id": "2",
                         "visual_membership": "required"}}
    reference = {"observation_id": capture["observation_id"], "entities": deepcopy(entities),
                 "views": deepcopy(capture["views"]), "task_spec": {"target_refs": ["pink"]}}
    return capture, [detection], entities, reference


def inventory(calls, *, value="pink", confidence=.99, legacy=False):
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = [{"name": question["name"], "type": "predicate", "probability":
                    .5 if value == 'unknown' and question['name'].startswith('match_') else confidence}
                   if question['type'] == 'predicate' else
                   {"name": question["name"], "type": "choice",
                    "choice": value if question["name"].startswith("identity_") else
                              "none" if question["name"] == "scene_inventory_closure" else "visible",
                    "confidence": confidence if question["name"].startswith("identity_") else .99}
                   for question in payload["questions"]]
        return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers})
    cls = VisualInventory if legacy else DemoVisualInventory
    return cls(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))


def run(calls, capture, detections, entities, reference, **options):
    return inventory(calls, **options).resolve(capture, detections, entities=entities, reference=reference,
                                               instruction="move all", purpose="initialize_inventory")


def test_exact_same_capture_proposes_identity_and_retains_original_visual_answer():
    capture, detections, entities, reference = same_capture_inputs()
    calls = []
    result = run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    provenance = context["same_capture_reference"]
    assert provenance["initial_reference_observation_id"] == provenance["current_observation_id"] == "frame-1"
    assert provenance["verified_camera_ids"] == ["2"]
    assert len(provenance["raw_rgb_sha256_by_camera"]["2"]) == 64
    proposal = context["same_capture_box_correspondence_proposals"][0]
    assert proposal["camera"] == "2" and proposal["candidate"] == "candidate-0"
    assert proposal["same_camera_reference_box_overlaps"] == [
        {"reference_entity_id": "pink", "reference_box": entities["pink"]["bbox"], "box_iou": 1.}]
    assert "not established identity" in context["box_correspondence_rule"]
    question = calls[0]["questions"][0]
    instruction = question['instructions'][0]['text']
    assert "same frozen capture" in provenance['meaning']
    assert "identical frozen images" not in instruction
    assert question['type'] == 'predicate'
    assert 'R1 in the left panel' in instruction
    assert 'C0 in the right panel' in instruction
    assert result["groups"][0][0]["entity_id"] == "pink"
    evidence = result['evidence']
    assert evidence['original_decisions_answers'][0] == {
        'name': 'identity_0', 'type': 'predicate', 'probability': .99}
    assert evidence['same_capture_identity_proposals'] == {'identity_0': 'pink'}
    assert evidence['answers'][0]['choice'] == 'pink'
    location = context["candidate_geometric_location_hints"][0]
    np.testing.assert_allclose(location["measured_centroid_world_m"], detections[0]["points"].mean(axis=0))
    assert "Robot world +Y" in context["geometric_location_rule"]
    assert "identity" not in location


def test_single_pass_same_capture_does_not_duplicate_comparison_with_full_mask_subject_images():
    import json

    capture, detections, entities, reference = same_capture_inputs()
    calls = []
    result = inventory(calls).resolve(capture, detections, entities=entities, reference=reference,
        instruction='move all', purpose='associate_inventory')
    assert result['groups'][0][0]['entity_id'] == 'pink'
    assert len(calls) == 1
    metadata = [json.loads(part['text']) for part in calls[0]['input'][0]['content']
                if part['type'] == 'input_text' and part['text'].startswith('{')]
    assert not any(row.get('image_role') == 'current_candidate_sam_subject' for row in metadata)
    assert not any(row.get('image_role') == 'same_capture_identity_comparison' for row in metadata)
    question = calls[0]['questions'][0]
    panel = json.loads(question['instructions'][1]['text'])
    assert panel['image_role'] == 'same_capture_identity_comparison'
    assert panel['question'] == question['name']
    assert question['instructions'][2]['type'] == 'input_image'


@pytest.mark.parametrize('clarified_probability', [.99, .5])
@pytest.mark.parametrize('purpose', ['initialize_inventory', 'associate_inventory'])
def test_all_predicate_identity_unknown_gets_one_same_capture_subject_clarification(clarified_probability, purpose):
    capture, detections, entities, reference = same_capture_inputs()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={'answers': [
            {'name': q['name'], 'type': 'predicate',
             'probability': .5 if len(calls) == 1 else clarified_probability}
            if q['type'] == 'predicate' else
            {'name': q['name'], 'type': 'choice', 'choice': 'none' if q['name'] == 'scene_inventory_closure'
             else 'visible', 'confidence': .99} for q in payload['questions']]})

    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    if clarified_probability < .8:
        with pytest.raises(VisualInventoryError, match='visual_identity_unknown'):
            client.resolve(capture, detections, entities=entities, reference=reference,
                           instruction='move all', purpose=purpose)
    else:
        result = client.resolve(capture, detections, entities=entities, reference=reference,
                                instruction='move all', purpose=purpose)
        assert result['groups'][0][0]['entity_id'] == 'pink'
    assert len(calls) == 2
    assert calls[0]['questions'][0]['type'] == calls[1]['questions'][0]['type'] == 'predicate'
    assert any('current_candidate_sam_subject' in part.get('text', '')
               for part in calls[1]['input'][0]['content'])
    assert json.loads(calls[1]['input'][0]['content'][0]['text'])['observation_id'] == capture['observation_id']


def test_same_capture_identity_questions_share_overlap_table_without_repeating_constraints():
    capture, detections, entities, reference = same_capture_inputs()
    detections = [deepcopy(detections[0]) for _ in range(3)]
    calls = []
    run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]['input'][0]['content'][0]['text'])
    assert len(context['same_capture_box_correspondence_proposals']) == 3
    for question in calls[0]['questions'][:3]:
        instructions = question['instructions'][0]['text']
        assert 'same_camera_reference_box_overlaps' not in instructions
        assert 'These same-camera rectangles are locating hints only' not in instructions
        assert 'Reference boxes are fallible' not in instructions
        assert 'Zero overlap does not rule out identity' not in instructions
        assert 'Reference boxes are fallible' in context['box_correspondence_rule']
        assert 'Zero overlap does not rule out identity' in context['box_correspondence_rule']
        assert 'Answer false for different subjects' in instructions
        assert 'remain uncertain' in instructions
        assert 'A box and an object inside it are separate objects' in instructions
        assert question['type'] == 'predicate'


@pytest.mark.parametrize("case", ["different_id", "different_pixels", "missing_reference",
                                 "missing_reference_id", "missing_reference_view"])
def test_same_capture_claim_is_omitted_without_both_provenance_and_pixel_equality(case):
    capture, detections, entities, reference = same_capture_inputs()
    if case == "different_id":
        reference["observation_id"] = "older-frame"
    elif case == "different_pixels":
        reference["views"]["2"]["rgb"][0, 0, 0] ^= 1
    elif case == "missing_reference":
        reference = None
    elif case == "missing_reference_id":
        reference.pop("observation_id")
    else:
        reference["views"] = {}
    calls = []
    run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert "same_capture_reference" not in context
    assert "same_capture_box_correspondence_proposals" not in context
    assert "identical frozen images" not in calls[0]["questions"][0]["instructions"]


def test_hints_are_scoped_to_verified_cameras_and_matching_reference_box_camera():
    capture, detections, entities, reference = same_capture_inputs()
    capture["views"]["1"] = deepcopy(capture["views"]["2"])
    reference["views"]["1"] = deepcopy(capture["views"]["1"])
    reference["views"]["1"]["rgb"][0, 0, 0] ^= 1
    detections.append({**deepcopy(detections[0]), "camera_id": "1"})
    reference["entities"]["other-camera"] = {
        **deepcopy(entities["pink"]), "id": "other-camera", "reference_camera_id": "1"}
    calls = []
    run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert context["same_capture_reference"]["verified_camera_ids"] == ["2"]
    proposals = context["same_capture_box_correspondence_proposals"]
    assert len(proposals) == 1
    assert [item["reference_entity_id"] for item in proposals[0]["same_camera_reference_box_overlaps"]] == ["pink"]
    assert "identical frozen images" not in calls[0]["questions"][1]["instructions"]


def test_reference_box_in_another_camera_does_not_produce_a_correspondence_proposal():
    capture, detections, entities, reference = same_capture_inputs()
    reference["entities"]["pink"]["reference_camera_id"] = "1"
    calls = []
    run(calls, capture, detections, entities, reference)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert context["same_capture_box_correspondence_proposals"] == []
    assert "identical frozen images" not in calls[0]["questions"][0]["instructions"]


@pytest.mark.parametrize("value,confidence", [("unknown", .99), ("pink", .79)])
def test_ambiguous_full_overlap_never_overrides_unknown_or_low_confidence(value, confidence):
    capture, detections, entities, reference = same_capture_inputs()
    entities["twin"] = {**deepcopy(entities["pink"]), "id": "twin"}
    reference["entities"]["twin"] = deepcopy(entities["twin"])
    calls = []
    with pytest.raises(VisualInventoryError, match="visual_identity_unknown"):
        run(calls, capture, detections, entities, reference, value=value, confidence=confidence)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    overlaps = context["same_capture_box_correspondence_proposals"][0]["same_camera_reference_box_overlaps"]
    assert [item["box_iou"] for item in overlaps] == [1., 1.]


def test_legacy_same_capture_does_not_receive_demo_grounding():
    capture, detections, entities, reference = same_capture_inputs()
    calls = []
    run(calls, capture, detections, entities, reference, legacy=True)
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    assert "same_capture_reference" not in context
    assert "same_capture_box_correspondence_proposals" not in context


@pytest.mark.parametrize("value,confidence,accepted", [
    ("pink", .99, True), ("pink", .79, False), ("unknown", .99, False),
])
def test_inaccurate_reference_box_never_controls_decisions_identity(value, confidence, accepted):
    # Recorded failure: a container reference covered only its upper edge while
    # another entity had the highest IoU. The image pixels had not changed.
    capture, detections, entities, reference = same_capture_inputs()
    entities["pink"]["bbox"] = [0., 0., 1., .01]
    reference["entities"]["pink"] = deepcopy(entities["pink"])
    reference["entities"]["overlapping-table"] = {
        "id": "overlapping-table", "kind": "table", "reference_camera_id": "2",
        "bbox": mask_box(detections[0]["silhouette_mask"])}
    calls = []
    if accepted:
        result = run(calls, capture, detections, entities, reference, value=value, confidence=confidence)
        assert result["groups"][0][0]["entity_id"] == "pink"
    else:
        with pytest.raises(VisualInventoryError, match="visual_identity_unknown") as caught:
            run(calls, capture, detections, entities, reference, value=value, confidence=confidence)
        current = caught.value.evidence
        assert current['answers'][0]['choice'] == 'unknown'
        assert current['answers'][0]['confidence'] == 0.
        initial = current['initial_identity_failure']['evidence']
        assert initial['answers'][0]['choice'] == value
        assert initial['answers'][0]['confidence'] == confidence
        assert current['request_hash'] != initial['request_hash']
        assert len(calls) == 2
    context = json.loads(calls[0]["input"][0]["content"][0]["text"])
    overlaps = context["same_capture_box_correspondence_proposals"][0]["same_camera_reference_box_overlaps"]
    assert [(row["reference_entity_id"], row["box_iou"]) for row in overlaps] == [
        ("pink", 0.), ("overlapping-table", 1.)]
    assert "Zero overlap does not rule out identity" in context["box_correspondence_rule"]
    question = calls[0]["questions"][0]
    assert "Reference boxes are fallible" not in question["instructions"]
    assert "not ranked identity candidates" in context["box_correspondence_rule"]
    assert [item["value"] for item in question["choices"]] == ["pink", "reject", "unknown"]
