"""Demo evidence presentation retains full frames and permanent identity semantics."""
import json
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from test_demo_same_capture import inventory
from test_visual_inventory import client_with_answers

from luna_mujoco.demo_perception import _same_capture_context
from luna_mujoco.visual_inventory import VisualInventory


def test_identity_question_carries_its_own_measured_location_without_assigning_an_id():
    from luna_mujoco.demo_identity_presentation import ground_identity_locations
    rows = [{'candidate': 'candidate-7', 'camera': '1',
             'measured_centroid_world_m': [.45, -.64, .55],
             'measured_bounds_world_m': [[.3, -1.34, 0.], [.98, -.42, .65]]}]
    payload = {'input': [{'content': [{'type': 'input_text', 'text': json.dumps({
        'candidate_geometric_location_hints': rows})}]}], 'questions': [
        {'type': 'choice', 'name': 'identity_7', 'instructions': 'Which instance?',
         'choices': [{'value': 'left'}, {'value': 'reject'}, {'value': 'unknown'}]},
        {'type': 'predicate', 'name': 'identity_2', 'instructions': 'Same instance?'},
        {'type': 'predicate', 'name': 'source_identity', 'instructions': 'Source?'}]}
    before = deepcopy(payload)
    result = ground_identity_locations(payload)
    assert payload == before
    assert result['input'] == before['input']
    assert result['questions'][0]['choices'] == before['questions'][0]['choices']
    assert result['questions'][1:] == before['questions'][1:]
    instruction = result['questions'][0]['instructions']
    assert json.dumps(rows[0]) in instruction
    assert 'same physical instance' in instruction
    assert '+Y is robot LEFT' in instruction
    assert 'partial surfaces' in instruction
    assert 'not an identity verdict' in instruction
    assert ground_identity_locations(result) == result


def source_payload():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    detections, entities = [], {}
    for index, column in enumerate((40, 70, 100)):
        rgb[50:70, column:column + 14] = [240, 80, 180]
        mask = np.zeros((120, 160), bool)
        mask[50:70, column:column + 14] = True
        identifier = f"permanent-pink-{index + 1}"
        entities[identifier] = {"id": identifier, "kind": "object", "color": "pink",
            "bbox": [column / 160, 50 / 120, (column + 14) / 160, 70 / 120],
            "reference_camera_id": "2", "visual_membership": "required", "display_id": index + 1}
        detections.append({"camera_id": "2", "mask": mask, "silhouette_mask": mask,
            "points": np.tile([.2, .5, .5], (280, 1)), "pixels": np.zeros((280, 3)),
            "pixels_rc": np.argwhere(mask), "kind": "object", "prompt": "pink object"})
    capture = {"observation_id": "same-capture", "generation": 2, "views": {"2": {"rgb": rgb}}}
    reference = {"observation_id": "same-capture", "views": deepcopy(capture["views"]),
        "entities": entities, "task_spec": {"target_refs": list(entities)}}
    context = _same_capture_context(capture, detections, reference)
    calls = []
    answers = {f"identity_{i}": identifier for i, identifier in enumerate(entities)}
    # Build pre-presentation inputs; byte dedup is the final API-boundary step.
    with patch('luna_mujoco.visual_image_selection.compact_decision_images', side_effect=lambda p: (p, {})):
        VisualInventory(api_key="test", client=client_with_answers(answers, calls)).resolve(
            capture, detections, instruction="move three", reference=reference, entities=entities,
            source_inventory={"source_id": "source-bin", "source_object_ids": list(entities)},
            identity_context=context)
    return calls[0], capture, reference, detections


def metadata(payload, role):
    records = []
    for part in payload["input"][0]["content"]:
        if part["type"] != "input_text":
            continue
        try:
            item = json.loads(part["text"])
        except ValueError:
            continue
        if item.get("image_role") == role:
            records.append(item)
    return records


def test_short_labels_keep_canonical_options_and_full_rgb_with_bounded_paired_zoom():
    from luna_mujoco.demo_identity_presentation import improve_identity_presentation
    payload, _, _, _ = source_payload()
    before = deepcopy(payload)
    result = improve_identity_presentation(payload)
    assert payload == before
    content = result["input"][0]["content"]
    context = json.loads(content[0]["text"])
    assert set(context["identity_display_legend"]) == {"Reference 1", "Reference 2", "Reference 3"}
    assert {row["entity_id"] for row in context["identity_display_legend"].values()} == {
        "permanent-pink-1", "permanent-pink-2", "permanent-pink-3"}
    assert context['image_label_legend']['R1']['entity_id'] == 'permanent-pink-1'
    assert context['image_label_legend']['C0']['candidate'] == 'candidate-0'
    assert 'unchanged full raw RGB' in context['display_label_rule']
    for original, updated in zip(payload["questions"], result["questions"], strict=True):
        assert original["name"] == updated["name"]
        assert [v["value"] for v in original["choices"]] == [v["value"] for v in updated["choices"]]
    assert "not established identity" in context["box_correspondence_rule"]
    assert result["questions"][0]["choices"][0]["description"].startswith("Reference 1")
    original_images = [part for part in payload["input"][0]["content"] if part["type"] == "input_image"]
    new_images = [part for part in content if part["type"] == "input_image"]
    assert new_images[0] == original_images[0] and new_images[2] == original_images[2]
    assert len(new_images) == len(original_images) + 2
    for role in ("initial_reference_annotations", "current_annotations"):
        row = metadata(result, role)[0]
        assert row["frame_color"] == [235, 235, 235]
        assert all(item["display_label"].startswith(('Reference ', 'Candidate ')) for item in row["items"])
        assert row['annotation_layout'] == 'near_box' and row['derived_image'] is True
        assert all(item['image_label'][0] in {'R', 'C'} for item in row['items'])
    zooms = metadata(result, "same_capture_source_group_zoom")
    assert len(zooms) == 2
    for zoom in zooms:
        assert zoom["source_camera_id"] == "2"
        assert zoom["source_object_ids"] == list(context["fixed_entities"])
        left, top, right, bottom = zoom["crop_bounds_xyxy_px"]
        assert 0 <= left <= 40 and 114 <= right <= 160
        assert 0 <= top <= 50 and 70 <= bottom <= 120
        assert zoom["source_rgb_size_px"] == [160, 120]
        assert len(zoom["source_image_sha256"]) == 64
        assert len(zoom["items"]) == 3
        assert "not new evidence" in zoom["meaning"]
        for index, item in enumerate(zoom["items"]):
            recovered = (np.asarray(item["box"]) * [right-left, bottom-top, right-left, bottom-top]
                         + [left, top, left, top]) / [160, 120, 160, 120]
            np.testing.assert_allclose(recovered, [(40+30*index)/160, 50/120,
                                                  (54+30*index)/160, 70/120])
    reordered = deepcopy(payload)
    original_context = json.loads(reordered["input"][0]["content"][0]["text"])
    original_context["fixed_entities"] = dict(reversed(list(original_context["fixed_entities"].items())))
    reordered["input"][0]["content"][0]["text"] = json.dumps(original_context)
    reordered_context = json.loads(improve_identity_presentation(reordered)["input"][0]["content"][0]["text"])
    assert reordered_context["identity_display_legend"] == context["identity_display_legend"]


@pytest.mark.parametrize("case", ["no_verified_camera", "later_capture", "different_pixels", "missing_source_box"])
def test_zoom_requires_verified_same_capture_and_every_source_reference(case):
    from luna_mujoco.demo_identity_presentation import improve_identity_presentation
    payload, _, _, _ = source_payload()
    content = payload["input"][0]["content"]
    context = json.loads(content[0]["text"])
    if case == "no_verified_camera":
        context["same_capture_reference"]["verified_camera_ids"] = []
    elif case == "later_capture":
        context["observation_id"] = "new-capture"
    elif case == "different_pixels":
        from luna_mujoco.visual_inventory import image_part
        image_indices = [i for i, part in enumerate(content) if part["type"] == "input_image"]
        content[image_indices[2]] = image_part(np.ones((120, 160, 3), dtype=np.uint8))
    else:
        context["source_inventory"]["source_object_ids"].append("unseen")
    content[0]["text"] = json.dumps(context)
    result = improve_identity_presentation(payload)
    assert metadata(result, "same_capture_source_group_zoom") == []


def test_runtime_demo_uses_presentation_while_unknown_identity_remains_unknown():
    _, capture, reference, detections = source_payload()
    calls = []
    with pytest.raises(ValueError, match="visual_identity_unknown") as caught:
        inventory(calls, value="permanent-pink-1", confidence=.67).resolve(
            capture, detections, instruction="move three", reference=reference,
            entities=reference["entities"], purpose="initialize_inventory")
    assert len(calls) == 2
    assert metadata(calls[0], "same_capture_source_group_zoom")
    assert caught.value.evidence['sam3_score'] is None
    assert 'sam3_score_not_recorded' in caught.value.evidence['unknowns']
    assert caught.value.evidence['confidence'] == .67


@pytest.mark.parametrize('current_capture', [False, True])
def test_actual_archive_retains_both_zoom_images_and_original_request(tmp_path, current_capture):
    import hashlib

    from test_experiments import begin

    from luna_mujoco.demo_identity_presentation import improve_identity_presentation
    from luna_mujoco.experiments import ExperimentStore
    payload = current_payload() if current_capture else source_payload()[0]
    request = improve_identity_presentation(payload)
    store = ExperimentStore(tmp_path)
    session = begin(store)
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = session.visual_decision({"request": request, "response": {"answers": []},
        "request_hash": key, "observation_id": "same-capture", "http_status": 200})
    session.finish({"status": "needs_attention"})
    saved = json.loads(store.file(session.id, path).read_text(encoding="utf-8"))
    parts = saved["request"]["input"][0]["content"]
    assert len([part for part in parts if part["type"] == "input_image"]) == 6
    for index, part in enumerate(parts):
        if part["type"] == "input_text" and any(role in part["text"] for role in (
                "same_capture_source_group_zoom", "current_candidate_detail_zoom")):
            image = parts[index + 1]["image_url"]
            assert hashlib.sha256(store.file(session.id, image["path"]).read_bytes()).hexdigest() == image["sha256"]


@pytest.mark.parametrize("later_capture", [False, True])
def test_visible_inventory_question_gets_all_reference_labels_without_asserting_current_matches(later_capture):
    from luna_mujoco.demo_identity_presentation import (
        add_visible_inventory_guide,
        improve_identity_presentation,
    )
    from luna_mujoco.visual_inventory import choice

    payload, _, _, _ = source_payload()
    context = json.loads(payload["input"][0]["content"][0]["text"])
    # Count requests still need every frozen source reference, including unselected neighbors.
    context["fixed_entities"]["permanent-pink-3"]["visual_membership"] = "excluded"
    if later_capture:
        context["observation_id"] = "later-capture"
    payload["input"][0]["content"][0]["text"] = json.dumps(context)
    payload["questions"].append(choice("scene_inventory_closure", "Judge visible unmatched subjects.",
                                       ["none", "unlisted", "unknown"]))
    result = improve_identity_presentation(payload)
    question = result["questions"][-1]
    instructions, encoded = question["instructions"].split(" Visible-source reference guide: ", 1)
    guide = json.loads(encoded)
    assert guide["listed_reference_count"] == 3
    assert guide["scope"] == "initial_reference_inventory_not_a_current_count_or_matching_verdict"
    assert [item["entity_id"] for item in guide["references"]] == context["source_inventory"]["source_object_ids"]
    assert [item["reference_display_label"] for item in guide["references"]] == ["Reference 1", "Reference 2", "Reference 3"]
    assert all(item["reference_boxes"][0]["camera_id"] == "2" for item in guide["references"])
    assert all(len(item["reference_boxes"][0]["box"]) == 4 for item in guide["references"])
    assert "all three full CURRENT RGB images, including outside any zoom crop" in instructions
    assert "Count each physical object once across cameras" in instructions
    assert "possibly distinct visible task subject remains unresolved" in instructions
    assert "does not establish the current number" in instructions
    assert [item["value"] for item in question["choices"]] == ["none", "unlisted", "unknown"]
    assert bool(metadata(result, "same_capture_source_group_zoom")) is not later_capture
    existing_presentation = deepcopy(result)
    existing_presentation["questions"][-1]["instructions"] = payload["questions"][-1]["instructions"]
    assert add_visible_inventory_guide(existing_presentation) == result


def test_visible_inventory_guidance_does_not_promote_none_probability_or_confidence():
    import httpx

    from luna_mujoco.demo_perception import DemoVisualInventory

    _, capture, reference, detections = source_payload()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for question in payload["questions"]:
            name = question["name"]
            if question['type'] == 'predicate':
                answers.append({'name': name, 'type': 'predicate', 'probability': .99})
                continue
            value = (f"permanent-pink-{int(name.removeprefix('identity_')) + 1}"
                     if name.startswith("identity_") else "none" if name == "scene_inventory_closure" else "visible")
            answers.append({"name": name, "type": "choice", "choice": value,
                            "confidence": .70 if name == "scene_inventory_closure" else .99,
                            "probabilities": [{"value": value, "probability": .80}]})
        return httpx.Response(200, json={"answers": answers, "model": "gpt-6-luna"})

    result = DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond))).resolve(
        capture, detections, instruction="move all", reference=reference,
        entities=reference["entities"], purpose="initialize_inventory")
    assert len(calls) == 1
    assert "Visible-source reference guide" in calls[0]["questions"][-1]["instructions"]
    assert result["demo_inventory_check"]["status"] == "unknown"
    assert result["demo_inventory_check"]["confidence"] == .70


def current_payload():
    payload, _, _, _ = source_payload()
    context = json.loads(payload['input'][0]['content'][0]['text'])
    context.update(purpose='associate_inventory', observation_id='later-current')
    payload['input'][0]['content'][0]['text'] = json.dumps(context)
    return payload


def test_current_capture_candidate_zooms_preserve_full_images_questions_and_neighbors():
    from luna_mujoco.demo_identity_presentation import improve_identity_presentation

    payload = current_payload()
    old = deepcopy(payload)
    result = improve_identity_presentation(payload)
    zooms = metadata(result, 'current_candidate_detail_zoom')
    assert len(zooms) == 2
    assert payload == old
    old_images = [p for p in payload['input'][0]['content'] if p['type'] == 'input_image']
    new_images = [p for p in result['input'][0]['content'] if p['type'] == 'input_image']
    assert new_images[0] == old_images[0] and new_images[2] == old_images[2]
    assert [(q['name'], [item['value'] for item in q['choices']]) for q in result['questions']] == [
        (q['name'], [item['value'] for item in q['choices']]) for q in payload['questions']]
    for index, zoom in enumerate(zooms):
        assert zoom['candidate_labels'] == ['Candidate ' + str(index)]
        assert zoom['source_camera_id'] == '2' and zoom['observation_id'] == 'later-current'
        assert len(zoom['source_image_sha256']) == 64
        left, top, right, bottom = zoom['crop_bounds_xyxy_px']
        column = 40 + index * 30
        assert 0 <= left < column and column + 14 < right <= 160
        assert 0 <= top < 50 and 70 < bottom <= 120
        assert zoom['display_size_px'][0] > right - left
        assert zoom['resampling'].startswith('nearest')
        assert 'not a mask' in zoom['meaning'] and 'does not prove inventory completeness' in zoom['meaning']
    assert improve_identity_presentation(result) == result


def test_two_detail_slots_are_not_spent_on_large_container_candidates():
    from luna_mujoco.demo_identity_presentation import improve_identity_presentation
    payload = current_payload()
    for part in payload['input'][0]['content']:
        if part['type'] != 'input_text':
            continue
        try:
            value = json.loads(part['text'])
        except ValueError:
            continue
        if value.get('image_role') == 'current_annotations':
            value['items'].insert(0, {'id': 'candidate-99', 'box': [0., 0., .8, .8]})
            part['text'] = json.dumps(value)
    zooms = metadata(improve_identity_presentation(payload), 'current_candidate_detail_zoom')
    assert len(zooms) == 2
    assert [zoom['candidate_labels'] for zoom in zooms] == [['Candidate 0'], ['Candidate 1']]


def test_detail_budget_preserves_each_camera_regardless_of_iteration_order():
    from luna_mujoco.demo_identity_presentation import _current_candidate_zooms
    from luna_mujoco.visual_inventory import image_part
    raw = {('current', camera): image_part(np.zeros((120, 160, 3), np.uint8))
           for camera in ('3', '2', '1')}
    annotations = {key: {'items': [{'id': f'{key[1]}-{n}', 'box': [.1 + n*.2, .2, .18 + n*.2, .3]}
                                 for n in range(3)]} for key in raw}
    parts = _current_candidate_zooms({'purpose': 'associate_inventory', 'observation_id': 'current'},
                                    raw, annotations, {}, 2)
    cameras = [json.loads(part['text'])['source_camera_id'] for part in parts if part['type'] == 'input_text']
    assert cameras == ['3', '3', '2', '2', '1', '1']


@pytest.mark.parametrize('purpose', ['initialize_inventory', 'verify_transfer'])
def test_current_details_only_serve_inventory_association(purpose):
    from luna_mujoco.demo_identity_presentation import improve_identity_presentation

    payload = current_payload()
    context = json.loads(payload['input'][0]['content'][0]['text'])
    context['purpose'] = purpose
    payload['input'][0]['content'][0]['text'] = json.dumps(context)
    assert not metadata(improve_identity_presentation(payload), 'current_candidate_detail_zoom')


def test_same_camera_duplicate_boxes_share_display_only_and_keep_candidate_questions():
    from luna_mujoco.demo_identity_presentation import improve_identity_presentation

    payload = current_payload()
    for part in payload['input'][0]['content']:
        if part['type'] != 'input_text':
            continue
        try:
            value = json.loads(part['text'])
        except ValueError:
            continue
        if value.get('image_role') == 'current_annotations':
            value['items'].append({**value['items'][0], 'id': 'candidate-9'})
            part['text'] = json.dumps(value)
    question = deepcopy(payload['questions'][0])
    question.update(name='identity_9', instructions='Identify candidate-9 without merging it.')
    payload['questions'].append(question)
    result = improve_identity_presentation(payload)
    zooms = metadata(result, 'current_candidate_detail_zoom')
    assert len(zooms) == 2
    assert zooms[0]['candidate_labels'] == ['Candidate 0', 'Candidate 9']
    assert {'identity_0', 'identity_9'} <= {q['name'] for q in result['questions']}
    assert len(result['questions']) == len(payload['questions'])
