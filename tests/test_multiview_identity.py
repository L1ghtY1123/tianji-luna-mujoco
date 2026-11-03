"""Only mutually measured current surfaces may share one identity question."""

from copy import deepcopy

import numpy as np
import pytest

from luna_mujoco.multiview_identity import (
    expand_identity_answers,
    measured_view_groups,
    reduce_identity_questions,
)


def scene():
    h, w = 48, 64
    yy, xx = np.mgrid[:h, :w]
    depth = np.ones((h, w))
    mask = np.zeros((h, w), bool)
    mask[8:40, 8:40] = True
    points = np.stack(((xx[mask] - 32) / 50, (yy[mask] - 24) / 50, depth[mask]), axis=1)
    view = {
        "rgb": np.zeros((h, w, 3), np.uint8),
        "depth_m": depth,
        "valid_depth": np.ones((h, w), bool),
        "intrinsics": {"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0},
        "T_world_camera": np.eye(4),
        "registered_to": "rgb",
        "calibration_id": "test",
        "frame_id": "now",
        "simulation_generation": 2,
    }
    capture = {
        "observation_id": "now",
        "generation": 2,
        "views": {c: {**deepcopy(view), "camera_id": c} for c in ("1", "2")},
    }
    detections = [
        {"camera_id": c, "points": points.copy(), "silhouette_mask": mask.copy()} for c in ("1", "2")
    ]
    return capture, detections


def test_current_surface_can_group_without_cad_or_any_permanent_id():
    capture, detections = scene()
    proof = measured_view_groups(capture, detections)
    assert proof["groups"][0]["members"] == ["candidate-0", "candidate-1"]
    assert proof["groups"][0]["representative"] == "candidate-0"
    assert len(proof["binding_hash"]) == 64
    assert "instance_id" not in str(proof)


@pytest.mark.parametrize("changed", ["stale", "occluded", "disjoint", "few", "ambiguous"])
def test_unproven_or_ambiguous_correspondence_keeps_questions_independent(changed):
    capture, detections = scene()
    if changed == "stale":
        capture["views"]["2"]["frame_id"] = "old"
    elif changed == "occluded":
        capture["views"]["2"]["depth_m"][:] = 0.5
    elif changed == "disjoint":
        detections[1]["points"][:, 0] += 1.0
    elif changed == "few":
        for d in detections:
            d["points"] = d["points"][:20]
    else:
        detections.append(deepcopy(detections[1]))
    assert not measured_view_groups(capture, detections).get("groups")


def test_group_answers_keep_model_choice_unknown_and_origin_provenance():
    capture, detections = scene()
    proof = measured_view_groups(capture, detections)
    questions = [
        {
            "type": "choice",
            "name": "identity_" + str(i),
            "instructions": "Which ID?",
            "choices": [{"value": v} for v in ("left", "right", "reject", "unknown")],
        }
        for i in range(2)
    ]
    reduced, aliases = reduce_identity_questions(questions, proof)
    assert [q["name"] for q in reduced] == ["identity_0"]
    for value, confidence in [("right", 0.99), ("unknown", 0.5)]:
        body = {
            "answers": [{"type": "choice", "name": "identity_0", "choice": value, "confidence": confidence}]
        }
        expanded, derivations = expand_identity_answers(body, questions, aliases, proof)
        assert [a["choice"] for a in expanded["answers"]] == [value, value]
        assert body["answers"][0]["name"] == "identity_0" and len(body["answers"]) == 1
        assert derivations["identity_1"]["source_question"] == "identity_0"
        assert derivations["identity_1"]["binding_hash"] == proof["binding_hash"]


def test_different_allowed_ids_do_not_collapse_into_an_unchecked_alias():
    capture, detections = scene()
    proof = measured_view_groups(capture, detections)
    questions = [
        {"type": "choice", "name": "identity_0", "choices": [{"value": "left"}, {"value": "unknown"}]},
        {"type": "choice", "name": "identity_1", "choices": [{"value": "right"}, {"value": "unknown"}]},
    ]
    reduced, aliases = reduce_identity_questions(questions, proof)
    assert reduced == questions and aliases == {}


def test_reference_camera_question_can_represent_a_measured_multiview_group():
    capture, detections = scene()
    proof = measured_view_groups(capture, detections)
    questions = [{'type': 'choice', 'name': 'identity_' + str(i),
                  'choices': [{'value': 'target'}, {'value': 'unknown'}]} for i in range(2)]
    reduced, aliases = reduce_identity_questions(questions, proof, preferred_questions={'identity_1'})
    assert [q['name'] for q in reduced] == ['identity_1']
    assert aliases == {'identity_0': 'identity_1'}
    for choice in ('target', 'unknown'):
        result, derivations = expand_identity_answers({'answers': [
            {'type': 'choice', 'name': 'identity_1', 'choice': choice, 'confidence': .92}]},
            questions, aliases, proof)
        assert [a['choice'] for a in result['answers']] == [choice, choice]
        assert derivations['identity_0']['source_question'] == 'identity_1'


def test_group_prefers_larger_visible_subject_without_changing_alternatives_or_evidence():
    questions = [{'type': 'choice', 'name': 'identity_' + str(i),
                  'choices': [{'value': 'red'}, {'value': 'unknown'}]} for i in range(2)]
    proof = {'groups': [{'members': ['candidate-0', 'candidate-1']}]}
    reduced, aliases = reduce_identity_questions(questions, proof,
        visible_pixels={'identity_0': 300, 'identity_1': 4000})
    assert reduced == [questions[1]]
    assert aliases == {'identity_0': 'identity_1'}
    reduced, aliases = reduce_identity_questions(questions, proof,
        visible_pixels={'identity_0': 300, 'identity_1': 4000}, preferred_questions={'identity_0'})
    assert reduced == [questions[0]]  # exact same-capture reference remains strongest
    assert aliases == {'identity_1': 'identity_0'}


def test_production_retains_same_reference_camera_comparison_before_group_reduction():
    import json

    import httpx

    from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
    from luna_mujoco.visual_inventory import mask_box

    capture, detections = scene()
    for detection in detections:
        view = capture['views'][detection['camera_id']]
        mask = detection['silhouette_mask']
        detection.update(mask=mask, pixels=view['rgb'][mask], pixels_rc=np.column_stack(np.nonzero(mask)),
                         kind='cube', color='red', prompt='cube', score=.95, view=view)
    entity = {'id': 'red', 'kind': 'cube', 'color': 'red', 'visual_membership': 'required',
              'reference_camera_id': '2', 'bbox': mask_box(detections[1]['silhouette_mask'])}
    reference = {'observation_id': capture['observation_id'], 'entities': {'red': entity},
                 'views': {camera: {'rgb': view['rgb']} for camera, view in capture['views'].items()}}
    calls = []
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        identities = [q for q in payload['questions'] if q['name'].startswith('identity_')]
        assert len(identities) == 1
        assert identities[0]['name'] == 'identity_1' and identities[0]['type'] == 'predicate'
        answers = [{'type': q['type'], 'name': q['name'], **(
            {'probability': .97} if q['type'] == 'predicate' else
            {'choice': 'none' if q['name'] == 'scene_inventory_closure' else 'visible', 'confidence': .97})}
            for q in payload['questions']]
        return httpx.Response(200, json={'answers': answers})
    client = DemoVisualInventory(api_key='test', client=httpx.Client(transport=httpx.MockTransport(respond)))
    result = client.resolve(capture, detections, entities={'red': entity}, reference=reference,
                            instruction='move red', purpose='associate_inventory')
    assert len(calls) == 1 and len(result['groups'][0]) == 2


@pytest.mark.parametrize("clarify", [False, True])
@pytest.mark.parametrize("preconfirmed_member", [False, True])
def test_production_wire_asks_once_per_group_and_preserves_derived_view_provenance(clarify, preconfirmed_member):
    import json

    import httpx

    from legacy_identity import ArchivedTwoStageInventory as DemoVisualInventory
    from luna_mujoco.visual_inventory import VisualInventory

    capture, detections = scene()
    for d in detections:
        view = capture["views"][d["camera_id"]]
        mask = d["silhouette_mask"]
        d.update(
            mask=mask,
            pixels=view["rgb"][mask],
            pixels_rc=np.column_stack(np.nonzero(mask)),
            kind="cube",
            color="red",
            prompt="cube",
            score=0.95,
            view=view,
        )
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = []
        for q in payload["questions"]:
            if q["name"].startswith("match_"):
                answers.append({"type": "predicate", "name": q["name"], "probability": 0.99})
            else:
                value = (
                    ("unknown" if clarify and len(calls) == 1 else "red")
                    if q["name"].startswith("identity_")
                    else ("none" if q["name"] == "scene_inventory_closure" else "visible")
                )
                answers.append({"type": "choice", "name": q["name"], "choice": value, "confidence": 0.99})
        return httpx.Response(200, json={"answers": answers})

    client = DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    resolve = client.resolve
    extra = {}
    if preconfirmed_member:
        # Same boundary as completed/container scoping: full measured proof,
        # but the independently confirmed member is absent from this request.
        extra = {'identity_context': {'current_measured_view_groups': measured_view_groups(capture, detections),
                                      'source_inventory': {'source_object_ids': []}},
                 'candidate_indices': [0]}
        detections = detections[:1]
        def resolve(*args, **kwargs):
            return VisualInventory.resolve(client, *args, **kwargs)
    result = resolve(
        capture,
        detections,
        entities={"red": {"kind": "cube", "color": "red", "visual_membership": "required"}},
        instruction="move red",
        purpose="associate_inventory",
        **extra,
    )
    assert len(calls) == (2 if clarify else 1)
    for payload in calls:
        assert all(
            q["name"] != "identity_1" and not q["name"].startswith("match_1_") for q in payload["questions"]
        )
        context = json.loads(payload['input'][0]['content'][0]['text'])
        if not preconfirmed_member:
            assert context['candidate_display_legend']['Candidate 1'] == 'candidate-1'
        identity_questions = [q for q in payload['questions']
                              if q['name'] == 'identity_0' or q['name'].startswith('match_0_')]
        assert identity_questions
        assert all(('Candidate 1' in q['instructions']) is not preconfirmed_member for q in identity_questions)
        if clarify and payload is calls[-1]:
            subjects = [json.loads(part['text']) for part in payload['input'][0]['content']
                        if part['type'] == 'input_text' and 'current_candidate_sam_subject' in part['text']]
            assert {row['candidate'] for row in subjects} == (
                {'candidate-0'} if preconfirmed_member else {'candidate-0', 'candidate-1'})
    assert len(result["groups"]) == 1 and len(result["groups"][0]) == (1 if preconfirmed_member else 2)
    if preconfirmed_member:
        return
    second = next(row for row in result["groups"][0] if row["camera_id"] == "2")["identity_evidence"][0]
    assert second["view_correspondence"]["source_question"] == "identity_0"
    assert second["original_decisions_answers"]
    assert all(
        a["name"] == "identity_0" or a["name"].startswith("match_0_")
        for a in second["original_decisions_answers"]
    )


@pytest.mark.parametrize("changed", ["capture", "generation", "hash"])
def test_group_proof_rejects_stale_or_modified_binding(changed):
    from luna_mujoco.multiview_identity import validate_group_proof
    from luna_mujoco.visual_api import VisualInventoryError

    capture, detections = scene()
    proof = measured_view_groups(capture, detections)
    validate_group_proof(proof, capture)
    if changed == "capture":
        capture["observation_id"] = "later"
    elif changed == "generation":
        capture["generation"] += 1
    else:
        proof["groups"][0]["members"].append("candidate-9")
    with pytest.raises(VisualInventoryError, match="multiview_identity_proof_invalid"):
        validate_group_proof(proof, capture)
