"""Post-place RGB checks bind to execution history, without SAM or fresh geometry."""
from copy import deepcopy

import httpx
import pytest

from luna_mujoco.demo_perception import DemoPerception, DemoVisualInventory
from luna_mujoco.demo_transfer_verification import release_result_payload, validate_release_capture
from luna_mujoco.visual_api import VisualInventoryError


def test_release_result_module_is_available():
    from luna_mujoco.demo_transfer_verification import build_release_binding, verify_release_result
    assert callable(build_release_binding) and callable(verify_release_result)


async def test_demo_post_place_does_not_observe_or_rebuild_geometry(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.demo_runner import DemoRunner

    class NoPostPlaceObservation(DemoVision):
        def observe(self, capture, **kwargs):
            assert kwargs.get("checkpoint") != "post_place", "Post-place must not run the sensing pipeline"
            return super().observe(capture, **kwargs)

    class RetainMeasuredScene(DemoRunner):
        async def execute(self, skill, context, arguments=None):
            before = {key: context.get(key) for key in ("observation", "inventory_observation", "scene_observation")}
            result = await super().execute(skill, context, arguments)
            if skill == "verify_place":
                assert all(context.get(key) is value for key, value in before.items())
                assert context["post_place_capture"]["observation_id"] != context["observation"]["observation_id"]
                assert context["post_place_result"]["geometry_updated"] is False
                assert self.state["facts"]["result_inspected"] is True
                assert self.state["facts"]["place_evidence_pending"] is False
            return result

    algorithms = NoPostPlaceObservation(1)
    runner = RetainMeasuredScene(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    assert algorithms.release_calls == 1
    assert len([entry for entry in algorithms.map_order if entry[0] == "map"]) == 1
    assert runner.task_progress.records["pink-1"]["destination_visual_evidence"]["identity_source"] == "execution_chain"


async def test_unknown_is_inspected_without_becoming_completed(tmp_path):
    from test_demo_runner import run_demo
    runner, _, _ = await run_demo(tmp_path, count=1, progress="unknown")
    assert runner.state["status"] == "needs_attention"
    assert runner.state["facts"]["result_inspected"] is True
    assert runner.state["facts"]["place_evidence_pending"] is False
    assert not runner.task_progress.completed_ids


async def test_confirmed_unpack_table_release_retains_private_capture_and_compact_binding(tmp_path):
    import json

    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.demo_lifecycle_perception import table_release_reference
    from luna_mujoco.demo_runner import DemoRunner

    class LocalObstacleVision(DemoVision):
        def publish_local_object(self, scene, capture, target, area, appearance):
            assert target == 'pink-1' and appearance == 'wrapped'
            return {'scene': scene, 'measurement': {'observation_id': capture['observation_id']}}

    class InspectUnpackRelease(DemoRunner):
        checked = False

        async def execute(self, skill, context, arguments=None):
            if skill != 'verify_place':
                return await super().execute(skill, context, arguments)
            original = deepcopy(self.demo_task)
            target = self.state['target_id']
            self.demo_task.update(workflow='unpack_transfer', item_lifecycle={target: {}})
            result = await super().execute(skill, context, arguments)
            assert result['success']
            before = context['post_place_capture']
            assert before is context['latest_capture']
            evidence = self.current_lifecycle()['table_visual_evidence']
            assert context['post_place_verification_input'] == table_release_reference(before, evidence, self.run_id)
            assert 'views' not in json.dumps(self.current_lifecycle())
            self.reset_transfer(context)
            assert context['post_place_capture'] is before
            assert context['post_place_verification_input']['request_hash'] == evidence['request_hash']
            self.checked = True
            self.demo_task = original
            return result

    runner = InspectUnpackRelease(DemoRobot(), LocalObstacleVision(1), None, tmp_path,
                                 planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.checked, runner.state.get('error')


@pytest.mark.parametrize("mutation", ["target", "arm", "generation", "scene", "slot", "layout",
                                      "held", "attachment", "collision", "collision_scene", "collision_owner",
                                      "release", "completed"])
async def test_invalid_release_chain_stops_before_visual_call(tmp_path, mutation):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.demo_runner import DemoRunner

    class InvalidChain(DemoRunner):
        async def execute(self, skill, context, arguments=None):
            if skill == "verify_place":
                context["place_plan"] = deepcopy(context["place_plan"])
                if mutation in {"target", "arm", "generation", "scene"}:
                    key = {"target": "target_id", "arm": "arm", "generation": "generation",
                           "scene": "scene_token"}[mutation]
                    context["place_plan"][key] = "changed"
                elif mutation == "slot":
                    context["place_plan"]["placement_request"]["slot_id"] = 2
                elif mutation == "layout":
                    context["place_plan"]["placement_request"]["layout_revision"] = "changed"
                elif mutation == "held":
                    context["holding"]["verification"]["status"] = "unknown"
                elif mutation == "attachment":
                    context["holding"]["attachment_evidence"]["attached"] = False
                elif mutation == "collision":
                    context["demo_place_execution"]["stages"]["release_collision_check"]["collision_free"] = False
                elif mutation in {"collision_scene", "collision_owner"}:
                    key = "scene_token" if mutation == "collision_scene" else "carrying_owner_id"
                    context["demo_place_execution"]["stages"]["release_collision_check"][key] = "other"
                elif mutation == "release":
                    context["demo_place_execution"]["open_confirmed"] = False
                elif mutation == "completed":
                    self.task_progress.records[self.state["target_id"]]["status"] = "completed"
            return await super().execute(skill, context, arguments)

    algorithms = DemoVision(1)
    runner = InvalidChain(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert runner.state["error"].startswith("release_verification_")
    assert algorithms.release_calls == 0


def release_inputs(final=False):
    from test_demo_runner import DemoRobot
    capture = DemoRobot().capture_rgbd()
    slot = {"slot_id": 1, "center_world_xy_m": [.4, 0.], "polygon_world_xy_m": [
        [.3, -.1], [.5, -.1], [.5, .1], [.3, .1]], "floor_z_m": .6, "layout_revision": "layout"}
    binding = {"arm": "A", "target_id": "pink-1", "generation": 1, "observation_id": "before",
        "binding_id": "proof", "release_feedback_id": "feedback-1-0.0", "release_feedback_time": 0.,
        "destination_id": "demo-destination", "region_id": "region:demo-destination", "slot_id": 1,
        "layout_revision": "layout", "assigned_slot": slot, "completed_slots": [],
        "identity_source": "execution_chain", "needs_visible_inventory_check": final}
    tcp = [[1., 0., 0., .4], [0., 1., 0., 0.], [0., 0., 1., .72], [0., 0., 0., 1.]]
    binding.update(release_tcp=tcp, evidence={'release_endpoint': {
        'source': 'measured_object_reference_attachment_gripper_and_actual_tcp',
        'at_release_goal': True, 'footprint_inside_region': True, 'actual_tcp_world': tcp,
        'tool_reference_point_m': [0., 0., 0.], 'actual_reference_world_m': [.4, 0., .72]}})
    reference = {"observation_id": "before", "views": deepcopy(capture["views"]),
        "entities": {"pink-1": {"id": "pink-1", "kind": "cylinder", "color": "pink",
                               "reference_camera_id": "2", "bbox": [.1, .1, .2, .2]}},
        "demo_task": {"source_object_ids": ["pink-1"], "source_bin_id": "bin"}}
    return capture, binding, reference


def decisions(answers, calls):
    def respond(request):
        import json
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": answers, "model": "gpt-6-luna"})
    return DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))


def answer(value="yes", confidence=.99, name="place_result"):
    return {"name": name, "type": "choice", "choice": value, "confidence": confidence}


def view_answers(value="yes", confidence=.99, cameras=("2", "3", "1")):
    return [answer(value, confidence, f"camera_{camera}_{question}") for camera in cameras
            for question in ("workpiece", "in_region", "separated", "on_table")]


@pytest.mark.parametrize("value,confidence,status", [
    ("yes", .8, "at_destination"), ("yes", .799, "unknown"),
    ("unknown", 1., "unknown"), ("no", .99, "unknown"),
])
def test_dedicated_decisions_has_no_segmentation_or_geometry(value, confidence, status):
    capture, binding, reference = release_inputs()
    calls = []
    perception = object.__new__(DemoPerception)
    perception.visual_inventory = decisions([answer(value, confidence)], calls)
    result = perception.verify_release(capture, binding, reference)
    assert result["visual_evidence"]["status"] == status
    assert result["visual_evidence"]["identity_source"] == "execution_chain"
    assert result["visual_evidence"]["release_feedback_id"] == binding["release_feedback_id"]
    assert result["geometry_updated"] is False
    assert not any(key in result for key in ("objects", "views", "table", "scene_token"))
    assert len(calls) == 1
    assert [q["name"] for q in calls[0]["questions"]] == ["place_result"]
    assert result["demo_inventory_check"]["status"] == "not_requested"


@pytest.mark.parametrize("bad", [None, {}, answer(confidence=True), answer(confidence=float("nan")),
                               answer(confidence=-.1), answer(confidence=1.1), answer("pink-1")])
def test_malformed_result_cannot_confirm_or_retry(bad):
    from luna_mujoco.demo_transfer_verification import verify_release_result
    capture, binding, reference = release_inputs()
    calls = []
    # JSON transport cannot represent NaN; use None as the invalid parsed value instead.
    if isinstance(bad, dict) and bad.get("confidence") != bad.get("confidence"):
        bad = {**bad, "confidence": None}
    inventory = decisions([bad], calls)
    for _ in range(2):
        with pytest.raises(VisualInventoryError, match="invalid_response"):
            verify_release_result(inventory, capture, binding, reference)
    assert len(calls) == 1


@pytest.mark.parametrize("confidence,status", [(.8, "none"), (.79, "unknown")])
def test_final_all_visible_inventory_is_in_same_request(confidence, status):
    from luna_mujoco.demo_transfer_verification import verify_release_result
    capture, binding, reference = release_inputs(final=True)
    calls = []
    result = verify_release_result(decisions([answer(), answer("none", confidence, "demo_unlisted_pink")], calls),
                                   capture, binding, reference)
    assert [q["name"] for q in calls[0]["questions"]] == [
        "place_result", "demo_unlisted_pink"]
    assert result["demo_inventory_check"]["status"] == status
    assert result["demo_inventory_check"]["request_hash"] == result["visual_evidence"]["request_hash"]
    assert result["demo_inventory_check"]["observation_id"] == capture["observation_id"]


def test_request_shows_only_current_slot_without_private_controller_context():
    import json

    from luna_mujoco.visual_inventory import image_part
    capture, binding, reference = release_inputs()
    previous = deepcopy(binding["assigned_slot"])
    previous.update(slot_id=2, center_world_xy_m=[.4, -.2])
    binding["completed_slots"] = [{"entity_id": "prior", "region_id": binding["region_id"],
                                   "slot_id": 2, "assigned_slot": previous}]
    payload = release_result_payload(capture, binding, reference, "move all")
    content = payload["input"][0]["content"]
    text = json.dumps(payload)
    assert all(value not in text for value in ("pink-1", "feedback-1", "region:demo-destination",
                                               "assigned_slot", "polygon_world", "prior"))
    assert "Only the workpiece in CURRENT counts" in content[1]["text"]
    assert "outside CURRENT do not count" in content[1]["text"]
    for camera in ("1", "2", "3"):
        assert image_part(capture["views"][camera]["rgb"]) in content
    assert len([item for item in content if item["type"] == "input_image"]) == 6
    assert [question["name"] for question in payload["questions"]] == ["place_result"]
    assert "all CURRENT views together" in payload["questions"][0]["instructions"]
    assert "One clear view is sufficient" in payload["questions"][0]["instructions"]
    # Completed slots must not change the CURRENT-only annotation or questions.
    binding["completed_slots"] = []
    assert release_result_payload(capture, binding, reference, "move all") == payload


def test_release_region_outline_has_no_center_pointer_or_object_marker():
    import numpy as np

    from luna_mujoco.demo_transfer_verification import release_region_annotation

    rgb = np.full((120, 160, 3), [60, 70, 80], dtype=np.uint8)
    region = {'center': [.5, .5], 'polygon': [[.2, .2], [.8, .2], [.8, .8], [.2, .8]],
              'region_id': 'CURRENT', 'slot_id': 1}
    before = rgb.copy()
    annotated = release_region_annotation(rgb, region)
    # Old label leader started at the center and crossed foreground objects.
    np.testing.assert_array_equal(annotated[55:66, 75:86], rgb[55:66, 75:86])
    np.testing.assert_array_equal(annotated[24, 32], [105, 226, 255])
    np.testing.assert_array_equal(rgb, before)
    region['center'] = [.3, .4]
    np.testing.assert_array_equal(release_region_annotation(rgb, region), annotated)


def test_release_prompt_distinguishes_plane_projection_and_visible_contact_from_robot_hardware():
    capture, binding, reference = release_inputs()
    reference['demo_task']['workflow'] = 'unpack_transfer'
    payload = release_result_payload(capture, binding, reference)
    context = payload['input'][0]['content'][1]['text']
    assert 'not the silhouette of the object' in context
    assert 'checked numerically from measured robot feedback' in context
    assert 'Source-box objects and robot parts do not count' in context
    assert 'One clear view is sufficient' in context
    question = payload['questions'][0]
    assert question['name'] == 'place_result' and question['type'] == 'predicate'
    assert 'visible on the main tabletop' in question['instructions']
    assert 'resting' not in question['instructions']
    assert payload['questions'][1]['name'] == 'workpiece_still_held'


def test_old_per_camera_result_cannot_be_reinterpreted_as_fused_evidence():
    from luna_mujoco.demo_transfer_verification import verify_release_result

    capture, binding, reference = release_inputs()
    rows = view_answers()
    for row in rows:
        if row['name'] == 'camera_2_in_region':
            row['confidence'] = .76
        elif row['name'] == 'camera_3_separated':
            row.update(choice='no', confidence=1.)
        elif row['name'].startswith('camera_1_'):
            row.update(choice='unknown', confidence=1.)
    calls = []
    inventory = decisions(rows, calls)
    for _ in range(2):
        with pytest.raises(VisualInventoryError, match='invalid_response'):
            verify_release_result(inventory, capture, binding, reference)
    assert len(calls) == 1


def test_out_of_view_region_does_not_remove_current_image_or_fused_question():
    capture, binding, reference = release_inputs()
    capture["views"]["1"]["intrinsics"]["cx"] = -100
    payload = release_result_payload(capture, binding, reference, "")
    assert [q["name"] for q in payload["questions"]] == ["place_result"]
    images = [item for item in payload["input"][0]["content"] if item["type"] == "input_image"]
    assert len(images) == 5  # All three originals, only two CURRENT annotations.


def test_unpack_table_release_questions_name_the_visible_wrapper_not_the_hidden_core():
    capture, binding, reference = release_inputs()
    legacy = release_result_payload(capture, binding, reference)
    reference['demo_task']['workflow'] = 'unpack_transfer'
    wrapped = release_result_payload(capture, binding, reference)
    context = wrapped['input'][0]['content'][1]['text']
    import json
    task_context = json.loads(wrapped['input'][0]['content'][0]['text'])['task_context']
    assert 'Pink foam packages a black workpiece' in task_context['interpretation']
    assert 'VLA unpacking follows successful table placement' in task_context['interpretation']
    assert 'even if the release action was recorded' in context
    for question in wrapped['questions']:
        assert 'separate pink workpiece' in question['instructions']
        assert 'foam' not in question['instructions'] and 'wrapped' not in question['instructions']
        assert 'material' not in question['instructions']
        assert 'elliptical-cylinder' not in question['instructions']
    assert 'foam material' not in wrapped['questions'][0]['instructions']
    assert [q['name'] for q in wrapped['questions']] == ['place_result', 'workpiece_still_held']
    assert [q['name'] for q in legacy['questions']] == ['place_result']
    assert wrapped['questions'][0]['type'] == 'predicate'
    assert legacy['questions'][0]['type'] == 'choice'
    current = [p for p in legacy['input'][0]['content'] if p['type'] == 'input_image']
    combined = [p for p in wrapped['input'][0]['content'] if p['type'] == 'input_image']
    assert all(image in combined for image in current)
    assert len(combined) == len(current) + 6  # Three INITIAL RGBs and their annotation copies.
    assert 'black core' not in legacy['input'][0]['content'][1]['text']


def test_boundary_only_or_invisible_region_has_no_result_question():
    from luna_mujoco.demo_transfer_verification import projection_intersects_image, verify_release_result
    assert not projection_intersects_image([[1., 0.], [2., 0.], [2., 1.], [1., 1.]])
    assert projection_intersects_image([[.9, -.1], [2., -.1], [2., 1.], [.9, 1.]])
    capture, binding, reference = release_inputs()
    for view in capture["views"].values():
        view["intrinsics"]["cx"] = -100
    calls = []
    with pytest.raises(VisualInventoryError, match="release_region_not_visible") as failure:
        verify_release_result(decisions([], calls), capture, binding, reference)
    assert failure.value.evidence["status"] == "unknown"
    assert not calls


def test_minimal_audit_markers_prevent_another_capture_or_binding_cache_hit():
    from luna_mujoco.demo_transfer_verification import verify_release_result
    capture, binding, reference = release_inputs()
    calls = []
    inventory = decisions([answer()], calls)
    first = verify_release_result(inventory, capture, binding, reference)
    repeated = verify_release_result(inventory, capture, binding, reference)
    assert first == repeated and len(calls) == 1
    capture["observation_id"] = "another-capture"
    second = verify_release_result(inventory, capture, binding, reference)
    binding["binding_id"] = "another-release"
    third = verify_release_result(inventory, capture, binding, reference)
    assert len(calls) == 3
    assert calls[0]["input"][0]["content"][1:] == calls[1]["input"][0]["content"][1:]
    assert second["visual_evidence"]["observation_id"] == "another-capture"
    assert third["visual_evidence"]["binding_id"] == "another-release"
    assert first["visual_evidence"]["request_hash"] != third["visual_evidence"]["request_hash"]


def test_fused_result_has_no_per_camera_component_gate():
    from luna_mujoco.demo_transfer_verification import verify_release_result
    capture, binding, reference = release_inputs()
    answers = [answer(confidence=.8)]
    result = verify_release_result(decisions(answers, []), capture, binding, reference)
    assert result["visual_evidence"]["status"] == "at_destination"
    assert result["visual_evidence"]["primary_required_checks"] == ["place_result"]
    assert result["visual_evidence"]["view_scope"] == "fused_current_three_rgb"
    assert result["visual_evidence"]["judgments"] == {"place_result": "yes"}
    assert result["visual_evidence"]["completion_semantics"] == "released_in_region_and_separate"
    assert result["evidence"]["answers"] == answers


@pytest.mark.parametrize("mutation", ["equal_time", "older_time", "generation", "capture_id", "missing_camera", "closed"])
def test_capture_must_follow_release_in_same_generation(mutation):
    capture, binding, _ = release_inputs()
    if mutation in {"equal_time", "older_time"}:
        capture["time"] = 0. if mutation == "equal_time" else -1.
    elif mutation == "generation":
        capture["generation"] = 2
    elif mutation == "capture_id":
        capture["observation_id"] = "before"
    elif mutation == "missing_camera":
        del capture["views"]["3"]
    else:
        capture["arms"]["A"]["gripper_closed_command"] = True
    with pytest.raises(ValueError, match="release_verification_"):
        validate_release_capture(capture, binding)


def test_equal_simulation_tick_requires_later_actual_acquisition():
    capture, binding, _ = release_inputs()
    binding["release_verified_perf_counter_ns"] = 123
    capture.update(time=0., captured_perf_counter_ns=124)
    validate_release_capture(capture, binding)
    for stamp in (None, 122, 123, float("nan")):
        capture["captured_perf_counter_ns"] = stamp
        with pytest.raises(ValueError, match="capture_not_after_release"):
            validate_release_capture(capture, binding)


@pytest.mark.parametrize("mutation", ["target_id", "release_feedback_id", "binding_id", "observation_id",
                                      "layout_revision", "identity_source", "confidence", "completed_slot"])
async def test_result_cannot_rebind_or_count_previous_object(tmp_path, mutation):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

    from luna_mujoco.demo_runner import DemoRunner

    class WrongResult(DemoVision):
        def verify_release(self, *args, **kwargs):
            result = super().verify_release(*args, **kwargs)
            key = "slot_id" if mutation == "completed_slot" else mutation
            result["visual_evidence"][key] = .79 if key == "confidence" else 2 if key == "slot_id" else "other"
            return result
    algorithms = WrongResult(1)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert not runner.task_progress.completed_ids
    assert algorithms.release_calls == 1
