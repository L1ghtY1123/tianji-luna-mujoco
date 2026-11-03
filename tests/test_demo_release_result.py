"""Post-place RGB checks bind to execution history, without SAM or fresh geometry."""
from copy import deepcopy

import httpx
import pytest

from luna_mujoco.demo_perception import DemoVisualInventory
from luna_mujoco.demo_transfer_verification import release_result_payload, validate_release_capture
from luna_mujoco.visual_api import VisualInventoryError


def test_release_result_module_is_available():
    from luna_mujoco.demo_transfer_verification import build_release_binding, verify_release_result
    assert callable(build_release_binding) and callable(verify_release_result)


async def test_demo_post_place_does_not_observe_or_rebuild_geometry(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController

    class NoPostPlaceObservation(DemoVision):
        def observe(self, capture, **kwargs):
            assert kwargs.get("checkpoint") != "post_place", "Post-place must not run the sensing pipeline"
            return super().observe(capture, **kwargs)

    class RetainMeasuredScene(TestPolicyController):
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
    assert len([entry for entry in algorithms.map_order if entry[0] == "map"]) == 2
    assert runner.task_progress.records["pink-1"]["destination_visual_evidence"]["identity_source"] == "execution_chain"


async def test_unknown_is_inspected_without_becoming_completed(tmp_path):
    from test_demo_runner import run_demo
    runner, _, _ = await run_demo(tmp_path, count=1, progress="unknown")
    assert runner.state["status"] == "needs_attention"
    assert runner.state["facts"]["result_inspected"] is True
    assert runner.state["facts"]["place_evidence_pending"] is False
    assert not runner.task_progress.completed_ids


async def test_place_unknown_does_not_move_and_explicit_recheck_keeps_original_release(tmp_path):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController

    class UncertainVision(DemoVision):
        def verify_release(self, *args, **kwargs):
            result = super().verify_release(*args, **kwargs)
            result['observation_assessment'] = {'needs_observation': True, 'status': 'unknown'}
            return result

    class ExplicitRecheck(TestPolicyController):
        checked = False

        async def supplement_result_observation(self, *args):
            raise AssertionError('verification must not move or automatically retry')

        async def execute(self, skill, context, arguments=None):
            if skill != 'verify_place':
                return await super().execute(skill, context, arguments)
            first = await super().execute(skill, context, arguments)
            assert first['status'] == 'unknown' and self.algorithms.release_calls == 1
            second = await super().execute(skill, context, arguments)
            assert second['status'] == 'unknown' and self.algorithms.release_calls == 2
            assert first['feedback_id'] == second['feedback_id']
            self.checked = True
            return second

    runner = ExplicitRecheck(DemoRobot(), UncertainVision(1, progress_status='unknown'), None, tmp_path,
                             planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.checked, runner.state.get('error')
    assert not runner.task_progress.completed_ids


async def test_confirmed_release_retains_private_capture_and_compact_binding(tmp_path):
    import json

    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController

    from luna_mujoco.robot_skills.evidence import table_release_reference

    class InspectRelease(TestPolicyController):
        checked = False

        async def execute(self, skill, context, arguments=None):
            target = self.state.get('target_id')
            result = await super().execute(skill, context, arguments)
            if skill != 'verify_place':
                return result
            assert result['success']
            before = context['post_place_capture']
            assert before is context['latest_capture']
            evidence = self.task_progress.records[target]['destination_visual_evidence']
            assert context['post_place_verification_input'] == table_release_reference(before, evidence, self.run_id)
            assert 'views' not in json.dumps(evidence)
            self.reset_transfer(context)
            assert context['post_place_capture'] is before
            assert context['post_place_verification_input']['request_hash'] == evidence['request_hash']
            self.checked = True
            return result

    runner = InspectRelease(DemoRobot(), DemoVision(1), None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('move all pink objects')
    await runner.worker
    assert runner.checked, runner.state.get('error')


@pytest.mark.parametrize("mutation", ["target", "arm", "generation", "scene", "slot", "layout",
                                      "held", "attachment", "collision", "collision_scene", "collision_owner",
                                      "release", "completed"])
async def test_invalid_release_chain_stops_before_visual_call(tmp_path, mutation):
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController

    class InvalidChain(TestPolicyController):
        async def execute_place(self, context):
            result = await super().execute_place(context)
            # Corrupt the execution proof before it is frozen; verification must
            # still reject missing held/attachment/collision evidence semantically.
            if result['released']:
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
                    result["stages"]["release_collision_check"]["collision_free"] = False
                elif mutation in {"collision_scene", "collision_owner"}:
                    key = "scene_token" if mutation == "collision_scene" else "carrying_owner_id"
                    result["stages"]["release_collision_check"][key] = "other"
                elif mutation == "release":
                    result["open_confirmed"] = False
                elif mutation == "completed":
                    self.task_progress.records[self.state["target_id"]]["status"] = "completed"
            return result

    algorithms = DemoVision(1)
    runner = InvalidChain(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all pink objects")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    reason = {
        'target': 'place_binding_changed', 'arm': 'place_binding_changed',
        'generation': 'place_binding_changed', 'scene': 'place_binding_changed',
        'slot': 'slot_changed', 'layout': 'slot_changed', 'held': 'held_evidence_missing',
        'attachment': 'attachment_evidence_missing', 'collision': 'place_execution_incomplete',
        'collision_scene': 'place_execution_incomplete', 'collision_owner': 'place_execution_incomplete',
        'release': 'place_execution_incomplete', 'completed': 'target_already_completed',
    }[mutation]
    assert runner.state['error_code'] == 'release_verification_' + reason
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
        "task_spec": {"target_refs": ["pink-1"], "source_ref": "bin"}}
    return capture, binding, reference


def decisions(answers, calls):
    def respond(request):
        import json
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"answers": answers, "model": "gpt-6-luna"})
    return DemoVisualInventory(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))


def answer(value="yes", confidence=.99, name="place_result"):
    return {"name": name, "type": "choice", "choice": value, "confidence": confidence}


def fused_answers(value='yes', confidence=.99):
    from decision_fixtures import decision_answer
    return [decision_answer({'name': name, 'type': 'predicate'}, result, score)
            for name, result, score in [('place_result', value, confidence),
                ('workpiece_still_held', 'no', .99)]]


def view_answers(value="yes", confidence=.99, cameras=("2", "3", "1")):
    return [answer(value, confidence, f"camera_{camera}_{question}") for camera in cameras
            for question in ("workpiece", "in_region", "separated", "on_table")]


@pytest.mark.parametrize("value,confidence,status", [
    ("yes", .8, "at_destination"), ("yes", .799, "unknown"),
    ("unknown", 1., "unknown"), ("no", .99, "rejected"),
])
def test_dedicated_decisions_has_no_segmentation_or_geometry(value, confidence, status):
    from luna_mujoco.demo_transfer_verification import verify_release_result
    capture, binding, reference = release_inputs()
    calls = []
    inventory = decisions(fused_answers(value, confidence), calls)
    result = verify_release_result(inventory, capture, binding, reference)
    assert result["visual_evidence"]["status"] == status
    assert result["visual_evidence"]["identity_source"] == "execution_chain"
    assert result["visual_evidence"]["release_feedback_id"] == binding["release_feedback_id"]
    assert result["geometry_updated"] is False
    assert not any(key in result for key in ("objects", "views", "table", "scene_token"))
    assert len(calls) == 1
    assert [q["name"] for q in calls[0]["questions"]] == ["place_result", "workpiece_still_held"]
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
    assert all(value not in text for value in ('feedback-1', 'assigned_slot', 'prior', 'completed_slots'))
    context = json.loads(content[0]['text'])
    assert context['target_id'] == 'pink-1'
    assert 'XY and orientation are checked separately' in context['scope']
    for camera in ('1', '2', '3'):
        assert image_part(capture['views'][camera]['rgb']) in content
    assert len([item for item in content if item['type'] == 'input_image']) == 4
    assert [question['name'] for question in payload['questions']] == [
        'place_result', 'workpiece_still_held']
    # Completed slots must not change the CURRENT-only annotation or questions.
    binding["completed_slots"] = []
    assert release_result_payload(capture, binding, reference, "move all") == payload


def test_release_crop_preserves_reference_pixels_without_region_or_location_labels():
    import numpy as np

    from luna_mujoco.demo_transfer_verification import release_appearance_reference
    from luna_mujoco.visual_inventory import image_part

    _, binding, reference = release_inputs()
    rgb = reference['views']['2']['rgb']
    before = rgb.copy()
    appearance, image = release_appearance_reference(reference, binding['target_id'])
    height, width = rgb.shape[:2]
    assert image == image_part(rgb[int(.1*height):int(np.ceil(.2*height)),
                                   int(.1*width):int(np.ceil(.2*width))])
    assert appearance == {'kind': 'cylinder', 'color': 'pink'}
    np.testing.assert_array_equal(rgb, before)


def test_release_prompt_scopes_vision_to_tabletop_without_invisible_xy_overlay():
    import json
    payload = release_result_payload(*release_inputs())
    context = json.loads(payload['input'][0]['content'][0]['text'])
    assert context['destination_surface'] == 'tabletop'
    assert 'XY and orientation are checked separately' in context['scope']
    assert 'release_position_check' not in context
    assert 'polygon' not in json.dumps(payload) and 'cyan' not in json.dumps(payload)
    assert 'tabletop in the CURRENT views' in payload['questions'][0]['instructions']
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
    assert [q["name"] for q in payload["questions"]] == ["place_result", "workpiece_still_held"]
    images = [item for item in payload["input"][0]["content"] if item["type"] == "input_image"]
    assert len(images) == 4  # One appearance reference and three current originals.


def test_release_questions_use_current_target_without_a_processing_stage():
    capture, binding, reference = release_inputs()
    reference['task_spec']['processing'] = ['vla_unpack']
    payload = release_result_payload(capture, binding, reference, 'unwrap then deliver')
    reference['task_spec']['processing'] = []
    assert release_result_payload(capture, binding, reference, 'ordinary transfer') == payload


def test_invisible_region_does_not_prevent_current_visual_judgment():
    capture, binding, reference = release_inputs()
    before = release_result_payload(capture, binding, reference)
    for view in capture['views'].values():
        view['intrinsics']['cx'] = -100
    assert release_result_payload(capture, binding, reference) == before


def test_minimal_audit_markers_prevent_another_capture_or_binding_cache_hit():
    from luna_mujoco.demo_transfer_verification import verify_release_result
    capture, binding, reference = release_inputs()
    calls = []
    inventory = decisions(fused_answers(), calls)
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


def test_result_keeps_original_multiview_model_answers():
    from luna_mujoco.demo_transfer_verification import verify_release_result
    capture, binding, reference = release_inputs()
    answers = fused_answers(confidence=.8)
    result = verify_release_result(decisions(answers, []), capture, binding, reference)
    assert result["visual_evidence"]["status"] == "at_destination"
    assert result["visual_evidence"]["primary_required_checks"] == ["place_result", "workpiece_still_held"]
    assert result["visual_evidence"]["view_scope"] == "fused_current_three_rgb"
    assert result["visual_evidence"]["judgments"] == {"place_result": "yes", "workpiece_still_held": "no"}
    assert result["visual_evidence"]["completion_semantics"] == "release_feedback_checked_and_visual_destination_and_object_separate"
    assert result["evidence"]["answers"] == answers


def test_release_unknown_report_preserves_each_decision_question_and_probability():
    from luna_mujoco.demo_transfer_verification import verify_release_result
    from luna_mujoco.runtime.diagnostics import failure_record
    from luna_mujoco.runtime.execution_state import PolicyStopped

    capture, binding, reference = release_inputs()
    answers = [{'name': name, 'type': 'predicate', 'probability': probability}
               for name, probability in [('place_result', .47), ('workpiece_still_held', 0.)]]
    result = verify_release_result(decisions(answers, []), capture, binding, reference)
    assert result['visual_evidence']['status'] == 'unknown'
    report = failure_record(PolicyStopped('graph stop', {
        'success': False, 'status': 'unknown', 'reason': 'visual_destination_unconfirmed',
        'visual_evidence': result['visual_evidence'],
        'observation_assessment': result['observation_assessment']}), {})['failure_report']
    checks = {row['name']: row for row in report['checks']}
    for name, probability in [('place_result', .47), ('workpiece_still_held', 0.)]:
        assert checks[name]['actual'] == probability
        assert checks[name]['question'] and checks[name]['threshold'] == .8
    assert checks['place_result']['status'] == 'unknown'


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
    from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision, TestPolicyController

    class WrongResult(DemoVision):
        def verify_release(self, *args, **kwargs):
            result = super().verify_release(*args, **kwargs)
            key = "slot_id" if mutation == "completed_slot" else mutation
            result["visual_evidence"][key] = .79 if key == "confidence" else 2 if key == "slot_id" else "other"
            return result
    algorithms = WrongResult(1)
    runner = TestPolicyController(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("move all")
    await runner.worker
    assert runner.state["status"] == "needs_attention"
    assert not runner.task_progress.completed_ids
    assert algorithms.release_calls == 1
