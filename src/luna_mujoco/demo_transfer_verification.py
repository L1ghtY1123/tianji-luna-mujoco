"""Bind a post-release RGB result to the executed transfer, without rebuilding geometry."""
import hashlib
import json
import math
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

from .decision_conditions import condition, parse_answers
from .observation_labels import project_placement_regions
from .verification import gripper_open_confirmed
from .visual_api import VisualAPI, VisualInventoryError
from .visual_inventory import choice, image_part, reference_content, valid_assigned_slot


def _require(condition, reason):
    if not condition:
        raise ValueError("release_verification_" + reason)


def _overlap(first, second):
    """Positive-area overlap of convex slot polygons; shared boundaries are allowed."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            axis = np.array([-edge[1], edge[0]])
            a, b = first @ axis, second @ axis
            if a.max() <= b.min() + 1e-10 or b.max() <= a.min() + 1e-10:
                return False
    return True


def build_release_binding(state, context, ledger, run_id, release):
    """Check the actual held/attachment/place/release records before any visual request."""
    target, arm, generation = state["target_id"], state["arm"], state["generation"]
    scene, plan, holding = context["scene"], context["place_plan"], context["holding"]
    execution = context.get("demo_place_execution", {})
    pending = context.get("pending_transfer", {})
    record = ledger.records[target]
    _require(target == ledger.current_id and record.get("status") != "completed"
             and target not in ledger.completed_ids and not pending.get("verified"), "target_already_completed")
    expected = {"arm": arm, "instance_id": target, "generation": generation,
                "scene_token": scene["scene_token"], "observation_id": scene["observation_id"], "owner_id": run_id}
    _require(arm in {"A", "B"} and ledger.bound_arm() == arm, "arm_changed")
    _require(scene.get("generation") == generation and scene.get("target_id") == target, "scene_changed")
    for binding in (holding, plan.get("carrying_binding", {})):
        _require(all(binding.get(key) == value for key, value in expected.items()), "holding_binding_changed")
    _require(plan.get("arm") == arm and plan.get("target_id") == target
             and plan.get("destination_id") == state["destination_id"]
             and all(plan.get(key) == expected[key] for key in ("generation", "scene_token", "observation_id")),
             "place_binding_changed")
    held = holding.get("verification", {})
    attachment = holding.get("attachment_evidence", {})
    _require(held.get("success") is True and held.get("status") == "held"
             and held.get("feedback_id") == f"feedback-{generation}-{held.get('feedback_time')}"
             and state["facts"].get("grasp_completed"), "held_evidence_missing")
    _require(attachment.get("attached") is True and attachment.get("scene_token") == scene["scene_token"],
             "attachment_evidence_missing")
    check = execution.get("stages", {}).get("release_collision_check", {})
    _require(execution.get("arm") == arm and execution.get("released") is True
             and execution.get("open_confirmed") is True
             and check.get("collision_free") is True and check.get("attached") is True
             and check.get("scene_token") == scene["scene_token"]
             and check.get("carrying_owner_id") == run_id
             and context.get("release_endpoint_evidence", {}).get("at_release_goal") is True,
             "place_execution_incomplete")
    _require(pending.get("target_id") == target and pending.get("destination_id") == state["destination_id"]
             and pending.get("scene_token") == scene["scene_token"]
             and np.array_equal(pending.get("release_tcp"), context.get("release_tcp")),
             "pending_transfer_changed")
    release_time, held_time = release.get("feedback_time"), held.get("feedback_time")
    _require(type(release_time) in (int, float) and math.isfinite(release_time)
             and type(held_time) in (int, float) and math.isfinite(held_time) and held_time <= release_time
             and release.get("feedback_id") == f"feedback-{generation}-{release_time}"
             and release.get("success") is True and release.get("open_confirmed") is True
             and release.get("released") is True and release.get("execution_completed") is True
             and record.get("release_feedback_id") == release["feedback_id"], "release_feedback_invalid")
    placement = ledger.placement_request(state["destination_id"])
    shared = placement.get('layout', {}).get('method') == 'shared_work_area'
    _require(plan.get("placement_request") == placement and placement.get("object_id") == target
             and placement.get("slot_id") == (1 if shared else record.get('execution_rank',
                                                                         record.get("display_id"))), "slot_changed")
    progress = ledger.visual_progress()
    current = progress[target]
    _require(valid_assigned_slot(current) and current["assigned_slot"].get("layout_revision")
             == placement.get("layout_revision"), "slot_invalid")
    completed = []
    for identifier in ledger.completed_ids:
        if shared and ledger.records[identifier].get('destination_id') != state['destination_id']:
            continue
        prior = progress[identifier]
        _require(valid_assigned_slot(prior) and prior.get("region_id") == current["region_id"]
                 and prior["assigned_slot"]["layout_revision"] == current["assigned_slot"]["layout_revision"],
                 "completed_slot_binding_invalid")
        _require(prior["slot_id"] != current["slot_id"] and not _overlap(
            prior["assigned_slot"]["polygon_world_xy_m"], current["assigned_slot"]["polygon_world_xy_m"]),
            "slot_already_occupied")
        completed.append({"entity_id": identifier, "region_id": prior["region_id"],
                          "slot_id": prior["slot_id"], "assigned_slot": deepcopy(prior["assigned_slot"])})
    binding = {**expected, "target_id": target, "destination_id": state["destination_id"],
        "region_id": current["region_id"], "slot_id": current["slot_id"],
        "layout_revision": placement["layout_revision"], "assigned_slot": deepcopy(current["assigned_slot"]),
        "stage_index": ledger.stage_index(target), "release_feedback_id": release["feedback_id"],
        "release_feedback_time": release_time, "held_feedback_id": held["feedback_id"],
        "release_tcp": deepcopy(context["release_tcp"]), "completed_slots": completed,
        "identity_source": "execution_chain", "source": "validated_held_attachment_place_release",
        "evidence": {"held": deepcopy(held), "attachment": deepcopy(attachment),
                     "release_endpoint": deepcopy(context["release_endpoint_evidence"]),
                     "release_collision_check": deepcopy(check), "physical_release": deepcopy(release)},
        "needs_visible_inventory_check": state["demo_task"].get('workflow') != 'unpack_transfer'
            and state["demo_task"]["scope"] == "all"
            and set(ledger.completed_ids) | {target} == set(state["demo_task"]["transport_ids"])}
    if "verified_perf_counter_ns" in release:
        binding["release_verified_perf_counter_ns"] = release["verified_perf_counter_ns"]
    binding["binding_id"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return binding


def validate_release_capture(capture, binding):
    _require(capture.get("generation") == binding["generation"]
             and capture.get("observation_id") and capture["observation_id"] != binding["observation_id"],
             "capture_binding_invalid")
    timestamp = capture.get("time")
    _require(type(timestamp) in (int, float) and math.isfinite(timestamp)
             and timestamp >= binding["release_feedback_time"], "capture_not_after_release")
    verified_at = binding.get("release_verified_perf_counter_ns")
    if verified_at is not None:
        captured_at = capture.get("captured_perf_counter_ns")
        _require(type(verified_at) is int and verified_at >= 0
                 and type(captured_at) is int
                 and captured_at > verified_at, "capture_not_after_release")
    else:
        # Historical captures can prove ordering with strictly increasing
        # simulation time; equal ticks require the real acquisition timestamp.
        _require(timestamp > binding["release_feedback_time"], "capture_not_after_release")
    _require(set(capture.get("views", {})) == {"1", "2", "3"}, "three_views_required")
    feedback = capture.get("arms", {}).get(binding["arm"], {})
    _require(feedback.get("gripper_closed_command") is False and gripper_open_confirmed(feedback),
             "capture_gripper_not_open")
    _require(valid_assigned_slot(binding), "slot_invalid")


def projection_intersects_image(polygon):
    """Clip the normalized convex slot polygon to the image; boundary contact is not area."""
    clipped = [np.asarray(point, float) for point in polygon]
    for axis, boundary, sign in ((0, 0., 1), (0, 1., -1), (1, 0., 1), (1, 1., -1)):
        output = []
        for first, second in zip(clipped, clipped[1:] + clipped[:1], strict=True):
            a, b = sign * (first[axis] - boundary), sign * (second[axis] - boundary)
            if (a >= 0) != (b >= 0):
                output.append(first + a / (a - b) * (second - first))
            if b >= 0:
                output.append(second)
        clipped = output
    if len(clipped) < 3:
        return False
    points = np.asarray(clipped)
    area = abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                      - points[:, 1] * np.roll(points[:, 0], -1))) / 2.
    return bool(area > 1e-12)


PRIMARY_RELEASE_CHECKS = ("place_result",)


def release_result_question(numeric_position=False):
    if numeric_position:
        return condition('place_result', 'A separate pink workpiece is visible on the main tabletop '
            'in the CURRENT images. Pink robot parts and objects in boxes do not count.')
    return choice("place_result", "Using all CURRENT views together, has the separate pink workpiece "
        "been released onto the main tabletop in the designated cyan CURRENT area, rather than "
        "remaining held by the gripper? Judge coarse placement using the object's base/support on the table; "
        "its raised top and sides need not fit inside the projected 2D outline. One clear view is sufficient; "
        "other views may be occluded or show projection overlap. Robot parts and pink objects still in the "
        "source box do not count. Answer no for clear placement outside the area, retention by the gripper, "
        "or a missing tabletop workpiece; unknown if the combined views cannot establish the result. "
        "No precise contact, millimetre accuracy, fixed object orientation or full silhouette is required.",
        ["yes", "no", "unknown"])


def release_position_check(binding):
    """Check release XY using actual robot feedback, never a simulator object pose."""
    endpoint = binding.get('evidence', {}).get('release_endpoint', {})
    try:
        actual = np.asarray(binding['release_tcp'], float)
        reference = np.asarray(endpoint['tool_reference_point_m'], float)
        point = actual[:3, :3] @ reference + actual[:3, 3]
        polygon = np.asarray(binding['assigned_slot']['polygon_world_xy_m'], float)
        _require(actual.shape == (4, 4) and reference.shape == (3,)
                 and np.isfinite(actual).all() and np.isfinite(reference).all()
                 and np.isfinite(polygon).all(), 'numeric_position_invalid')
        hull = ConvexHull(polygon)
        front_target = endpoint.get('source') == 'calibrated_finger_front_attachment_gripper_and_actual_tcp'
        _require((front_target or endpoint['source'] == 'measured_object_reference_attachment_gripper_and_actual_tcp')
                 and endpoint['at_release_goal'] is True
                 and endpoint['front_inside_region' if front_target else 'footprint_inside_region'] is True
                 and np.allclose(endpoint['actual_tcp_world'], actual, atol=1e-9, rtol=0)
                 and np.allclose(endpoint['actual_reference_world_m'], point, atol=1e-9, rtol=0)
                 and np.all(hull.equations[:, :2] @ point[:2] + hull.equations[:, 2] <= 1e-9),
                 'numeric_position_invalid')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('release_verification_numeric_position_missing') from error
    return {'source': endpoint['source'], 'time_scope': 'at_gripper_opening_before_current_images',
        'reference_xy_m': point[:2].tolist(), 'target_region_polygon_xy_m': polygon.tolist(),
        'reference_kind': 'finger_front' if front_target else 'legacy_object_reference',
        'reference_inside_region': True,
        'meaning': 'Numerical release-location check from measured robot feedback and the declared tool reference, '
                   'not a post-release observation. CURRENT images must still show the released workpiece.'}


def release_region_annotation(rgb, projection):
    """Outline the table plane without pointing to foreground object pixels."""
    height, width = rgb.shape[:2]
    image = Image.new('RGB', (width + 220, height), '#10212d')
    image.paste(Image.fromarray(np.asarray(rgb, dtype=np.uint8)), (0, 0))
    drawing = ImageDraw.Draw(image)
    polygon = [(x * width, y * height) for x, y in projection['polygon']]
    drawing.line(polygon + polygon[:1], fill='#69e2ff', width=2)
    drawing.text((width + 10, 8), 'CURRENT table area', fill='#69e2ff', font=ImageFont.load_default(size=14))
    return np.asarray(image)


def release_task_context(reference, binding, instruction):
    """Send recorded identities and stages, never simulator truth or commanded poses."""
    task = reference['demo_task']
    entities = reference.get('entities', {})
    if binding['target_id'] not in entities or set(reference.get('views', {})) != {'1', '2', '3'}:
        raise VisualInventoryError('release_initial_reference_required')
    fields = ('id', 'display_id', 'kind', 'color', 'description', 'reference_camera_id', 'bbox', 'initial_parent_id')
    initial = {identifier: {key: deepcopy(entity[key]) for key in fields if key in entity}
               for identifier, entity in entities.items()}
    products = {identifier: {key: deepcopy(entity[key]) for key in fields if key in entity}
                for identifier, entity in task.get('product_entities', {}).items()}
    states = {identifier: {key: record[key] for key in
                          ('phase', 'appearance', 'appearance_revision', 'to_table_arm', 'to_right_bin_arm')
                          if key in record}
              for identifier, record in task.get('item_lifecycle', {}).items()}
    processes = {identifier: {key: record[key] for key in ('input_entity_id', 'output_entity_id', 'status')
                             if key in record}
                 for identifier, record in task.get('process_records', {}).items()}
    return {'instruction': instruction, 'workflow': 'unpack_transfer',
        'active_target_id': binding['target_id'], 'initial_inventory': initial,
        'product_inventory': products, 'last_confirmed_lifecycle': states, 'process_records': processes,
        'source_object_ids': task.get('source_object_ids', []),
        'selected_input_ids': task.get('transport_ids', []),
        'completed_process_ids': task.get('completed_process_ids', []),
        'initial_observation_id': reference['observation_id'],
        'current_stage': 'awaiting_table_release_verification_before_unpack',
        'interpretation': 'INITIAL images, boxes and descriptions locate original entities; they are historical, '
            'not claims about their CURRENT positions. The active target was the subject of the recorded '
            'grasp and table-release action; its last confirmed lifecycle may still say wrapped_source '
            'because table release is awaiting this decision. The action does not prove visual success. '
            'Compare its INITIAL appearance with all CURRENT views. Other pink source items may remain '
            'in the box. Evaluate only this active transfer, not whether all inventory items are on the table. '
            'Pink foam packages a black workpiece; VLA unpacking follows successful table placement. '
            'Black outputs use independent IDs linked by the process record. No current pose is supplied.'}


def release_result_payload(capture, binding, reference, instruction=""):
    validate_release_capture(capture, binding)
    wrapped = reference.get('demo_task', {}).get('workflow') == 'unpack_transfer'
    numeric_position = release_position_check(binding) if wrapped else None
    stage_context = (
        "This stage checks the visible separate pink object before unpacking. "
        "Pink foam covers a black core; VLA removes this foam in the later unpacking stage. "
        "According to the task prior, the black core is still covered at this stage; it is not the subject "
        "of these questions. Do not require black pixels or a complete side profile to identify the pink object. "
        "Do not require visual proof of foam material or wrapping structure; these are task priors only. "
        "The prior is not proof that an object is visible; judge the CURRENT images. "
        if wrapped else "")
    slot = binding["assigned_slot"]
    regions = [{"region_id": "CURRENT", "layout": {"cells": [slot], "floor_z_m": slot["floor_z_m"]}}]
    projections = {row["camera_id"]: row for row in project_placement_regions(
        capture, regions, require_visible_support=False) if projection_intersects_image(row["polygon"])}
    if not projections:
        raise VisualInventoryError("release_region_not_visible", evidence={"status": "unknown"})
    content = [{"type": "input_text", "text": json.dumps({
        "observation_id": capture["observation_id"], "binding_id": binding["binding_id"],
        **({'task_context': release_task_context(reference, binding, instruction)} if wrapped else {})})},
        {"type": "input_text", "text":
        "The two identifiers above are audit markers, not visual evidence. Combine all three CURRENT views "
        "of the same scene into one result. Repeated full images and annotated copies are not extra objects. "
        + stage_context +
        "Each camera has its original RGB and, when visible, "
        "a copy marking the CURRENT tabletop region in cyan. Only the workpiece in CURRENT counts; "
        "other workpieces outside CURRENT do not count. The outline marks the projected tabletop plane, "
        "not an object identity pointer or evidence that an object is there. A raised top or side need not fit inside the 2D outline; "
        "judge coarse placement on that table area, not exact footprint geometry. "
        "Robot hardware, including pink rings or endcaps attached to the robot, never counts as the separate object, "
        "even when it overlaps this outline in the image. The tabletop may be occluded by foreground robot parts. "
        "A view that cannot see the workpiece does not veto another clear view. Answer unknown only when "
        "the combined evidence is insufficient or conflicting; do not infer contact merely from projection overlap."}]
    if wrapped:
        context = json.loads(content[0]['text'])
        context['release_position_check'] = numeric_position
        feedback = capture['arms'][binding['arm']]
        context['measured_gripper_state'] = {
            'source': 'actual_robot_joint_feedback_at_current_capture',
            'observation_id': capture['observation_id'], 'arm': binding['arm'],
            'position_rad': feedback['gripper_position_rad'],
            'open_limit_rad': feedback['gripper_limits_rad'][0],
            'closed_limit_rad': feedback['gripper_limits_rad'][1],
            'open_confirmed': True,
            'meaning': 'The fingers are physically open, not merely commanded open. '
                'This does not establish object location: verify the tabletop object from the CURRENT RGB. '
                'An object below the open gripper may appear between its fingers in a wrist-camera '
                'projection without being held.'}
        content[0]['text'] = json.dumps(context)
        content[1]['text'] = (
            'Check the result of placing ONE pink workpiece from the left box onto the main tabletop. '
            'INITIAL is historical; CURRENT shows the scene after opening the gripper and retreating. '
            'Use all three CURRENT cameras together. One clear view is sufficient; other views may be occluded. '
            'The cyan outline is the configured table region, not the silhouette of the object. '
            'Release XY was checked numerically from measured robot feedback and held-object geometry. '
            'Your task is to check the CURRENT visible outcome, not to measure contact or fit raised object pixels '
            'inside the projected outline. Source-box objects and robot parts do not count. '
            'A clear object on another surface or still held is a failure, even if the release action was recorded.')
    questions = [release_result_question(wrapped)]
    if wrapped:
        questions.append(condition('workpiece_still_held',
            'The CURRENT images show the separate pink workpiece still physically retained against the gripper '
            'fingers, despite the measured open gripper and completed retreat. Positive evidence means '
            'attachment/contact supporting the workpiece; appearing between open fingers in the wrist '
            'camera, or overlapping the robot in the head camera, is not evidence of retention. '
            'A separate object resting below the open gripper on the tabletop is not held.'))
    cameras = ("2", "3", "1")
    if wrapped:
        content.extend(reference_content(reference))
        # Keep both wrists and finish with the head's overall table context.
        cameras = ("1", "2", "3")
    for camera in cameras:
        rgb = capture["views"][camera]["rgb"]
        content.extend([{"type": "input_text", "text": "CURRENT camera " + camera + " — original RGB"}, image_part(rgb)])
        if camera not in projections:
            continue
        # CURRENT marks only the configured area. Initial identity annotations
        # are supplied separately; the full execution binding stays in the archive.
        image = release_region_annotation(rgb, projections[camera])
        content.extend([{"type": "input_text", "text": "CURRENT camera " + camera + " — cyan region"}, image_part(image)])
    if binding["needs_visible_inventory_check"]:
        source_ids = reference["demo_task"]["source_object_ids"]
        display_entities = {f"S{index + 1}": {
            key: value for key, value in reference["entities"].get(identifier, {}).items()
            if key in {"bbox", "reference_camera_id"}} for index, identifier in enumerate(source_ids)}
        content.extend(reference_content({"views": reference["views"], "entities": display_entities}))
        questions.append(choice("demo_unlisted_pink",
            "Across all CURRENT images, is any visible pink workpiece additional to those in INITIAL references? "
            "Count each physical workpiece once across cameras. Moved listed pieces are not extra; robot parts "
            "do not count. none covers visible subjects only, not hidden contents. If uncertain choose unknown.",
            ["none", "extra", "unknown"]))
    return {"model": "gpt-6-luna", "input": [{"role": "user", "content": content}], "questions": questions}


def verify_release_result(inventory, capture, binding, reference, instruction=""):
    payload = release_result_payload(capture, binding, reference, instruction)

    def parse(body, evidence):
        judgments, answers = parse_answers(body, payload['questions'], inventory.minimum_confidence)
        by_name = {answer["name"]: answer for answer in answers}
        predicate = payload['questions'][0]['type'] == 'predicate'
        confidence = by_name["place_result"]['probability' if predicate else 'confidence']
        status = "at_destination" if judgments["place_result"] == "yes" else "unknown"
        if predicate:
            confidence = min(confidence, 1. - by_name['workpiece_still_held']['probability'])
            if judgments['workpiece_still_held'] != 'no':
                status = 'unknown'
        visual = {"source": "openai_decisions", "identity_source": "execution_chain",
            "observation_id": capture["observation_id"], "status": status, "confidence": confidence,
            "confidence_rule": 'min_table_presence_and_not_held_probability' if predicate else 'place_result_answer_confidence',
            "view_scope": "fused_current_three_rgb",
            "primary_required_checks": ['place_result', 'workpiece_still_held'] if predicate else list(PRIMARY_RELEASE_CHECKS),
            "completion_semantics": 'release_xy_verified_and_tabletop_object_separate' if predicate else 'released_in_region_and_separate',
            **({'release_position_check': release_position_check(binding)} if predicate else {}),
            "judgments": judgments, "answers": answers,
            "target_id": binding["target_id"], "destination_id": binding["destination_id"],
            "region_id": binding["region_id"], "slot_id": binding["slot_id"],
            "layout_revision": binding["layout_revision"], "binding_id": binding["binding_id"],
            "release_feedback_id": binding["release_feedback_id"], "request_hash": evidence["request_hash"]}
        check = {"source": "openai_decisions", "observation_id": capture["observation_id"],
                 "request_hash": evidence["request_hash"], "status": "not_requested"}
        if binding["needs_visible_inventory_check"]:
            check.update(status=judgments["demo_unlisted_pink"],
                         confidence=by_name["demo_unlisted_pink"]["confidence"])
        return {"visual_evidence": visual, "demo_inventory_check": check,
                "binding_id": binding["binding_id"], "observation_id": capture["observation_id"],
                "evidence": {**evidence, "answers": answers}, "geometry_updated": False}

    return VisualAPI.request(inventory, payload, capture, "verify_release_result", parse)
