"""Bind a post-release RGB result to the executed transfer, without rebuilding geometry."""
import hashlib
import json
import math
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull

from .decision_conditions import condition, parse_answers
from .inventory_reference import current_inventory_entities
from .verification import gripper_open_confirmed
from .visual_api import VisualAPI, VisualInventoryError
from .visual_inventory import image_part, valid_assigned_slot

PLACEMENT_VERIFICATION_SCOPE = (
    ' This predicate checks only the current transfer placement relationship and the original '
    'language location/orientation constraints applicable to this transfer. It does not check '
    'CAD selection, grasp algorithms, earlier processing steps, later observation poses, final '
    'robot return or overall task completion; those have separate checks. Historical appearance '
    'descriptions may mention a former support or location, which is not required now. '
    'Use all three CURRENT RGB views together to locate the reference target and judge its '
    'support/containment and separation from the gripper. An occluded view does not negate a '
    'clear view; actual contradictory evidence must remain uncertain. Do not re-identify the '
    'whole inventory or require exact coordinates. The operator accepts approximate visual '
    'placement; configured regions have no visible marker and their planned XY is checked '
    'against execution feedback separately. Do not require a post-release depth measurement. '
    'An action receipt alone does not prove landing.')


def _require(condition, reason):
    if not condition:
        raise ValueError("release_verification_" + reason)


def _execution_key(evidence):
    return hashlib.sha256(json.dumps({key: value for key, value in evidence.items() if key != 'sha256'},
        sort_keys=True, allow_nan=False, default=lambda value: value.tolist()).encode()).hexdigest()


def freeze_release_execution(state, context, run_id):
    """Keep historical release proof without any executable trajectory."""
    plan, execution = context['place_plan'], context['demo_place_execution']
    evidence = {'owner_id': run_id, **{key: state[key] for key in (
        'target_id', 'destination_id', 'arm', 'generation')},
        'scene': {key: context['scene'][key] for key in (
            'scene_token', 'observation_id', 'generation', 'target_id')},
        'place': {key: deepcopy(plan[key]) for key in (
            'arm', 'target_id', 'destination_id', 'generation', 'scene_token', 'observation_id',
            'carrying_binding', 'placement_request')},
        'release_reference': deepcopy(plan['goal']['release_reference']),
        # Raw sensing/map snapshots remain in their original archives. Repeating
        # them here serializes all RGBD arrays on every integrity check. Keep
        # every held execution/geometry field and its existing source IDs.
        'held': {key: deepcopy(value) for key, value in context['holding'].items()
                 if key not in {'observation', 'scene', 'scene_observation'}},
        'execution': {key: deepcopy(execution[key]) for key in ('arm', 'released', 'open_confirmed')},
        **{key: deepcopy(context[key]) for key in (
            'pending_transfer', 'release_tcp', 'release_endpoint_evidence')},
        'completed_feedback': deepcopy(execution['completed_feedback'])}
    if 'placement_zone' in plan:
        evidence['place']['placement_zone'] = deepcopy(plan['placement_zone'])
    evidence['execution']['stages'] = {
        'release_collision_check': deepcopy(execution['stages']['release_collision_check'])}
    evidence['sha256'] = _execution_key(evidence)
    return evidence


def release_execution_context(state, context, run_id):
    evidence = context.get('release_execution_evidence')
    if evidence is None:
        return context
    _require(evidence.get('owner_id') == run_id
             and all(evidence.get(key) == state.get(key) for key in (
                 'target_id', 'destination_id', 'arm', 'generation'))
             and evidence.get('sha256') == _execution_key(evidence), 'execution_evidence_changed')
    return {**context, 'scene': evidence['scene'], 'place_plan': evidence['place'],
        'holding': evidence['held'], 'demo_place_execution': evidence['execution'],
        **{key: evidence[key] for key in ('pending_transfer', 'release_tcp', 'release_endpoint_evidence')}}


def released_target_map_is_current(context, target_id):
    """A visual verdict alone cannot certify a new map or a retained old layer."""
    evidence = context.get('release_execution_evidence', {})
    completed = evidence.get('completed_feedback', {})
    scene, measured = context['scene'], context.get('scene_observation', {})
    record = scene.get('inventory_map', {}).get('entities', {}).get(target_id, {})
    if (scene.get('coverage', {}).get('inventory_complete') is not True
            or context.get('scene_refresh_required')
            or record.get('status') not in {'observed', 'retained'}
            or any(scene.get(key) != measured.get(key) for key in ('observation_id', 'generation'))
            or scene.get('generation') != completed.get('generation')
            or record.get('last_generation') != completed.get('generation')
            or type(completed.get('time')) not in (int, float) or not math.isfinite(completed['time'])):
        return False
    for name in ('scene_observation', 'observation', 'inventory_observation', 'latest_capture',
                 'initial_capture', 'post_place_capture', 'fine_capture'):
        capture = context.get(name, {})
        if (isinstance(capture, dict) and capture.get('observation_id') == record.get('last_observation_id')
                and capture.get('generation') == completed['generation']
                and type(capture.get('time')) in (int, float) and math.isfinite(capture['time'])
                and capture['time'] >= completed['time']):
            return True
    return False


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
    context = release_execution_context(state, context, run_id)
    target, arm, generation = state["target_id"], state["arm"], state["generation"]
    scene, plan, holding = context['scene'], context['place_plan'], context['holding']
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
        if ledger.records[identifier].get('destination_id') != state['destination_id']:
            continue
        prior = progress[identifier]
        _require(valid_assigned_slot(prior) and prior.get("region_id") == current["region_id"],
                 "completed_slot_binding_invalid")
        if not placement.get('free_placement'):
            _require(prior["assigned_slot"]["layout_revision"] == current["assigned_slot"]["layout_revision"],
                     "completed_slot_binding_invalid")
            _require(prior["slot_id"] != current["slot_id"] and not _overlap(
                prior["assigned_slot"]["polygon_world_xy_m"], current["assigned_slot"]["polygon_world_xy_m"]),
                "slot_already_occupied")
        # A shared support polygon is not an occupied object footprint. Free
        # placement retains the exact executed goal and collision evidence above;
        # it must not reserve the entire table after the first delivery.
        completed.append({"entity_id": identifier, "region_id": prior["region_id"],
                          "slot_id": prior["slot_id"], "assigned_slot": deepcopy(prior["assigned_slot"])})
    binding = {**expected, "target_id": target, "destination_id": state["destination_id"],
        "region_id": current["region_id"], "slot_id": current["slot_id"],
        "layout_revision": placement["layout_revision"], "assigned_slot": deepcopy(current["assigned_slot"]),
        "transfer_index": ledger.transfer_index(target), "release_feedback_id": release["feedback_id"],
        "release_feedback_time": release_time, "held_feedback_id": held["feedback_id"],
        "release_tcp": deepcopy(context["release_tcp"]), "completed_slots": completed,
        "identity_source": "execution_chain", "source": "validated_held_attachment_place_release",
        "evidence": {"held": deepcopy(held), "attachment": deepcopy(attachment),
                     "release_endpoint": deepcopy(context["release_endpoint_evidence"]),
                     "release_collision_check": deepcopy(check), "physical_release": deepcopy(release)}}
    if "verified_perf_counter_ns" in release:
        binding["release_verified_perf_counter_ns"] = release["verified_perf_counter_ns"]
    if 'placement_zone' in plan:
        binding['placement_zone'] = deepcopy(plan['placement_zone'])
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


def release_result_question(has_reference_image=True, placement_intent=None):
    reference = 'REFERENCE crop' if has_reference_image else 'REFERENCE appearance description'
    if placement_intent is not None:
        relation = 'inside the identified destination container' if placement_intent['relation'] == 'inside' else 'resting on the identified destination support surface'
        return condition('place_result', f'The target object identified by the {reference} is '
            f'{relation} in the CURRENT views and visually satisfies the location/orientation clause '
            'for this transfer in the supplied placement_intent and original instruction. '
            'Use current_transfer to distinguish a final destination from a declared intermediate work area. '
            'An intermediate release must respect original handling/location constraints and its bound work area, '
            'but need not satisfy the later final destination or later processing effects; '
            'this does not establish final task completion. A different destination alone grants no such permission. '
            'Check visually distinguishable relative location and orientation against the original wording. '
            'Do not reinterpret a specified viewpoint using a wrist camera after it moved. '
            'An ambiguous target, destination or requested relationship stays uncertain. '
            'Prior candidate acceptance is not evidence that the released object actually landed there; '
            'release motion and an open gripper alone do not prove this relationship.'
            + PLACEMENT_VERIFICATION_SCOPE)
    return condition('place_result', f'The target object identified by the {reference} '
                     'is visible on the tabletop in the CURRENT views.')


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


def release_appearance_reference(reference, target_id):
    """Use a historical appearance crop, without carrying its old location forward."""
    target = current_inventory_entities(reference).get(target_id)
    if not target or set(reference.get('views', {})) != {'1', '2', '3'}:
        raise VisualInventoryError('release_initial_reference_required')
    appearance = {key: target[key] for key in ('kind', 'color') if key in target}
    if target.get('reference_camera_id') is None and target.get('bbox') is None:
        # A newly verified product need not exist in the initial RGB. Never
        # fabricate its crop from the retired input's frame or bounding box.
        if target.get('description'):
            appearance['description'] = target['description']
        return appearance, None
    camera, box = target.get('reference_camera_id'), np.asarray(target.get('bbox'), float)
    if (camera not in reference['views'] or box.shape != (4,) or not np.isfinite(box).all()
            or np.any(box < 0) or np.any(box > 1) or np.any(box[2:] <= box[:2])):
        raise VisualInventoryError('release_target_reference_invalid')
    rgb = reference['views'][camera]['rgb']
    height, width = rgb.shape[:2]
    x0, y0 = np.floor(box[:2] * [width, height]).astype(int)
    x1, y1 = np.ceil(box[2:] * [width, height]).astype(int)
    return appearance, image_part(rgb[y0:y1, x0:x1])


def release_result_payload(capture, binding, reference, instruction=""):
    validate_release_capture(capture, binding)
    release_position_check(binding)
    appearance, reference_image = release_appearance_reference(reference, binding['target_id'])
    context = {'observation_id': capture['observation_id'], 'binding_id': binding['binding_id'],
        'target_id': binding['target_id'], 'destination_id': binding['destination_id'],
        'appearance': appearance, 'destination_surface': 'tabletop',
        'scope': 'Judge this object in the CURRENT views only. REFERENCE is a historical appearance crop; '
            'it does not locate the object now. Use all CURRENT views together. Objects inside boxes '
            'do not count as placed on the tabletop. XY and orientation are checked separately; '
            'no visible placement marker is required.'}
    placement_intent = binding.get('placement_zone', {}).get('intent')
    from .release_region_reference import release_region_reference

    region_reference = release_region_reference(capture, binding)
    if region_reference is not None:
        context['target_region_reference'] = region_reference
    if placement_intent is not None:
        zone = binding['placement_zone']
        identities = [zone['support_entity_id'], *placement_intent.get('reference_ids', [])]
        context.update(placement_intent=deepcopy(placement_intent),
            destination_surface=placement_intent['relation'],
            destination_and_reference_appearances={identifier: {key: value for key, value in
                reference.get('entities', {}).get(identifier, {}).items()
                if key in {'kind', 'color', 'description', 'appearance_keywords'}} for identifier in identities},
            scope='Judge the target in the CURRENT view against the supplied placement relationship. '
                'REFERENCE is historical appearance only. On a support and inside a container are '
                'different relationships. Missing or occluded identity/relationship evidence stays unknown; '
                'a commanded pose, release action or open gripper does not establish the final location.')
        if 'original_requirement' in zone:
            context.update(original_placement_requirement=deepcopy(zone['original_requirement']),
                active_object_placement_requirement=deepcopy(zone.get('active_object_requirement')),
                current_transfer=deepcopy(zone.get('current_transfer')),
                placement_semantic_scope=zone.get('semantic_scope'),
                inventory_reference_descriptions={identifier: {key: value for key, value in entity.items()
                    if key in {'kind', 'color', 'description', 'appearance_keywords'}}
                    for identifier, entity in reference.get('entities', {}).items()})
    context['scope'] += PLACEMENT_VERIFICATION_SCOPE
    feedback = capture['arms'][binding['arm']]
    context['measured_gripper_state'] = {
        'source': 'actual_robot_joint_feedback_at_current_capture',
        'observation_id': capture['observation_id'], 'arm': binding['arm'],
        'position_rad': feedback['gripper_position_rad'],
        'open_limit_rad': feedback['gripper_limits_rad'][0],
        'closed_limit_rad': feedback['gripper_limits_rad'][1],
        'open_confirmed': True,
        'meaning': 'The fingers are physically open, not merely commanded open. '
            'This does not establish object location: verify the requested destination relation from CURRENT RGB. '
            'An object below the open gripper may appear between its fingers in a wrist-camera '
            'projection without being held.'}
    content = [{'type': 'input_text', 'text': json.dumps(context)},
               {'type': 'input_text', 'text': 'REFERENCE target appearance before transfer'}]
    if reference_image is not None:
        content.append(reference_image)
    else:
        context['scope'] = context['scope'].replace('a historical appearance crop',
                                                   'the appearance description of this registered identity')
        content[0]['text'] = json.dumps(context)
        content.append({'type': 'input_text', 'text': json.dumps(appearance)})
    questions = [release_result_question(reference_image is not None, placement_intent)]
    if region_reference is not None:
        questions[0] = condition('place_result',
            'The reference target is resting on the identified support surface or inside the identified '
            'container, as required by placement_intent.relation, in the CURRENT RGB. Judge only this '
            'support/containment relationship; proximity to the requested subregion and gripper retention '
            'are separate questions. Do not rejudge CAD, processing history or task completion.')
        questions.append(condition('placement_near_target',
            'The released reference target is approximately at or near the requested region in CURRENT RGB. '
            'Use target_region_reference and its projected support-plane outlines as spatial references, '
            'not as observed object boundaries. Respect original location/orientation constraints for '
            'current_transfer only; an intermediate work area need not be the final destination. '
            'Allow qualitative placement error, perspective and object height; no exact XY is required. '
            'If the needed location cannot be judged from any current view, remain uncertain. '
            'This question does not check CAD identity, whether still held, or the whole task.'))
    questions.append(condition('workpiece_still_held',
        'The CURRENT images show the active task object still physically retained against the gripper '
        'fingers, despite the measured open gripper and completed retreat. Positive evidence means '
        'attachment/contact supporting the workpiece; appearing between open fingers in the wrist '
        'camera, or overlapping the robot in the head camera, is not evidence of retention. '
        'A separate object resting below the open gripper at its destination is not held.'))
    # These two predicates are the actual visual completion criteria. Each has
    # its own unknown probability interval; an extra meta-predicate would judge
    # the same evidence again without adding a required observable property.
    for camera in ('1', '2', '3'):
        content.extend([{'type': 'input_text', 'text': 'CURRENT RGB camera ' + camera},
                        image_part(capture['views'][camera]['rgb'])])
        projected = [row for row in (region_reference or {}).get('projections', []) if row['camera_id'] == camera]
        if projected:
            from .observation_labels import raster_visual_annotations

            content.extend([{'type': 'input_text', 'text': json.dumps({
                'image_role': 'release_target_region_reference', 'camera_id': camera,
                'observation_id': capture['observation_id'], 'meaning': region_reference['meaning']})},
                image_part(raster_visual_annotations(capture['views'][camera]['rgb'], [], regions=projected))])
    return {"model": "gpt-6-luna", "input": [{"role": "user", "content": content}], "questions": questions}


def verify_release_result(inventory, capture, binding, reference, instruction=""):
    from .observation_requirements import assessment
    payload = release_result_payload(capture, binding, reference, instruction)
    if binding.get('placement_zone', {}).get('semantic_scope') == 'entire_region':
        from .release_decision_input import scoped_release_payload

        payload = scoped_release_payload(payload, binding['placement_zone']['support_entity_id'])
    questions = payload['questions']

    def parse(body, evidence):
        judgments, answers = parse_answers(body, questions, inventory.minimum_confidence)
        by_name = {answer["name"]: answer for answer in answers}
        confidence = min(by_name['place_result']['probability'],
                         1. - by_name['workpiece_still_held']['probability'])
        status = "at_destination" if judgments["place_result"] == "yes" else "unknown"
        if judgments['workpiece_still_held'] != 'no':
            status = 'unknown'
        required = {'place_result': 'yes', 'workpiece_still_held': 'no'}
        if 'placement_near_target' in judgments:
            required['placement_near_target'] = 'yes'
            confidence = min(confidence, by_name['placement_near_target']['probability'])
            if judgments['placement_near_target'] != 'yes':
                status = 'unknown'
        assessment_judgments, assessment_required = dict(judgments), dict(required)
        observation = assessment(assessment_judgments, assessment_required,
                                 capture['observation_id'], require_sufficiency=False)
        if observation and observation['status'] != 'confirmed':
            status = 'rejected' if observation['status'] == 'rejected' else 'unknown'
        visual = {"source": "openai_decisions", "identity_source": "execution_chain",
            "diagnostic_checks": deepcopy(evidence.get('diagnostic_checks', [])),
            "observation_id": capture["observation_id"], "status": status, "confidence": confidence,
            "confidence_rule": 'min_destination_relationship_and_not_held_probability',
            "view_scope": "fused_current_three_rgb",
            "camera_ids": list(evidence['camera_ids']),
            "primary_required_checks": list(required),
            "completion_semantics": 'release_feedback_checked_and_visual_destination_and_object_separate',
            "post_release_metric_pose_measured": False,
            'release_position_check': release_position_check(binding),
            "judgments": judgments, "answers": answers,
            "target_id": binding["target_id"], "destination_id": binding["destination_id"],
            "region_id": binding["region_id"], "slot_id": binding["slot_id"],
            "layout_revision": binding["layout_revision"], "binding_id": binding["binding_id"],
            "release_feedback_id": binding["release_feedback_id"], "request_hash": evidence["request_hash"],
            **({'observation_assessment': observation} if observation else {})}
        check = {"source": "openai_decisions", "observation_id": capture["observation_id"],
                 "request_hash": evidence["request_hash"], "status": "not_requested"}
        return {"visual_evidence": visual, "demo_inventory_check": check,
                "binding_id": binding["binding_id"], "observation_id": capture["observation_id"],
                "evidence": {**evidence, "answers": answers}, "geometry_updated": False,
                **({'observation_assessment': observation} if observation else {})}

    result = VisualAPI.request(inventory, payload, capture, "verify_release_result", parse)
    return result
