"""Bind a post-release RGB result to the executed transfer, without rebuilding geometry."""
import hashlib
import json
import math
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull

from .core.placement import ReleaseLedgerSnapshot
from .decision_conditions import condition, parse_answers
from .visual_api import VisualAPI, VisualDecisionError, merge_decision_evidence
from .visual_identity import choice, valid_placement_region


def _require(condition, reason):
    if not condition:
        raise ValueError("release_verification_" + reason)


def _execution_key(evidence):
    return hashlib.sha256(json.dumps({key: value for key, value in evidence.items() if key != 'sha256'},
        sort_keys=True, allow_nan=False, default=lambda value: value.tolist()).encode()).hexdigest()


def valid_container_entry(evidence, target_id, destination_id, feedback_id):
    return bool(evidence.get('source') == 'visual_container_entry'
        and evidence.get('status') == 'inside'
        and evidence.get('placement_outcome') in {'direct_success', 'inferred_success'}
        and evidence.get('rule') == 'decisions_placement_outcome_and_bound_release'
        and evidence.get('target_id') == target_id and evidence.get('destination_id') == destination_id
        and feedback_id and evidence.get('release_feedback_id') == feedback_id
        and evidence.get('receipt_id') and evidence.get('binding_id') and evidence.get('observation_id')
        and evidence.get('visual_evidence', {}).get('source') == 'openai_decisions'
        and evidence.get('visual_evidence', {}).get('status') == 'at_destination'
        and evidence.get('visual_evidence', {}).get('request_hash')
        and evidence.get('post_release_metric_pose_measured') is False)


def container_entry_receipt(binding, release_reference, capture, visual):
    """Bind a positive visual containment verdict to the actual release."""
    zone = binding.get('placement_zone', {})
    intent = zone.get('intent', {})
    _require(intent.get('release_mode') == 'container_release' and intent.get('relation') == 'inside'
             and zone.get('semantic_scope') == 'entire_region', 'container_entry_rule_not_applicable')
    _require(release_reference.get('release_mode') == 'container_release'
             and release_reference.get('scene_token') == binding['scene_token']
             and release_reference.get('rim_provenance', {}).get('entity_id') == binding['destination_id'],
             'container_entry_geometry_binding_changed')
    validate_release_capture(capture, binding)
    _require(visual.get('source') == 'openai_decisions' and visual.get('status') == 'at_destination'
        and visual.get('binding_id') == binding['binding_id']
        and visual.get('observation_id') == capture['observation_id']
        and visual.get('target_id') == binding['target_id']
        and visual.get('destination_id') == binding['destination_id']
        and visual.get('request_hash')
        and visual.get('placement_outcome') in {'direct_success', 'inferred_success'},
        'visual_container_entry_unconfirmed')
    position = release_position_check(binding)
    receipt = {key: deepcopy(binding[key]) for key in (
        'target_id', 'destination_id', 'arm', 'generation', 'region_id', 'placement_region',
        'layout_revision', 'binding_id', 'release_feedback_id', 'release_feedback_time')}
    receipt.update(source='visual_container_entry', status='inside',
        placement_outcome=visual['placement_outcome'],
        rule='decisions_placement_outcome_and_bound_release', observation_id=capture['observation_id'],
        pre_release_measurement_observation_id=binding['observation_id'],
        identity_source='execution_chain', post_release_metric_pose_measured=False,
        visual_evidence=deepcopy(visual),
        release_position_check={key: value for key, value in position.items() if key != 'meaning'},
        container_geometry=deepcopy(release_reference['rim_provenance']),
        release_endpoint=deepcopy(binding['evidence']['release_endpoint']))
    receipt['receipt_id'] = _execution_key({key: value for key, value in receipt.items() if key != 'observation_id'})
    return receipt


def freeze_release_execution(state, context, run_id, *, scene):
    """Keep historical release proof without any executable trajectory."""
    plan, execution = context['place_plan'], context['demo_place_execution']
    evidence = {'owner_id': run_id, **{key: state[key] for key in (
        'target_id', 'destination_id', 'arm', 'generation')},
        'destination_ref': context.get('destination', {}).get('reference_id', state['destination_id']),
        'scene': {key: deepcopy(scene[key]) for key in (
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
        'execution': {key: deepcopy(execution[key]) for key in ('arm', 'released', 'release_commanded')},
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
    return {**context, 'release_scene': deepcopy(evidence['scene']), 'place_plan': evidence['place'],
        'holding': evidence['held'], 'demo_place_execution': evidence['execution'],
        **{key: evidence[key] for key in ('pending_transfer', 'release_tcp', 'release_endpoint_evidence')}}


def released_target_map_is_current(context, target_id, *, collision):
    """A visual verdict alone cannot certify a new map or a retained old layer."""
    evidence = context.get('release_execution_evidence', {})
    completed = evidence.get('completed_feedback', {})
    scene, measured = collision.scene, collision.observation
    record = scene.get('object_layers', {}).get('entities', {}).get(target_id, {})
    if (scene.get('coverage', {}).get('objects_complete') is not True
            or collision.refresh_required
            or record.get('status') not in {'observed', 'retained'}
            or any(scene.get(key) != measured.get(key) for key in ('observation_id', 'generation'))
            or scene.get('generation') != completed.get('generation')
            or record.get('last_generation') != completed.get('generation')
            or type(completed.get('time')) not in (int, float) or not math.isfinite(completed['time'])):
        return False
    for name in ('scene_observation', 'observation', 'objects_observation', 'latest_capture',
                 'initial_capture', 'post_place_capture', 'fine_capture'):
        capture = measured if name == 'scene_observation' else context.get(name, {})
        if (isinstance(capture, dict) and capture.get('observation_id') == record.get('last_observation_id')
                and capture.get('generation') == completed['generation']
                and type(capture.get('time')) in (int, float) and math.isfinite(capture['time'])
                and capture['time'] >= completed['time']):
            return True
    return False



def build_release_binding(state, context, ledger: ReleaseLedgerSnapshot, run_id, release, *, scene):
    """Check the actual held/attachment/place/release records before any visual request."""
    context = release_execution_context(state, context, run_id)
    target, arm, generation = state["target_id"], state["arm"], state["generation"]
    scene = context.get('release_scene', scene)
    plan, holding = context['place_plan'], context['holding']
    execution = context.get("demo_place_execution", {})
    pending = context.get("pending_transfer", {})
    record = ledger.record
    _require(target == ledger.target_id == ledger.current_id and record.get("status") != "completed"
             and target not in ledger.completed_ids and not pending.get("verified"), "target_already_completed")
    expected = {"arm": arm, "instance_id": target, "generation": generation,
                "scene_token": scene["scene_token"], "observation_id": scene["observation_id"], "owner_id": run_id}
    _require(arm in {"A", "B"} and ledger.bound_arm == arm, "arm_changed")
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
             and execution.get("release_commanded") is True
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
             and release.get("success") is True and release.get("release_commanded") is True
             and release.get("released") is True and release.get("execution_completed") is True
             and record.get("release_feedback_id") == release["feedback_id"], "release_feedback_invalid")
    placement = ledger.placement_request
    _require(plan.get("placement_request") == placement and placement.get("object_id") == target,
             "placement_region_changed")
    current = ledger.visual_progress
    _require(valid_placement_region(current) and current["placement_region"].get("layout_revision")
             == placement.get("layout_revision"), "placement_region_invalid")
    binding = {**expected, "target_id": target, "destination_id": state["destination_id"],
        "region_id": current["region_id"],
        "layout_revision": placement["layout_revision"], "placement_region": deepcopy(current["placement_region"]),
        "transfer_index": ledger.transfer_index, "release_feedback_id": release["feedback_id"],
        "release_feedback_time": release_time, "held_feedback_id": held["feedback_id"],
        "release_tcp": deepcopy(context["release_tcp"]),
        "identity_source": "execution_chain", "source": "validated_held_attachment_place_release",
        "evidence": {"held": deepcopy(held), "attachment": deepcopy(attachment),
                     "release_endpoint": deepcopy(context["release_endpoint_evidence"]),
                     "release_collision_check": deepcopy(check), "physical_release": deepcopy(release)}}
    if "verified_perf_counter_ns" in release:
        binding["release_verified_perf_counter_ns"] = release["verified_perf_counter_ns"]
    if 'placement_zone' in plan:
        binding['placement_zone'] = deepcopy(plan['placement_zone'])
    rim = plan.get('goal', {}).get('release_reference', {}).get('rim_provenance')
    if rim is not None:
        binding['destination_geometry'] = deepcopy(rim)
    binding["binding_id"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return binding


def validate_release_capture(capture, binding):
    digest = hashlib.sha256(json.dumps({key: value for key, value in binding.items() if key != 'binding_id'},
                                      sort_keys=True, allow_nan=False).encode()).hexdigest()
    _require(binding.get('binding_id') == digest, 'release_binding_integrity_changed')
    _require(release_identity_confirmed(binding), 'held_identity_binding_missing')
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
    _require(valid_placement_region(binding), "placement_region_invalid")


def release_identity_confirmed(binding):
    """A held verdict from another object/run/scene cannot support inferred delivery."""
    held = binding.get('evidence', {}).get('held', {})
    identity = held.get('identity_binding') or {}
    return (held.get('success') is True and held.get('status') == 'held'
        and all(held.get(key) == binding.get(key) and binding.get(key) is not None
                for key in ('target_id', 'arm', 'generation'))
        and identity.get('source') == 'continuous_grasp_execution'
        and identity.get('visual_identity_verified') is False
        and all(identity.get(key) == binding.get(key) and binding.get(key) is not None
                for key in ('target_id', 'arm', 'generation', 'owner_id', 'scene_token'))
        and identity.get('source_observation_id') == binding.get('observation_id')
        and bool(identity.get('reference_observation_id')))


def release_result_question(has_reference_image=True, placement_intent=None, *, target_id=None,
                            support_id=None):
    reference = f'target ID {target_id}' if target_id is not None else 'described target'
    container = (placement_intent or {}).get('relation') == 'inside'
    relation = 'inside the destination container' if container else 'on the destination support'
    region = (' The configured release square constrains the robot release position for both table and '
              'container placements; do not require the resting object to fit within that square. '
              'Judge whether delivery to the named support or container succeeded. '
              'For container delivery, do not require identifying which of several identical contents '
              'is the newly delivered object. The bound execution chain supports inferred_success.')
    reference_rule = (f'Historical boxes labelled {target_id} identify the target appearance. '
                      f'Boxes labelled {support_id} identify the destination support/container. '
                      if has_reference_image else 'Use the supplied target and destination descriptions. ')
    return choice('place_result', f'Assess placement of {reference} {relation} (ID {support_id}).'
        + region + ' ' + reference_rule
        + 'Use CURRENT images, dated BEFORE_RELEASE images and the bound execution evidence. '
        'direct_success means the current target is visibly at the destination under the stated acceptance scope. '
        'inferred_success means the execution evidence confirms holding this target, reaching the configured '
        'release XYZ within tolerance and releasing it there; the gripper is confirmed empty now and '
        'there is no visible evidence contradicting delivery. Occlusion is allowed but is not required. '
        'Inspect visible surroundings and the before/after change. Visible contrary evidence takes precedence '
        'over planned or measured release location. Missing required execution evidence requires unknown. '
        'A clearly visible target outside the destination acceptance scope or still carried is failure.', [
            {'value': 'direct_success', 'description': '直接看到目标已放在指定桌面支撑或进入指定盒子。'},
            {'value': 'inferred_success', 'description': '执行证据确认持有目标、到达释放 XYZ 并释放，当前空爪，未发现交付失败的反证。'},
            {'value': 'failure', 'description': '目标在目的地外或仍被携带等明确失败证据。'},
            {'value': 'unknown', 'description': '证据不足。无法满足直接确认或推测成功条件，也缺少明确失败证据。'}])


def release_position_check(binding):
    """Check measured release XYZ and configured XY region, never simulator object truth."""
    endpoint = binding.get('evidence', {}).get('release_endpoint', {})
    try:
        actual = np.asarray(binding['release_tcp'], float)
        reference = np.asarray(endpoint['tool_reference_point_m'], float)
        point = actual[:3, :3] @ reference + actual[:3, 3]
        polygon = np.asarray(binding['placement_region']['polygon_world_xy_m'], float)
        _require(actual.shape == (4, 4) and reference.shape == (3,)
                 and np.isfinite(actual).all() and np.isfinite(reference).all()
                 and np.isfinite(polygon).all(), 'numeric_position_invalid')
        hull = ConvexHull(polygon)
        container = binding.get('placement_zone', {}).get('intent', {}).get('release_mode') == 'container_release'
        _require(endpoint.get('source') == 'configured_front_tcp_and_actual_tcp'
                 and endpoint['at_release_goal'] is True
                 and (endpoint.get('release_mode') == 'container_release') == container
                 and endpoint['object_geometry_inside_region'] is True
                 and np.allclose(endpoint['actual_tcp_world'], actual, atol=1e-9, rtol=0)
                 and np.allclose(endpoint['actual_reference_world_m'], point, atol=1e-9, rtol=0)
                 and np.all(hull.equations[:, :2] @ point[:2] + hull.equations[:, 2] <= 1e-9),
                 'numeric_position_invalid')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('release_verification_numeric_position_missing') from error
    return {'source': endpoint['source'], 'time_scope': 'at_gripper_opening_before_current_images',
        'reference_xy_m': point[:2].tolist(), 'reference_world_m': point.tolist(),
        'position_error_m': endpoint['position_error_m'], 'position_tolerance_m': .005,
        'within_target_xyz_tolerance': True, 'target_region_polygon_xy_m': polygon.tolist(),
        'reference_kind': 'configured_front_tcp',
        'reference_inside_region': True,
        'meaning': 'Numerical release-location check from measured robot feedback and the declared tool reference, '
                   'Combine this historical measurement with current placement classification.'}




def release_result_payload(capture, binding, reference, instruction="", *, empty_gripper_confirmed=False):
    from .raw_visual_input import capture_content, reference_content
    target = binding['target_id']
    remembered = reference.get('entities', {}).get(target, {})
    zone = binding.get('placement_zone', {})
    support_id = zone.get('support_entity_id', binding['destination_id'])
    destination = reference.get('entities', {}).get(support_id, {})
    release_position = release_position_check(binding)
    context = {'observation_id': capture['observation_id'], 'target_id': target,
        'executing_arm': binding['arm'],
        'destination_id': binding['destination_id'], 'binding_id': binding['binding_id'],
        'target_description': remembered.get('description'),
        'target_description_scope': 'historical appearance; former locations are not required now',
        'destination_support_id': support_id, 'destination_kind': destination.get('kind'),
        'destination_description': destination.get('description'),
        'placement_intent': zone.get('intent'),
        'placement_scope': 'named_support_or_container',
        'configured_release_region_role': 'robot_release_position_only',
        'reference_labels': {'target': target, 'destination_support_or_container': support_id},
        'execution_evidence': {
            'target_id': target, 'binding_id': binding['binding_id'],
            'held_feedback_id': binding.get('held_feedback_id'),
            'release_feedback_id': binding['release_feedback_id'],
            'pre_release_target_confirmed': binding.get('evidence', {}).get('held', {}).get('success') is True
                and binding.get('evidence', {}).get('held', {}).get('status') == 'held',
            'held_target_identity': deepcopy(binding.get('evidence', {}).get('held', {}).get('identity_binding')),
            'held_target_identity_interpretation': 'Held presence was verified using measured gripper position or RGB. Target identity was inherited '
                'from the confirmed pregrasp target and its bound completed grasp, not re-identified after grasp.',
            'release_inside_requested_region': release_position['reference_inside_region'],
            'actual_release_xyz_check': release_position,
            'current_empty_gripper_confirmed': empty_gripper_confirmed},
        'rules': 'Judge CURRENT placement from the supplied RGB. BEFORE and reference frames are dated. '
            'Use target descriptions and labelled references for identity; former locations belong to those historical frames. '
            'Configured release XYZ is checked from actual robot feedback for both table and container placement. '
            'The configured release square is not a visual boundary for the final resting object. '
            'Choose direct_success for clearly visible correct placement. Use inferred_success for '
            'delivery supported by actual release XYZ, the bound execution chain and current empty gripper, with '
            'no visible outside-drop evidence. Preserve failure or unknown when their conditions apply.'}
    content = [{'type': 'input_text', 'text': json.dumps(context, default=lambda v: v.tolist())}]
    content.extend(capture_content(capture, modalities=('rgb',)))
    baseline = capture.get('release_baseline')
    if baseline:
        content.extend(capture_content(baseline, modalities=('rgb',), role='before_release'))
    relevant = {'entities': {target: remembered, support_id: destination}, 'identity_references': {
        identifier: stored for identifier, stored in reference.get('identity_references', {}).items()
        if identifier in {target, support_id}}}
    content.extend(reference_content(relevant))
    question = release_result_question(target in relevant['identity_references'], zone.get('intent'),
        target_id=target, support_id=support_id)
    return {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}],
            'questions': [question]}


def gripper_release_payload(capture, arm, *, modalities=('rgb',), evidence=None):
    from .raw_visual_input import capture_content, gripper_detail_content
    from .robot_skills.grasp_verification import (
        HOLDING_VISUAL_CRITERIA,
        WRIST_CAMERAS,
        gripper_evidence_context,
        gripper_feedback_state,
        object_held_question,
    )

    camera = WRIST_CAMERAS[arm]
    wrist = {**capture, 'views': {camera: capture['views'][camera]}}
    annotation = gripper_evidence_context(capture, evidence) if evidence is not None else {}
    content = [{'type': 'input_text', 'text': json.dumps({
        'observation_id': capture['observation_id'], 'selected_arm': arm,
        'primary_wrist_camera_id': camera,
        'holding_visual_criteria': HOLDING_VISUAL_CRITERIA,
        'gripper_feedback': [gripper_feedback_state(capture, arm)],
        'holding_segmentation': annotation.get('holding_segmentation', []),
        'rules': 'Judge the CURRENT gripper using images and same-capture joint feedback. Each image has its own capture and role. '
                 'Only current contact images are supplied. Excluded pixels are unobserved.'}, ensure_ascii=False)}]
    content.extend(gripper_detail_content(capture, arm) or
                   capture_content(wrist, modalities=modalities, masks=annotation.get('visual_masks')))
    question = object_held_question(arm)
    return {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}],
            'questions': [condition(question.name, question.instructions)]}


def verify_release_result(objects, capture, binding, reference, instruction="", *, gripper_evidence=None):
    from .observation_requirements import assessment
    from .robot_skills.grasp_verification import holding_from_feedback

    validate_release_capture(capture, binding)
    release_position_check(binding)
    measured = holding_from_feedback(capture, binding['arm'])
    requests = []
    if measured['requires_visual']:
        requests.append(gripper_release_payload(capture, binding['arm'], evidence=gripper_evidence))
    elif measured['status'] == 'no':
        requests.append(release_result_payload(capture, binding, reference, instruction,
                                              empty_gripper_confirmed=True))
    support_id = binding.get('placement_zone', {}).get('support_entity_id', binding['destination_id'])
    questions = [question for request in requests for question in request['questions']]

    def parse(body, evidence):
        visual_judgments, answers = parse_answers(body, questions[:len(body['answers'])])
        inverse = {'yes': 'no', 'no': 'yes', 'unknown': 'unknown'}
        holding = visual_judgments.get('object_held', measured['status'])
        holding_source = 'openai_decisions' if measured['requires_visual'] else measured['source']
        placement_outcome = visual_judgments.get('place_result', 'unknown')
        inference_chain_confirmed = release_identity_confirmed(binding) and holding == 'no'
        if placement_outcome == 'inferred_success' and not inference_chain_confirmed:
            placement_outcome = 'unknown'
        placement_status = {'direct_success': 'yes', 'inferred_success': 'yes',
                            'failure': 'no', 'unknown': 'unknown'}[placement_outcome]
        judgments = {'target_released': inverse[holding], 'place_result': placement_status}
        by_name = {answer["name"]: answer for answer in answers}
        status = "at_destination" if judgments["place_result"] == "yes" else "unknown"
        required = {'target_released': 'yes', 'place_result': 'yes'}
        # Refusal supplies no probability. Keep it missing rather than fabricate
        # a zero, or let it raise a KeyError before the graph sees unknown.
        confidence = by_name.get('place_result', {}).get('confidence')
        assessment_judgments, assessment_required = dict(judgments), dict(required)
        observation = assessment(assessment_judgments, assessment_required,
                                 capture['observation_id'])
        if observation and observation['status'] != 'confirmed':
            status = 'rejected' if observation['status'] == 'rejected' else 'unknown'
        visual = {"source": evidence['source'], "identity_source": "execution_chain",
            "support_entity_id": support_id, "generation": binding['generation'],
            "diagnostic_checks": deepcopy(evidence.get('diagnostic_checks', [])),
            "observation_id": capture["observation_id"], "status": status, "confidence": confidence,
            "confidence_rule": 'placement_choice_confidence_archive_only',
            "placement_outcome": placement_outcome,
            "inference_chain_confirmed": inference_chain_confirmed,
            "view_scope": "measured_gripper_then_destination_views",
            "camera_ids": list(evidence['camera_ids']),
            "primary_required_checks": list(required),
            "completion_semantics": 'measured_gripper_then_four_way_placement',
            "not_requested_checks": [name for name in required
                if name == 'place_result' and name not in by_name],
            "holding_presence": {'status': holding, 'source': holding_source,
                'rule': 'measured_position_then_closed_gripper_vision', 'feedback': measured['feedback']},
            "visual_judgments": visual_judgments,
            "derived_judgments": {'target_released': {'source': holding_source,
                **({'source_question': 'object_held'} if measured['requires_visual'] else {}),
                'operation': 'logical_not', 'source_status': holding, 'status': inverse[holding]}},
            "post_release_metric_pose_measured": False,
            'release_position_check': release_position_check(binding),
            "judgments": judgments, "answers": answers,
            "target_id": binding["target_id"], "destination_id": binding["destination_id"],
            "region_id": binding["region_id"],
            "placement_region": deepcopy(binding["placement_region"]),
            "layout_revision": binding["layout_revision"], "binding_id": binding["binding_id"],
            "release_feedback_id": binding["release_feedback_id"], "request_hash": evidence["request_hash"],
            **({'observation_assessment': observation} if observation else {})}
        check = {"source": evidence['source'], "observation_id": capture["observation_id"],
                 "request_hash": evidence["request_hash"], "status": "not_requested"}
        return {"visual_evidence": visual, "demo_objects_check": check,
                "binding_id": binding["binding_id"], "observation_id": capture["observation_id"],
                "evidence": {**evidence, "answers": answers}, "geometry_updated": False,
                **({'observation_assessment': observation} if observation else {})}

    answers, records = [], []
    for single in requests:
        try:
            body, evidence = VisualAPI.request(objects, single, capture, 'verify_release_result',
                                              lambda body, evidence: (body, evidence))
        except VisualDecisionError as error:
            error.evidence['completed_questions'] = merge_decision_evidence(records, capture, 'verify_release_result')
            raise
        answers.extend(body.get('answers', []))
        records.append({**evidence, 'answers': body.get('answers', [])})
        judgments, _ = parse_answers(body, single['questions'])
        if 'object_held' in judgments and judgments['object_held'] != 'no':
            break
        if 'object_held' in judgments:
            placement_payload = release_result_payload(capture, binding, reference, instruction,
                                                       empty_gripper_confirmed=True)
            questions.extend(placement_payload['questions'])
            requests.append(placement_payload)
    evidence = merge_decision_evidence(records, capture, 'verify_release_result')
    if not records:
        evidence.update(source=measured['source'], component='gripper_feedback', request_hash=None)
    evidence['gripper_feedback'] = measured['feedback']
    return parse({'answers': answers}, evidence)
