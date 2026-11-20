"""Bind gripper feedback and measured release position to the selected transfer."""
import hashlib
import json
import math
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull

from .core.placement import ReleaseLedgerSnapshot
from .decision_conditions import condition, parse_answers
from .visual_api import VisualAPI, VisualDecisionError
from .visual_identity import valid_placement_region


def _require(condition, reason):
    if not condition:
        raise ValueError("release_verification_" + reason)


def _execution_key(evidence):
    return hashlib.sha256(json.dumps({key: value for key, value in evidence.items() if key != 'sha256'},
        sort_keys=True, allow_nan=False, default=lambda value: value.tolist()).encode()).hexdigest()


def release_execution_evidence_id(evidence):
    if (evidence.get('source') != 'bound_release_execution' or not evidence.get('execution_evidence_id')
            or evidence.get('post_release_metric_pose_measured') is not False
            or not all(evidence.get(k) for k in ('observation_id', 'binding_id', 'target_id',
                                                  'destination_id', 'release_feedback_id'))):
        return None
    receipt = {k: v for k, v in evidence.items() if k != 'execution_evidence_id'}
    digest = _execution_key(receipt)
    return digest if evidence['execution_evidence_id'] == digest else None


def valid_container_entry(evidence, target_id, destination_id, feedback_id):
    if evidence.get('source') == 'bound_container_release':
        visual = evidence.get('visual_evidence') or {}
        fields = ('target_id', 'destination_id', 'observation_id', 'binding_id', 'release_feedback_id', 'generation')
        return bool(evidence.get('receipt_id') and evidence.get('status') == 'inside'
            and evidence.get('rule') == 'bound_release_and_empty_gripper'
            and evidence.get('placement_outcome') == visual.get('placement_outcome') == 'execution_assumed'
            and visual.get('status') == 'at_destination' and visual.get('holding_presence', {}).get('status') == 'no'
            and evidence.get('target_id') == target_id and evidence.get('destination_id') == destination_id
            and feedback_id and evidence.get('release_feedback_id') == feedback_id
            and all(evidence.get(key) is not None and evidence[key] == visual.get(key) for key in fields)
            and evidence.get('post_release_metric_pose_measured') is False and release_execution_evidence_id(visual))
    # Archived visual receipts remain readable.
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
    """Record the configured container as the destination of the completed release."""
    zone = binding.get('placement_zone', {})
    intent = zone.get('intent', {})
    _require(intent.get('release_mode') == 'container_release' and intent.get('relation') == 'inside'
             and zone.get('semantic_scope') == 'entire_region', 'container_entry_rule_not_applicable')
    _require(release_reference.get('release_mode') == 'container_release'
             and release_reference.get('scene_token') == binding['scene_token']
             and release_reference.get('rim_provenance', {}).get('entity_id') == binding['destination_id'],
             'container_entry_geometry_binding_changed')
    validate_release_capture(capture, binding)
    _require(release_execution_evidence_id(visual) and visual.get('status') == 'at_destination'
        and visual.get('binding_id') == binding['binding_id']
        and visual.get('observation_id') == capture['observation_id']
        and visual.get('target_id') == binding['target_id'] and visual.get('destination_id') == binding['destination_id']
        and visual.get('placement_outcome') == 'execution_assumed', 'container_release_unconfirmed')
    position = release_position_check(binding)
    receipt = {key: deepcopy(binding[key]) for key in (
        'target_id', 'destination_id', 'arm', 'generation', 'region_id', 'placement_region',
        'layout_revision', 'binding_id', 'release_feedback_id', 'release_feedback_time')}
    receipt.update(source='bound_container_release', status='inside',
        placement_outcome=visual['placement_outcome'],
        rule='bound_release_and_empty_gripper', observation_id=capture['observation_id'],
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
                 if key not in {'observation', 'scene', 'scene_observation', 'identity_reference', 'scene_entities'}},
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
    from .robot_skills.grasp_verification import WRIST_CAMERAS
    cameras = set(capture.get('views', {}))
    if capture.get('source') in {'gripper_feedback', 'gripper_wrist_rgbd'}:
        expected = {WRIST_CAMERAS[binding['arm']]} if capture['source'] == 'gripper_wrist_rgbd' else set()
        _require(cameras == expected and capture.get('arm') == binding['arm']
                 and capture.get('feedback_id') and capture.get('session_id'), 'gripper_sample_binding_invalid')
    else:
        _require(cameras == {'1', '2', '3'}, 'release_capture_source_invalid')
    _require(valid_placement_region(binding), "placement_region_invalid")


def release_identity_confirmed(binding):
    """Accept holding evidence for this target, arm, scene and run."""
    held = binding.get('evidence', {}).get('held', {})
    identity = held.get('identity_binding') or {}
    if identity.get('source') == 'continuous_grasp_execution':
        # Historical experiment receipts stored execution inference without an outcome field.
        accepted = (identity.get('visual_identity_verified') is False and (
            'outcome' not in identity or (
                identity['outcome'] == held.get('target_identity_outcome') == 'execution_assumed'
                and held.get('holding_presence', {}).get('status') == 'yes'
                and bool(held.get('observation_id'))
                and identity.get('holding_observation_id') == held.get('observation_id')
                    == held.get('holding_presence', {}).get('observation_id'))))
    else:
        outcome = identity.get('outcome')
        evidence = identity.get('identity_evidence') or {}
        accepted = (identity.get('source') == 'post_lift_target_identity'
            and outcome in {'direct_success', 'inferred_success'} and held.get('target_identity_outcome') == outcome
            and held.get('holding_presence', {}).get('status') == 'yes'
            and identity.get('visual_identity_verified') is (outcome == 'direct_success')
            and evidence.get('source') in {'post_lift_rgbd', 'openai_decisions'}
            and evidence.get('target_id') == held.get('target_id') and bool(held.get('observation_id'))
            and evidence.get('observation_id') == identity.get('holding_observation_id')
                == held['holding_presence'].get('observation_id') == held['observation_id'])
    return (accepted and held.get('success') is True and held.get('status') == 'held'
        and all(held.get(key) == binding.get(key) and binding.get(key) is not None
                for key in ('target_id', 'arm', 'generation'))
        and all(identity.get(key) == binding.get(key) and binding.get(key) is not None
                for key in ('target_id', 'arm', 'generation', 'owner_id', 'scene_token'))
        and identity.get('source_observation_id') == binding.get('observation_id')
        and bool(identity.get('reference_observation_id')))


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
                 and 0 <= float(endpoint['position_error_m']) <= .005
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
        'meaning': 'Release position uses measured robot feedback and the configured tool reference. '
                   'The empty-gripper check completes the bound release.'}


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


def verify_release_result(objects, capture, binding):
    """A measured release at the bound destination and an empty gripper complete delivery."""
    from .observation_requirements import assessment
    from .robot_skills.grasp_verification import holding_from_feedback

    validate_release_capture(capture, binding)
    position = release_position_check(binding)
    measured = holding_from_feedback(capture, binding['arm'])
    holding = measured['status']
    raw_answers = []
    evidence = {'source': measured['source'], 'request_hash': None, 'camera_ids': [], 'model_called': False}
    if measured['requires_visual'] and capture.get('views'):
        payload = gripper_release_payload(capture, binding['arm'])
        try:
            body, evidence = VisualAPI.request(objects, payload, capture, 'verify_release_result',
                                               lambda body, record: (body, record))
            answers, raw_answers = parse_answers(body, payload['questions'])
            holding = answers['object_held']
        except VisualDecisionError as error:
            evidence = {**error.evidence, 'error': str(error)}
            holding = 'unknown'
    released = {'no': 'yes', 'yes': 'no', 'unknown': 'unknown'}[holding]
    outcome = {'yes': 'execution_assumed', 'no': 'failure', 'unknown': 'unknown'}[released]
    status = {'yes': 'at_destination', 'no': 'rejected', 'unknown': 'unknown'}[released]
    judgments = {'target_released': released, 'place_result': released}
    observation = assessment(judgments, {'target_released': 'yes', 'place_result': 'yes'}, capture['observation_id'])
    visual = {key: deepcopy(binding[key]) for key in ('binding_id', 'target_id', 'destination_id', 'arm',
        'generation', 'region_id', 'layout_revision', 'placement_region', 'release_feedback_id')}
    visual.update(source='bound_release_execution', identity_source='execution_chain', status=status,
        observation_id=capture['observation_id'], post_release_metric_pose_measured=False,
        support_entity_id=(binding.get('placement_zone') or {}).get('support_entity_id', binding['destination_id']),
        placement_outcome=outcome, verification_steps={'holding_presence': holding,
            'destination': outcome if released == 'yes' else 'not_checked'},
        holding_presence={'status': holding, 'source': evidence.get('source'), 'feedback': measured['feedback']},
        verification_time=capture['time'], verification_perf_counter_ns=capture.get('captured_perf_counter_ns'),
        wrist_error=capture.get('wrist_error'), release_position_check=position, judgments=judgments, answers=raw_answers,
        request_hash=evidence.get('request_hash'), camera_ids=evidence.get('camera_ids', []),
        observation_assessment=observation)
    visual['execution_evidence_id'] = _execution_key(visual)
    return {'visual_evidence': visual, 'evidence': {**evidence, 'answers': raw_answers},
            'demo_objects_check': {'status': 'not_requested'}, 'observation_assessment': observation,
            'binding_id': binding['binding_id'], 'observation_id': capture['observation_id'], 'geometry_updated': False}
