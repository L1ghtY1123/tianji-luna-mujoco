"""Execute frozen-frame grounding and publish accepted bindings at the runtime boundary."""
import hashlib
import json
import time
from copy import deepcopy

from ..core.grounding import identity_provenance
from ..object_memory import restored_object_record
from ..robot_contracts import RobotFeedbackUnavailable
from ..robot_skills.observation import capture_reuse_rejection, stationary_capture_reference
from ..visual_api import merge_decision_evidence
from .compiled_task import bind_compiled_task
from .diagnostics import DiagnosticError


def fixed_targets_delivered(spec, progress):
    """Keep the selected task roles once every member of a finite selection is delivered."""
    if spec is None or spec.scope not in {'ids', 'count'}:
        return False
    delivered = progress.completed_input_ids()
    if not delivered or delivered != set(progress.input_ids):
        return False
    if spec.quantity is not None and len(delivered) != spec.quantity:
        return False
    return delivered == set(spec.target_refs) if spec.scope == 'ids' else spec.quantity is not None


def _geometry_binding(client, capture, result, registration, entities, sam3_queries):
    records = result.evidence['decision_records']
    evidence = merge_decision_evidence(records, capture, 'rgbd_grounding')
    evidence.update(source='rgbd_instance_association', observation_id=capture['observation_id'],
        generation=capture['generation'], registration_basis='scene_association_v3',
        identity_derivation='scene_identity_then_task_selection',
        grounding=result.evidence, missing_evidence=list(result.missing_evidence))
    evidence['committed_group_ids'] = dict(registration['group_ids'])
    evidence['association_digest'] = hashlib.sha256(json.dumps(
        [capture['observation_id'], capture['generation'], registration['group_ids']], sort_keys=True).encode()).hexdigest()
    matches = {match.group_id: match for match in result.history_matches}
    detections, confirmed, names, judgments = [], [], [], {}
    for group in result.groups:
        identifier = registration['group_ids'].get(group.group_id)
        if identifier is None:
            continue
        for index in group.candidate_indices:
            candidate = result.observation.candidates[index]
            entry = dict(candidate)
            entry.update(kind=entities[identifier]['kind'], entity_id=identifier)
            confirmed.append(entry)
            if entry['kind'] in {'table', 'marker'}:
                continue
            entry['identity_evidence'] = [{**identity_provenance(matches[group.group_id]), 'entity_id': identifier,
                'candidate': candidate['candidate_id'], 'camera': candidate['camera_id'],
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'registration_basis': 'scene_association_v3', 'group_id': group.group_id}]
            judgments['identity_' + str(len(detections))] = identifier
            names.append(candidate['candidate_id'])
            detections.append(entry)
    visual = client._groups(detections, names, entities, judgments, evidence, purpose='associate_objects')
    visual.update(membership=registration['membership'], registered_entities={}, progress={})
    return {'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'confirmed_candidates': confirmed, 'visual_result': visual,
        'sam3_queries': deepcopy(sam3_queries)}


def _check_capture(controller, context, capture, run_id):
    controller.check_generation()
    if controller.run_id != run_id or controller.state['status'] != 'running':
        raise DiagnosticError('grounding_cancelled', {'run_id': run_id})
    feedback = controller.simulation.robot_snapshot(include_tcp=False)
    idle = capture.get('active_commands') == [] and feedback.get('active_commands') == []
    if idle and (not feedback['available'] or any(
            'gripper_position_rad' not in feedback['arms'].get(arm, {}) for arm in ('A', 'B'))):
        # Refresh missing telemetry without losing an active command already seen at either check.
        try:
            feedback = controller.simulation.robot_snapshot(expected_generation=controller.state['generation'])
        except RobotFeedbackUnavailable as error:
            raise DiagnosticError('grounding_capture_invalidated', {
                **error.details, 'observation_id': capture['observation_id'], 'reason': 'robot_feedback_unavailable',
            }) from error
    reason = capture_reuse_rejection(capture, feedback,
        epoch_matches=context.get('capture_epochs', {}).get(capture['observation_id']) == controller.observation_epoch,
        now=time.monotonic(), max_age_s=None)
    if reason:
        raise DiagnosticError('grounding_capture_invalidated', {
            'reason': reason, 'observation_id': capture['observation_id'],
            'capture_active_commands': capture.get('active_commands'),
            'feedback_active_commands': feedback.get('active_commands'), 'feedback_error': feedback.get('physics_error'),
        })


async def prepare_grounding(controller, context, capture, *, initial=False, scene_only=False):
    """No capture or robot movement occurs here. The calling graph owns those effects."""
    from ..agent.schema import object_task_specs

    plan = context['initial_plan']
    # A recovery can re-enter ground_task after an earlier input was already claimed or delivered.
    initial = initial and not controller.task_progress.input_ids
    run_id = controller.run_id
    revision = controller.object_database.revision
    key = (capture['observation_id'], capture['generation'], revision,
           json.dumps(context.get('grounded_refs', {}), sort_keys=True))
    _check_capture(controller, context, capture, run_id)
    cached = context.get('_grounding_attempt', {})
    if not scene_only and cached.get('key') == key:
        if cached['status'] != 'success':
            raise DiagnosticError('task_grounding_unknown', cached['details'])
        return cached['summary']
    perception = controller.algorithms.perception
    await controller.call_algorithm(controller.algorithms.prepare_observation_memory)
    memory = controller.object_registrar.entities()
    reference = controller.object_database.reference(memory)
    stopped = (controller.state.get('failure_return') or {}).get('motion_stopped') if scene_only else None
    continuity = stationary_capture_reference(context.get('registered_capture'), capture, stopped=stopped)
    if continuity:
        reference['stationary_capture'] = continuity
    grounded_refs = context.get('grounded_refs', {})
    retained = controller.object_memory.entities()
    proposals = [] if scene_only else [selector for selector in plan.task_selectors if not (
        isinstance(grounded_refs.get(selector.ref), str)
        and retained.get(grounded_refs[selector.ref], {}).get('status') in {'consumed', 'removed'})]
    audit, done = controller.visual_audit_sink()
    perception.visual_identity.audit = audit
    from ..candidate_observation import reobserved_proposal_hints
    origin = context['initial_capture']
    same_epoch = context.get('capture_epochs', {}).get(origin['observation_id']) == controller.observation_epoch
    hints = reobserved_proposal_hints(origin, capture, plan.scene_inventory) if same_epoch else []
    try:
        result = await controller.call_algorithm(perception.ground, capture, proposals=proposals,
            entities=memory, reference=reference, initial=initial, inventory=plan.scene_inventory,
            declared_objects=plan.declared_objects, reobserved_hints=hints,
            proposal_observation_id=context['initial_capture']['observation_id'])
        _check_capture(controller, context, capture, run_id)
        registration = controller.object_registrar.commit_grounding(capture, result, expected_revision=revision)
        context['registered_capture'] = capture
        registered_revision = controller.object_database.revision
        roles = {'identifiers': {}, 'membership': {}, 'unresolved_proposals': [], 'decision_records': []}
        if not scene_only:
            roles = await controller.call_algorithm(perception.bind_task_roles, capture, result,
                registration['group_ids'], registration['entities'], proposals, instruction=controller.state['instruction'],
                bound_refs=grounded_refs, initial=initial, inventory=plan.scene_inventory,
                proposal_observation_id=origin['observation_id'])
            _check_capture(controller, context, capture, run_id)
            if controller.object_database.revision != registered_revision:
                raise DiagnosticError('grounding_database_revision_changed', {'observation_id': capture['observation_id']})
            if not roles['unresolved_proposals']:
                controller.object_registrar.apply_geometry_policies(proposals, roles['identifiers'],
                                                                   expected_revision=registered_revision)
                if controller.object_database.revision != registered_revision:
                    registration['entities'] = controller.object_registrar.entities()
    finally:
        done.set()
    registration.update(roles)
    refs = {**grounded_refs, **roles['identifiers']}
    registration['identifiers'] = refs
    task_status = 'not_checked' if scene_only else 'unknown' if roles['unresolved_proposals'] else 'success'
    controller.emit('task_grounding_evidence', '当前实例与 Decisions 角色绑定', {
        'status': task_status, 'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'registered_ids': registration['group_ids'], 'role_bindings': roles['identifiers'],
        'missing_evidence': roles['unresolved_proposals'], 'role_decisions': roles['decision_records'],
        'evidence': result.evidence})
    controller.object_memory.restore(tuple(restored_object_record(row) for row in registration['entities'].values()))
    controller.object_memory.observation_id = capture['observation_id']
    entities = controller.object_memory.entities()
    context['grounded_refs'] = refs
    context['grounded_observation'] = _geometry_binding(perception.visual_identity, capture, result,
        registration, entities, perception.segmentation_evidence)
    context['identity_reference'] = {**controller.object_database.reference(entities),
        'observation_id': capture['observation_id'], 'generation': capture['generation'], 'time': capture['time'],
        'object_memory_snapshot': controller.object_memory.snapshot(), 'owner_id': controller.run_id}
    from ..observation_labels import registered_reference_labels
    controller.record_annotation(capture, registered_reference_labels(capture, context['identity_reference']))
    scene_summary = {'observation_id': capture['observation_id'], 'group_ids': registration['group_ids'],
        'unresolved_candidates': registration['unresolved_candidates'], 'missing_tracks': registration['missing_tracks'],
        'coverage': registration['coverage'], 'task_status': task_status}
    controller.emit('scene_association_committed', '已保存场景实例与观测', scene_summary)
    controller.state['scene_association'] = scene_summary
    if scene_only:
        return scene_summary
    if roles['unresolved_proposals']:
        details = {'missing_evidence': roles['unresolved_proposals'], 'observation_id': capture['observation_id'],
                   'confirmed_scene_ids': list(registration['group_ids'].values())}
        context['_grounding_attempt'] = {'key': (capture['observation_id'], capture['generation'],
            controller.object_database.revision, json.dumps(refs, sort_keys=True)), 'status': 'unknown', 'details': details}
        controller.state['facts']['task_grounded'] = False
        raise DiagnosticError('task_grounding_unknown', details)
    binding = bind_compiled_task(plan, registration, entities)
    controller.task_spec = plan.task.model_validate(binding['task_spec'])
    controller.task_bindings = binding['task_bindings']
    controller.object_tasks = object_task_specs(binding['policy_graph'])
    controller.task_progress.register_entities(entities, roles['membership'])
    active = set(controller.task_progress.input_ids) | controller.task_progress.completed_input_ids()
    for identifier, membership in roles['membership'].items():
        record = controller.task_progress.records.get(identifier)
        if record and identifier not in active and controller.task_progress.membership.get(identifier) != membership:
            controller.task_progress.membership[identifier] = record['visual_membership'] = membership
            controller.task_progress.ledger_revision += 1
    if initial:
        controller.task_progress.configure_task(controller.task_spec, [], object_tasks=controller.object_tasks)
    else:
        controller.task_progress.task_spec = controller.task_spec
        controller.task_progress.object_tasks = controller.object_tasks
    executor = controller.graph_executor
    if executor is not None:
        executor.update_entity_refs(refs)
        if controller.active_object_task is not None:
            controller.active_object_task = controller.task_spec.model_validate(
                executor.bind(controller.active_object_task.model_dump(mode='json')))
            controller.state['active_object_task'] = controller.active_object_task.model_dump(mode='json')
        executor.bindings.update({'$source': controller.task_bindings.get('source_id'),
            '$destination': controller.task_bindings.get('destination_id'), '$work_area': controller.task_bindings.get('work_area_id')})
    controller.state.update(task_spec=controller.task_spec.model_dump(mode='json'),
                            task_bindings=deepcopy(controller.task_bindings))
    controller.state['facts']['task_grounded'] = True
    summary = {'success': True, 'status': 'success', 'observation_id': capture['observation_id'],
        'entity_ref_bindings': deepcopy(refs), 'registration_basis': 'scene_association_v3'}
    context['_grounding_attempt'] = {'key': (capture['observation_id'], capture['generation'],
        controller.object_database.revision, json.dumps(refs, sort_keys=True)), 'status': 'success', 'summary': summary}
    controller.emit('task_grounding_committed', '已提交 RGBD 身份与任务引用', summary)
    controller.state['grounding'] = summary
    return summary
