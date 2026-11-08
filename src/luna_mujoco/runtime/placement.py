"""Placement composition: Decisions owns meaning; measured skills own geometry."""
from collections import Counter
from copy import deepcopy

import numpy as np

from ..agent.placement import placement_region_questions, select_placement_candidates
from ..algorithms import AlgorithmError
from ..core.decisions import ChoiceRequest
from ..object_obb import held_geometry_points
from ..placement import observed_object
from ..robot_skills.placement_resolution import digest, project_placement_candidates
from ..robot_skills.placement_zone import perceive_placement_zone, placement_frames
from .decision_inputs import destination_reference_evidence


def invalidate_placement(controller, context, *, zone=False):
    keys = ['placement_resolution', 'place_plan', 'place_plan_binding', 'place_plan_observation_id']
    controller.state.setdefault('facts', {}).update(placement_resolved=False, place_plan_ready=False)
    if zone:
        keys.extend(['placement_zone', 'placement_candidate_view'])
        controller.state['facts']['placement_zone_ready'] = False
    for key in keys:
        context.pop(key, None)


def declared_destination_binding(controller, observation, target, destination_id, local_task):
    """Reuse the compiled task relation and the current measured instance binding."""
    record = controller.task_progress.records[target]
    root = record.get('root_input_id', target)
    coverage = observation.get('visual_inventory', {}).get('identity_coverage', {})
    destination = observed_object(observation, destination_id)
    role = ('final_destination' if destination_id == local_task.get('destination_ref') else
            'declared_work_area' if destination_id == local_task.get('work_area_ref') else None)
    if (role is None or root not in controller.task_progress.input_ids
            or record.get('destination_ref') != destination_id):
        return None
    if destination.get('kind') not in {'table', 'marker'} and (
            coverage.get('observation_id') != observation.get('observation_id')
            or destination_id not in coverage.get('accepted_entity_ids', [])):
        return None
    # Confirmed products inherit their input's destination, but not an unperformed
    # processing requirement. Walk the ledger rather than trusting a root ID alone.
    current, visited = root, set()
    while current != target:
        if current in visited:
            return None
        visited.add(current)
        parent = controller.task_progress.records.get(current, {})
        child = parent.get('output_entity_id')
        child_record = controller.task_progress.records.get(child, {})
        if (parent.get('status') != 'consumed' or child_record.get('input_entity_id') != current
                or not all(child_record.get('creation_evidence', {}).get(key)
                           for key in ('observation_id', 'request_hash', 'action_id'))):
            return None
        current = child
    actual, required = Counter(record.get('processing_history', [])), Counter(local_task.get('processing', []))
    if role == 'final_destination' and any(actual[name] < count for name, count in required.items()):
        return None
    return {'source': 'compiled_task_requirement_and_current_identity',
        'task_sha256': digest(local_task), 'transfer_role': role,
        'root_input_id': root, 'target_id': target, 'destination_id': destination_id,
        'current_identity_observation_id': observation['observation_id'],
        'current_identity_coverage': deepcopy(coverage), 'historical_relation_reused': True,
        'current_pose_or_completion_claimed': False}


def accepted_intermediate_binding(context, target, destination, task):
    """Reuse permission for this exact transfer; do not rejudge it as final delivery."""
    permission = context.get('destination_permission', {})
    evidence = permission.get('evidence', {})
    if (permission.get('target_id') != target
            or permission.get('destination_id') != destination['id']
            or permission.get('reference_id') != destination.get('reference_id', destination['id'])
            or permission.get('task_sha256') != digest(task)
            or permission.get('observation_id') != context['observation']['observation_id']
            or permission.get('region_sha256') != digest(destination.get('placement_reference', {}))
            or evidence.get('judgments', {}).get('intermediate_destination_allowed', {}).get('status') != 'yes'
            or not evidence.get('evidence', {}).get('request_hash')):
        return None
    return {'source': 'accepted_intermediate_destination_permission',
            'transfer_role': 'permitted_intermediate', **deepcopy(permission),
            'current_pose_or_completion_claimed': False}


async def perceive_for_controller(controller, context):
    invalidate_placement(controller, context, zone=True)
    target, destination_id = controller.state['target_id'], controller.state['destination_id']
    observation = context['observation']
    destination = observed_object(observation, destination_id)
    record = controller.task_progress.records[target]
    reference = destination.get('placement_reference', {})
    try:
        # This is the measured destination's physical mode, not a language parse.
        zone = perceive_placement_zone(observation, destination_id,
            {'relation': 'inside' if destination['kind'] == 'bin' else 'on',
             'release_mode': record['release_mode'],
             'preference': record.get('placement_preference', 'clearance')})
    except ValueError as error:
        return {'status': 'unknown', 'reason': str(error),
            'target_id': target, 'destination_id': destination_id,
            'observation_id': observation['observation_id'], 'generation': observation.get('generation'),
            'details': {'component': 'placement_geometry', 'stage': 'perceive_placement_zone',
                'placement_geometry_issue': deepcopy(destination.get('placement_geometry_issue'))},
            'required_evidence': 'Current identified support/reference geometry and unambiguous language; '
                                 'the graph may choose an observation capability. No fallback region was selected.'}
    requirement = controller.task_spec.model_dump(mode='json')
    active = controller.active_object_task
    zone.update(original_requirement=requirement,
                active_object_requirement=active.model_dump(mode='json') if active else None,
                bound_destination_ref=record.get('destination_ref', destination_id))
    local = zone['active_object_requirement'] or requirement
    bound, final, work = zone['bound_destination_ref'], local.get('destination_ref'), local.get('work_area_ref')
    zone['current_transfer'] = {'bound_destination_ref': bound, 'final_destination_ref': final,
        'work_area_ref': work, 'scope': 'final_destination' if bound == final else
        'declared_work_area' if bound == work else 'other_transfer_requires_semantic_permission'}
    zone['semantic_frames'] = placement_frames(context['latest_capture'])
    confirmed = declared_destination_binding(controller, observation, target, destination_id, local)
    if confirmed is None:
        confirmed = accepted_intermediate_binding(context, target, context.get('destination', destination), local)
    if confirmed and confirmed['transfer_role'] == 'permitted_intermediate':
        zone['current_transfer']['scope'] = 'permitted_intermediate'
    questions = placement_region_questions(configured=reference.get('source') == 'configured_xy_measured_support')
    if confirmed:
        questions = [question for question in questions if question.name != 'placement_destination_matches']
    semantic_key = digest({'task': local, 'destination_id': destination_id,
        'transfer_scope': zone['current_transfer']['scope'], 'configured': reference.get('source') == 'configured_xy_measured_support'})
    cache = context.setdefault('placement_semantics', {})
    if confirmed and semantic_key in cache:
        grounding = deepcopy(cache[semantic_key])
        grounding['language_scope_reused'] = True
    else:
        grounding = await controller._judge(context, questions, extra={
            'verification_scope': 'placement_region_scope', 'original_requirement': requirement,
            'active_object_requirement': zone['active_object_requirement'],
            'target_id': target, 'destination_id': destination_id,
            'bound_destination_ref': zone['bound_destination_ref'],
            'current_transfer': zone['current_transfer'],
            'declared_destination_references': destination_reference_evidence(
                observation, controller.compiler.configured_regions, active or controller.task_spec),
            'measured_region': {key: value for key, value in zone.items() if key != 'measured_support_xy_m'},
            'configured_region_constraints': {key: reference[key] for key in (
                'source', 'precise_xy', 'position_world_m', 'xy_tolerance_m', 'support_entity_id',
                'region_polygon_world_xy_m', 'floor_z_m', 'required_robot_configurations') if key in reference}})
        if confirmed:
            cache[semantic_key] = deepcopy(grounding)
    judgments = grounding.get('judgments', {})
    destination_match = 'yes' if confirmed else judgments.get('placement_destination_matches', {}).get('status')
    if confirmed:
        grounding['reused_destination_binding'] = confirmed
    additional = judgments.get('placement_additional_spatial_constraint', {}).get('status')
    if destination_match == 'no':
        return {'status': 'failure', 'reason': 'placement_destination_mismatch',
                'evidence': grounding, 'current_transfer': zone['current_transfer']}
    if destination_match != 'yes' or additional not in ('yes', 'no'):
        return {'status': 'unknown', 'reason': 'placement_region_semantics_unknown',
                'evidence': grounding, 'current_transfer': zone['current_transfer'],
                'required_evidence': 'Resolve permission for this current transfer before proceeding; '
                                     'no placement region has been authorized.'}
    entire = additional == 'no'
    zone.update(semantic_scope='entire_region' if entire else 'candidate_selection',
                semantic_evidence=deepcopy(grounding))
    context['placement_zone'] = zone
    record.update(placement_intent=deepcopy(zone['intent']))
    return {'status': 'success', 'intent': zone['intent'], 'placement_zone': zone,
            'semantic_scope': zone['semantic_scope'], 'evidence': grounding}


def _candidate_view_binding(scene, zone, capture, epoch):
    if (not scene.get('scene_token') or scene.get('observation_id') != zone.get('observation_id')
            or scene.get('generation') != zone.get('generation')
            or capture.get('generation') != zone.get('generation')):
        return None
    return {'source': 'explicit_current_capture_over_verified_scene',
        'capture_observation_id': capture['observation_id'], 'action_epoch': epoch,
        'geometry_observation_id': zone['observation_id'], 'generation': zone['generation'],
        'scene_token': scene['scene_token'], 'placement_zone_sha256': digest(zone)}


async def capture_candidate_view(controller, context, zone):
    """Called only for the model-selected fresh argument, after verify_scene."""
    capture = await controller.capture_observation(context)
    binding = _candidate_view_binding(controller.spatial_memory.scene, zone, capture, controller.observation_epoch)
    if binding is None:
        return {'status': 'unknown', 'reason': 'placement_support_geometry_stale'}
    context['placement_candidate_view'] = binding
    controller.emit('placement_candidate_capture', '采集放置候选当前图像', binding)
    return {'status': 'success', 'capture_binding': binding}


def candidate_capture_rejection(controller, context, zone):
    """Reject missing candidate evidence before expensive endpoint computation."""
    if zone.get('semantic_scope') != 'candidate_selection':
        return None
    capture = context.get('latest_capture', {})
    binding = context.get('placement_candidate_view')
    expected = (_candidate_view_binding(controller.spatial_memory.scene, zone, capture, controller.observation_epoch)
                if capture.get('observation_id') else None)
    distinct_bound_capture = binding is not None and binding == expected
    if (not capture.get('observation_id')
            or context.get('capture_epochs', {}).get(capture['observation_id']) != controller.observation_epoch
            or (any(capture.get(key) != zone.get(key) for key in ('observation_id', 'generation'))
                and not distinct_bound_capture)):
        return {'status': 'unknown', 'reason': 'placement_candidate_capture_stale',
                'required_evidence': 'Select resolve_placement with capture_mode=fresh to acquire current '
                    'candidate images over the verified scene, without full observation or movement. '
                    'Changing numerical preference cannot refresh this evidence.'}
    return None


async def select_for_controller(controller, context, resolution, *, publish=True):
    """Bind semantics to numerical candidates before making a resolution executable."""
    zone = resolution['placement_zone']
    if zone['semantic_scope'] == 'entire_region':
        selection = {'status': 'success', 'scope': 'entire_region',
            'candidate_ids': [candidate['candidate_id'] for candidate in resolution['candidates']],
            'evidence': zone['semantic_evidence']}
    else:
        capture = context['latest_capture']
        rejection = candidate_capture_rejection(controller, context, zone)
        if rejection is not None:
            return rejection
        try:
            projected = project_placement_candidates(capture, resolution, context['holding'],
                geometry_binding=context.get('placement_candidate_view'))
        except ValueError as error:
            return {'status': 'unknown', 'reason': str(error)}

        async def judge(requests, extra):
            return await controller._judge(context, requests, extra=extra)

        selection = await select_placement_candidates(judge, projected['candidates'], context={
            'original_requirement': zone['original_requirement'],
            'active_object_requirement': zone['active_object_requirement'],
            'target_id': controller.state['target_id'], 'destination_id': zone['destination_id'],
            'bound_destination_ref': zone['bound_destination_ref'],
            'current_transfer': zone.get('current_transfer'),
            'semantic_frames': zone.get('semantic_frames'),
            'physical_relation': zone['intent']['relation'],
            'current_inventory_entities': [{key: item[key] for key in
                ('id', 'kind', 'color', 'description', 'position', 'visible_centroid_world', 'identity_source')
                if key in item} for item in context.get('observation', {}).get('objects', [])],
            'candidate_projection': projected,
            'occupancy_evidence': resolution['selection_evidence']})
        selection['projection'] = projected
        if selection['status'] != 'success':
            return selection
    accepted = set(selection['candidate_ids'])
    result = deepcopy(resolution)
    result['candidates'] = [candidate for candidate in result['candidates'] if candidate['candidate_id'] in accepted]
    result['semantic_selection'] = selection
    result.pop('resolution_id', None)
    result['resolution_id'] = digest(result)
    if publish:
        context['placement_resolution'] = result
    return result


def xy_proposal_context(zone, attempts):
    """Keep numerical failure traces in the archive, not in an XY choice question."""
    region = {key: zone[key] for key in ('destination_id', 'support_entity_id', 'observation_id',
        'generation', 'intent', 'floor_z_m', 'polygon_world_xy_m', 'frame_axes_world_xy',
        'halfplanes_world_xy', 'geometry_source', 'semantic_scope') if key in zone}
    failures = []
    for attempt in attempts:
        row = {key: attempt[key] for key in ('subdivision_cells', 'candidate_indices', 'reason') if key in attempt}
        row['candidates'] = []
        for failed in (attempt.get('details') or {}).get('candidate_failures', []):
            detail = failed.get('details') or {}
            row['candidates'].append({'candidate_index': failed.get('candidate_index'), 'reason': failed.get('code'),
                **{key: detail[key] for key in ('initial_goal_position_m', 'required_clearance_m',
                    'maximum_height_increase_m', 'world_clear_sample_count', 'held_ik_sample_count',
                    'trajectory_checked') if key in detail}})
        failures.append(row)
    return region, failures


async def resolve_xy_then_z(controller, context, observation, request):
    """Decisions selects XY at each subdivision level before any Z/IK search."""
    zone, holding = context['placement_zone'], context['holding']
    precise = request.get('precise_xy') or not request.get('free_placement')
    levels = (4,) if precise else (4, 8, 16)
    attempts = []
    searches, deferred = [], []
    for level in levels:
        try:
            requests, evidence = controller.algorithms.placement_xy_candidates(observation, controller.state['target_id'], holding,
                request, zone, context['planning_settings'], scene=controller.spatial_memory.scene, subdivision_cells=level)
        except AlgorithmError as error:
            if error.code != 'placement_no_candidate':
                raise
            attempts.append({'subdivision_cells': level, 'reason': error.code, 'details': error.details})
            continue
        remaining = list(range(len(requests)))
        ranking = evidence.get('ranking') or {}
        gaps = {tuple(row['position_world_xy_m']): row['measured_obstacle_gap_m']
                for row in ranking.get('candidates', [])}
        desired = ranking.get('desired_clearance_m')
        if (not precise and ranking.get('preference') == 'clearance' and desired is not None
                and any(gap is not None for gap in gaps.values())):
            preferred = [i for i in remaining
                if gaps.get(tuple(requests[i]['selected_reference_world_xy_m'])) is not None
                and gaps[tuple(requests[i]['selected_reference_world_xy_m'])] >= desired]
            for index in remaining:
                if index not in preferred:
                    gap = gaps.get(tuple(requests[index]['selected_reference_world_xy_m']))
                    deferred.append((gap if gap is not None else -1., level, requests, evidence, [index]))
            remaining = preferred
        searches.append((level, requests, evidence, remaining))
    deferred.sort(key=lambda item: item[0], reverse=True)
    searches.extend(item[1:] for item in deferred)
    for level, requests, evidence, remaining in searches:
        while remaining:
            if precise:
                selection = {'status': 'success', 'scope': 'configured_xy',
                    'candidate_ids': ['placement-0'], 'evidence': zone['semantic_evidence']}
            elif zone['semantic_scope'] == 'entire_region':
                choices = tuple('placement-' + str(i) for i in remaining)
                region, failures = xy_proposal_context(zone, attempts)
                current_xy = np.asarray(observation['arms'][controller.state['arm']]['tcp'], float)[:2, 3]
                ranking = evidence.get('ranking') or {}
                gaps = {tuple(row['position_world_xy_m']): row['measured_obstacle_gap_m']
                        for row in ranking.get('candidates', [])}
                judgment = await controller._judge(context, [ChoiceRequest(name='placement_xy',
                    choices=(*choices, 'unknown'), selection_mode='proposal', instructions=(
                        'Select one proposed XY within the region already authorized by perceive_placement_zone. '
                        'Destination identity and the current transfer language were resolved by that prior judgment. '
                        'Only choose a supplied candidate ID; do not invent coordinates or decide Z. '
                        'The skill will check collision, workspace and held IK after your selection. '
                        'This is an operation proposal, not proof of free space or completed placement. '
                        'The whole region is already authorized. Several equally suitable candidates are normal; '
                        'For placement_preference=clearance, prefer greater measured_obstacle_gap_m up to '
                        'desired_clearance_m, then shorter travel. This helps separate successive placements. '
                        'For preference=reach, prefer shorter travel. A null gap supplies no separation evidence. '
                        'A failed candidate says nothing about the other positions. Use unknown only if the '
                        'supplied region or candidates conflict with the location requirement or cannot be interpreted, '
                        'not because collision or IK has not yet been tested.'))],
                    include_task=False, extra={'verification_scope': 'placement_xy_proposal',
                        'scene_token': controller.spatial_memory.scene['scene_token'],
                        'target_id': controller.state['target_id'],
                        'region_authorization': {'source': 'perceive_placement_zone', 'scope': 'entire_region',
                            'judgments': zone['semantic_evidence'].get('judgments', {}),
                            'request_hash': zone['semantic_evidence'].get('evidence', {}).get('request_hash')},
                        'current_transfer': zone.get('current_transfer'),
                        'placement_zone': region, 'subdivision_cells': level,
                        'placement_preference': request.get('placement_preference', zone['intent']['preference']),
                        'desired_clearance_m': ranking.get('desired_clearance_m'),
                        'xy_candidates': [{'candidate_id': name,
                            'tcp_world_xy_m': requests[i]['selected_reference_world_xy_m'],
                            'measured_obstacle_gap_m': gaps.get(tuple(requests[i]['selected_reference_world_xy_m'])),
                            'travel_xy_m': float(np.linalg.norm(
                                np.asarray(requests[i]['selected_reference_world_xy_m']) - current_xy))}
                            for i, name in zip(remaining, choices, strict=True)],
                        'previous_candidate_failures': failures})
                answer = judgment.get('judgments', {}).get('placement_xy', {})
                chosen = answer.get('value')
                if chosen not in choices:
                    return {'status': 'unknown', 'reason': 'placement_xy_selection_unknown', 'evidence': judgment}
                selection = {'status': 'success', 'scope': 'xy_before_z',
                    'candidate_ids': [chosen], 'evidence': judgment}
            else:
                # This pose is only a projection of the XY footprint on the support
                # plane. It is never a collision-checked endpoint or a plan input.
                pose = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'], float).copy()
                pose[2, 3] = zone['floor_z_m'] - (held_geometry_points(holding) @ pose[:3, :3].T)[:, 2].min()
                proposals = []
                for i in remaining:
                    proposed = pose.copy()
                    proposed[:2, 3] = requests[i]['selected_reference_world_xy_m']
                    proposals.append({'candidate_id': 'placement-' + str(i),
                        'goal': {'T_world_tcp': proposed.tolist()}, 'request': requests[i]})
                selected = await select_for_controller(controller, context, {'status': 'success', 'placement_zone': zone,
                    'candidates': proposals, 'selection_evidence': evidence}, publish=False)
                if selected.get('status') != 'success':
                    return selected
                selection = selected['semantic_selection']
            indices = [i for i in remaining if 'placement-' + str(i) in selection['candidate_ids']]
            try:
                result = await controller.call_algorithm(controller.algorithms.resolve_placement,
                    observation, controller.state['target_id'], controller.state['arm'], controller.spatial_memory.scene,
                    holding, controller.state['destination_id'], placement_request=request,
                    placement_zone=zone, planning_settings=context['planning_settings'],
                    subdivision_cells=level, candidate_indices=indices, report_progress=True)
            except AlgorithmError as error:
                if error.code != 'placement_no_candidate':
                    raise
                attempts.append({'subdivision_cells': level, 'candidate_indices': indices,
                    'reason': error.code, 'details': error.details})
                remaining = [i for i in remaining if i not in indices]
                continue
            if any(item['candidate_id'] not in selection['candidate_ids'] for item in result['candidates']):
                raise ValueError('placement_numerical_result_changed_selected_xy')
            result['semantic_selection'] = selection
            result['selection_evidence'].update(search_order='decisions_xy_then_numeric_z',
                subdivision_levels=[4, 8, 16] if not precise else [],
                maximum_xy_candidates=1 if precise else 28, previous_attempts=attempts)
            result.pop('resolution_id', None)
            result['resolution_id'] = digest(result)
            context['placement_resolution'] = result
            return result
    raise AlgorithmError('No selected XY passed the bounded placement search', code='placement_no_candidate',
        details={'attempts': attempts, 'maximum_xy_candidates': 1 if precise else 28,
            'whole_destination_infeasible': False,
            'effective_arguments': {'preference': request.get('placement_preference')}})
