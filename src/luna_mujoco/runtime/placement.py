"""Placement uses compiled task permission and measured fixed-goal geometry."""
from collections import Counter
from copy import deepcopy

from ..placement import observed_object
from ..robot_skills.placement_resolution import digest
from ..robot_skills.placement_zone import perceive_placement_zone


def invalidate_placement(controller, context, *, zone=False):
    keys = ['placement_resolution', 'place_plan', 'place_plan_binding', 'place_plan_observation_id']
    controller.state.setdefault('facts', {}).update(place_plan_ready=False)
    if zone:
        keys.extend(['placement_zone', 'placement_candidate_view'])
        controller.state['facts']['placement_zone_ready'] = False
    for key in keys:
        context.pop(key, None)


def declared_destination_binding(controller, observation, target, destination_id, local_task):
    """Reuse the compiled task relation and the current measured instance binding."""
    record = controller.task_progress.records[target]
    root = record.get('root_input_id', target)
    coverage = observation.get('visual_identity', {}).get('identity_coverage', {})
    destination = observed_object(observation, destination_id)
    role = ('final_destination' if destination_id == local_task.get('destination_ref') else
            'declared_work_area' if destination_id == local_task.get('work_area_ref') else
            'permitted_intermediate' if destination_id in local_task.get('allowed_intermediate_refs', []) else None)
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


async def perceive_for_controller(controller, context):
    invalidate_placement(controller, context, zone=True)
    target, destination_id = controller.state['target_id'], controller.state['destination_id']
    observation = context['observation']
    destination = observed_object(observation, destination_id)
    record = controller.task_progress.records[target]
    if destination['kind'] == 'bin' and record['release_mode'] == 'container_release':
        from ..container_release import configure_release_region

        configure_release_region(destination, context['planning_settings'])
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
    confirmed = declared_destination_binding(controller, observation, target, destination_id, local)
    if not confirmed:
        return {'status': 'unknown', 'reason': 'compiled_destination_or_current_identity_unavailable'}
    if local.get('unsupported_placement_constraints'):
        return {'status': 'failure', 'reason': 'unsupported_fixed_placement_requirement'}
    grounding = {'source': 'compiled_task_and_current_identity', 'binding': confirmed, 'model_called': False}
    zone.update(semantic_scope='entire_region', semantic_evidence=grounding)
    context['placement_zone'] = zone
    record.update(placement_intent=deepcopy(zone['intent']))
    return {'status': 'success', 'intent': zone['intent'], 'placement_zone': zone,
            'semantic_scope': zone['semantic_scope'], 'evidence': grounding}


async def select_for_controller(controller, context, resolution, *, publish=True):
    """Bind semantics to numerical candidates before making a resolution executable."""
    zone = resolution['placement_zone']
    if zone['semantic_scope'] != 'entire_region':
        return {'status': 'failure', 'reason': 'placement_semantics_not_compiled'}
    selection = {'status': 'success', 'scope': 'entire_region',
        'candidate_ids': [candidate['candidate_id'] for candidate in resolution['candidates']],
        'evidence': zone['semantic_evidence']}
    accepted = set(selection['candidate_ids'])
    result = deepcopy(resolution)
    result['candidates'] = [candidate for candidate in result['candidates'] if candidate['candidate_id'] in accepted]
    result['semantic_selection'] = selection
    result.pop('resolution_id', None)
    result['resolution_id'] = digest(result)
    if publish:
        context['placement_resolution'] = result
    return result


async def prepare_fixed_placement(controller, context, observation, request):
    """Bind the operator's fixed TCP XY and measured-support relative height and validate semantic permission."""
    result = await controller.call_algorithm(controller.algorithms.prepare_fixed_placement,
        observation, controller.state['target_id'], controller.state['arm'], controller.spatial_memory.scene,
        context['holding'], controller.state['destination_id'], placement_request=request,
        placement_zone=context['placement_zone'], planning_settings=context['planning_settings'], report_progress=True)
    return await select_for_controller(controller, context, result)
