"""Project physical outcome evidence for graph and semantic consumers."""
from collections import Counter
from copy import deepcopy


def scalar_fields(record):
    return {key: value for key, value in record.items()
            if value is None or isinstance(value, (str, bool, int, float))}


def grasp_search_evidence(evidence):
    if isinstance(evidence.get('grasp_search'), dict):
        return deepcopy(evidence['grasp_search'])
    sampling = evidence.get('sampling_evidence', evidence)
    if not isinstance(sampling, dict) or not isinstance(sampling.get('rounds'), list):
        return None
    rounds = []
    for item in sampling['rounds']:
        row = {key: deepcopy(item[key]) for key in ('sampling_round', 'tested_count', 'valid_count', 'outcome',
            'source_ik_success_count', 'candidate_filter_counts') if key in item}
        failures = []
        for failure in item.get('planner_failure', {}).get('failures', []):
            detail = failure.get('details') or {}
            record = {key: deepcopy(failure[key]) for key in
                ('candidate_index', 'stage', 'code', 'status', 'phase_success') if key in failure}
            for key in ('first_world_collision', 'closure_collision', 'opposing_contact_predicted', 'pads',
                        'driver_limit_rad', 'contact_probe_rad'):
                if key in detail:
                    record[key] = deepcopy(detail[key])
            failures.append(record)
        row['failures'] = failures
        rounds.append(row)
    return {'arm': sampling.get('arm'), 'target_id': sampling.get('target_id'),
            'scene_token': sampling.get('scene_token'), 'rounds': rounds,
            'total_tested_count': sampling.get('total_tested_count'),
            'scope': 'bounded_candidates_with_current_arm_geometry_and_settings'}


def placement_search_evidence(evidence):
    """Keep the limits and numerical rejection counts of a bounded XY search."""
    existing = evidence.get('placement_search')
    if isinstance(existing, dict):
        return deepcopy(existing)
    attempts = evidence.get('attempts')
    direct_candidates = evidence.get('candidate_failures')
    resolved_path = not isinstance(attempts, list) and isinstance(direct_candidates, list)
    if resolved_path:
        attempts = [{'details': evidence}]
    if not isinstance(attempts, list):
        return None
    failures = []
    generation_failures = Counter()
    for attempt in attempts:
        detail = attempt.get('details') or {}
        candidates = detail.get('candidate_failures', [])
        failures.extend(candidates)
        if not candidates:
            generation_failures[attempt.get('reason', 'unknown')] += 1
    counts = Counter(row.get('code', 'unknown') for row in failures)
    phases = Counter((row.get('code', 'unknown'), (row.get('details') or {}).get('compound_phase'),
                      (row.get('details') or {}).get('status')) for row in failures
                     if (row.get('details') or {}).get('compound_phase'))
    result = {
        'scope': ('resolved_pose_path_search' if resolved_path
                  else 'sampled_xy_with_current_grasp_orientation_and_scene'),
        'attempt_count': len(attempts), 'tested_candidate_count': len(failures),
        'maximum_xy_candidates': evidence.get('maximum_xy_candidates'),
        'whole_destination_infeasible': evidence.get('whole_destination_infeasible'),
        'failure_counts': [{'code': code, 'count': count} for code, count in counts.items()],
        'candidate_generation_failures': [{'code': code, 'count': count}
                                          for code, count in generation_failures.items()],
    }
    if resolved_path:
        result['resolution_id'] = evidence.get('resolution_id')
    if phases:
        result['phase_failures'] = [{'code': code, 'phase': phase, 'sdk_status': status, 'count': count}
                                   for (code, phase, status), count in phases.items()]
    return result


def outcome_state(result):
    if not isinstance(result, dict):
        return result
    # Result-level IDs, status and reason belong to the skill's public outcome.
    # Free-form exception messages can contain serialized candidate trajectories.
    summary = scalar_fields(result)
    message = summary.pop('message', None)
    if isinstance(message, str) and result.get('success') is False:
        summary['message'] = message.splitlines()[0][:480] if message else ''
    if isinstance(result.get('arguments'), dict):
        summary['arguments'] = deepcopy(result['arguments'])
    if isinstance(result.get('partial_execution'), dict):
        summary['partial_execution'] = scalar_fields(result['partial_execution'])
    evidence = result.get('details') or result.get('evidence') or result.get('visual_evidence') or {}
    if not isinstance(evidence, dict):
        return summary
    fields = ('component', 'stage', 'purpose', 'reason', 'error_code', 'status', 'arm',
              'target_id', 'destination_id', 'observation_id', 'request_hash', 'retryable',
              'binding_id', 'release_feedback_id', 'region_id', 'layout_revision')
    details = {key: deepcopy(evidence[key]) for key in fields if key in evidence}
    if isinstance(evidence.get('camera_errors'), list):
        details['camera_errors'] = [scalar_fields(row) for row in evidence['camera_errors'] if isinstance(row, dict)]
        details['synchronization'] = scalar_fields(evidence.get('synchronization', {}))
        if 'attempts' in evidence:
            details['attempts'] = evidence['attempts']
    if isinstance(evidence.get('settling'), dict):
        details['settling'] = scalar_fields(evidence['settling'])
        for key in ('closed_command', 'elapsed_s', 'position_rad', 'command_position_rad'):
            if key in evidence:
                details[key] = deepcopy(evidence[key])
    geometry_issue = evidence.get('placement_geometry_issue')
    if isinstance(geometry_issue, dict):
        details['placement_geometry_issue'] = {key: deepcopy(geometry_issue[key]) for key in (
            'reason', 'normal_world', 'minimum_up_component', 'observation_id', 'generation', 'meaning')
            if key in geometry_issue}
    for key in ('known_facts', 'unknowns', 'missing_entity_ids', 'stale_entity_ids', 'unresolved_membership'):
        if key in evidence:
            details[key] = deepcopy(evidence[key])
    if evidence.get('unassociated_candidates'):
        details['unassociated_candidates'] = [{**scalar_fields(row), 'answer': deepcopy(row.get('answer'))}
                                             for row in evidence['unassociated_candidates']]
    if evidence.get('unresolved_proposals'):
        details['unresolved_proposals'] = [{key: deepcopy(row[key]) for key in ('proposal', 'status', 'reason')
                                           if key in row} for row in evidence['unresolved_proposals']]
    judgments = evidence.get('judgments', result.get('judgments'))
    if isinstance(judgments, dict):
        details['judgments'] = {name: scalar_fields(answer) if isinstance(answer, dict) else answer
                                for name, answer in judgments.items()}
    checks = []
    for check in evidence.get('diagnostic_checks', []):
        row = scalar_fields(check)
        for key in ('answer', 'actual', 'expected'):
            if isinstance(check.get(key), dict):
                row[key] = scalar_fields(check[key])
        checks.append(row)
    if checks:
        details['diagnostic_checks'] = checks
    failures = evidence.get('failures', [])
    # This projection is applied again when building a router's question. Already
    # projected evidence must retain the same meaning on the second pass.
    if isinstance(evidence.get('failure_counts'), list):
        details['failure_counts'] = deepcopy(evidence['failure_counts'])
    if isinstance(failures, list) and failures:
        counts = Counter((row.get('stage'), row.get('code')) for row in failures)
        details['failure_counts'] = [{'stage': stage, 'code': code, 'count': count}
                                    for (stage, code), count in counts.items()]
    placement_search = placement_search_evidence(evidence)
    if placement_search is not None:
        details['placement_search'] = placement_search
    grasp_search = grasp_search_evidence(evidence)
    if grasp_search is not None:
        details['grasp_search'] = grasp_search
    pool = evidence.get('pool_exhaustion')
    if isinstance(pool, dict):
        details['pool_exhaustion'] = {key: pool[key] for key in (
            'scope', 'arm', 'target_id', 'planned_candidate_count', 'sampling_round', 'outcome',
            'candidate_indices_count', 'unplanned_candidate_indices_count') if key in pool}
        for key in ('candidate_indices', 'unplanned_candidate_indices'):
            if key in pool:
                details['pool_exhaustion'][key + '_count'] = len(pool[key])
    summary['details'] = details
    return summary


