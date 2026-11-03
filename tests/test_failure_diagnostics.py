from luna_mujoco.runtime.execution_state import PolicyStopped


def test_incomplete_inventory_map_reports_stale_geometry_without_guessing_visibility():
    from luna_mujoco.runtime.diagnostics import build_failure_report

    report = build_failure_report('inventory_map_incomplete', {
        'component': 'perception_mapping', 'missing_entity_ids': [],
        'stale_entity_ids': ['product-1'], 'observation_id': 'capture-2'}, {})
    assert '地图' in report['summary']
    assert any('product-1' in fact for fact in report['known_facts'])
    assert 'Decisions' in report['next_step']
    assert not any('遮挡导致' in fact for fact in report['known_facts'])


def test_cuda_memory_failure_preserves_allocator_evidence_separately_from_planning():
    from luna_mujoco.runtime.diagnostics import build_failure_report
    report = build_failure_report('CUROBO_CUDA_OUT_OF_MEMORY', {
        'resource': 'cuda_memory', 'allocator_message': 'Tried to allocate 626 MiB; 334 MiB free'},
        {'current_step': {'skill': 'build_scene'}})
    assert report['component'] == 'curobo_resource'
    assert report['stage'] == 'build_scene'
    assert '显存不足' in report['summary']
    assert report['checks'][0]['actual'] == 'Tried to allocate 626 MiB; 334 MiB free'


def test_world_collision_is_not_reported_as_diagnostic_coverage_rejection():
    from luna_mujoco.runtime.diagnostics import build_failure_report
    collision = {'link': 'gripL_Link3_L', 'required_clearance_m': .01,
                 'public_collision_cost': .002, 'cost_is_physical_penetration_depth': False}
    report = build_failure_report('planning_no_valid_candidate', {'failures': [{
        'candidate_index': 0, 'stage': 'grasp', 'code': 'execution_collision', 'details': {
            'first_world_collision': collision, 'unknown_space_motion_gate': False,
            'observation_coverage': [{'covered': False, 'first_unobserved_link': 'TCP_Link_L'}]}}]}, {})
    world = next(check for check in report['checks'] if check['name'] == 'world_collision')
    coverage = next(check for check in report['checks'] if check['name'] == 'observed_motion_coverage')
    assert world['status'] == 'failed' and world['actual'] == collision
    assert coverage['status'] == 'diagnostic'
    assert not any('覆盖拒绝' in value for value in report['unknowns'])


def test_policy_stop_preserves_actual_failure_and_coverage_evidence():
    from luna_mujoco.runtime.diagnostics import failure_record
    coverage = {'covered': False, 'first_unobserved_sample_index': 0,
                'first_unobserved_sphere_index': 2,
                'first_unobserved_sphere_world_m': [0, .16, 1.12, .058],
                'minimum_capture_containment_gap_m': 1.184e-9}
    error = PolicyStopped('stop as declared', {'status': 'failure',
        'error_code': 'planning_no_valid_candidate', 'details': {'arm': 'B', 'failures': [
            {'candidate_index': 169, 'stage': 'command_validation',
             'code': 'execution_observation_coverage_insufficient',
             'details': {'observation_coverage': [coverage]}},
            {'candidate_index': 10, 'stage': 'approach', 'status': 'IK_FAIL'}]}})
    record = failure_record(error, {'current_step': {'skill': 'plan_grasp'}, 'target_id': 'parcel'})
    assert record['error_code'] == 'planning_no_valid_candidate'
    assert record['policy_failure']['code'] == 'policy_stopped'
    report = record['failure_report']
    assert report['component'] == 'motion_planning'
    assert report['stage'] == 'plan_grasp'
    assert report['checks'][0]['candidate_id'] == 169
    assert report['checks'][0]['actual']['minimum_capture_containment_gap_m'] == 1.184e-9
    assert 'SAM3' not in report['summary']
    assert len(report['failure_counts']) == 2


def test_camera_depth_reason_and_lift_bounds_survive_report_preview():
    from luna_mujoco.algorithms import AlgorithmError
    from luna_mujoco.runtime.diagnostics import failure_record
    camera = {'camera': 'front', 'reason': 'first_return_before_required_depth',
              'minimum_depth_m': .5, 'required_free_depth_m': 1.03}
    report = failure_record(AlgorithmError('coverage', code='execution_observation_coverage_insufficient',
        details={'observation_coverage': [{'covered': False, 'first_unobserved_link': 'Link1_L',
                                          'camera_certificates': [camera]}]}), {})['failure_report']
    assert report['checks'][0]['actual']['first_unobserved_link'] == 'Link1_L'
    assert report['checks'][1]['actual'] == camera
    report = failure_record(AlgorithmError('source region', code='grasp_lift_geometry_invalid',
        details={'stage': 'source_clearance_lift', 'reason': 'grasp_lift_target_outside_source_region',
                 'target_centroid_world_m': [.4, .2, .6]}), {})['failure_report']
    assert report['component'] == 'motion_planning'
    assert report['checks'][0]['actual']['target_centroid_world_m'] == [.4, .2, .6]


def test_transport_reach_failure_reports_candidate_distances_and_limit():
    from luna_mujoco.algorithms import AlgorithmError
    from luna_mujoco.runtime.diagnostics import failure_record
    error = AlgorithmError('no reachable grasp', code='grasp_no_transport_reachable_candidate', details={
        'rounds': [{'sampling_round': 1, 'transport_reach_filter': {
            'input_count': 176, 'accepted_count': 0, 'reach_screening_limit_m': .597,
            'candidate_evidence': [{'nominal_transport_wrist_distance_m': .76},
                                   {'nominal_transport_wrist_distance_m': .83}]}}]})
    report = failure_record(error, {})['failure_report']
    assert report['component'] == 'motion_planning'
    assert report['checks'][0]['actual']['candidate_wrist_distance_range_m'] == [.76, .83]
    assert report['checks'][0]['actual']['reach_screening_limit_m'] == .597
    assert not any('扫掠体积' in fact for fact in report['known_facts'])
    assert not any('覆盖拒绝' in fact for fact in report['unknowns'])


def test_visual_exception_evidence_is_preserved_without_speculative_reason():
    from luna_mujoco.runtime.diagnostics import failure_record
    from luna_mujoco.visual_api import VisualInventoryError
    error = VisualInventoryError('visual_identity_unknown')
    error.evidence = {'component': 'decisions', 'observation_id': 'capture-1',
        'request_hash': 'hash', 'diagnostic_checks': [
            {'name': 'identity', 'question': 'Which entity is C0?', 'actual': .59,
             'threshold': .8, 'comparison': '>=', 'candidate_id': 'C0', 'camera_id': '1'}]}
    record = failure_record(error, {'current_step': {'skill': 'observe_scene'}})
    report = record['failure_report']
    assert report['request_hash'] == 'hash'
    assert report['checks'][0]['actual'] == .59
    assert report['checks'][0]['threshold'] == .8
    assert report['unknowns']
    assert record['error_details'] == error.evidence


def test_sam_no_candidates_does_not_invent_score_or_internal_reason():
    from luna_mujoco.runtime.diagnostics import DiagnosticError, failure_record
    record = failure_record(DiagnosticError('sam3_no_match', {
        'component': 'sam3', 'diagnostic_checks': [{'name': 'candidate_count', 'actual': 0}],
        'unknowns': ['Sidecar did not return its internal filter threshold.']}), {})
    report = record['failure_report']
    assert report['component'] == 'sam3'
    assert 'threshold' not in report['checks'][0]
    assert report['unknowns'] == ['Sidecar did not return its internal filter threshold.']


def test_policy_stop_without_failed_skill_is_not_relabelled_as_model_failure():
    from luna_mujoco.runtime.diagnostics import failure_record
    record = failure_record(PolicyStopped('task is unsupported', {'success': True}), {})
    assert record['error_code'] == 'policy_stopped'
    assert record['failure_report']['component'] == 'policy_runtime'


def test_verification_unknown_keeps_top_level_visual_evidence():
    from luna_mujoco.runtime.diagnostics import failure_record
    result = {'success': False, 'status': 'unknown', 'reason': 'unpack_result_unknown',
        'observation_id': 'obs', 'request_hash': 'request', 'visual_evidence': {
            'component': 'decisions', 'diagnostic_checks': [
                {'name': 'unpack_result', 'actual': .6, 'threshold': .8}]}}
    report = failure_record(PolicyStopped('stop', result), {})['failure_report']
    assert report['component'] == 'decisions'
    assert report['request_hash'] == 'request'
    assert report['checks'][0]['actual'] == .6


def test_geometry_failure_after_positive_visual_judgment_is_not_model_failure():
    from luna_mujoco.runtime.diagnostics import failure_record
    result = {'success': False, 'reason': 'unpacked_object_outside_execution_workspace',
        'visual_evidence': {'component': 'decisions'},
        'workspace_evidence': {'reachable': False, 'feasible_arms': []}}
    report = failure_record(PolicyStopped('stop', result), {})['failure_report']
    assert report['component'] == 'motion_planning'
    assert report['checks'][0]['actual']['reachable'] is False


def test_judgment_failure_preserves_raw_choice_and_configured_gate():
    from types import SimpleNamespace

    from luna_mujoco.runtime.diagnostics import judgment_details
    check = {'name': 'arm', 'answer': {'choice': 'B', 'confidence': .79},
             'actual': .79, 'threshold': .85, 'metric': 'decisions_confidence'}
    result = {'evidence': {'diagnostic_checks': [check]},
              'judgments': {'arm': {'value': 'unknown', 'status': 'unknown', 'confidence': .79}}}
    details = judgment_details(result, [SimpleNamespace(name='arm', instructions='Choose arm')])
    assert details['diagnostic_checks'][0]['answer']['choice'] == 'B'
    assert details['diagnostic_checks'][0]['threshold'] == .85
