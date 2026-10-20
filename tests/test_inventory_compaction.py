import json
from copy import deepcopy

from jev_mujoco.decisions import compact_algorithm_result, compact_decision_state


def test_repeated_unknown_verification_is_referenced_without_losing_evidence():
    verification = {'status': 'unknown', 'reason': 'placement_not_visible',
        'observation_id': 'post-place', 'missing_ids': ['source'],
        'camera_attempts': [{'camera_id': camera, 'verification': {'status': 'unknown',
            'reason': 'identity_unresolved', 'candidate_ids': ['other']}} for camera in ('3', '2', '1')]}
    state = {'verification': verification, 'recent_results': [
        {'title': 'verify_place', 'data': {'result': deepcopy(verification), 'duration_s': 1.}}]}
    original = deepcopy(state)
    result = compact_decision_state(state, skill_selection=True)
    assert result['verification'] == verification
    assert result['recent_results'][0]['data'] == {'result_reference': 'verification', 'duration_s': 1.}
    assert state == original
    assert compact_decision_state(result, skill_selection=True) == result


def test_different_verification_history_keeps_its_own_evidence():
    state = {'verification': {'status': 'unknown', 'observation_id': 'new'},
        'recent_results': [{'title': 'verify_place', 'data': {
            'result': {'status': 'unknown', 'observation_id': 'old', 'reason': 'missing_depth'}}}]}
    result = compact_decision_state(state, skill_selection=True)
    assert result['recent_results'][0]['data']['result'] == state['recent_results'][0]['data']['result']


def test_context_compaction_preserves_ids_constraints_and_unknowns():
    objects = [{'id': f'part-{i}', 'display_id': i + 1, 'kind': 'cube', 'color': 'red',
                'status': 'completed' if i < 20 else 'observed', 'association_status': 'unknown',
                'visibility': 'unseen', 'visual_semantics': {'description': 'identical part ' * 20}}
               for i in range(30)]
    state = {'instruction': '左臂搬运 21 至 30，保留红色盒子', 'requested_arm': 'A',
             'task_progress': {'objects': objects, 'completed': objects[:20], 'deferred': [],
                 'unseen_ids': [o['id'] for o in objects[20:]], 'unseen_objects': objects[20:],
                 'unseen_relevance': {'part-20': 'unknown'}}}
    original = deepcopy(state)
    result = compact_decision_state(state)
    assert result['instruction'] == state['instruction'] and result['requested_arm'] == 'A'
    assert result['task_progress']['objects'] == objects
    assert result['task_progress']['unseen_relevance'] == {'part-20': 'unknown'}
    assert [o['id'] for o in result['task_progress']['completed']] == [o['id'] for o in objects[:20]]
    assert len(json.dumps(result)) < .65 * len(json.dumps(state))
    assert state == original


def test_successful_algorithm_logs_do_not_repeat_geometry_in_skill_input():
    observation = {'observation_id': 'frame', 'objects': [{'id': 'one', 'position': [1, 2, 3]}]}
    state = {'observation': observation, 'recent_results': [
        {'title': 'observe_geometry', 'data': {'result': observation | {'views': {'pixels': list(range(5000))}}}},
        {'title': 'generate_grasps', 'data': {'result': {'candidates': {'count': 200, 'first': [list(range(1000))]}}}},
        {'title': 'build_scene', 'data': {'result': {'ok': True, 'scene_token': 'map', 'target_id': 'one',
            'source_restoration': {'one': {'restored': True}}, 'voxel_diagnostics': list(range(1000))}}}]}
    result = compact_decision_state(state, skill_selection=True)
    assert len(json.dumps(result)) < 1500
    assert result['observation'] == observation
    assert result['recent_results'][2]['data']['result']['scene_token'] == 'map'


def test_latest_assessment_reuses_identical_ledger_but_preserves_history():
    progress = {'ledger_revision': 4, 'objects': [{'id': 'inside', 'status': 'completed'}],
                'membership': {'inside': 'required', 'outside': 'excluded'}}
    event = {'title': 'assess_task', 'data': {'result': {
        'judgment': 'complete', 'task_progress': deepcopy(progress)}}}
    compact = compact_algorithm_result(event, {'task_progress': progress})
    assert compact['data']['result'] == {'judgment': 'complete', 'task_progress_reference': 'task_progress'}
    assert event['data']['result']['task_progress'] == progress
    # A historical assessment is evidence, not an alias to a later changed ledger.
    changed = {**progress, 'ledger_revision': 5}
    historical = compact_algorithm_result(event, {'task_progress': changed})
    assert historical['data']['result']['task_progress']['ledger_revision'] == 4


def test_historical_assessment_references_unchanged_evidence_but_keeps_unknown_history():
    objects = [{'id': f'part-{i}', 'status': 'observed', 'position': [.4, .2, .6],
                'measured_relationships': [{'object_id': 'bin', 'relation': 'above_observed_bin_floor',
                    'visible_xy_fraction': .6, 'complete_containment_verified': False}]} for i in range(8)]
    old = {'ledger_revision': 2, 'objects': objects, 'membership': {'part-0': 'unknown'},
           'failure': None}
    current = old | {'ledger_revision': 3, 'membership': {'part-0': 'required'},
                     'failure': {'status': 'unknown', 'code': 'task_progress_unknown'}}
    state = {'task_progress': current, 'recent_results': [
        {'title': 'assess_task', 'data': {'result': {'judgment': 'unknown', 'task_progress': old}}}]}
    result = compact_decision_state(state, skill_selection=True)
    history = result['recent_results'][0]['data']['result']
    assert history['judgment'] == 'unknown'
    assert history['task_progress']['ledger_revision'] == 2
    assert history['task_progress']['membership']['part-0'] == 'unknown'
    assert history['task_progress']['objects'] == {'evidence_reference': 'task_progress.objects'}
    assert result['task_progress']['objects'] == objects
    assert len(json.dumps(history)) < 450
