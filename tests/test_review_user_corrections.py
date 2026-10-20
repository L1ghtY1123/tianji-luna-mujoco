import copy
import json
from types import SimpleNamespace

import pytest
from test_geometry_service import Runtime, request_payload

from jev_mujoco import geometry_service
from jev_mujoco.decisions import verification_evidence
from jev_mujoco.planning_settings import PlanningSettings


@pytest.mark.parametrize('count', [1, 49, 72, 100])
def test_shared_ik_uses_one_fixed_capacity_and_returns_only_real_goals(count):
    class BatchRuntime(Runtime):
        def solve_ik_batch_native(self, request, lock_timeout_s):
            assert len(request.goal_matrices_world) == 100
            assert all(goal == request.goal_matrices_world[count - 1]
                       for goal in request.goal_matrices_world[count:])
            return SimpleNamespace(to_payload=lambda: {
                'ok': True, 'scene_token': 'scene', 'batch_size': 100, 'batch_capacity': 100,
                'items': [{'candidate_index': i, 'success': i % 2 == 0} for i in range(100)]})

    payload = request_payload(PlanningSettings())
    payload['goal_matrices_world'] *= count
    result = geometry_service.run_operation(BatchRuntime(), '/tools/solve_ik_batch', payload)
    assert result['batch_size'] == count and result['batch_capacity'] == 100
    assert len(result['items']) == count
    assert [item['candidate_index'] for item in result['items']] == list(range(count))


def test_wrapped_cuda_oom_is_a_resource_error_not_a_no_solution():
    error_type = type('OutOfMemoryError', (RuntimeError,), {'__module__': 'torch'})
    try:
        try:
            raise error_type('CUDA out of memory. private process details')
        except RuntimeError as cause:
            raise RuntimeError('cuRobo backend place batch failed') from cause
    except RuntimeError as error:
        status, payload = geometry_service.operation_error_response(error)
    assert status == 503
    assert payload['error']['code'] == 'CUROBO_CUDA_OUT_OF_MEMORY'
    assert 'private process details' not in json.dumps(payload)


def test_verification_summary_preserves_visual_unknown_without_attachment_arrays():
    evidence = {'success': False, 'status': 'unknown', 'reason': 'visual_destination_unconfirmed',
                'open_confirmed': True, 'visual_evidence': {
                    'status': 'unknown', 'source': 'openai_decisions', 'observation_id': 'current'},
                'attachment': {'attached': False, 'attachment_spheres_tool_m': [[0., 0., 0., .01]] * 64}}
    original = copy.deepcopy(evidence)
    result = verification_evidence(evidence)
    assert evidence == original
    assert len(json.dumps(result)) < 2500
    assert result['status'] == 'unknown' and result['open_confirmed'] is True
    assert result['visual_evidence'] == evidence['visual_evidence']
    assert result['attachment'] == {'attached': False}
    assert verification_evidence(result) == result
