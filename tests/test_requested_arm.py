import numpy as np
import pytest
from test_algorithms import snapshot

from luna_mujoco.algorithms import Algorithms


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_explicit_arm_sends_only_that_arm_to_gpu(arm):
    calls = []

    def request(path, body):
        calls.append(body['arm'])
        return {'ok': True, 'scene_token': 'test', 'batch_size': 1,
                'items': [{'candidate_index': 0, 'success': True}]}

    result = Algorithms(curobo_request=request).evaluate_arms(
        snapshot(), 'red', {'candidates': [{'index': 17, 'score': .9,
        'T_object_grasp': np.eye(4).tolist()}]}, {'scene_token': 'test', 'target_id': 'red',
        'generation': 1, 'observation_id': 'capture-4'}, requested_arm=arm)
    assert calls == [arm]
    assert set(result) == {arm}
