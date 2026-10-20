import copy

import numpy as np
from test_rgbd_algorithms import observed_scene

from jev_mujoco.algorithms import Algorithms
from jev_mujoco.perception_mapping import decode_mapping_payload


def test_painted_marker_keeps_depth_without_requiring_its_own_collision_volume():
    observation = observed_scene()
    observation['objects'].append({'id': 'marker-yellow', 'kind': 'marker'})
    for view in observation['views'].values():
        view['instance_masks'] = dict(view['instance_masks'], **{'marker-yellow': np.eye(8, dtype=bool)})
    calls = []

    def request(path, body):
        calls.append(copy.deepcopy(body))
        return {'ok': True, 'scene_token': 'map'}

    Algorithms(curobo_request=request).build_scene(observation, 'seen-red')
    decoded, _ = decode_mapping_payload(calls[0])
    for view in decoded['views'].values():
        assert 'marker-yellow' not in view['instance_masks']
        assert 'seen-red' in view['instance_masks']
        assert np.count_nonzero(view['depth_left_m']) > 0
