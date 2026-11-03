"""Instance ownership, camera frames, motion invalidation, and mesh centering."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_cached_foundationpose import CachedFoundationPoseBackend, backend_fixture
from test_cad_perception import BoxCatalog, Match, Pose, fixture

from luna_mujoco.cad_perception import CadGeometryResolver
from luna_mujoco.pose_estimation import CadPoseEstimator


@pytest.mark.parametrize('change, tracked', [('none', True), ('motion', False), ('unknown', False), ('generation', False)])
def test_current_pose_seed_is_same_instance_and_invalidated_by_manipulation(change, tracked):
    capture, group, target, pose = fixture()
    capture['cad_motion_context'] = {target['id']: {'status': 'observed'}}
    calls = []

    class RecordingPose(Pose):
        def estimate_instance(self, *args, **kwargs):
            calls.append(deepcopy(kwargs))
            return super().estimate_instance(*args, **kwargs)

    resolver = CadGeometryResolver(BoxCatalog(), Match('box'), RecordingPose(pose, 'source-hash'))
    resolver.resolve(capture, group, target, instruction='move')
    capture['observation_id'], capture['time'] = 'next', 2.
    capture['views']['1'].update(frame_id='next', capture_time=2.)
    if change == 'motion':
        capture['cad_motion_context'][target['id']] = {'status': 'released', 'release_feedback_id': 'release-new'}
    elif change == 'unknown':
        capture.pop('cad_motion_context')
    elif change == 'generation':
        capture['generation'] = 3
        capture['views']['1']['simulation_generation'] = 3
    resolver.resolve(capture, group, target, instruction='move')
    assert ('tracking_seed' in calls[-1]) is tracked
    if tracked:
        assert calls[-1]['tracking_seed']['instance_id'] == target['id']
        assert calls[-1]['tracking_seed']['observation_id'] == 'observation'
        np.testing.assert_allclose(calls[-1]['tracking_seed']['T_world_object'], pose)
    target['id'] = group[0]['entity_id'] = 'another-box'
    resolver.resolve(capture, group, target, instruction='move')
    assert 'tracking_seed' not in calls[-1]


def test_previous_world_pose_is_transformed_into_current_camera_frame():
    capture, group, _, pose = fixture()
    view = capture['views']['1']
    view['T_world_camera'][:3, 3] = [.2, -.1, .4]
    calls = []

    def track(*args, seed, **kwargs):
        calls.append((args, seed, kwargs))
        return {'capture_id': view['frame_id'], 'camera_id': view['camera_id'],
                'T_camera_object': seed['T_camera_object'], 'T_source_canonical_m': np.eye(4)}

    estimator = CadPoseEstimator(pose_client=SimpleNamespace(track=track))
    seed = {'T_world_object': pose, 'instance_id': 'box-1', 'source_sha256': 'source', 'observation_id': 'previous'}
    result = estimator.estimate_instance(view, group[0]['mask'], .9, model_id='box',
        mask_capture_id=view['frame_id'], mask_camera_id=view['camera_id'], tracking_seed=seed)
    np.testing.assert_allclose(result['T_world_object'], pose)
    np.testing.assert_allclose(np.asarray(calls[0][1]['T_camera_object'])[:3, 3], [-.2, .1, .6])
    np.testing.assert_array_equal(calls[0][0][0], view['rgb'])


def test_shared_model_resets_explicit_instance_seed_and_preserves_centered_mesh_frame(tmp_path):
    base, request, calls, _ = backend_fixture(tmp_path)
    transform = np.eye(4)
    transform[:3, 3] = [.01, -.02, .03]
    base.torch.as_tensor = lambda value, **kwargs: np.asarray(value)
    base.torch.float32 = np.float32
    base.torch.linalg = SimpleNamespace(inv=np.linalg.inv)

    class Estimator(base.estimator_class):
        def get_tf_to_centered_mesh(self):
            return transform

        def track_one(self, **kwargs):
            calls.append(('track', self.pose_last.copy()))
            assert kwargs['iteration'] == 2
            return self.pose_last @ transform

    base.estimator_class = Estimator
    backend = CachedFoundationPoseBackend(base)
    try:
        for identifier, x in [('left', .3), ('right', -.4), ('left', .35)]:
            pose = np.eye(4)
            pose[0, 3] = x
            result = backend.estimate(request, tracking_seed={'instance_id': identifier,
                'source_sha256': 'source', 'observation_id': 'previous', 'T_camera_object': pose})
            np.testing.assert_allclose(result['T_camera_object'], pose, atol=1e-9)
            np.testing.assert_allclose(calls[-1][1], pose @ np.linalg.inv(transform))
            assert result['estimation_mode'] == 'track_one'
        assert sum(row[0] == 'init' for row in calls) == 1
        assert not any(row[0] == 'register' for row in calls)
    finally:
        backend.close()
