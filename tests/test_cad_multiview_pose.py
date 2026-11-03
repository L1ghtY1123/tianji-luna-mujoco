"""One permanent instance/model, independently estimated views, one verified pose."""
from copy import deepcopy

import numpy as np
import pytest
from test_cad_perception import BoxCatalog, Match, Pose, fixture, cad_resolver


def multiview():
    capture, group, target, pose = fixture()
    for camera in ('2', '3'):
        capture['views'][camera] = deepcopy(capture['views']['1'])
        capture['views'][camera]['camera_id'] = camera
        group.append(deepcopy(group[0]) | {'camera_id': camera})
    return capture, group, target, pose


class ViewPoses(Pose):
    def __init__(self, pose, failing=(), unavailable=()):
        super().__init__(pose, 'source-hash')
        self.failing, self.unavailable, self.calls = failing, unavailable, []

    def estimate_instance(self, view, mask, score, **kwargs):
        camera = view['camera_id']
        self.calls.append(camera)
        if camera in self.unavailable:
            raise RuntimeError('service unavailable')
        result = super().estimate_instance(view, mask, score, **kwargs)
        if camera in self.failing:
            bad = np.asarray(result['T_world_object'])
            bad[0, 3] += .08
            result.update(T_world_object=bad.tolist(), T_world_source=bad.tolist())
        return result


def test_best_depth_view_pose_is_used_directly_without_multiview_refinement():
    capture, group, target, pose = multiview()
    estimator = ViewPoses(pose, failing={'1'})
    result = cad_resolver(BoxCatalog(), Match('box'), estimator).resolve(
        capture, group, target, instruction='move object')
    assert estimator.calls == ['1']
    cad = result['cad']
    assert cad['instance_id'] == target['id'] and cad['asset_id'] == 'box'
    assert cad['pose_evidence']['camera_id'] == '1'
    assert cad['pose_validation']['status'] == 'accepted_by_policy'
    expected = pose.copy()
    expected[0, 3] += .08
    np.testing.assert_array_equal(cad['T_world_object'], expected)


def test_successful_main_view_does_not_run_two_more_pose_estimates():
    capture, group, target, pose = multiview()
    estimator = ViewPoses(pose)
    result = cad_resolver(BoxCatalog(), Match('box'), estimator).resolve(
        capture, group, target, instruction='move object')
    assert estimator.calls == ['1']
    assert result['cad']['view_estimation']['selected_camera_id'] == '1'


def test_service_error_is_not_relabelled_as_a_bad_view_or_silently_retried():
    capture, group, target, pose = multiview()
    estimator = ViewPoses(pose, unavailable={'1'})
    with pytest.raises(ValueError, match='cad_pose_estimation_failed'):
        cad_resolver(BoxCatalog(), Match('box'), estimator).resolve(
            capture, group, target, instruction='move object')
    assert estimator.calls == ['1']


def test_new_observation_uses_new_pose_instead_of_reusing_previous_pose():
    capture, group, target, pose = fixture()
    estimator = ViewPoses(pose)
    resolver = cad_resolver(BoxCatalog(), Match('box'), estimator)
    resolver.resolve(capture, group, target, instruction='move object')
    capture['observation_id'] = 'next'
    capture['time'] = 2.
    capture['views']['1'].update(frame_id='next', capture_time=2.)
    estimator.failing = {'1'}
    result = resolver.resolve(capture, group, target, instruction='move object')
    cad = result['cad']
    assert cad['observation_id'] == 'next'
    assert cad['pose_validation']['accepted'] is True
    expected = pose.copy()
    expected[0, 3] += .08
    np.testing.assert_array_equal(cad['T_world_object'], expected)
    assert estimator.calls == ['1', '1']


def test_same_cad_for_two_ids_caches_independent_model_bindings_and_current_poses():
    capture, group, target, pose = fixture()
    estimator = ViewPoses(pose)
    resolver = cad_resolver(BoxCatalog(), Match('box'), estimator)
    first = resolver.resolve(capture, group, target, instruction='move objects')
    other = deepcopy(target) | {'id': 'object-2'}
    other_group = [deepcopy(group[0]) | {'entity_id': 'object-2'}]
    second = resolver.resolve(capture, other_group, other, instruction='move objects')
    assert first['cad']['asset_id'] == second['cad']['asset_id'] == 'box'
    assert first['cad']['instance_id'] != second['cad']['instance_id']
    assert set(resolver.confirmed_models) == {(2, 'object-1'), (2, 'object-2')}
    assert estimator.calls == ['1', '1']
    constraints = resolver.identity_constraints(2)
    assert set(constraints) == {'object-1', 'object-2'}
    assert constraints['object-1']['asset_id'] == constraints['object-2']['asset_id']
    assert 'T_world_object' not in constraints['object-1']
    assert resolver.identity_constraints(3) == {}
