"""Production uses the current FoundationPose result without a second registration."""

import pytest
from test_cad_multiview_pose import multiview, ViewPoses
from test_cad_perception import BoxCatalog, Match, Pose, api, fixture


def test_direct_pose_is_current_and_no_cpu_geometry_validation_runs(monkeypatch):
    capture, group, target, pose = multiview()
    estimator = ViewPoses(pose, failing={'1'})
    resolver = api().CadGeometryResolver(BoxCatalog(), Match('box'), estimator)
    result = resolver.resolve(capture, group, target, instruction='move object')
    assert estimator.calls == ['1']
    assert result['cad']['pose_validation']['status'] == 'accepted_by_policy'
    assert result['cad']['T_world_object'][0][3] == pytest.approx(.08)
    assert 'full_max_residual_m' not in result['cad']['pose_validation']['surface']
    capture['observation_id'] = 'next'
    capture['views']['1']['frame_id'] = 'next'
    capture['views']['2']['frame_id'] = 'next'
    capture['views']['3']['frame_id'] = 'next'
    resolver.resolve(capture, group, target, instruction='move object')
    assert estimator.calls == ['1', '1']


def test_direct_pose_still_requires_matching_source_and_current_capture():
    capture, group, target, pose = fixture()
    with pytest.raises(ValueError, match='cad_pose_provenance_mismatch'):
        api().CadGeometryResolver(BoxCatalog(), Match('box'), Pose(pose, 'wrong')).resolve(
            capture, group, target, instruction='move object')


def test_direct_pose_attachment_does_not_require_measured_residual():
    from luna_mujoco.carrying import _cad_residual
    capture, group, target, pose = fixture()
    result = api().CadGeometryResolver(BoxCatalog(), Match('box'), Pose(pose, 'source-hash')).resolve(
        capture, group, target, instruction='move object')
    assert _cad_residual(result['cad']['pose_validation'], len(target['points_world_m'])) == 0
