"""Attachment evidence follows the uploaded CAD, OBB, or legacy surface binding."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_release_clearance import backend, request
from test_visible_surface_gpu_contract import measured_binding

from luna_mujoco import backend_builder
from luna_mujoco.backend_builder import CuroboToolError
from luna_mujoco.release_clearance import release_clearance_evidence
from luna_mujoco.resident_robot import ResidentRobotGeometry

COVERAGE = [
    (None, 'measured_surface_voxels_with_uncertainty'),
    ('obb_tool', 'observed_obb_volume_with_uncertainty'),
    ('cad_tool', 'registered_cad_volume_with_uncertainty'),
]


def held_geometry(kind, scene_token):
    fields = {'obb_tool': None, 'cad_tool': None, 'scene_token': scene_token}
    if kind:
        fields[kind] = {'instance_id': 'part'}
    return replace(measured_binding(), **fields)


@pytest.mark.parametrize('kind,coverage', COVERAGE)
def test_release_model_label_does_not_change_uploaded_spheres_or_world_cost(kind, coverage):
    spheres = [[.02, 0., -.05, .013]]
    seen = []
    def query(world):
        seen.append(world.copy())
        return np.array([[[.001]]])
    result = release_clearance_evidence(spheres, request(), query, scene_token='current',
        geometry_coverage=coverage)
    assert result['geometry_coverage'] == coverage
    assert result['scope'] == ('attached_surface_spheres_at_requested_release_goal' if kind is None
                               else 'attached_model_volume_spheres_at_requested_release_goal')
    assert result['conflict'] and result['world_collision_costs'] == [.001]
    assert result['full_object_coverage'] is False
    assert result['min_clearance_m'] is None
    np.testing.assert_allclose(seen[0][0, 0], [[.52, .2, .65, .013]])


@pytest.mark.parametrize('kind,coverage', COVERAGE)
def test_attachment_upload_propagates_geometry_label_into_release_check(monkeypatch, kind, coverage):
    # Geometry validation/fit has its own contract tests. This boundary receives
    # its result and must describe the exact uploaded float32 spheres truthfully.
    monkeypatch.setattr(backend_builder, 'carrying_spheres', lambda binding: [[.01, 0., 0., .01]])
    value = backend([[[.001]]])
    value._state = lambda joints: object()
    value.motion_planner.attachment_manager = SimpleNamespace(update=lambda *args, **kwargs: None)
    value.attach_measured_carrying(held_geometry(kind, value.carrying_scene_token), [0.] * 7)
    with pytest.raises(CuroboToolError) as error:
        value.plan_place_native(request())
    assert error.value.code == 'release_clearance_conflict'
    evidence = error.value.details
    assert evidence['geometry_coverage'] == coverage
    assert evidence['full_object_coverage'] is False


def test_resident_attach_switch_and_detach_refresh_geometry_coverage(monkeypatch):
    monkeypatch.setattr('luna_mujoco.resident_robot.carrying_geometry',
        lambda binding: {'spheres': [[0., 0., 0., .01]], 'fit_method': 'fixture'})
    value = SimpleNamespace(effective_config={'geometry_coverage': None})
    for kind, coverage in COVERAGE + [(None, 'measured_surface_voxels_with_uncertainty')]:
        ResidentRobotGeometry._evidence(value, held_geometry(kind, 'current'))
        assert value.effective_config['geometry_coverage'] == coverage
        assert value.carrying_geometry_coverage == coverage
    ResidentRobotGeometry._evidence(value, None)
    assert value.effective_config['geometry_coverage'] is None
    assert value.carrying_geometry_coverage is None
    assert value.effective_config['optimizer_attachment'] is False
