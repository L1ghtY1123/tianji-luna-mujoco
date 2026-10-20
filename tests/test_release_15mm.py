import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_region_placement import region_inputs
from test_world_yaw_placement import tilted_inputs

from jev_mujoco.placement import release_candidates, release_endpoint_evidence
from jev_mujoco.planning_settings import PlanningSettings, load_settings


@pytest.mark.parametrize('region', [True, False])
def test_eccentric_grip_keeps_tcp_xy_and_collision_threshold_gap(region):
    target, destination, holding, spheres = tilted_inputs()
    if region:
        _, measured, _, _ = region_inputs()
        destination.update(points_world_m=measured['points_world_m'])
        destination['placement_reference'].update(measured['placement_reference'])
    original = spheres.copy()
    goals = release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)
    assert len(goals) == 18
    grasp = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    for goal, degrees in zip(goals, range(0, 360, 20), strict=True):
        tcp = np.asarray(goal['T_world_tcp'])
        np.testing.assert_allclose((tcp @ [0., 0., .185, 1.])[:2], [.7, -.14 if region else -.2], atol=1e-12)
        expected_rotation = Rotation.from_euler('z', degrees, degrees=True).as_matrix() @ grasp[:3, :3]
        np.testing.assert_allclose(tcp[:3, :3], expected_rotation, atol=1e-12)
        relative_bottom = np.min((spheres[:, :3] @ expected_rotation.T)[:, 2] - spheres[:, 3])
        assert tcp[2, 3] == pytest.approx(.615 + .012 - relative_bottom)
        assert release_endpoint_evidence(goal, holding, tcp)['attachment_floor_gap_m'] == pytest.approx(.012)
        np.testing.assert_array_equal(goal['release_reference']['attachment_spheres_tool_m'], original)
    np.testing.assert_array_equal(spheres, original)


def test_endpoint_rejects_changed_gap_even_when_tcp_reaches_tampered_goal():
    target, destination, holding, spheres = region_inputs()
    goal = release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)[0]
    goal['T_world_tcp'][2][3] -= .005
    actual = np.asarray(goal['T_world_tcp'])
    evidence = release_endpoint_evidence(goal, holding, actual)
    assert evidence['position_error_m'] == 0.
    assert evidence['attachment_floor_gap_m'] == pytest.approx(.007)
    assert not evidence['at_release_goal']


@pytest.mark.parametrize('clearance', [.01, .012, .02, .03])
def test_release_height_is_derived_from_collision_threshold(clearance):
    settings = PlanningSettings(trajectory_collision_clearance_m=clearance, place_release_height_m=.05)
    assert settings.place_release_height_m == clearance + .002


@pytest.mark.parametrize('gap', [.007, .01, .011, .02, .05])
def test_saved_release_height_is_retired_without_changing_collision_threshold(tmp_path, gap):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'place_release_height_m': gap, 'time_scale': 4}), encoding='utf-8')
    settings = load_settings(path)
    assert settings.place_release_height_m == .012
    assert settings.trajectory_collision_clearance_m == .01
    assert settings.time_scale == 4
    assert json.loads(path.read_text(encoding='utf-8'))['place_release_height_m'] == gap


@pytest.mark.parametrize('legacy_name', ['release_margin_m', 'place_release_margin_m'])
def test_legacy_release_margin_is_retired(legacy_name):
    settings = PlanningSettings.model_validate({legacy_name: .007, 'trajectory_collision_clearance_m': .012})
    assert settings.place_release_height_m == .014
