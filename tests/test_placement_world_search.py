"""Placement searches existing 3D collision geometry, with bounded release gaps."""
from types import SimpleNamespace

import numpy as np
import pytest
from test_resolve_placement import setup

from luna_mujoco.algorithms import AlgorithmError
from luna_mujoco.demo_placement import resolve_release_height
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.robot_skills.placement_resolution import candidate_requests


def test_xy_envelope_does_not_reject_space_below_measured_overhang(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    obs['objects'].append({'id': 'overhang', 'kind': 'object',
        'points_world_m': [[.3, -.5, 1.2], [.9, .2, 1.3]]})
    candidates, _ = candidate_requests(obs, 'seen-red', held, request, zone,
        PlanningSettings().model_dump(), scene=scene)
    assert candidates
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    assert result['candidates']
    assert service.calls
    assert result['selection_evidence']['occupancy_rule'] == 'existing_scene_attachment_and_gripper_spheres'


def test_height_query_batches_z_and_selects_first_clear_sample():
    queries = []
    def query(world):
        queries.append(world.copy())
        return np.maximum(.524 - (world[..., 2] - world[..., 3]), 0.)
    request = SimpleNamespace(goal_position_m=[.5, .1, .54], goal_quaternion_wxyz=[1, 0, 0, 0],
                              trajectory_collision_clearance_m=.005)
    result = resolve_release_height([[0, 0, 0, .02]], [[.1, 0, 0, .01]], request, query, scene_token='scene')
    assert result['height_increase_m'] == pytest.approx(.004)
    assert len(queries) == 1
    assert queries[0].shape[1] > 1


def test_world_clear_above_blocker_is_not_accepted_as_low_surface_placement(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    service.resolved_height_increase = .08
    with pytest.raises(AlgorithmError) as caught:
        algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_zone=zone)
    assert caught.value.code == 'placement_no_candidate'


def test_outside_workspace_candidates_do_not_query_world(monkeypatch):
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    positions = []
    def unreachable(arm, values):
        positions.extend(values)
        return [{'position_world_m': point, 'reachable': False} for point in values]
    algorithm.workspace_lookup = unreachable
    with pytest.raises(AlgorithmError) as caught:
        algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_zone=zone)
    assert caught.value.code == 'placement_no_candidate'
    assert positions and not service.calls


def test_gripper_limited_clear_pose_must_still_respect_configured_object_drop_gap(monkeypatch):
    algorithm, _, obs, scene, held, request, zone = setup(monkeypatch)
    monkeypatch.setattr(algorithm, 'gripper_release_spheres', lambda *args: [[0., 0., .2, .02]])
    with pytest.raises(AlgorithmError) as caught:
        algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_zone=zone)
    assert {item['code'] for item in caught.value.details['candidate_failures']} == {'placement_release_too_high'}
    result = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone, planning_settings={'place_max_release_gap_m': .09})
    assert result['candidates']
