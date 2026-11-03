"""A placement pre-screen cannot attach, move, or certify a whole trajectory."""
import hashlib
from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.world_sphere_query import prepare_query, query_result


def inputs():
    scene = {'scene_token': 'scene', 'observation_id': 'capture', 'generation': 4}
    payload = {**scene, 'arm': 'B', 'start_joints_deg': [0.] * 7,
        'other_joints_deg': [1.] * 7, 'gripper_joint_positions_rad': {'A': 0., 'B': .3},
        'spheres_world_m': [[[.4, -.6, .6, .02], [.5, -.6, .6, .03]],
                            [[.4, -.6, .7, .02], [.5, -.6, .7, .03]]]}
    return payload, scene


@pytest.mark.parametrize('field,value', [('scene_token', 'old'), ('generation', 3),
    ('observation_id', 'old'), ('arm', 'C'), ('start_joints_deg', [0.] * 6),
    ('other_joints_deg', [float('nan')] * 7), ('carrying_binding', {'instance_id': 'object'})])
def test_query_rejects_unbound_inputs(field, value):
    payload, scene = inputs()
    payload[field] = value
    with pytest.raises(ValueError):
        prepare_query(payload, scene)


@pytest.mark.parametrize('spheres', [[], [[[0, 0, 0, 0]]], [[[0, 0, float('nan'), .1]]],
    [[0, 0, 0, .1]], np.ones((1, 262145, 4)), np.ones((1, 1, 4), dtype=bool)])
def test_query_rejects_invalid_or_unbounded_geometry(spheres):
    payload, scene = inputs()
    payload['spheres_world_m'] = spheres
    with pytest.raises(ValueError):
        prepare_query(payload, scene)


def test_cost_summary_preserves_every_pose_and_binds_exact_query_bytes():
    payload, scene = inputs()
    original = deepcopy(payload)
    spheres = prepare_query(payload, scene)
    result = query_result(spheres, np.array([[0., 1e-9], [0., 1e-8]]), scene['scene_token'], .005)
    assert result['pose_collision_free'] == [True, False]
    assert result['maximum_world_collision_cost'] == [1e-9, 1e-8]
    assert result['query_sha256'] == hashlib.sha256(np.asarray(payload['spheres_world_m'], dtype='<f8').tobytes()).hexdigest()
    assert result['attached'] is False and result['trajectory_checked'] is False
    assert payload == original


@pytest.mark.parametrize('costs', [[[0.]], [[0., 0.], [0., -1.]], [[0., 0.], [0., float('nan')]]])
def test_missing_or_invalid_checker_results_do_not_become_free_space(costs):
    payload, scene = inputs()
    spheres = prepare_query(payload, scene)
    with pytest.raises(ValueError):
        query_result(spheres, costs, scene['scene_token'], .005)


def runtime_fixture():
    from luna_mujoco.geometry_service import GeometryRuntime
    from luna_mujoco.planning_settings import PlanningSettings

    payload, scene = inputs()
    settings = PlanningSettings()
    calls = []

    def costs(world):
        calls.append(world.copy())
        return np.zeros(world.shape[:-1])

    @contextmanager
    def carrying_scope(binding, feedback):
        assert binding is None
        assert feedback == payload['gripper_joint_positions_rad']
        yield

    runtime = SimpleNamespace(_carrying_scope_lock=RLock(), gpu_lock=RLock(),
        logical_carrying_binding=None, carrying_binding=None,
        geometry_solver_profile=settings.solver_payload(),
        _scene_snapshot_unlocked=lambda: scene,
        configure_planning=lambda _: None, carrying_scope=carrying_scope,
        _prepare_scene_cache_unlocked=lambda: scene,
        _compound_backend_unlocked=lambda *_: SimpleNamespace(query_world_sphere_costs=costs),
        _refresh_scene_unlocked=lambda **_: scene['scene_token'],
        _refresh_peer_collision_unlocked=lambda **_: None,
        _require_scene_token_unchanged_unlocked=lambda token: None)
    return GeometryRuntime.query_world_spheres, runtime, payload, scene, settings, calls


@pytest.mark.parametrize('binding', ['carrying_binding', 'logical_carrying_binding'])
def test_pregrasp_query_cannot_temporarily_detach_a_held_object(binding):
    query, runtime, payload, _, settings, calls = runtime_fixture()
    held = object()
    setattr(runtime, binding, held)
    with pytest.raises(ValueError, match='empty hand'):
        query(runtime, payload, settings)
    assert getattr(runtime, binding) is held
    assert not calls


def test_query_rechecks_capture_after_entering_planner_scope():
    query, runtime, payload, scene, settings, calls = runtime_fixture()
    runtime._prepare_scene_cache_unlocked = lambda: {**scene, 'observation_id': 'new'}
    with pytest.raises(ValueError, match='observation_id'):
        query(runtime, payload, settings)
    assert not calls


def test_query_requires_current_clearance_profile():
    query, runtime, payload, _, settings, calls = runtime_fixture()
    runtime.geometry_solver_profile = {}
    with pytest.raises(ValueError, match='settings changed'):
        query(runtime, payload, settings)
    assert not calls


def test_runtime_returns_bounded_world_evidence_without_attachment_or_motion():
    query, runtime, payload, scene, settings, calls = runtime_fixture()
    original = deepcopy(payload)
    result = query(runtime, payload, settings)
    assert result['scene_token'] == scene['scene_token']
    assert result['pose_collision_free'] == [True, True]
    assert calls[0].shape == (1, 2, 2, 4)
    assert result['attached'] is False and result['trajectory_checked'] is False
    assert runtime.logical_carrying_binding is None and runtime.carrying_binding is None
    assert payload == original


def test_bounded_batch_fits_its_http_limit_without_expanding_other_endpoints():
    import json

    from luna_mujoco.geometry_service import request_size_limit

    payload, _ = inputs()
    payload['spheres_world_m'] = np.full((2383, 110, 4), 1.2345678912345678).tolist()
    size = len(json.dumps(payload).encode())
    assert size > 8 * 1024 * 1024
    assert size < request_size_limit('/tools/query_world_spheres')
    assert request_size_limit('/tools/solve_ik_batch') == 8 * 1024 * 1024
