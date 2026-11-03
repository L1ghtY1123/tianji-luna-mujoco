from types import SimpleNamespace

import pytest

from luna_mujoco.geometry_service import GeometryRuntime
from luna_mujoco.planning_timing import collect_timings, timed_stage


def test_timings_include_failure_and_reset_between_requests():
    with collect_timings() as first, pytest.raises(ValueError), timed_stage('sdk'):
        raise ValueError('no result')
    with collect_timings() as second, timed_stage('sdk'):
        pass
    assert first['sdk']['calls'] == second['sdk']['calls'] == 1
    assert first is not second
    assert first['sdk']['wall_s'] >= 0


def test_peer_cache_reuses_only_exact_backend_joint_and_geometry_snapshot():
    runtime = GeometryRuntime()
    calls = []
    backend = SimpleNamespace(resident_geometry=SimpleNamespace(key='open'),
        update_peer_collision_model=lambda *args: calls.append(args))
    def refresh(**changes):
        values = {'arm': 'A', 'backend': backend, 'active_joints_deg': [0.] * 7,
                  'other_joints_deg': [1.] * 7}
        return runtime._refresh_peer_collision_unlocked(**(values | changes))
    assert refresh() is True
    assert refresh() is False
    assert len(calls) == 1
    assert refresh(other_joints_deg=[2.] * 7) is True
    assert refresh(active_joints_deg=[3.] * 7) is True
    backend.resident_geometry.key = 'fingers_changed'
    assert refresh() is True
    assert refresh(cache_snapshot=False) is True
    assert refresh() is True  # Temporary update must invalidate the old cache.
    assert refresh() is False
    replacement = SimpleNamespace(resident_geometry=SimpleNamespace(key='fingers_changed'),
        update_peer_collision_model=lambda *args: calls.append(args))
    assert refresh(backend=replacement) is True
    runtime._peer_collision_snapshot_by_arm.clear()
    assert refresh() is True


def test_peer_failed_update_never_leaves_a_reusable_snapshot():
    runtime = GeometryRuntime()
    backend = SimpleNamespace(resident_geometry=SimpleNamespace(key='open'))
    args = {'arm': 'A', 'backend': backend, 'active_joints_deg': [0.] * 7, 'other_joints_deg': [1.] * 7}
    backend.update_peer_collision_model = lambda *args: None
    runtime._refresh_peer_collision_unlocked(**args)
    def fail(*args):
        raise ValueError('upload failed')
    backend.update_peer_collision_model = fail
    with pytest.raises(ValueError):
        runtime._refresh_peer_collision_unlocked(**(args | {'cache_snapshot': False}))
    with pytest.raises(ValueError):
        runtime._refresh_peer_collision_unlocked(**args)


def test_peer_cache_never_skips_current_attachment_upload():
    runtime = GeometryRuntime()
    runtime.carrying_binding = object()
    uploads = []
    backend = SimpleNamespace(resident_geometry=SimpleNamespace(key='held'),
        update_peer_collision_model=lambda *args: None,
        attach_measured_carrying=lambda *args: uploads.append(args))
    args = {'arm': 'A', 'backend': backend, 'active_joints_deg': [0.] * 7, 'other_joints_deg': [1.] * 7}
    assert runtime._refresh_peer_collision_unlocked(**args) is True
    assert runtime._refresh_peer_collision_unlocked(**args) is False
    assert uploads == [(runtime.carrying_binding, [0.] * 7)] * 2


def test_compound_timing_does_not_intercept_sdk_internal_calls():
    from luna_mujoco.public_planner import PublicPlanner
    class SDK:
        def plan_pose(self):
            return 'trajectory'
        def plan_grasp(self):
            return self.plan_pose()
    planner = PublicPlanner(SDK(), SimpleNamespace())
    with collect_timings() as timings:
        assert planner.plan_grasp() == 'trajectory'
    assert set(timings) == {'public_plan_grasp'}


def test_client_retains_server_stages_and_separate_roundtrip_decode():
    import threading

    import httpx

    from luna_mujoco.algorithms import Algorithms
    algorithm = object.__new__(Algorithms)
    algorithm._client_lock = threading.Lock()
    algorithm.curobo_url = 'http://test'
    algorithm._client = httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200, json={'ok': True, 'request_timings': {'stages': {'operation': {'wall_s': .02}}}},
                       headers={'Server-Timing': 'encode;dur=0.1'})))
    try:
        result = algorithm._request('/tools/test', {'arm': 'A'})
    finally:
        algorithm.close()
    assert result['request_timings']['stages']['operation']['wall_s'] == .02
    assert result['transport_timings']['response_bytes'] > 0
    assert result['transport_timings']['response_decode_s'] >= 0
    assert result['transport_timings']['server_timing'] == 'encode;dur=0.1'


def test_model_update_invalidates_peer_cache_even_when_geometry_key_repeats():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene', 'target_id': 'target'}
    runtime.geometry_grids = []
    runtime.geometry_model = {'cuboid': {}}
    calls = []
    backend = SimpleNamespace(resident_geometry=SimpleNamespace(key='open', update=lambda *args: True),
        update_peer_collision_model=lambda *args: calls.append(args), update_voxel_scene=lambda *args, **kwargs: None)
    args = {'arm': 'A', 'backend': backend, 'active_joints_deg': [0.] * 7, 'other_joints_deg': [1.] * 7}
    runtime._refresh_peer_collision_unlocked(**args)
    runtime._refresh_scene_unlocked(arm='A', backend=backend)
    assert runtime._refresh_peer_collision_unlocked(**args) is True
    assert len(calls) == 2


def test_concurrent_requests_keep_timing_counts_separate():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    def request(count):
        with collect_timings() as timings:
            barrier.wait()
            for _ in range(count):
                with timed_stage('public_plan_pose'):
                    pass
        return timings
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request, count) for count in (1, 3)]
        results = [future.result() for future in futures]
    assert [result['public_plan_pose']['calls'] for result in results] == [1, 3]


def test_compact_collision_capacity_retains_every_obstacle_and_grows_in_buckets():
    from luna_mujoco.geometry_service import geometry_collision_cache
    empty = geometry_collision_cache(None)
    assert (empty['mesh'], empty['cuboid'], empty['voxel']['layers']) == (4, 5, 16)
    assert geometry_collision_cache({'mesh': {str(i): {} for i in range(3)}}) == empty
    scene = {'mesh': {f'mesh-{i}': {} for i in range(17)},
             'cuboid': {f'box-{i}': {} for i in range(9)}}
    cache = geometry_collision_cache(scene)
    assert (cache['mesh'], cache['cuboid']) == (32, 16)
    assert len(scene['mesh']) == 17 and len(scene['cuboid']) == 9


def test_compact_voxel_capacity_reuses_backend_then_rebuilds_without_dropping_geometry():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene', 'target_id': 'target'}
    layer = {'dims': [1., 1., 1.], 'voxel_size': .02}
    runtime.geometry_model = {'voxel': {str(i): dict(layer) for i in range(4)},
        'cuboid': {'semantic_table::table': {}}}
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache['voxel']['layers'] == 16
    destroyed = []
    backend = SimpleNamespace(destroy=lambda: destroyed.append(True))
    runtime._batch_backends['A'] = backend
    runtime._prepare_scene_cache_unlocked()
    assert runtime._batch_backends['A'] is backend and not destroyed
    runtime.geometry_model['voxel']['4'] = dict(layer)
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache['voxel']['layers'] == 16 and not destroyed
    assert len(runtime._scene_model['voxel']) == 5
    assert runtime._batch_backends['A'] is backend
    runtime.geometry_model['voxel'].update({str(i): dict(layer) for i in range(17)})
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache['voxel']['layers'] == 32 and destroyed == [True]
    assert len(runtime._scene_model['voxel']) == 17
    assert not runtime._batch_backends


def test_voxel_capacity_covers_uploaded_background_even_if_metadata_omits_it():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene', 'target_id': 'target'}
    runtime.geometry_model = {'voxel': {f'object-{index}': {
        'dims': [1., 1., 1.], 'voxel_size': .02} for index in range(4)}}
    runtime.geometry_grids = [SimpleNamespace(name=name) for name in
                              ('background', 'object-0', 'object-1', 'object-2', 'object-3')]
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache['voxel']['layers'] == 16
    assert len(runtime.geometry_grids) == 5 and len(runtime._scene_model['voxel']) == 4
