import numpy as np
import pytest


def test_full_surface_bounds_include_extreme_discarded_by_preview_sampling():
    from luna_mujoco.instance_mapping import surface_geometry_metadata

    centers = np.zeros((20001, 3), dtype=np.float32)
    centers[1] = [-.4, .7, 1.2]  # Odd index is omitted by the bounded preview.
    original = centers.copy()
    metadata = surface_geometry_metadata(centers, .02)
    assert len(metadata["centers_base_m"]) <= 20000
    assert np.max(np.asarray(metadata["centers_base_m"])[:, 2]) == 0.
    np.testing.assert_allclose(metadata["surface_cell_bounds_world_m"],
                               [[-.41, -.01, -.01], [.01, .71, 1.21]])
    assert metadata["surface_cell_count"] == 20001
    assert metadata["surface_cell_bounds_source"] == "all_native_surface_cells_before_preview_sampling"
    np.testing.assert_array_equal(centers, original)


def test_local_mapping_inputs_translate_cameras_and_table_without_changing_depth(monkeypatch):
    import sys
    from types import SimpleNamespace

    from luna_mujoco.instance_mapping import local_mapping_inputs

    monkeypatch.setitem(sys.modules, 'curobo.types', SimpleNamespace(
        CameraObservation=SimpleNamespace, Pose=SimpleNamespace))
    positions = np.array([[1., 2., 3.], [-1., 0., 1.]])
    observation = SimpleNamespace(depth_image=np.ones((2, 3, 4)), rgb_image=None,
        intrinsics=np.eye(3), depth_to_meter=1.,
        pose=SimpleNamespace(position=positions, quaternion=np.array([[1., 0., 0., 0.]] * 2)))
    table = {'pose': [.5, -.2, .4, 1., 0., 0., 0.], 'dims': [1., 1., .3]}
    local, cropped_table = local_mapping_inputs(observation, np.array([.4, -.1, .6]), table)
    np.testing.assert_allclose(local.pose.position, positions - [.4, -.1, .6])
    np.testing.assert_allclose(cropped_table['pose'][:3], [.1, -.1, -.2])
    assert local.depth_image is observation.depth_image
    assert local.intrinsics is observation.intrinsics
    assert local.pose.quaternion is observation.pose.quaternion
    np.testing.assert_array_equal(observation.pose.position, positions)
    assert table['pose'] == [.5, -.2, .4, 1., 0., 0., 0.]


def test_esdf_bounds_keep_query_halo_without_tsdf_allocation_slack():
    from luna_mujoco.instance_mapping import instance_grid_layout
    layout = instance_grid_layout(np.array([[.41, -.013, .61], [.45, .027, .65]]), .19, voxel_size_m=.01)
    np.testing.assert_allclose(layout['center'], [.43, .01, .63])
    np.testing.assert_allclose(layout['tsdf_extent'], [.16, .16, .16])
    np.testing.assert_allclose(layout['esdf_extent'], [.42, .44, .43])
    assert layout['esdf_cells'] == 42*44*43
    assert layout['legacy_esdf_cells'] == 48*48*49
    assert layout['esdf_cells'] < layout['legacy_esdf_cells']


@pytest.mark.parametrize('points', [[], [[np.nan, 0., 1.]], [[1., 2.]]])
def test_invalid_measured_points_never_allocate_gpu_maps(points):
    from luna_mujoco.instance_mapping import instance_grid_layout
    with pytest.raises(ValueError, match='instance_points'):
        instance_grid_layout(np.array(points), .19)


def test_grid_centers_preserve_native_tsdf_phase_for_asymmetric_measurements():
    from luna_mujoco.instance_mapping import instance_grid_layout
    points = np.array([[.403, -.127, .603], [.457, .069, .637]])
    layout = instance_grid_layout(points, .19, voxel_size_m=.01)
    center, extent = np.asarray(layout['center']), np.asarray(layout['esdf_extent'])
    assert np.all(extent / 2 - np.abs(points - center).max(axis=0) >= .19 - 1e-9)
    legacy_shape = np.ceil(np.asarray(layout['legacy_esdf_extent']) / .01).astype(int)
    np.testing.assert_array_equal(np.asarray(layout['esdf_shape']) % 2, legacy_shape % 2)


def test_semantic_scene_builds_all_instances_together_without_empty_background():
    from test_semantic_builder import builder, request
    value, _, mapper, _ = builder()

    class Pool:
        def __init__(self):
            self.calls = 0
            self.grids = []
            self.metrics = {}
        def build(self, prepared, instances, **kwargs):
            self.calls += 1
            assert kwargs.pop('floor_heights_m') == {}
            result = {name: mapper.build_instance_esdf(prepared, name, points, **kwargs)
                      for name, points in instances.items()}
            self.grids = mapper.scene_voxel_grids[1:]
            return result
    pool = Pool()
    value.instance_mapper = pool
    payload = request() | {'experimental_semantic_only_mapping': True, 'full_observation_complete': True}
    scene = value.build(payload)
    assert pool.calls == 1
    assert mapper.integration_count == 0
    assert scene.scene_data.voxel == ['semantic_instance::source', 'semantic_instance::other']
    assert value.diagnostics['background_grid_omitted'] is True


def test_semantic_crop_uses_measured_instance_samples_without_projecting_background():
    from dataclasses import replace

    from test_semantic_builder import Segmenter, request
    from tianji_curobo.robot_segmenter import FfsDepthObservationRequest, SemanticProjectionRequest

    from luna_mujoco.semantic_builder import crop_instance_samples
    segmenter = Segmenter()
    view = request()['views']['1']
    prepared = segmenter.remove_semantic_instances(
        segmenter.prepare_ffs_observation(FfsDepthObservationRequest.from_mapping(view)),
        SemanticProjectionRequest.from_mapping(view))
    prepared = replace(prepared, instance_points_base_m={'source': np.array([[0., 0., .61], [0., 0., .59], [2., 0., .7]])},
        instance_depth_samples={'source': (np.array([1, 2, 3]), np.array([.3, .4, .5]))})
    result = crop_instance_samples(prepared, {'pose': [0, 0, 1.1, 1, 0, 0, 0], 'dims': [1, 1, 1]})
    np.testing.assert_array_equal(result.instance_points_base_m['source'], np.array([[0., 0., .61]], dtype=np.float32))
    np.testing.assert_array_equal(result.instance_depth_samples['source'][0], [1])
    np.testing.assert_array_equal(result.instance_depth_samples['source'][1], [.3])
    assert result.camera_observation is prepared.camera_observation
    # Background is intentionally untouched here; semantic-only composition
    # never integrates it or sends it through support-depth reconstruction.
    assert result.background_points_base_m is prepared.background_points_base_m


@pytest.mark.parametrize('failure_stage', ['allocation', 'packing'])
def test_preparation_failure_bounds_cache_and_waits_for_owned_streams(monkeypatch, failure_stage):
    import sys
    from types import SimpleNamespace

    from luna_mujoco.instance_mapping import InstanceMapper

    streams = []

    class Stream:
        def __init__(self):
            self.synchronized = False
            streams.append(self)

        def wait_stream(self, producer):
            pass

        def synchronize(self):
            self.synchronized = True

    class Mapper:
        def __init__(self, config):
            if failure_stage == 'allocation' and config.grid_center[0] > 9:
                raise RuntimeError('injected_preparation_failure')

    def stack_observations(observations, name):
        if failure_stage == 'packing' and name.endswith('-9'):
            raise RuntimeError('injected_preparation_failure')
        return SimpleNamespace(depth_image=np.ones((3, 2, 2)))

    # CUDA allocation is unavailable in this CPU test; cache ownership and
    # exception cleanup still run through the real InstanceMapper.build.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        current_stream=lambda: object(), Stream=Stream)))
    monkeypatch.setitem(sys.modules, 'curobo.perception', SimpleNamespace(
        Mapper=Mapper, MapperCfg=lambda **values: SimpleNamespace(**values)))
    monkeypatch.setitem(sys.modules, 'tianji_curobo.contracts', SimpleNamespace(CuroboToolError=RuntimeError))
    monkeypatch.setitem(sys.modules, 'tianji_curobo.mixed_esdf_scene', SimpleNamespace(InstanceGeometry=object))
    segmenter = SimpleNamespace(device_cfg=SimpleNamespace(to_device=lambda value: value),
                                stack_instance_observations=stack_observations)
    mapper = InstanceMapper(segmenter, local_frame=False)
    for attempt in range(2):
        instances = {f'object-{attempt}-{index}': {'3': [[index + .2, 0., 1.]]} for index in range(10)}
        with pytest.raises(RuntimeError, match='injected_preparation_failure'):
            mapper.build(None, instances, padding_m=.19, table_model=None)
        assert len(mapper.cache) <= 8
        assert streams and all(stream.synchronized for stream in streams)
        assert mapper.grids == []


def mapping_with_sync_failures(monkeypatch, sync_failures):
    import sys
    from types import SimpleNamespace

    from luna_mujoco.instance_mapping import InstanceMapper

    streams = []

    class Stream:
        def __init__(self):
            self.failures = list(sync_failures.get(len(streams), []))
            self.synchronize_calls = 0
            streams.append(self)

        def wait_stream(self, producer):
            pass

        def synchronize(self):
            self.synchronize_calls += 1
            if self.failures:
                raise self.failures.pop(0)

    class Surface:
        def __init__(self, values=None):
            self.values = np.array([[.2, 0., 1.]]) if values is None else values
            self.shape = self.values.shape

        def __len__(self):
            return len(self.values)

        def __getitem__(self, item):
            return Surface(self.values[item])

        def amin(self, dim):
            return Surface(self.values.min(axis=dim))

        def amax(self, dim):
            return Surface(self.values.max(axis=dim))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Mapper:
        def __init__(self, config):
            self.integrator = SimpleNamespace(get_voxel_grid=self.get_voxel_grid)

        def reset(self):
            pass

        def get_voxel_grid(self):
            features = SimpleNamespace(clone=lambda: SimpleNamespace(numel=lambda: 1))
            return SimpleNamespace(feature_tensor=features, pose=[0., 0., 0., 1., 0., 0., 0.],
                                   dims=[.01, .01, .01], voxel_size=.01)

        def extract_occupied_voxels(self, *, surface_only):
            return SimpleNamespace(centers=Surface())

    # Only the unavailable CUDA/SDK boundary is replaced. Real build workers,
    # result export, stream cleanup, cache ownership and publication run below.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        current_stream=lambda: object(), Stream=Stream)))
    monkeypatch.setitem(sys.modules, 'curobo.perception', SimpleNamespace(
        Mapper=Mapper, MapperCfg=lambda **values: SimpleNamespace(**values)))
    monkeypatch.setitem(sys.modules, 'tianji_curobo.contracts', SimpleNamespace(CuroboToolError=RuntimeError))
    monkeypatch.setitem(sys.modules, 'tianji_curobo.mixed_esdf_scene', SimpleNamespace(
        InstanceGeometry=lambda *values: values))
    segmenter = SimpleNamespace(device_cfg=SimpleNamespace(to_device=lambda value: value),
        stack_instance_observations=lambda observations, name: SimpleNamespace(
            name=name, depth_image=np.ones((3, 2, 2))))
    mapper = InstanceMapper(segmenter, local_frame=False)
    monkeypatch.setattr(mapper, 'integrate', lambda mapper, stream, observation, table: {})
    instances = {f'object-{index}': {'3': [[index + .2, 0., 1.]]} for index in range(3)}
    return mapper, streams, instances


def test_worker_failure_preserved_when_multiple_stream_cleanups_fail(monkeypatch):
    primary_error = ValueError('injected_worker_failure')
    mapper, streams, instances = mapping_with_sync_failures(monkeypatch, {
        0: [RuntimeError('injected_first_cleanup_failure')],
        1: [RuntimeError('injected_second_cleanup_failure')]})

    def integrate(mapper, stream, observation, table):
        if observation.name == 'object-0':
            raise primary_error
        return {}

    monkeypatch.setattr(mapper, 'integrate', integrate)
    with pytest.raises(ValueError, match='injected_worker_failure') as caught:
        mapper.build(None, instances, padding_m=.19, table_model=None)
    assert caught.value is primary_error
    notes = '\n'.join(caught.value.__notes__)
    assert 'object-0' in notes and 'injected_first_cleanup_failure' in notes
    assert 'object-1' in notes and 'injected_second_cleanup_failure' in notes
    assert [stream.synchronize_calls for stream in streams] == [1, 1, 1]
    assert mapper.cache == {}
    assert mapper.grids == []


def test_cleanup_failure_prevents_publication_and_cached_mapper_reuse(monkeypatch):
    sync_error = RuntimeError('injected_cleanup_failure')
    mapper, streams, instances = mapping_with_sync_failures(monkeypatch, {0: [sync_error]})
    with pytest.raises(RuntimeError, match='injected_cleanup_failure') as caught:
        mapper.build(None, instances, padding_m=.19, table_model=None)
    assert caught.value is sync_error
    assert [stream.synchronize_calls for stream in streams] == [1, 1, 1]
    assert mapper.grids == []
    assert mapper.cache == {}
    assert 'object-0' in '\n'.join(caught.value.__notes__)

    geometries = mapper.build(None, instances, padding_m=.19, table_model=None)
    assert set(geometries) == set(instances)
    assert len(mapper.grids) == 3
    assert len(streams) == 6
    assert not any(metric['cache_hit'] for metric in mapper.metrics.values())
    mapper.build(None, instances, padding_m=.19, table_model=None)
    assert len(streams) == 6
    assert all(metric['cache_hit'] for metric in mapper.metrics.values())


def test_worker_sync_failure_invalidates_cache_even_if_cleanup_succeeds(monkeypatch):
    sync_error = RuntimeError('injected_worker_sync_failure')
    mapper, streams, instances = mapping_with_sync_failures(monkeypatch, {0: [sync_error]})

    def integrate(mapper, stream, observation, table):
        stream.synchronize()
        return {}

    monkeypatch.setattr(mapper, 'integrate', integrate)
    with pytest.raises(RuntimeError, match='injected_worker_sync_failure') as caught:
        mapper.build(None, instances, padding_m=.19, table_model=None)
    assert caught.value is sync_error
    assert [stream.synchronize_calls for stream in streams] == [2, 2, 2]
    assert mapper.cache == {}
    assert mapper.grids == []


def test_head_and_three_camera_layers_use_separate_sdk_configurations(monkeypatch):
    import sys
    from types import SimpleNamespace

    mapper, _, instances = mapping_with_sync_failures(monkeypatch, {})
    camera_count = [1]
    configurations = []

    def config(**values):
        configurations.append(values)
        return SimpleNamespace(**values)

    monkeypatch.setattr(sys.modules['curobo.perception'], 'MapperCfg', config)
    mapper.segmenter.stack_instance_observations = lambda observations, name: SimpleNamespace(
        name=name, depth_image=np.ones((camera_count[0], 2, 2)))
    mapper.build(None, instances, padding_m=.19, table_model=None, floor_heights_m={'object-0': .61})
    assert all(cfg['num_cameras'] == 1 for cfg in configurations)
    assert all(cfg['voxel_size'] == .02 and cfg['esdf_voxel_size'] == .02 for cfg in configurations)
    assert all(item['voxel_size_m'] == .02 for item in mapper.metrics.values())
    camera_count[0] = 3
    mapper.build(None, instances, padding_m=.19, table_model=None, floor_heights_m={'object-0': .61})
    assert [cfg['num_cameras'] for cfg in configurations] == [1, 1, 1, 3, 3, 3]
    assert all(cfg['voxel_size'] == .02 and cfg['esdf_voxel_size'] == .02 for cfg in configurations)
    camera_count[0] = 1
    mapper.build(None, instances, padding_m=.19, table_model=None, floor_heights_m={'object-0': .61})
    assert len(configurations) == 6 and all(item['cache_hit'] for item in mapper.metrics.values())
