"""Cache model resources, never an observation's pose or acceptance result."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'services'))
from luna_foundationpose.backend import CachedFoundationPoseBackend


def backend_fixture(tmp_path):
    mesh = tmp_path / 'model.ply'
    trimesh.creation.box(extents=[40, 50, 60]).export(mesh)
    record = {'cache_key': 'model', 'source_sha256': 'source',
              'mesh_sha256': hashlib.sha256(mesh.read_bytes()).hexdigest(),
              'T_source_canonical_m': np.eye(4).tolist(), 'pose_frame': 'aabb_center_axes_preserved'}
    calls = []

    class Estimator:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs['mesh'].vertices.copy()))
            self.rot_grid = np.zeros((252, 4, 4))
            self.mesh_path = None

        def register(self, **kwargs):
            calls.append(('register', kwargs['rgb'].copy()))
            pose = np.eye(4)
            pose[0, 3] = float(kwargs['rgb'][0, 0, 0]) / 100
            return pose

    cuda = SimpleNamespace(synchronize=lambda: None, memory_allocated=lambda: 123,
                           max_memory_allocated=lambda: 456, reset_peak_memory_stats=lambda: None)
    base = SimpleNamespace(registry=SimpleNamespace(get=lambda name: (record, tmp_path)),
        torch=SimpleNamespace(cuda=cuda), estimator_class=Estimator,
        erode_depth=lambda x, **kwargs: x, filter_depth=lambda x, **kwargs: x,
        scorer=object(), refiner=object(), context=object(), checkpoint_sha256='weights',
        search_profile='default', search_parameters={'min_n_views': 40, 'inplane_step': 60})
    request = {'model_id': 'box', 'rgb': np.full((4, 4, 3), 20, np.uint8),
               'depth_m': np.ones((4, 4)), 'mask': np.ones((4, 4), bool), 'K': np.eye(3)}
    return base, request, calls, record


def test_estimator_is_reused_but_each_capture_is_registered(tmp_path):
    base, request, calls, _ = backend_fixture(tmp_path)
    backend = CachedFoundationPoseBackend(base)
    first = backend.estimate(request)
    request['rgb'][:] = 60
    second = backend.estimate(request)
    assert [row[0] for row in calls] == ['init', 'register', 'register']
    assert first['T_camera_object'][0][3] == .2
    assert second['T_camera_object'][0][3] == .6
    assert first['cache_hit'] is False and second['cache_hit'] is True
    assert np.ptp(calls[0][1], axis=0).tolist() == pytest.approx([.04, .05, .06])


def test_changed_mesh_invalidates_cached_estimator(tmp_path):
    base, request, calls, record = backend_fixture(tmp_path)
    backend = CachedFoundationPoseBackend(base)
    backend.estimate(request)
    trimesh.creation.box(extents=[30, 40, 50]).export(tmp_path / 'model.ply')
    record['mesh_sha256'] = hashlib.sha256((tmp_path / 'model.ply').read_bytes()).hexdigest()
    backend.estimate(request)
    assert sum(row[0] == 'init' for row in calls) == 2


@pytest.mark.parametrize('change', ['source', 'mesh', 'path', 'frame', 'units'])
def test_reduced_mesh_requires_source_hash_and_scoped_content(tmp_path, change):
    base, request, _, record = backend_fixture(tmp_path)
    reduced = tmp_path / 'reduced'
    reduced.mkdir()
    raw = (tmp_path / 'model.ply').read_bytes()
    (reduced / 'box.ply').write_bytes(raw)
    entry = {'source_sha256': record['source_sha256'], 'original_mesh_sha256': record['mesh_sha256'],
             'mesh_sha256': hashlib.sha256(raw).hexdigest(), 'path': 'box.ply', 'face_count': 12,
             'units': 'mm', 'frame': 'original_registry_canonical', 'frame_transform': np.eye(4).tolist()}
    if change == 'source':
        entry['source_sha256'] = 'another'
    elif change == 'mesh':
        entry['mesh_sha256'] = 'wrong'
    elif change == 'path':
        entry['path'] = '../model.ply'
    elif change == 'frame':
        entry['frame_transform'][0][3] = 1
    else:
        entry['units'] = 'm'
    (reduced / 'manifest.json').write_text(json.dumps({'version': 1, 'models': {'box': entry}}))
    backend = CachedFoundationPoseBackend(base, mesh_directory=reduced)
    with pytest.raises(ValueError, match='registration_mesh'):
        backend.estimate(request)


def test_valid_reduced_mesh_keeps_original_registration_source_frame(tmp_path):
    base, request, calls, record = backend_fixture(tmp_path)
    manifest = {'version': 1, 'models': {'box': {
        'source_sha256': record['source_sha256'], 'original_mesh_sha256': record['mesh_sha256'],
        'mesh_sha256': record['mesh_sha256'], 'path': 'model.ply', 'units': 'mm',
        'frame': 'original_registry_canonical', 'frame_transform': np.eye(4).tolist()}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    backend = CachedFoundationPoseBackend(base, mesh_directory=tmp_path)
    result = backend.estimate(request)
    assert result['source_sha256'] == record['source_sha256']
    assert result['T_source_canonical_m'] == record['T_source_canonical_m']
    assert result['registration_mesh']['mesh_sha256'] == record['mesh_sha256']
    assert len(calls) == 2
