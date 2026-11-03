"""Reuse official GPU resources; estimate every pose from current RGBD."""
import hashlib
import json
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import trimesh


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedFoundationPoseBackend:
    def __init__(self, base, *, mesh_directory=None, max_models=4):
        if type(max_models) is not int or max_models < 1:
            raise ValueError('invalid_model_cache_capacity')
        self.base = base
        self.registry = base.registry
        self.search_profile = base.search_profile
        self.search_parameters = base.search_parameters
        self.checkpoint_sha256 = base.checkpoint_sha256
        self.mesh_directory = Path(mesh_directory).resolve() if mesh_directory else None
        self.max_models = max_models
        self.models = OrderedDict()

    def registration_mesh(self, model_id, record, cache):
        path = cache / 'model.ply'
        metadata = {'mesh_sha256': record['mesh_sha256'], 'source': 'original_registry'}
        if self.mesh_directory:
            manifest = json.loads((self.mesh_directory / 'manifest.json').read_text())
            if manifest.get('version') != 1 or model_id not in manifest.get('models', {}):
                raise ValueError('registration_mesh_manifest_invalid')
            metadata = manifest['models'][model_id]
            path = (self.mesh_directory / metadata['path']).resolve()
            if (not path.is_relative_to(self.mesh_directory)
                    or metadata['source_sha256'] != record['source_sha256']
                    or metadata['original_mesh_sha256'] != record['mesh_sha256']
                    or metadata.get('units') != 'mm'
                    or metadata.get('frame') != 'original_registry_canonical'
                    or metadata.get('frame_transform') != np.eye(4).tolist()
                    or not path.is_file() or digest(path) != metadata['mesh_sha256']):
                raise ValueError('registration_mesh_provenance_mismatch')
        return path, metadata

    def estimator(self, record, path, metadata):
        key = (record['cache_key'], record['mesh_sha256'], metadata['mesh_sha256'])
        if key in self.models:
            self.models.move_to_end(key)
            return self.models[key]['estimator'], True
        mesh = trimesh.load(path, force='mesh')
        mesh.apply_scale(.001)
        directory = tempfile.TemporaryDirectory(prefix='luna-foundationpose-')
        try:
            estimator = self.base.estimator_class(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                scorer=self.base.scorer, refiner=self.base.refiner,
                glctx=self.base.context, debug=0, debug_dir=directory.name)
            if self.search_profile != 'default':
                estimator.make_rotation_grid(**self.search_parameters)
        except Exception:
            directory.cleanup()
            raise
        self.models[key] = {'estimator': estimator, 'directory': directory}
        while len(self.models) > self.max_models:
            _, entry = self.models.popitem(last=False)
            self._release(entry)
        return estimator, False

    @staticmethod
    def _release(entry):
        # The official constructor creates this one export; never glob other jobs' files.
        path = getattr(entry['estimator'], 'mesh_path', None)
        if path and Path(path).parent == Path(tempfile.gettempdir()):
            Path(path).unlink(missing_ok=True)
        entry['directory'].cleanup()
        entry.clear()

    def close(self):
        while self.models:
            _, entry = self.models.popitem()
            self._release(entry)

    def estimate(self, request, *, tracking_seed=None):
        started, cpu_started = time.perf_counter(), time.process_time()
        cuda = self.base.torch.cuda
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
        record, cache = self.registry.get(request['model_id'])
        path, metadata = self.registration_mesh(request['model_id'], record, cache)
        estimator, cache_hit = self.estimator(record, path, metadata)
        cuda.synchronize()
        initialized = time.perf_counter()
        depth = request['depth_m']
        filtered = self.base.filter_depth(self.base.erode_depth(depth, radius=2, device='cuda'),
                                         radius=2, device='cuda')
        if np.count_nonzero(request['mask'] & (filtered >= .001)) < 4:
            from tianji_foundationpose.contracts import PoseError
            raise PoseError('insufficient_depth')
        cuda.synchronize()
        filtered_at = time.perf_counter()
        if tracking_seed is None:
            mode, iterations = 'register', 5
            pose = estimator.register(K=request['K'], rgb=request['rgb'], depth=depth,
                                      ob_mask=request['mask'], iteration=iterations)
        else:
            from tianji_foundationpose.contracts import PoseError, validate_pose

            if tracking_seed['source_sha256'] != record['source_sha256']:
                raise PoseError('tracking_source_mismatch')
            # The official estimator caches a CENTERED mesh pose. Always replace
            # it with this instance's explicit seed: two boxes share resources,
            # but must never inherit one another's tracking state.
            seed = validate_pose(tracking_seed['T_camera_object'])
            torch = self.base.torch
            estimator.pose_last = torch.as_tensor(seed, dtype=torch.float32, device='cuda') @ torch.linalg.inv(
                estimator.get_tf_to_centered_mesh())
            mode, iterations = 'track_one', 2
            pose = estimator.track_one(K=request['K'], rgb=request['rgb'], depth=depth, iteration=iterations)
        cuda.synchronize()
        from tianji_foundationpose.contracts import validate_pose
        pose = validate_pose(pose)
        finished = time.perf_counter()
        return {'T_camera_object': pose.tolist(), 'pose_score': None,
                'search_profile': self.search_profile, 'search_parameters': self.search_parameters,
                'rotation_candidates': int(estimator.rot_grid.shape[0]) if tracking_seed is None else 1,
                'refinement_iterations': iterations, 'estimation_mode': mode,
                'tracking_seed': tracking_seed,
                'score_kind': 'not_exposed_by_official_pose_api', 'model_key': record['cache_key'],
                'source_sha256': record['source_sha256'],
                'T_source_canonical_m': record['T_source_canonical_m'],
                'checkpoint_sha256': self.checkpoint_sha256, 'pose_frame': record['pose_frame'],
                'cache_hit': cache_hit, 'registration_mesh': metadata,
                'timings_s': {'model_resources': initialized - started,
                              'depth_validation': filtered_at - initialized,
                              mode: finished - filtered_at,
                              'total': finished - started, 'process_cpu': time.process_time() - cpu_started},
                'cuda_allocated_bytes': cuda.memory_allocated(),
                'cuda_peak_allocated_bytes': cuda.max_memory_allocated()}
