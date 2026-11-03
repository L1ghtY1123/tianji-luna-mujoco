from copy import deepcopy
from types import SimpleNamespace

import pytest

from luna_mujoco.public_planner import PublicPlanner


class Sdk:
    def __init__(self, failure=False):
        self.world = 'original'
        self.updates = []
        self.disabled = []
        self.failure = failure
        self.scene_collision_checker = SimpleNamespace(data=SimpleNamespace(
            meshes=SimpleNamespace(set_enabled=self.set_enabled),
            cuboids=SimpleNamespace(set_enabled=self.set_enabled)))

    def set_enabled(self, name, *, enabled, env_idx):
        assert not enabled and env_idx == 0
        self.disabled.append((self.world, name))

    def update_world(self, scene):
        self.world = scene
        self.updates.append(scene)

    def plan_grasp(self):
        assert self.world == 'reduced'
        if self.failure:
            raise ValueError('optimizer failed')
        return 'trajectory'


@pytest.mark.parametrize('failure', [False, True])
def test_optimizer_world_is_scoped_and_original_restored_before_validation(failure):
    sdk = Sdk(failure)
    planner = PublicPlanner(sdk, SimpleNamespace())
    planner.set_planning_scene('original', 'reduced', {'mesh': ['held']}, {'mesh': ['held::optimizer']})
    if failure:
        with pytest.raises(ValueError, match='optimizer failed'):
            planner.plan_grasp()
    else:
        assert planner.plan_grasp() == 'trajectory'
    assert sdk.world == 'original'
    assert sdk.updates == ['reduced', 'original']
    assert sdk.disabled == [('reduced', 'held::optimizer'), ('original', 'held')]
    assert not planner.scene_restore_failed


def test_restore_failure_cannot_leave_next_planning_call_using_reduced_world():
    sdk = Sdk()
    original_update = sdk.update_world

    def failing_update(scene):
        if scene == 'original':
            raise RuntimeError('upload failed')
        original_update(scene)

    sdk.update_world = failing_update
    planner = PublicPlanner(sdk, SimpleNamespace())
    planner.set_planning_scene('original', 'reduced', {}, {})
    with pytest.raises(RuntimeError, match='upload failed'):
        planner.plan_grasp()
    assert planner.scene_restore_failed
    with pytest.raises(RuntimeError, match='original_collision_scene_not_restored'):
        planner.plan_grasp()


def test_reduced_scene_preserves_original_mesh_pose_names_and_other_obstacles(tmp_path, monkeypatch):
    import hashlib
    import json
    import numpy as np
    import trimesh
    from luna_mujoco import planning_meshes
    from luna_mujoco.planning_meshes import optimizer_scene

    # Legacy proxy support uses a fixture; production now has one low-poly CAD.
    directory = tmp_path / 'planning_meshes'
    directory.mkdir()
    proxy = directory / 'blue_box.ply'
    trimesh.creation.box().export(proxy)
    monkeypatch.setattr(planning_meshes, 'cad_resource_root', lambda: tmp_path)
    manifest = {'version': 1, 'meshes': {
        '2a4330bc0634afb10d2daa6d5ffab215166e1660996cb78398949b64e640d71a': {
            'path': proxy.name, 'units': 'm', 'frame': 'source_visual_canonical',
            'frame_transform': np.eye(4).tolist(), 'asset_id': 'blue_box',
            'source_sha256': 'adca308b03be7f29eb1a3f06abfd06a94f2ae54a30154ba80e12f932026f9f9c',
            'mesh_sha256': hashlib.sha256(proxy.read_bytes()).hexdigest(), 'original_faces': 24}}}
    (directory / 'manifest.json').write_text(json.dumps(manifest))

    original = {'mesh': {'semantic_instance::a': {
        'geometry_source': 'registered_cad', 'asset_id': 'blue_box',
        'file_path': '/original/blue_box.stl',
        'mesh_sha256': '2a4330bc0634afb10d2daa6d5ffab215166e1660996cb78398949b64e640d71a',
        'source_sha256': 'adca308b03be7f29eb1a3f06abfd06a94f2ae54a30154ba80e12f932026f9f9c',
        'vertices': [[0, 0, 0]], 'faces': [[0, 0, 0]], 'pose': [1, 2, 3, 1, 0, 0, 0]}},
        'cuboid': {'table': {'dims': [1, 2, .1], 'pose': [0, 0, 0, 1, 0, 0, 0]}}}
    frozen = deepcopy(original)
    reduced, evidence = optimizer_scene(original)
    assert original == frozen
    assert 'semantic_instance::a' not in reduced['mesh']
    mesh = reduced['mesh'][evidence[0]['optimizer_obstacle']]
    assert len(mesh['faces']) == 12
    assert mesh['file_path'].replace('\\', '/').endswith('/planning_meshes/blue_box.ply')
    assert mesh['pose'] == original['mesh']['semantic_instance::a']['pose']
    assert reduced['cuboid'] == original['cuboid']
    assert evidence[0]['validation_geometry'] == 'original_cad'
    original['mesh']['semantic_instance::a']['source_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='planning_mesh_source_mismatch'):
        optimizer_scene(original)


def test_failed_optimizer_restore_forces_full_scene_upload_despite_cached_token(monkeypatch):
    from test_geometry_scene_reuse import DistanceBackend, mapped_runtime

    runtime, _ = mapped_runtime(monkeypatch)
    backend = DistanceBackend()
    with runtime.carrying_scope(None, {'A': 0., 'B': 0.}):
        token = runtime._refresh_scene_unlocked(arm='A', backend=backend)
        assert backend.features[0] == .4
        backend.features[:] = -.3  # A different world was left active.
        backend.motion_planner.scene_restore_failed = True
        assert runtime._refresh_scene_unlocked(arm='A', backend=backend) == token
    assert backend.features[0] == .4
