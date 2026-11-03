from copy import deepcopy
import json

import pytest

from luna_mujoco.collision_archive import freeze_collision_snapshot, archive_collision_snapshot
from luna_mujoco.experiments import archive_value


def scene():
    return {'status': 'available', 'scene_token': 'scene-1', 'observation_id': 'capture-1', 'generation': 2,
        'cuboid_layers': [{'name': 'object', 'model': {'pose': [1, 2, 3, 1, 0, 0, 0]}}],
        'mesh_layers': {'cad': {'mesh_sha256': 'a' * 64, 'instance_id': 'permanent-id',
            'pose_wxyz': [1, 0, 0, 1, 0, 0, 0], 'vertices_local_m': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            'faces': [[0, 1, 2]], 'enabled': True}}}


def test_archive_binds_each_pose_to_stage_and_shares_only_static_mesh(tmp_path):
    live = scene()
    before = deepcopy(live)
    first = freeze_collision_snapshot(live)
    assert live == before
    assert 'vertices_local_m' not in first['mesh_layers']['cad']
    live['scene_token'] = 'scene-2'
    live['mesh_layers']['cad']['pose_wxyz'][0] = 9
    second = freeze_collision_snapshot(live)
    assert len(first['mesh_assets']) == 1
    assert not list(tmp_path.iterdir())
    archive_collision_snapshot(tmp_path / 'run', 'collision/stage-1', first)
    archive_collision_snapshot(tmp_path / 'run', 'collision/stage-2', second)
    saved = json.loads((tmp_path / 'run/collision/stage-1.json').read_text())
    assert saved['scene_token'] == 'scene-1'
    assert saved['mesh_layers']['cad']['pose_wxyz'][0] == 1
    assert 'mesh_assets' not in saved
    asset = saved['mesh_layers']['cad']['mesh_artifact']
    assert json.loads((tmp_path / 'run' / asset).read_text())['faces'] == [[0, 1, 2]]
    assert len(list((tmp_path / 'run/collision/assets').glob('*.json'))) == 1


def test_archive_rejects_unsafe_asset_paths(tmp_path):
    snapshot = scene()
    snapshot['mesh_layers']['cad']['mesh_artifact'] = '../../elsewhere.json'
    with pytest.raises(ValueError, match='collision_asset'):
        archive_collision_snapshot(tmp_path, 'collision/stage', snapshot)


def test_frozen_mesh_is_independent_of_live_arrays():
    live = scene()
    frozen = freeze_collision_snapshot(live)
    live['mesh_layers']['cad']['vertices_local_m'][0][0] = 9
    artifact = frozen['mesh_layers']['cad']['mesh_artifact']
    assert frozen['mesh_assets'][artifact]['vertices_local_m'][0][0] == 0


def test_algorithm_and_stage_records_share_assets_without_duplicate_mesh_arrays(tmp_path):
    frozen = freeze_collision_snapshot(scene())
    archive_value(tmp_path, 'artifacts/build', {'result': {'collision_snapshot': frozen}})
    archive_collision_snapshot(tmp_path, 'collision/stage', frozen)
    raw = json.loads((tmp_path / 'artifacts/build.json').read_text())['result']['collision_snapshot']
    assert 'mesh_assets' not in raw
    assert len(list((tmp_path / 'collision/assets').glob('*.json'))) == 1
    assert (tmp_path / raw['mesh_layers']['cad']['mesh_artifact']).is_file()
    assert 'mesh_assets' in frozen
