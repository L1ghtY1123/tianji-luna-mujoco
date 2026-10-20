import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from jev_mujoco import reachability

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_workspaces_match_the_current_robot_model():
    from jev_mujoco.workspace import WorkspaceAtlas, model_fingerprint

    atlas = WorkspaceAtlas(ROOT)
    for arm in ('A', 'B'):
        metadata, grid, _, _ = atlas._load(arm)
        assert metadata['model_fingerprint'] == model_fingerprint(ROOT)
        assert metadata['reachable_points'] == int(grid.sum())


def atlas_fixture(root):
    from jev_mujoco.workspace import model_fingerprint
    for name in ('resources/curobo/marvin_gui_dual.yml', 'assets/marvin_urdf/marvin_with_grippers.urdf'):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    folder = root / 'resources/workspaces'
    folder.mkdir()
    for arm in ('A', 'B'):
        reachable = np.zeros((3, 3, 3), bool)
        reachable[1, 1, 1] = arm == 'A'
        metadata = {'schema_version': 1, 'arm': arm, 'model_fingerprint': model_fingerprint(root),
            'origin_m': [.4, 0, .6], 'resolution_m': .01, 'frame': 'Link_Base',
            'reference_offset_m': [0, 0, .185], 'collision_links': ['Link_Base', 'Link_Stand'], 'complete': True}
        np.savez_compressed(folder / f'arm_{arm}.npz', reachable=reachable,
                            metadata=np.array(json.dumps(metadata)))
    return root


def test_lookup_has_bounded_grid_error_and_does_not_invent_boundary_cells(tmp_path):
    from jev_mujoco.workspace import WorkspaceAtlas
    atlas = WorkspaceAtlas(atlas_fixture(tmp_path))
    hit, miss = atlas.query('A', [[.411, .008, .612], [.43, .01, .61]])
    outside, = atlas.query('A', [[5, 0, 2]])
    assert hit['reachable'] and hit['grid_distance_m'] == pytest.approx(.003)
    assert not miss['reachable'] and not outside['reachable']
    assert hit['orientation_constrained'] is False
    assert hit['collision_scope'] == 'single_arm_and_fixed_robot_body'
    assert hit['environment_collision_checked'] is False
    assert not atlas.query('B', [[.41, .01, .61]])[0]['reachable']


def test_model_change_rejects_stale_atlas(tmp_path):
    from jev_mujoco.workspace import WorkspaceAtlas
    root = atlas_fixture(tmp_path)
    (root / 'resources/curobo/marvin_gui_dual.yml').write_text('changed model')
    with pytest.raises(ValueError, match='workspace_model_mismatch'):
        WorkspaceAtlas(root).query('A', [[.41, .01, .61]])


def test_missing_atlas_has_explicit_error(tmp_path):
    from jev_mujoco.workspace import WorkspaceAtlas
    root = atlas_fixture(tmp_path)
    (root / 'resources/workspaces/arm_A.npz').unlink()
    with pytest.raises(ValueError, match='workspace_file_missing'):
        WorkspaceAtlas(root).query('A', [[.41, .01, .61]])


def test_partial_benchmark_cannot_be_loaded_as_a_complete_workspace(tmp_path):
    from jev_mujoco.workspace import WorkspaceAtlas
    root = atlas_fixture(tmp_path)
    path = root / 'resources/workspaces/arm_A.npz'
    with np.load(path) as data:
        metadata, grid = json.loads(str(data['metadata'])), data['reachable']
    metadata['complete'] = False
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), reachable=grid)
    with pytest.raises(ValueError, match='invalid_workspace_artifact'):
        WorkspaceAtlas(root).query('A', [[.41, .01, .61]])


def test_offline_model_excludes_peer_but_keeps_stand_and_head_camera():
    import yaml

    from jev_mujoco.workspace_builder import workspace_robot
    base = yaml.safe_load((ROOT / 'resources/curobo/marvin_gui_dual.yml').read_text())
    for arm, own, peer in [('A', '_L', '_R'), ('B', '_R', '_L')]:
        model = workspace_robot(ROOT, arm)
        kin = model['robot_cfg']['kinematics']
        assert {'Link_Base', 'Link_Stand', 'TCP_Link' + own} <= set(kin['collision_link_names'])
        assert all('Link' + str(i) + peer not in kin['collision_link_names'] for i in range(1, 8))
        assert kin['collision_spheres']['Link_Stand'] == base['robot_cfg']['kinematics']['collision_spheres']['Link_Stand']
        assert len(kin['cspace']['joint_names']) == 7
        assert kin['tool_frames'] == ['jev_workspace_center_' + arm]
        assert kin['extra_links'][kin['tool_frames'][0]]['fixed_transform'][:3] == [0., 0., .185]


def test_invalid_points_rejected_by_shared_contract():
    with pytest.raises(ValueError):
        reachability.validate_query({'arm': 'A', 'positions_world_m': [[float('nan'), 0, 0]]})


def test_surface_preview_preserves_cells_holes_and_lookup(tmp_path):
    from jev_mujoco.workspace import WorkspaceAtlas
    root = atlas_fixture(tmp_path)
    path = root / 'resources/workspaces/arm_A.npz'
    with np.load(path) as data:
        metadata = json.loads(str(data['metadata']))
    grid = np.ones((3, 3, 3), bool)
    grid[1, 1, 1] = False
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), reachable=grid)
    atlas = WorkspaceAtlas(root)
    preview = atlas.preview()
    assert preview['source'] == 'offline_single_arm_workspace'
    assert preview['surface_policy'] == 'exposed_voxel_faces'
    left, right = preview['arms']
    assert left['arm'] == 'A' and right['arm'] == 'B'
    assert left['reachable_points'] == 26 and right['reachable_points'] == 0
    assert sum(len(face['indices']) for face in left['faces']) == 60  # outer 54 + cavity 6
    for face in left['faces']:
        expected = []
        for index in np.flatnonzero(grid):
            cell = np.array(np.unravel_index(index, grid.shape))
            neighbor = cell.copy()
            neighbor[face['axis']] += face['sign']
            if not np.all((neighbor >= 0) & (neighbor < grid.shape)) or not grid[tuple(neighbor)]:
                expected.append(int(index))
        assert face['indices'] == expected
    assert atlas.preview() is preview
    assert not atlas.query('A', [[.41, .01, .61]])[0]['reachable']
    np.testing.assert_array_equal(atlas.maps['A'][1], grid)
