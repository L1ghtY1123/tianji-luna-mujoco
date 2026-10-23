"""Known CAD shape is located by current depth, never a simulator body pose."""
import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from luna_mujoco.cad_assets import cad_asset


def prior():
    asset = cad_asset('blue_box')
    return {'kind': 'bin', 'color': 'blue', 'instance_id': 'bin',
        'observation_id': 'capture', 'generation': 3,
        'source': 'user_supplied_cad_shape', 'asset_id': 'blue_box',
        'cad_sha256': asset['source']['sha256'], 'outer_dimensions_m': asset['size_m']}


def surfaces(*, walls=4, yaw=31., translation=(.23, .58, .4725)):
    asset = cad_asset('blue_box')
    sections = asset['cavity_center_sections_m']
    z = np.array([section['z'] for section in sections])
    widths = np.array([section['half_size_xy'] for section in sections])
    slope, intercept = np.polyfit(z, widths, 1)
    x, y = np.meshgrid(np.linspace(-.16, .16, 37), np.linspace(-.11, .11, 29))
    points = [np.c_[x.ravel(), y.ravel(), np.full(x.size, asset['cavity_floor_z_m'])]]
    for axis, sign in ((0, 1), (1, 1), (0, -1), (1, -1))[:walls]:
        lateral, height = np.meshgrid(np.linspace(-.10, .10, 31), np.linspace(-.05, .06, 29))
        wall = np.empty((lateral.size, 3))
        wall[:, axis] = sign*(intercept[axis]+slope[axis]*height.ravel())
        wall[:, 1-axis] = lateral.ravel()
        wall[:, 2] = height.ravel()
        points.append(wall)
    rotation = Rotation.from_euler('z', yaw, degrees=True)
    return rotation.apply(np.concatenate(points))+translation, rotation, np.asarray(translation)


def fit(points):
    from luna_mujoco.demo_bin_geometry import fit_bin_shell
    return fit_bin_shell(points, instance_id='bin', observation_id='capture', generation=3, prior=prior())


def test_current_depth_locates_sloped_cad_and_keeps_all_collision_parts_and_empty_cavity():
    points, rotation, center = surfaces()
    original = points.copy()
    result = fit(points)
    assert result['source'] == 'known_cad_measured_registration'
    assert result['synthetic_geometry_is_measured'] is False
    assert result['asset_id'] == 'blue_box'
    assert result['cad_sha256'] == prior()['cad_sha256']
    assert set(result['meshes']) == {'cad_surface'}
    transform = np.asarray(result['T_world_cad'])
    np.testing.assert_allclose(transform[:3, 3], center, atol=.001)
    np.testing.assert_allclose(transform[:3, :3], rotation.as_matrix(), atol=.002)
    assert result['registration']['model_agreement_fraction'] >= .85
    assert result['measured_point_count'] == len(points)
    np.testing.assert_array_equal(points, original)
    # Every published mesh stays in the native centered CAD frame. Its pose
    # alone supplies the measured rigid placement; no box fills the cavity.
    for mesh in result['meshes'].values():
        np.testing.assert_allclose(mesh['pose'][:3], center, atol=.001)
        assert 'vertices' in mesh and 'faces' in mesh and 'dims' not in mesh
    assert result['floor_height_m'] == pytest.approx(.405, abs=.001)


@pytest.mark.parametrize('damage', ['one_wall', 'ambiguous_adjacent_walls', 'tilted', 'wrong_size', 'nonfinite'])
def test_unobservable_or_inconsistent_depth_does_not_invent_a_cad_pose(damage):
    points, _, center = surfaces(walls={'one_wall': 1, 'ambiguous_adjacent_walls': 2}.get(damage, 4))
    if damage == 'tilted':
        points = Rotation.from_euler('x', 12, degrees=True).apply(points-center)+center
    elif damage == 'wrong_size':
        points[:, 0] = center[0]+1.3*(points[:, 0]-center[0])
    elif damage == 'nonfinite':
        points[0, 1] = np.nan
    with pytest.raises(ValueError, match='demo_bin'):
        fit(points)


@pytest.mark.parametrize('field,value', [('cad_sha256', 'wrong'), ('asset_id', 'black_workpiece'),
                                       ('observation_id', 'old'), ('generation', 2)])
def test_cad_prior_requires_current_inventory_and_exact_known_shape(field, value):
    from luna_mujoco.demo_bin_geometry import validate_bin_priors
    valid = prior()
    validate_bin_priors({'bin': valid}, {'bin'}, 'capture', 3)
    invalid = copy.deepcopy(valid)
    invalid[field] = value
    with pytest.raises(ValueError, match='demo_bin'):
        validate_bin_priors({'bin': invalid}, {'bin'}, 'capture', 3)


@pytest.mark.parametrize('name,expected_y', [('source', .605), ('right', -.605)])
def test_real_registered_depth_recovers_cad_pose_without_receiving_ground_truth(name, expected_y):
    points = np.load(Path(__file__).parent/'fixtures/cad_bin_registered_depth.npz')[name]
    result = fit(points)
    transform = np.asarray(result['T_world_cad'])
    # Expected scene pose is test-only, never an input to registration.
    np.testing.assert_allclose(transform[:3, 3], [.20, expected_y, .4725], atol=.001)
    assert abs(transform[1, 0]) > .99999
    assert result['registration']['model_agreement_fraction'] > .99
    assert result['floor_height_m'] == pytest.approx(.405, abs=.001)
    assert result['rim_height_m'] == pytest.approx(.545, abs=.001)


def test_real_depth_registration_is_equivariant_to_unconfigured_yaw_and_translation():
    points = np.load(Path(__file__).parent/'fixtures/cad_bin_registered_depth.npz')['source']
    change = Rotation.from_euler('z', -23, degrees=True)
    shift = np.array([.11, -.30, .07])
    result = fit(change.apply(points)+shift)
    np.testing.assert_allclose(np.asarray(result['T_world_cad'])[:3, 3],
                               change.apply([.20, .605, .4725])+shift, atol=.001)
    assert result['registration']['model_agreement_fraction'] > .99


@pytest.mark.parametrize('name,expected_y', [('source', .573), ('right', -.625)])
def test_v16_nearby_wall_fits_describe_one_box_without_requiring_submillimeter_uniqueness(name, expected_y):
    points = np.load(Path(__file__).parent/'fixtures/cad_bin_v16_inventory.npz')[name]
    before = points.copy()
    result = fit(points)
    np.testing.assert_allclose(np.asarray(result['T_world_cad'])[:3, 3], [.481, expected_y, .571], atol=.001)
    np.testing.assert_array_equal(points, before)
    assert result['registration']['model_agreement_fraction'] > .999
    assert result['registration']['candidate_count'] >= 3
    assert .003 < result['registration']['maximum_candidate_surface_displacement_m'] < .006
    assert result['registration']['pose_equivalence_tolerance_m'] == .006
    assert set(result['meshes']) == {'cad_surface'}


def test_v16_registration_uses_measured_pose_after_world_frame_change():
    points = np.load(Path(__file__).parent/'fixtures/cad_bin_v16_inventory.npz')['source']
    rotation = Rotation.from_euler('z', 17, degrees=True)
    shift = np.array([-.07, .13, -.025])
    result = fit(rotation.apply(points)+shift)
    np.testing.assert_allclose(np.asarray(result['T_world_cad'])[:3, 3],
                               rotation.apply([.481, .573, .571])+shift, atol=.001)


def test_pose_equivalence_accounts_for_rotated_rim_not_only_center():
    from luna_mujoco.demo_bin_cad import _candidate_surface_spread
    candidates = [(1., np.eye(3), np.zeros(3)),
                  (1., Rotation.from_euler('z', 2, degrees=True).as_matrix(), np.zeros(3))]
    assert _candidate_surface_spread(candidates, cad_asset('blue_box')['size_m']) > .006


def test_pose_equivalence_uses_every_pair_without_chain_merging():
    from luna_mujoco.demo_bin_cad import _candidate_surface_spread
    candidates = [(1., np.eye(3), np.array([x, 0., 0.])) for x in (0., -.005, .005)]
    assert _candidate_surface_spread(candidates, cad_asset('blue_box')['size_m']) == pytest.approx(.01)


def test_half_turn_does_not_create_false_shape_ambiguity():
    from luna_mujoco.demo_bin_cad import _candidate_surface_spread
    candidates = [(1., np.eye(3), np.zeros(3)),
                  (1., Rotation.from_euler('z', 180, degrees=True).as_matrix(), np.zeros(3))]
    assert _candidate_surface_spread(candidates, cad_asset('blue_box')['size_m']) < 1e-12
