import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_visible_placement import measured_inputs

from jev_mujoco import placement


def tilted_inputs():
    target, destination, holding = measured_inputs()
    grasp = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    grasp[:3, :3] = Rotation.from_euler('xyz', [155., 15., 23.], degrees=True).as_matrix()
    holding['grasp_support_reference']['T_world_tcp_at_grasp'] = grasp.tolist()
    # Measured spheres in the grasp frame, one protrudes 14 mm below support.
    world_centers = np.array([[.51, .07, .626], [.48, .10, .645]])
    local = (world_centers - grasp[:3, 3]) @ grasp[:3, :3]
    spheres = np.column_stack((local, [.04, .02])).astype(np.float32)
    return target, destination, holding, spheres


def test_world_yaw_keeps_tcp_position_with_tilted_off_center_grasp():
    target, destination, holding, spheres = tilted_inputs()
    before = copy.deepcopy((target, destination, holding))
    candidates = placement.release_candidates(target, destination, holding,
        spheres_tool_m=spheres, clearance_m=.01, release_height_m=.015)
    assert len(candidates) == 18
    grasp = np.asarray(holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    center = np.mean(holding['points_tool_m'], axis=0)
    positions = []
    for candidate, degrees in zip(candidates, range(0, 360, 20)):
        tcp = np.asarray(candidate['T_world_tcp'])
        expected_rotation = Rotation.from_euler('z', degrees, degrees=True).as_matrix() @ grasp[:3, :3]
        np.testing.assert_allclose(tcp[:3, :3], expected_rotation, atol=1e-12)
        positions.append((tcp @ [0., 0., .185, 1.])[:3])
        np.testing.assert_allclose(candidate['object_reference_world_m'], tcp[:3, :3] @ center + tcp[:3, 3])
        assert candidate['yaw_delta_world_rad'] == pytest.approx(np.deg2rad(degrees))
        assert candidate['frame'] == 'world'
    np.testing.assert_allclose(positions, np.broadcast_to(positions[0], (18, 3)), atol=1e-12)
    np.testing.assert_allclose(positions[0][:2], destination['placement_reference']['position_world_m'][:2])
    assert not np.allclose(candidates[0]['T_world_tcp'], candidates[2]['T_world_tcp'])
    assert before == (target, destination, holding)


def test_release_height_preserves_requested_gap_despite_conservative_radii():
    target, destination, holding, spheres = tilted_inputs()
    candidates = placement.release_candidates(target, destination, holding,
        spheres_tool_m=spheres, clearance_m=.01, release_height_m=.015)
    for candidate in candidates:
        tcp = np.asarray(candidate['T_world_tcp'])
        centers = spheres[:, :3] @ tcp[:3, :3].T + tcp[:3, 3]
        bottom = np.min(centers[:, 2] - spheres[:, 3])
        floor = destination['placement_reference']['floor_z_m']
        assert bottom - floor == pytest.approx(.012, abs=1e-7)
        ref = candidate['release_reference']
        assert ref['requested_release_height_m'] == .012
        assert ref['release_height_m'] == pytest.approx(.012, abs=1e-7)
        assert ref['attachment_bottom_overhang_m'] == pytest.approx(.014, abs=1e-7)
        assert ref['nominal_object_floor_gap_m'] == pytest.approx(.026, abs=1e-7)


def test_legacy_requested_height_does_not_override_collision_threshold():
    target, destination, holding, spheres = tilted_inputs()
    candidates = placement.release_candidates(target, destination, holding,
        spheres_tool_m=spheres, clearance_m=.01, release_height_m=.05)
    assert all(c['release_reference']['release_height_m'] == .012 for c in candidates)


@pytest.mark.parametrize('spheres', [[], [[0., 0., 0., -1.]], [[0., 0., 0., float('nan')]], [[0., 0., 0.]]])
def test_bad_uploaded_spheres_do_not_produce_a_release_pose(spheres):
    target, destination, holding = measured_inputs()
    with pytest.raises(ValueError, match='spheres'):
        placement.release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)


def test_esdf_voxel_band_cannot_raise_release_goal():
    target, destination, holding, spheres = tilted_inputs()
    candidates = placement.release_candidates(target, destination, holding,
        spheres_tool_m=spheres, clearance_m=.01, release_height_m=.015, map_resolution_m=.01)
    for candidate in candidates:
        tcp = np.asarray(candidate['T_world_tcp'])
        bottom = np.min((spheres[:, :3] @ tcp[:3, :3].T + tcp[:3, 3])[:, 2] - spheres[:, 3])
        assert bottom - destination['placement_reference']['floor_z_m'] == pytest.approx(.012, abs=1e-7)
        assert candidate['release_reference']['map_resolution_allowance_m'] == .01
        assert candidate['release_reference']['release_height_m'] == pytest.approx(.012, abs=1e-7)
