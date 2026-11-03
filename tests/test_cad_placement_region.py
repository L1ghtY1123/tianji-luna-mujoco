from copy import deepcopy

import numpy as np
import pytest
import trimesh

from luna_mujoco.cad_placement_region import complete_placement_reference


def example(mesh=None):
    mesh = mesh if mesh is not None else trimesh.creation.box(extents=[.4, .3, .2])
    geometry = {'vertices_m': mesh.vertices, 'faces': mesh.faces, 'mesh_sha256': 'mesh'}
    target = {'id': 'container', 'cad': {'instance_id': 'container', 'asset_id': 'asset',
        'mesh_sha256': 'mesh', 'observation_id': 'capture', 'generation': 1,
        'pose_validation': {'status': 'validated'}, 'T_world_object': np.eye(4).tolist()},
        'placement_reference': {'observation_id': 'capture', 'floor_z_m': .1,
            'position_world_m': [0, 0, .1], 'source': 'measured_floor',
            'region_polygon_world_xy_m': [[-.03, -.03], [.03, -.03], [.03, .03], [-.03, .03]]}}
    return target, geometry


def test_current_cad_completes_patch_without_claiming_it_was_seen_or_free():
    target, geometry = example()
    original = deepcopy(target)
    result = complete_placement_reference(target, geometry, .002)
    reference = result['placement_reference']
    assert reference['floor_z_m'] == .1
    assert reference['region_measurement']['area_m2'] == pytest.approx(.12)
    assert reference['source'] == 'validated_cad_plane_with_measured_height'
    assert reference['region_measurement']['unseen_area_observed'] is False
    assert reference['region_measurement']['collision_checked'] is False
    assert reference['observed_region_polygon_world_xy_m'] == target['placement_reference']['region_polygon_world_xy_m']
    assert target == original


@pytest.mark.parametrize('field,value', [('instance_id', 'other'), ('observation_id', 'old'),
    ('mesh_sha256', 'wrong'), ('pose_validation', {'status': 'rejected'})])
def test_unverified_or_stale_cad_cannot_expand_region(field, value):
    target, geometry = example()
    target['cad'][field] = value
    with pytest.raises(ValueError, match='binding_mismatch'):
        complete_placement_reference(target, geometry, .002)


def test_a_planar_ring_cannot_be_filled_into_a_support_polygon():
    target, geometry = example(trimesh.creation.annulus(r_min=.05, r_max=.2, height=.2))
    target['placement_reference']['position_world_m'] = [.1, 0, .1]
    result = complete_placement_reference(target, geometry, .002)
    assert result == target


def test_plane_must_match_measured_height_and_contain_measured_anchor():
    for update in ({'floor_z_m': .15}, {'position_world_m': [.5, 0, .1]}):
        target, geometry = example()
        target['placement_reference'].update(update)
        assert complete_placement_reference(target, geometry, .002) == target


def test_registered_blue_container_uses_inner_floor_not_top_rim():
    from luna_mujoco.cad_perception import CadCatalog

    geometry = CadCatalog.from_file().geometry('blue_box')
    target, _ = example()
    target['cad']['mesh_sha256'] = geometry['mesh_sha256']
    target['placement_reference'].update(floor_z_m=-.0675, position_world_m=[0, 0, -.0675])
    result = complete_placement_reference(target, geometry, .00725)
    ref = result['placement_reference']
    assert ref['floor_z_m'] == -.0675
    assert ref['region_measurement']['area_m2'] == pytest.approx(.0949251025)
    assert 0 < ref['region_measurement']['face_count'] < 114


def known_container_without_visible_floor():
    from luna_mujoco.cad_perception import CadCatalog

    catalog = CadCatalog.from_file()
    geometry = catalog.geometry('blue_box')
    target, _ = example()
    target.pop('placement_reference')
    target.update(kind='bin',
                  observation_provenance={'observation_id': 'capture', 'generation': 1})
    target['cad'].update(asset_id='blue_box', mesh_sha256=geometry['mesh_sha256'])
    spec = {'kind': 'interior_floor', 'point_canonical_m': [0., 0., -.0675],
            'normal_canonical': [0., 0., 1.], 'mesh_sha256': geometry['mesh_sha256']}
    return target, geometry, spec


def test_registered_container_floor_does_not_require_visible_floor_pixels():
    target, geometry, spec = known_container_without_visible_floor()
    result = complete_placement_reference(target, geometry, .00725, support_spec=spec)
    ref = result['placement_reference']
    assert ref['source'] == 'validated_cad_support_plane'
    assert ref['floor_z_m'] == pytest.approx(-.0675)
    assert ref['region_measurement']['area_m2'] == pytest.approx(.0949251025)
    assert ref['region_measurement']['unseen_area_observed'] is False
    assert ref['region_measurement']['collision_checked'] is False
    assert ref['observation_id'] == target['observation_provenance']['observation_id']
    assert 'placement_reference' not in target


def test_visible_partial_floor_does_not_shrink_registered_container_after_release():
    target, geometry, spec = known_container_without_visible_floor()
    # A released object hides the near side. Its uncovered depth patch is not
    # the container's full interior, nor necessarily an exactly horizontal fit.
    target['placement_reference'] = {'source': 'sam3_bin_mask_and_measured_horizontal_floor_patch',
        'observation_id': 'capture', 'floor_z_m': -.055,
        'position_world_m': [.1, .07, -.055],
        'region_polygon_world_xy_m': [[.05, .02], [.15, .02], [.15, .12], [.05, .12]],
        'region_measurement': {'method': 'plane_hull_inscribed_rectangle'}}
    original = deepcopy(target)
    ref = complete_placement_reference(target, geometry, .00725, support_spec=spec)['placement_reference']
    assert ref['source'] == 'validated_cad_support_plane'
    assert ref['floor_z_m'] == pytest.approx(-.0675)
    assert ref['region_measurement']['area_m2'] == pytest.approx(.0949251025)
    assert ref['observed_floor_z_m'] == -.055
    assert ref['observed_region_polygon_world_xy_m'] == target['placement_reference']['region_polygon_world_xy_m']
    assert target == original


def test_visible_patch_cannot_allow_stale_registered_cad_support():
    target, geometry, spec = known_container_without_visible_floor()
    target['placement_reference'] = {'floor_z_m': -.0675, 'observation_id': 'capture',
        'position_world_m': [0., 0., -.0675],
        'region_polygon_world_xy_m': [[-.03, -.03], [.03, -.03], [.03, .03], [-.03, .03]]}
    target['cad']['observation_id'] = 'previous'
    with pytest.raises(ValueError, match='binding_mismatch'):
        complete_placement_reference(target, geometry, .00725, support_spec=spec)


@pytest.mark.parametrize('change', ['old_pose', 'wrong_mesh', 'wrong_id', 'rejected_pose'])
def test_catalog_support_needs_current_instance_and_validated_mesh_pose(change):
    target, geometry, spec = known_container_without_visible_floor()
    if change == 'old_pose':
        target['cad']['observation_id'] = 'previous'
    elif change == 'wrong_mesh':
        spec['mesh_sha256'] = 'different'
    elif change == 'wrong_id':
        target['cad']['instance_id'] = 'other'
    else:
        target['cad']['pose_validation']['status'] = 'rejected'
    with pytest.raises(ValueError, match='binding_mismatch'):
        complete_placement_reference(target, geometry, .00725, support_spec=spec)


def test_missing_model_support_metadata_does_not_invent_a_floor():
    target, geometry, _ = known_container_without_visible_floor()
    assert complete_placement_reference(target, geometry, .00725) == target


def test_nonhorizontal_accepted_pose_reports_placement_limit_not_missing_cad():
    from scipy.spatial.transform import Rotation
    from luna_mujoco.robot_skills.placement_zone import perceive_placement_zone

    target, geometry, spec = known_container_without_visible_floor()
    target['cad']['pose_validation'] = {'accepted': True, 'status': 'accepted_by_policy',
                                       'method': 'foundationpose_direct'}
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('x', 50, degrees=True).as_matrix()
    target['cad']['T_world_object'] = pose.tolist()
    result = complete_placement_reference(target, geometry, .00725, support_spec=spec)
    assert result['cad'] == target['cad']  # Pose remains directly accepted.
    assert result['placement_geometry_issue']['reason'] == 'cad_support_not_horizontal'
    assert 'placement_reference' not in result
    with pytest.raises(ValueError, match='cad_support_not_horizontal:container'):
        perceive_placement_zone({'objects': [result]}, 'container', {'relation': 'inside'})


def test_catalog_support_uses_current_pose_and_conservative_world_height():
    from scipy.spatial.transform import Rotation

    target, geometry, spec = known_container_without_visible_floor()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [1, 0, 90], degrees=True).as_matrix()
    pose[:3, 3] = [.46, -.62, .57]
    target['cad']['T_world_object'] = pose.tolist()
    ref = complete_placement_reference(target, geometry, .00725, support_spec=spec)['placement_reference']
    polygon = np.asarray(ref['region_polygon_world_xy_m'])
    from shapely.geometry import Polygon
    # The vertex average depends on triangulation; the area centroid does not.
    assert np.allclose(Polygon(polygon).centroid.coords[0], [.46, -.62], atol=.003)
    assert .5025 < ref['floor_z_m'] < .51
    assert ref['region_measurement']['T_world_object'] == pose.tolist()


def test_catalog_plane_cannot_fill_a_ring_or_use_a_nonexistent_surface():
    target, geometry = example(trimesh.creation.annulus(r_min=.05, r_max=.2, height=.2))
    target.pop('placement_reference')
    target.update(kind='bin', observation_id='capture', generation=1)
    for height in (.1, .15):
        spec = {'kind': 'interior_floor', 'point_canonical_m': [.1, 0., height],
                'normal_canonical': [0., 0., 1.], 'mesh_sha256': 'mesh'}
        result = complete_placement_reference(target, geometry, .002, support_spec=spec)
        assert 'placement_reference' not in result
