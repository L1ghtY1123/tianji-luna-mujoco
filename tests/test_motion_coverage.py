"""Unknown space never becomes free through identity, scope, or endpoint checks."""
import importlib.util

import numpy as np
import pytest


def guard(depth=None):
    assert importlib.util.find_spec('luna_mujoco.motion_coverage'), 'strict motion coverage is missing'
    from luna_mujoco.motion_coverage import MotionCoverage
    depth = np.full((101, 101), 3.) if depth is None else depth
    view = {'depth_left_m': depth, 'valid_left_mask': np.ones_like(depth),
            'left_intrinsics': [[50., 0., 50.], [0., 50., 50.], [0., 0., 1.]],
            'T_base_left_camera': np.eye(4)}
    return MotionCoverage({'head': view}, 'capture-map')


def test_measured_clear_motion_is_accepted_without_semantic_masks():
    coverage = guard()
    result = coverage.check(np.array([[[0., 0., 1., .05]], [[.1, 0., 1., .05]]]),
                            np.array([.1, 0.]), scene_token='capture-map')
    assert result['covered'] is True
    assert result['continuous_segment_envelope_checked'] is True


@pytest.mark.parametrize('sphere', [[0, 0, 3.1, .01], [4, 0, 1, .01], [0, 0, -1, .01],
                                    [1.95, 0, 2, .15]])
def test_occluded_outside_and_partial_fov_are_unknown(sphere):
    result = guard().check(np.array([[sphere]]), np.array([0.]), scene_token='capture-map')
    assert result['covered'] is False
    assert result['failure_reason'] == 'observation_coverage_insufficient'


@pytest.mark.parametrize('invalid', [0., np.nan, np.inf])
def test_invalid_depth_does_not_certify_free_space(invalid):
    depth = np.full((101, 101), 3.)
    depth[50, 50] = invalid
    result = guard(depth).check(np.array([[[0, 0, 1, .02]]]), np.array([0.]), scene_token='capture-map')
    assert result['covered'] is False


def test_motion_envelope_is_checked_even_with_clear_endpoints():
    depth = np.full((101, 101), 3.)
    depth[50, 50] = .7
    spheres = np.array([[[-.2, 0, 1, .01]], [[.2, 0, 1, .01]]])
    assert guard(depth).check(spheres, np.zeros(2), scene_token='capture-map')['covered']
    assert not guard(depth).check(spheres, np.array([.4, 0]), scene_token='capture-map')['covered']


def test_every_link_and_attachment_must_be_covered():
    spheres = np.array([[[0., 0., 1., .02], [4., 0., 1., .02]]])
    result = guard().check(spheres, np.zeros(1), scene_token='capture-map')
    assert not result['covered']
    assert result['first_unobserved_sphere_index'] == 1


def test_stale_scene_and_invalidated_capture_require_observation():
    coverage = guard()
    spheres = np.array([[[0., 0., 1., .02]]])
    assert not coverage.check(spheres, np.zeros(1), scene_token='later-map')['covered']
    coverage.invalidate('object_motion')
    assert not coverage.check(spheres, np.zeros(1), scene_token='capture-map')['covered']


def test_joint_motion_bound_includes_attachment_and_revolute_arc(tmp_path):
    assert importlib.util.find_spec('luna_mujoco.motion_coverage'), 'strict motion coverage is missing'
    from luna_mujoco.motion_coverage import joint_motion_bounds
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('<robot><joint name="shoulder" type="revolute"><parent link="base"/>'
                    '<child link="arm"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>'
                    '<joint name="tool" type="fixed"><parent link="arm"/>'
                    '<child link="tcp"/><origin xyz="1 0 0"/></joint></robot>')
    config = {'robot_cfg': {'kinematics': {'collision_spheres': {'load': [{'center': [1, 0, 0], 'radius': .1}]},
        'extra_links': {'load': {'parent_link_name': 'tcp', 'fixed_transform': [0, 0, 0, 1, 0, 0, 0]}}}}}
    bounds = joint_motion_bounds(urdf, config, ['shoulder'])
    np.testing.assert_allclose(bounds, [2.])
    # Arc travels away from the endpoints: its maximum displacement is still bounded.
    assert bounds[0] * np.pi >= 4.


def test_per_sphere_bound_uses_rotation_axis_and_fixed_transforms(tmp_path):
    from luna_mujoco.motion_coverage import joint_motion_bounds

    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('<robot><joint name="shoulder" type="revolute"><parent link="base"/>'
        '<child link="arm"/><axis xyz="0 0 1"/></joint>'
        '<joint name="tool" type="fixed"><parent link="arm"/><child link="tool"/>'
        '<origin xyz="0 0 2" rpy="0 0 1.5707963267948966"/></joint></robot>')
    config = {'robot_cfg': {'kinematics': {'collision_spheres': {'tool': [
        {'center': [0, 0, 3], 'radius': .1}, {'center': [.2, 0, 3], 'radius': .1}]}}}}
    result = joint_motion_bounds(urdf, config, ['shoulder'], per_sphere=True)
    np.testing.assert_allclose(result['tool'], [[0.], [.2]], atol=1e-15)
    np.testing.assert_allclose(joint_motion_bounds(urdf, config, ['shoulder']), [.2])


def test_capture_identity_does_not_absorb_any_spatial_tolerance():
    from luna_mujoco.motion_coverage import MotionCoverage

    coverage = MotionCoverage({}, 'capture-map')
    sphere = np.array([-.00045805415720678866, .1613452285528183,
                       1.1213138103485107, .058698270469903946], dtype=np.float32)
    coverage.capture_robot_spheres = sphere[None].copy()
    def check(value, envelope):
        return coverage.check(value[None, None], np.array([envelope]),
                              scene_token='capture-map', robot_sphere_indices=[0])
    assert check(sphere, 0.)['covered']
    assert not check(sphere, 1.1840881224957301e-9)['covered']
    assert not check(sphere, float(np.spacing(sphere[3])))['covered']
    moved = sphere.copy()
    moved[0] += .00015
    assert not check(moved, 0.)['covered']
    moved = sphere.copy()
    moved[0] = np.nextafter(moved[0], np.float32(np.inf))
    assert not check(moved, 0.)['covered']
    # A float64 model cannot inherit the much coarser float32 radius resolution.
    assert not check(sphere.astype(np.float64), 1.1840881224957301e-9)['covered']


def test_only_identical_influencing_fk_inputs_have_zero_motion_bound():
    from luna_mujoco.motion_coverage import segment_motion_bounds

    first = np.array([1., 2., 3.])
    same = first + np.spacing(first.astype(np.float32)).astype(float) * .2
    changed = same.copy()
    changed[2] = float(np.nextafter(np.float32(3.), np.float32(np.inf)))
    positions = np.array([first, same, changed])
    coefficients = np.array([[1., 2., 0.], [1., 2., 3.]])
    bounds, stationary = segment_motion_bounds(positions, coefficients, np.float32)
    np.testing.assert_array_equal(stationary, [[True, True], [True, False], [True, True]])
    assert bounds[0].tolist() == [0., 0.]
    assert bounds[1, 0] == 0.
    # The actual represented jump is greater than the requested delta here.
    assert bounds[1, 1] >= 3 * float(np.spacing(np.float32(3.)))
    precise, _ = segment_motion_bounds(positions, coefficients, np.float64)
    assert np.all(precise[0] > 0.)


def test_orbit_bounds_enclose_multijoint_arcs_and_mimic_travel(tmp_path):
    from scipy.spatial.transform import Rotation

    from luna_mujoco.motion_coverage import joint_motion_bounds

    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('<robot><joint name="first" type="revolute"><parent link="base"/>'
        '<child link="arm"/><axis xyz="0 0 1"/></joint>'
        '<joint name="second" type="revolute"><parent link="arm"/><child link="tool"/>'
        '<origin xyz="0.4 0 0.8" rpy="0.2 0.3 0.4"/><axis xyz="0 1 0"/></joint></robot>')
    point = np.array([.2, .3, .5])
    config = {'robot_cfg': {'kinematics': {'collision_spheres': {'tool': [
        {'center': point.tolist(), 'radius': .1}]}}}}
    bounds = joint_motion_bounds(urdf, config, ['first', 'second'], per_sphere=True)['tool'][0]
    rotation = Rotation.from_euler('xyz', [.2, .3, .4]).as_matrix()
    def fk(q):
        return Rotation.from_rotvec([0, 0, q[0]]).apply(
            rotation @ Rotation.from_rotvec([0, q[1], 0]).apply(point) + [.4, 0, .8])
    rng = np.random.default_rng(317)
    for _ in range(60):
        start, delta = rng.uniform(-np.pi, np.pi, 2), rng.uniform(-1, 1, 2)
        for t in [.2, .5, .8, 1.]:
            assert np.linalg.norm(fk(start + t * delta) - fk(start)) <= bounds @ np.abs(delta) + 1e-14
    urdf.write_text(urdf.read_text().replace('<axis xyz="0 1 0"/>',
        '<axis xyz="0 1 0"/><mimic joint="first" multiplier="-2"/>'))
    mimic = joint_motion_bounds(urdf, config, ['first'], per_sphere=True)['tool'][0, 0]
    assert mimic == pytest.approx(bounds[0] + 2 * bounds[1])


def test_prismatic_ancestor_bound_includes_full_translational_range(tmp_path):
    from luna_mujoco.motion_coverage import joint_motion_bounds

    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('<robot><joint name="turn" type="revolute"><parent link="base"/>'
        '<child link="arm"/><axis xyz="0 0 1"/></joint>'
        '<joint name="slide" type="prismatic"><parent link="arm"/><child link="tool"/>'
        '<axis xyz="1 0 0"/><limit lower="0" upper="2"/></joint></robot>')
    config = {'robot_cfg': {'kinematics': {'collision_spheres': {'tool': [
        {'center': [1., 0., 0.], 'radius': .1}]}}}}
    np.testing.assert_allclose(joint_motion_bounds(urdf, config, ['turn', 'slide']), [3., 1.])


def test_actual_command_boundary_rejects_missing_capture_even_when_sdk_reports_collision_free(monkeypatch):
    from test_review_place_coverage_scope import command_boundary

    from luna_mujoco.carrying import check_command_path
    from luna_mujoco.motion_coverage import MotionCoverage
    runtime, planner, payload, binding, calls = command_boundary(monkeypatch, 'observation')
    runtime.motion_coverage = MotionCoverage({}, 'scene')
    result = check_command_path(runtime, planner, payload, binding)
    assert not result['collision_free']
    assert result['failure_reason'] == 'observation_coverage_insufficient'
    assert {kind for _, kind in calls} == {'world', 'self', 'bounds'}


def test_actual_command_boundary_accepts_fully_observed_geometry(monkeypatch):
    from test_review_place_coverage_scope import command_boundary

    from luna_mujoco.carrying import check_command_path
    runtime, planner, payload, binding, _ = command_boundary(monkeypatch, 'observation')
    result = check_command_path(runtime, planner, payload, binding)
    assert result['collision_free']
    assert all(proof['continuous_segment_envelope_checked'] for proof in result['observation_coverage'])


def test_legacy_unbounded_scene_mutation_is_rejected_without_relabeling_free_rays():
    from luna_mujoco.geometry_service import GeometryRuntime

    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'capture-map', 'generation': 1,
                              'target_id': 'item', 'observation_id': 'capture'}
    runtime.geometry_model = {'voxel': {}, 'cuboid': {}, 'mesh': {}}
    runtime.geometry_grids = []
    runtime.motion_coverage = guard()
    import pytest
    with pytest.raises(ValueError, match='scene_update_operation_invalid'):
        runtime.update_perception({'scene_token': 'capture-map', 'generation': 1, 'target_id': 'item',
            'operation': 'unpack_view_motion', 'exclusion_state': 'released', 'action_id': 'unpack',
            'source': 'user_authorized_unpack_visibility_fallback'})
    assert runtime.geometry_scene['scene_token'] == 'capture-map'


def test_unchanged_robot_volume_is_not_permission_for_sweep_growth_or_attachment():
    coverage = guard(np.zeros((101, 101)))
    coverage.capture_robot_spheres = np.array([[0., 0., 1., .2]])
    spheres = np.array([[[0., 0., 1., .2]]])
    assert coverage.check(spheres, np.zeros(1), scene_token='capture-map', robot_sphere_indices=[0])['covered']
    assert not coverage.check(spheres, np.array([.01]), scene_token='capture-map', robot_sphere_indices=[0])['covered']
    assert not coverage.check(spheres, np.zeros(1), scene_token='capture-map')['covered']


def test_measured_free_rays_and_unchanged_self_volume_jointly_cover_motion():
    depth = np.full((101, 101), 3.)
    depth[:, :50] = .7
    coverage = guard(depth)
    coverage.capture_robot_spheres = np.array([[-.1, 0., 1., .22]])
    spheres = np.array([[[.10, 0., 1., .15]]])
    assert not coverage.check(spheres, np.zeros(1), scene_token='capture-map')['covered']
    result = coverage.check(spheres, np.zeros(1), scene_token='capture-map', robot_sphere_indices=[0])
    assert result['covered']
    assert result['initial_robot_volume_is_observed_free'] is False


def test_capture_union_checks_sphere_cell_intersection_not_external_cube_corners():
    from luna_mujoco.motion_coverage import MotionCoverage

    coverage = MotionCoverage({}, 'capture-map')
    coverage.capture_robot_spheres = np.array([[-.5, 0., 0., np.sqrt(1.25)],
                                               [.5, 0., 0., np.sqrt(1.25)]])
    # Each half of the unit ball is inside the nearer captured sphere. Cube
    # corners beyond that ball are irrelevant and are not known free space.
    assert coverage.check(np.array([[[0., 0., 0., 1.]]]), np.zeros(1),
        scene_token='capture-map', robot_sphere_indices=[0])['covered']
    coverage.capture_robot_spheres[:, 3] -= .001
    assert not coverage.check(np.array([[[0., 0., 0., 1.]]]), np.zeros(1),
        scene_token='capture-map', robot_sphere_indices=[0])['covered']


def test_contact_permission_is_limited_to_target_region_and_designated_hand():
    coverage = guard(np.zeros((101, 101)))
    sphere = np.array([[[0., 0., 1., .02]]])
    box = [[-.1, -.1, .9], [.1, .1, 1.1]]
    assert not coverage.check(sphere, np.zeros(1), scene_token='capture-map',
                              contact_bounds=box, contact_sphere_indices=[])['covered']
    assert coverage.check(sphere, np.zeros(1), scene_token='capture-map',
                           contact_bounds=box, contact_sphere_indices=[0])['covered']
    assert not coverage.check(sphere, np.array([.2]), scene_token='capture-map',
                              contact_bounds=box, contact_sphere_indices=[0])['covered']


def test_command_guard_uses_public_link_indices_without_giving_attachment_robot_exemption(monkeypatch):
    from test_review_place_coverage_scope import command_boundary

    from luna_mujoco.carrying import check_command_path

    runtime, planner, payload, binding, _ = command_boundary(monkeypatch, 'observation')
    runtime.motion_coverage = guard(np.zeros((101, 101)))
    runtime.motion_coverage.scene_token = 'scene'
    runtime.motion_coverage.capture_robot_spheres = np.array([[0., 0., 1., .02]])
    payload['positions_rad'] = [[0.] * 7]
    assert check_command_path(runtime, planner, payload, binding)['collision_free']
    runtime, planner, payload, binding, _ = command_boundary(monkeypatch, 'transport')
    # The fixture only creates bindings for place phases; obtain a real bound envelope.
    from dataclasses import asdict

    from test_visible_surface_gpu_contract import measured_binding
    binding = measured_binding()
    payload['carrying_binding'] = asdict(binding)
    payload['positions_rad'] = [[0.] * 7]
    runtime.motion_coverage = guard(np.zeros((101, 101)))
    runtime.motion_coverage.scene_token = 'scene'
    runtime.motion_coverage.capture_robot_spheres = np.array([[0., 0., 1., .02]])
    result = check_command_path(runtime, planner, payload, binding)
    assert result['collision_free']
    assert result['failure_reason'] is None
    assert result['observation_coverage_incomplete']
    assert not result['observation_coverage'][0]['covered']
    assert result['observation_coverage'][0]['first_unobserved_link'] == 'jev_carried_object_A'


def test_repeated_chunks_reuse_exact_coverage_results_and_capture_change_invalidates(monkeypatch):
    coverage = guard(np.zeros((101, 101)))
    spheres = np.tile([0., 0., 1., .02], (128, 20, 1))
    calls = []
    original = coverage.covered_by_union
    def count(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(coverage, 'covered_by_union', count)
    for _ in range(2):
        result = coverage.check(spheres, np.zeros(128), scene_token='capture-map', robot_sphere_indices=range(20))
        assert not result['covered']
        assert result['unobserved_sphere_count'] == 128 * 20
    assert len(calls) == 1
    coverage.capture_robot_spheres = np.array([[0., 0., 1., .02]])
    assert coverage.check(spheres, np.zeros(128), scene_token='capture-map', robot_sphere_indices=range(20))['covered']


def test_coverage_cache_cannot_reuse_contact_permission_or_round_motion_bounds():
    coverage = guard(np.zeros((101, 101)))
    spheres = np.array([[[0., 0., 1., .02]]])
    contact = [[-.02, -.02, .98], [.02, .02, 1.02]]
    assert coverage.check(spheres, np.zeros(1), scene_token='capture-map',
                          contact_bounds=contact, contact_sphere_indices=[0])['covered']
    assert not coverage.check(spheres, np.zeros(1), scene_token='capture-map',
                              contact_bounds=contact, contact_sphere_indices=[])['covered']
    assert not coverage.check(spheres, np.array([.0000001]), scene_token='capture-map',
                              contact_bounds=contact, contact_sphere_indices=[0])['covered']


def test_bulk_coverage_agrees_with_independent_scalar_composition():
    random = np.random.default_rng(42)
    coverage = guard(np.full((101, 101), 1.5))
    coverage.capture_robot_spheres = np.array([[-.5, 0., 1., .3]])
    spheres = np.column_stack((random.uniform(-1., 1., 90), random.uniform(-.8, .8, 90),
                               random.uniform(.2, 2., 90), random.uniform(.005, .05, 90))).reshape(3, 30, 4)
    bounds = random.uniform(0., .02, (3, 30))
    expected = []
    for sample in range(3):
        for index in range(30):
            center = spheres[sample, index, :3]
            radius = spheres[sample, index, 3] + bounds[sample, index]
            expected.append(coverage.sphere_observed(center, radius)
                or coverage.covered_by_union(center, radius, index % 2 == 0, None))
    result = coverage.check(spheres, bounds, scene_token='capture-map', robot_sphere_indices=range(0, 30, 2))
    assert result['unobserved_sphere_count'] == expected.count(False)
    assert (result['first_unobserved_sample_index'], result['first_unobserved_sphere_index']) == divmod(expected.index(False), 30)


def test_command_fail_fast_preserves_first_unknown_without_claiming_full_evaluation():
    coverage = guard(np.zeros((101, 101)))
    spheres = np.tile([0., 0., 1., .02], (128, 20, 1))
    full = coverage.check(spheres, np.zeros(128), scene_token='capture-map')
    fast = coverage.check(spheres, np.zeros(128), scene_token='capture-map', stop_on_first_unknown=True)
    assert not fast['covered']
    assert fast['first_unobserved_sample_index'] == full['first_unobserved_sample_index']
    assert fast['first_unobserved_sphere_index'] == full['first_unobserved_sphere_index']
    assert fast['checked_sphere_count'] == 1
    assert fast['requested_sphere_count'] == 2560
    assert fast['unobserved_sphere_count'] == 1
    assert not fast['coverage_evaluation_complete']
    assert fast['unobserved_sphere_count_is_lower_bound']


@pytest.mark.parametrize('depth, reason', [(0., 'invalid_depth_in_projection'),
    (.5, 'first_return_before_required_depth')])
def test_coverage_failure_explains_camera_evidence_without_claiming_collision(depth, reason):
    coverage = guard(np.full((101, 101), depth))
    result = coverage.check(np.array([[[0., 0., 1., .02]]]), np.zeros(1), scene_token='capture-map')
    assert not result['covered']
    diagnostic = result['camera_certificates'][0]
    assert diagnostic['reason'] == reason
    assert diagnostic['required_free_depth_m'] == pytest.approx(1.03)
    assert result['diagnosis_scope'] == 'whole_sphere_certificate_failure_not_proof_of_physical_collision'
def test_task_scope_includes_camera_descendants_and_all_extra_collision_links(tmp_path):
    from luna_mujoco.motion_coverage import task_interaction_links
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="test"><link name="base"/><link name="tcp"/>
        <link name="camera_mount"/><joint name="tool" type="fixed"><parent link="base"/>
        <child link="tcp"/></joint><joint name="camera" type="fixed"><parent link="tcp"/>
        <child link="camera_mount"/></joint></robot>''')
    kin = {'base_link': 'base', 'collision_link_names': ['base', 'tcp', 'camera_mount', 'peer_attachment'],
           'collision_spheres': {'tcp': [{'center': [0, 0, 0], 'radius': .01}]},
           'extra_links': {'peer_attachment': {'parent_link_name': 'base'}}}
    assert task_interaction_links(path, {'robot_cfg': {'kinematics': kin}}, ['tcp']) == {
        'tcp', 'camera_mount', 'peer_attachment'}
    kin['collision_spheres']['tcp'][0]['radius'] = -1.
    with pytest.raises(ValueError, match='no finite enabled sphere'):
        task_interaction_links(path, {'robot_cfg': {'kinematics': kin}}, ['tcp'])
    kin['collision_spheres']['tcp'][0]['radius'] = .01
    kin['extra_links']['peer_attachment']['parent_link_name'] = 'missing'
    with pytest.raises(ValueError, match='invalid robot chain'):
        task_interaction_links(path, {'robot_cfg': {'kinematics': kin}}, ['tcp'])
    with pytest.raises(ValueError, match='TCP roots missing'):
        task_interaction_links(path, {'robot_cfg': {'kinematics': kin}}, ['missing'])


@pytest.mark.parametrize('indices', [[], [-1], [2], [0.5]])
def test_task_scope_cannot_silently_certify_empty_or_invalid_selection(indices):
    with pytest.raises(ValueError, match='sphere scope'):
        guard().check(np.array([[[0., 0., 1., .02]]]), [0.], scene_token='capture-map',
                      required_sphere_indices=indices)


def test_task_scope_all_disabled_is_not_a_coverage_certificate():
    with pytest.raises(ValueError, match='no enabled geometry'):
        guard().check(np.array([[[0., 0., 1., -1.]]]), [0.], scene_token='capture-map',
                      required_sphere_indices=[0])

