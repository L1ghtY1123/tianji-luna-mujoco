"""A clear tool endpoint is not a valid held-arm release pose."""
import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco import demo_placement


def api():
    name = 'luna_mujoco.release_height_search'
    assert importlib.util.find_spec(name), 'held release search must use configured object-gap bounds'
    return importlib.import_module(name)


def inputs():
    pose = np.eye(4)
    pose[:3, 3] = [.5, .2, .6]
    points = np.array([[-.01, -.01, -.075], [.01, .01, -.025]])
    bounds = api().height_search_bounds(pose, points, .5, .05)
    search = {'bounds': bounds, 'input_sha256': 'bound-input', 'arm': 'A',
              'ik_position_threshold_m': .005, 'ik_rotation_threshold_rad': .05,
              'workspace': [{'position_world_m': p, 'reachable': True}
                            for p in api().height_positions(bounds)]}
    request = SimpleNamespace(goal_position_m=pose[:3, 3], goal_quaternion_wxyz=[1., 0., 0., 0.],
                              trajectory_collision_clearance_m=.005)
    world = demo_placement.resolve_release_height([[0, 0, -.075, .005]], [[0, 0, 0, .01]],
        request, lambda spheres: np.zeros(spheres.shape[:-1]), scene_token='scene', search=search)
    return pose, points, search, world


def result(poses, *, success_at=.621, actual_delta=0.):
    items = []
    for index, matrix in enumerate(poses):
        pose = np.asarray(matrix).copy()
        success = bool(pose[2, 3] >= success_at - 1e-9)
        pose[2, 3] += actual_delta
        item = {'candidate_index': index, 'success': success, 'joints_rad': [0.] * 7 if success else None,
                'joint_names': [f'Joint{i}_L' for i in range(1, 8)],
                'position_error_m': abs(actual_delta) if success else .006, 'rotation_error_rad': 0.}
        if success:
            item['achieved_tool_pose'] = {'source': 'curobo_public_compute_kinematics', 'frame': 'world',
                'arm': 'A', 'tool_frame': 'gripper_front_A', 'scene_token': 'scene',
                'joints_rad': [0.] * 7, 'joint_names': item['joint_names'], 'T_world_tcp': pose.tolist()}
        items.append(item)
    return {'scene_token': 'scene', 'attached': True, 'items': items,
            'collision_evidence': {'optimizer_attachment': True},
            'effective_config': {'ik_position_threshold_m': .005, 'ik_rotation_threshold_rad': .05},
            'solve_time_s': .01}


def test_world_clear_low_samples_continue_past_old_20mm_limit_without_fake_collisions():
    _, points, search, world = inputs()
    calls = []
    def solve(poses):
        calls.append(deepcopy(poses))
        return result(poses)
    proof = api().select_held_height(world, search, points, solve)
    assert proof['height_increase_m'] == pytest.approx(.021)
    assert proof['first_world_clear_height']['height_increase_m'] == 0.
    assert [len(poses) for poses in calls] == [1, 25]
    assert all(not row['conflicting_sphere_indices'] for row in proof['tested_heights'])
    assert proof['tested_heights'][20]['held_ik']['success'] is False
    assert proof['selected_object_floor_gap_m'] == pytest.approx(.046)
    assert proof['status'] == 'first_world_and_held_ik_valid_sample'


def test_lowest_valid_height_stops_before_testing_higher_ik_goals():
    _, points, search, world = inputs()
    calls = []

    def solve(poses):
        calls.append(deepcopy(poses))
        return result(poses, success_at=0.)

    proof = api().select_held_height(world, search, points, solve)
    assert [len(poses) for poses in calls] == [1]
    assert proof['selected_sample_index'] == 0
    assert proof['held_ik_sample_count'] == 1


def test_default_release_gap_adds_no_independent_height_above_clearance():
    from luna_mujoco.planning_settings import PlanningSettings
    assert PlanningSettings().place_release_height_m == 0.


def test_achieved_fk_gap_rejects_goal_with_legal_nominal_gap():
    _, points, search, world = inputs()
    with pytest.raises(ValueError, match='release_height_no_held_ik_goal') as caught:
        api().select_held_height(world, search, points,
            lambda poses: result(poses, success_at=.623, actual_delta=.003))
    details = caught.value.details
    assert details['tested_heights'][23]['object_floor_gap_m'] == pytest.approx(.048)
    assert details['tested_heights'][23]['achieved_object_floor_gap_m'] == pytest.approx(.051)
    assert details['tested_heights'][23]['rejection'] == 'achieved_object_floor_gap_exceeded'
    assert details['world_clear_sample_count'] == 26


def test_workspace_uses_same_configured_height_list_beyond_20mm():
    from luna_mujoco.robot_skills.placement_selection import check_height_workspace
    _, _, search, _ = inputs()
    positions = api().height_positions(search['bounds'])
    rows = check_height_workspace(lambda arm, values: [
        {'position_world_m': p, 'reachable': p[2] > .6205} for p in values],
        'A', positions[0], positions=positions)
    assert len(rows) == 26 and rows[21]['reachable']


def test_non_grid_upper_bound_is_included_without_exceeding_configured_gap():
    pose = np.eye(4)
    pose[2, 3] = .6
    bounds = api().height_search_bounds(pose, [[0, 0, -.0746]], .5, .05)
    positions = api().height_positions(bounds)
    assert positions[-1][2] == pytest.approx(.6246)
    assert positions[-2][2] == pytest.approx(.624)
    assert len(positions) == 26


def test_production_resolution_requires_held_ik_and_binds_feedback(monkeypatch):
    from test_resolve_placement import setup

    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    reference = resolution['candidates'][0]['goal']['release_reference']
    assert reference['height_resolution'].get('scope') == 'held_ik_release_endpoint'
    obs['arms']['B']['joints_rad'][0] += .01
    service.calls.clear()
    from luna_mujoco.planning_settings import PlanningSettings
    from luna_mujoco.robot_skills.placement_resolution import resolution_input, validate_resolution

    # The old endpoint certificate remains stale; a future plan must recompute
    # and validate its full current-feedback trajectory before use.
    with pytest.raises(ValueError, match='placement_resolution_changed'):
        validate_resolution(resolution, resolution_input(obs, 'seen-red', 'A', scene, held,
            'demo-destination', request, PlanningSettings().model_dump()))
    assert service.calls == []


def test_actual_release_gap_is_checked_against_full_geometry(monkeypatch):
    from test_resolve_placement import setup

    algorithm, _, obs, scene, held, request, zone = setup(monkeypatch)
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    goal = resolution['candidates'][0]['goal']
    actual = np.asarray(goal['T_world_tcp']).copy()
    assert demo_placement.release_endpoint_evidence(goal, held, actual)['at_release_goal']
    actual[2, 3] += .03
    evidence = demo_placement.release_endpoint_evidence(goal, held, actual)
    assert evidence['object_floor_gap_m'] > .05
    assert evidence['object_floor_gap_valid'] is False and evidence['at_release_goal'] is False


def test_modified_selected_workspace_cannot_validate_as_held_proof(monkeypatch):
    from test_resolve_placement import setup

    algorithm, _, obs, scene, held, request, zone = setup(monkeypatch)
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    goal = resolution['candidates'][0]['goal']
    reference = goal['release_reference']
    reference['height_search']['workspace'][0]['reachable'] = False
    reference['height_resolution']['search']['workspace'][0]['reachable'] = False
    reference['height_resolution']['tested_heights'][0]['workspace']['reachable'] = False
    assert not demo_placement.release_endpoint_evidence(goal, held, goal['T_world_tcp'])['at_release_goal']


def test_production_route_reuses_held_ik_without_nested_gpu_lock(monkeypatch):
    from dataclasses import asdict

    from test_carrying import binding
    from test_pre_release_place import runtime_fixture

    from luna_mujoco.geometry_service import run_operation
    from luna_mujoco.planning_settings import PlanningSettings

    runtime, backend, _, _ = runtime_fixture()
    settings = PlanningSettings()
    runtime.geometry_scene = {'scene_token': 'current', 'observation_id': 'rgbd-2',
                              'generation': 2, 'target_id': 'sam-source'}
    runtime.geometry_solver_profile = settings.solver_payload()
    runtime.configure_planning = lambda value: None
    runtime._select_planner_configuration_unlocked = lambda *args: None
    runtime._refresh_peer_collision_unlocked = lambda **kwargs: None
    pose, points, search, _ = inputs()
    payload = {'arm': 'A', 'scene_token': 'current', 'planning_settings': settings.model_dump(),
        'start_joints_deg': [0.] * 7, 'other_joints_deg': [1.] * 7,
        'gripper_joint_positions_rad': {'A': .4, 'B': .1},
        'carrying_binding': binding(scene_token='current', points_tool_m=points.tolist()),
        'goal_position_m': pose[:3, 3].tolist(), 'goal_quaternion_wxyz': [1., 0., 0., 0.],
        'gripper_spheres_tool_m': [[0, 0, 0, .01]]}
    search['input_sha256'] = api().height_input_digest(payload)
    payload['height_search'] = search
    def world_query(request, gripper, *, search):
        assert runtime.gpu_lock.locked()
        return demo_placement.resolve_release_height([[0, 0, -.075, .005]], gripper, request,
            lambda spheres: np.zeros(spheres.shape[:-1]), scene_token='current', search=search)
    backend.resolve_release_height = world_query
    counts = []
    def solve(request, lock_timeout_s):
        assert not runtime.gpu_lock.locked(), 'public IK acquires the non-reentrant GPU lock itself'
        assert asdict(runtime.carrying_binding)['points_tool_m'] == tuple(map(tuple, points.tolist()))
        assert runtime.gripper_feedback == {'A': .4, 'B': .1}
        counts.append(len(request.goal_matrices_world))
        response = result(request.goal_matrices_world)
        response['scene_token'] = 'current'
        for item in response['items']:
            if item['success']:
                item['achieved_tool_pose']['scene_token'] = 'current'
        runtime.effective_config = {'A': {'ik': {'optimizer_attachment': True,
            'ik_position_threshold_m': .005, 'ik_rotation_threshold_rad': .05}}}
        return SimpleNamespace(to_payload=lambda: response)
    monkeypatch.setattr(runtime, 'solve_ik_batch_native', solve)
    proof = run_operation(runtime, '/tools/resolve_release_height', payload)
    assert proof['height_increase_m'] == pytest.approx(.021)
    assert proof['planner'] == 'release_height_world_and_held_ik'
    assert counts == [100, 100] and proof['held_ik_sample_count'] == 26
    assert runtime.carrying_binding is None and not runtime.gpu_lock.locked()


@pytest.mark.parametrize('case_index,accepted,expected_gap', [(0, True, .049395292), (1, False, .050416), (2, False, .051)])
def test_frozen_b3_full_cad_gap_uses_achieved_fk(case_index, accepted, expected_gap):
    from luna_mujoco.object_obb import held_geometry_points

    frozen = json.loads((Path(__file__).parent / 'fixtures/black_b3_held_release_gap.json').read_text())
    from luna_mujoco.cad_perception import CadCatalog
    cad = frozen['carrying_binding']['cad_tool']
    current = CadCatalog.from_file().geometry(cad['mesh_key'])
    cad.update({key: current[key] for key in ('source_sha256', 'mesh_sha256')})
    case = frozen['cases'][case_index]
    points = held_geometry_points(frozen['carrying_binding'])
    bounds = api().height_search_bounds(case['goal'], points, frozen['floor_z_m'], .05)
    search = {'bounds': bounds, 'arm': 'B', 'input_sha256': 'frozen-diagnostic',
        'ik_position_threshold_m': .005, 'ik_rotation_threshold_rad': .05,
        'workspace': [{'position_world_m': p, 'reachable': True} for p in api().height_positions(bounds)]}
    pose = np.asarray(case['goal'])
    from scipy.spatial.transform import Rotation

    quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    world = demo_placement.resolve_release_height([[0, 0, 0, .01]], [[0, 0, 0, .01]],
        SimpleNamespace(goal_position_m=pose[:3, 3], goal_quaternion_wxyz=quaternion,
                        trajectory_collision_clearance_m=.005),
        lambda spheres: np.zeros(spheres.shape[:-1]), scene_token=frozen['scene_token'], search=search)
    def solve(poses):
        items = [{'candidate_index': index, 'success': False, 'joints_rad': None,
                  'position_error_m': .006, 'rotation_error_rad': 0.} for index in range(len(poses))]
        items[0] = {**deepcopy(case['ik']), 'candidate_index': 0}
        return {'items': items, 'attached': True, 'scene_token': frozen['scene_token'],
                'collision_evidence': frozen['collision_evidence'], 'effective_config': frozen['effective_config']}
    if accepted:
        proof = api().select_held_height(world, search, points, solve)
        assert proof['achieved_object_floor_gap_m'] == pytest.approx(expected_gap, abs=1e-6)
        assert proof['selected_sample_index'] == 0
    else:
        with pytest.raises(ValueError, match='release_height_no_held_ik_goal') as caught:
            api().select_held_height(world, search, points, solve)
        row = caught.value.details['tested_heights'][0]
        assert row['object_floor_gap_m'] < .05
        assert row['achieved_object_floor_gap_m'] > .05
        assert row['rejection'] == 'achieved_object_floor_gap_exceeded'


def test_no_world_clear_samples_never_call_ik():
    _, points, search, world = inputs()
    for row in world['tested_heights']:
        row['conflicting_sphere_indices'] = [0]
        row['world_collision_costs'] = [.01, 0.]
        row['maximum_world_collision_cost'] = .01
    with pytest.raises(demo_placement.ReleaseHeightError) as caught:
        api().select_held_height(world, search, points, lambda poses: pytest.fail('colliding XY entered IK'))
    assert caught.value.details['world_clear_sample_count'] == 0
    assert caught.value.details['held_ik_sample_count'] == 0


def test_production_cannot_publish_geometry_only_response(monkeypatch):
    from test_resolve_placement import setup

    from luna_mujoco.algorithms import AlgorithmError

    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    def compatibility(path, payload):
        payload = {key: value for key, value in payload.items() if key != 'height_search'}
        return service(path, payload)
    algorithm.request = compatibility
    with pytest.raises(AlgorithmError) as error:
        algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_zone=zone)
    assert error.value.code == 'algorithm_invalid_response'


def test_signed_geometry_only_candidate_cannot_enter_production_planner(monkeypatch):
    from test_resolve_placement import setup

    from luna_mujoco.robot_skills.placement_resolution import digest

    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    resolution['candidates'][0]['goal']['release_reference'].pop('height_search')
    resolution.pop('resolution_id')
    resolution['resolution_id'] = digest(resolution)
    service.calls.clear()
    with pytest.raises(ValueError, match='placement_resolution_requires_held_ik'):
        algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination',
            placement_request=request, placement_resolution=resolution)
    assert service.calls == []


@pytest.mark.parametrize('change', ['start_joints_deg', 'other_joints_deg', 'gripper', 'owner', 'settings', 'geometry'])
def test_search_rejects_changed_feedback_geometry_or_settings(change):
    pose, points, search, _ = inputs()
    from luna_mujoco.planning_settings import PlanningSettings

    payload = {'arm': 'A', 'scene_token': 'scene', 'start_joints_deg': [0.] * 7, 'other_joints_deg': [1.] * 7,
        'gripper_joint_positions_rad': {'A': .4, 'B': .1}, 'carrying_binding': {'owner_id': 'owner'},
        'planning_settings': PlanningSettings().model_dump(), 'goal_position_m': pose[:3, 3].tolist(),
        'goal_quaternion_wxyz': [1., 0., 0., 0.]}
    search['input_sha256'] = api().height_input_digest(payload)
    api().validate_search(search, points, payload)
    if change in {'start_joints_deg', 'other_joints_deg'}:
        payload[change][0] += .1
    elif change == 'gripper':
        payload['gripper_joint_positions_rad']['B'] += .1
    elif change == 'owner':
        payload['carrying_binding']['owner_id'] = 'other'
    elif change == 'settings':
        payload['planning_settings']['place_max_release_gap_m'] += .01
    else:
        points = points.copy()
        points[0, 2] += .001
    with pytest.raises(ValueError, match='binding_changed'):
        api().validate_search(search, points, payload)


def test_large_configured_interval_splits_batches_without_dropping_or_retrying_samples():
    pose, points, search, _ = inputs()
    search['bounds'] = api().height_search_bounds(pose, points, .5, .15)
    search['workspace'] = [{'position_world_m': p, 'reachable': True}
                           for p in api().height_positions(search['bounds'])]
    world = demo_placement.resolve_release_height([[0, 0, -.075, .005]], [[0, 0, 0, .01]],
        SimpleNamespace(goal_position_m=pose[:3, 3], goal_quaternion_wxyz=[1., 0., 0., 0.],
                        trajectory_collision_clearance_m=.005),
        lambda spheres: np.zeros(spheres.shape[:-1]), scene_token='scene', search=search)
    seen = []
    def solve(poses):
        seen.extend(matrix[2][3] for matrix in poses)
        return result(poses, success_at=.72)
    proof = api().select_held_height(world, search, points, solve)
    assert proof['height_increase_m'] == pytest.approx(.12)
    assert [batch['sample_count'] for batch in proof['ik_batches']] == [1, 100, 25]
    assert len(set(seen)) == len(seen) == 126


def test_frozen_resolve_plan_feedback_jitter_preserves_resolution_and_is_reported(monkeypatch):
    from test_resolve_placement import setup

    frozen = json.loads((Path(__file__).parent / 'fixtures/black_b3_held_release_gap.json').read_text())
    algorithm, service, obs, scene, held, request, zone = setup(monkeypatch)
    for arm in ('A', 'B'):
        obs['arms'][arm].update(deepcopy(frozen['resolve_feedback'][arm]))
    resolution = algorithm.resolve_placement(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_zone=zone)
    for arm in ('A', 'B'):
        obs['arms'][arm].update(deepcopy(frozen['plan_feedback'][arm]))
    plan = algorithm.plan_place(obs, 'seen-red', 'A', scene, held, 'demo-destination',
        placement_request=request, placement_resolution=resolution)
    proof = plan['resolution_feedback_check']
    assert proof['consistent'] is True
    assert proof['arms']['B']['gripper_delta_rad'] == pytest.approx(1.422616716650893e-7)
    assert proof['arms']['B']['maximum_joint_delta_rad'] < 1e-12
    sent = next(payload for path, payload in reversed(service.calls) if path == '/tools/plan_place')
    assert sent['gripper_joint_positions_rad']['B'] == frozen['plan_feedback']['B']['gripper_position_rad']
