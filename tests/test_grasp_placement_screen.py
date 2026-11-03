"""Fixed robot geometry may reject sampled endpoints before attachment exists."""
import hashlib
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from luna_mujoco.demo_algorithms import DemoAlgorithms
from luna_mujoco.grasp_placement_screen import (
    fixed_gripper_spheres,
    placement_endpoint_poses,
    screen_grasp_placement,
)
from luna_mujoco.planning_settings import PlanningSettings


def geometry():
    return np.array(list(product((-.02, .02), (-.03, .03), (-.04, .04)))) + [.4, .2, .6]


def reference():
    return {'observation_id': 'capture', 'generation': 4, 'floor_z_m': .5,
        'region_polygon_world_xy_m': [[.2, -.8], [.7, -.8], [.7, -.3], [.2, -.3]],
        'source': 'measured_visible_floor_patch'}


def candidate(index=75):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('xyz', [15., 62., -40.], degrees=True).as_matrix()
    pose[:3, 3] = [.4, .2, .64]
    return {'candidate_index': index, 'score': .8, 'source': 'cad_offline_library',
        'grasp_id': 'B_' + str(index), 'T_world_tcp': pose.tolist(), 'ik': {'success': True}}


def query_response(world, *, free=True):
    count = len(world)
    return {'ok': True, 'scene_token': 'scene', 'attached': False,
        'query_sha256': hashlib.sha256(np.asarray(world, dtype='<f8').tobytes()).hexdigest(),
        'pose_collision_free': [free] * count,
        'maximum_world_collision_cost': [0. if free else .01] * count,
        'required_clearance_m': .005}


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_fixed_spheres_match_authored_robot_for_different_closures(arm):
    kin = yaml.safe_load((Path(__file__).parents[1] / 'resources/curobo/marvin_gui_dual.yml')
        .read_text())['robot_cfg']['kinematics']
    prefix, tool = ('gripL_', 'TCP_Link_L') if arm == 'A' else ('gripR_', 'TCP_Link_R')
    indices, offset = [], 0
    for link in kin['collision_link_names']:
        if link.startswith(prefix) or link == tool:
            count = len(kin['collision_spheres'][link])
            if link in (tool, prefix + 'Base'):
                indices.extend(range(offset, offset + count))
            offset += count
    fixed = fixed_gripper_spheres(arm)
    assert fixed.shape == (110, 4)
    previous = None
    for angle in (.05, .7):
        observation = {'arms': {name: {'gripper_position_rad': angle} for name in ('A', 'B')}}
        full = np.asarray(DemoAlgorithms.gripper_release_spheres(observation, arm))
        # Direct URDF rotation and MuJoCo's imported robot differ by <20 nm.
        np.testing.assert_allclose(fixed, full[indices], atol=3e-8, rtol=0)
        if previous is not None:
            np.testing.assert_allclose(previous, full[indices], atol=1e-12, rtol=0)
        previous = full[indices]


def test_endpoints_keep_rotation_full_object_footprint_and_configured_height_cap():
    grasp = candidate()
    points = geometry()
    settings = PlanningSettings(place_max_release_gap_m=.037).model_dump()
    poses, evidence = placement_endpoint_poses(grasp['T_world_tcp'], points, reference(), settings,
        uncertainty_m=.005)
    assert 1 <= evidence['xy_sample_count'] <= 28
    np.testing.assert_allclose(poses[:, :3, :3], np.broadcast_to(np.asarray(grasp['T_world_tcp'])[:3, :3],
        poses[:, :3, :3].shape), atol=0)
    relative = points - np.asarray(grasp['T_world_tcp'])[:3, 3]
    world = relative[None] + poses[:, None, :3, 3]
    assert np.all(world[:, :, 0] >= .21 - 1e-10)
    assert np.all(world[:, :, 0] <= .69 + 1e-10)
    assert np.all(world[:, :, 1] >= -.79 - 1e-10)
    assert np.all(world[:, :, 1] <= -.31 + 1e-10)
    gaps = world[:, :, 2].min(axis=1) - .5
    assert gaps.min() == pytest.approx(.005)
    assert gaps.max() == pytest.approx(.037)
    assert evidence['whole_destination_infeasible'] is False


def test_screen_batches_with_bound_hash_and_preserves_candidate_records():
    candidates = [candidate(index) for index in (75, 42, 11)]
    before = deepcopy(candidates)
    calls = []
    def query(world):
        calls.append(world.copy())
        return query_response(world)
    kept, evidence = screen_grasp_placement(candidates, geometry(), reference(), fixed_gripper_spheres('B'),
        query, scene_token='scene', settings=PlanningSettings().model_dump(), uncertainty_m=.005)
    assert kept == before and candidates == before
    assert len(calls) == 1 and len(calls[0]) == len(candidates)
    assert all(np.prod(world.shape[:2]) <= 262144 and world.shape[1:] == (110, 4) for world in calls)
    assert evidence['accepted_count'] == 3
    assert evidence['attached'] is False and evidence['trajectory_checked'] is False
    assert evidence['whole_destination_infeasible'] is False
    assert all(row['selected_endpoint'] is not None for row in evidence['candidate_evidence'])


def test_all_sampled_base_collisions_reject_candidate_without_global_claim():
    kept, evidence = screen_grasp_placement([candidate()], geometry(), reference(), fixed_gripper_spheres('B'),
        lambda world: query_response(world, free=False), scene_token='scene',
        settings=PlanningSettings().model_dump(), uncertainty_m=.005)
    assert kept == []
    row = evidence['candidate_evidence'][0]
    assert row['accepted'] is False
    assert row['reason'] == 'fixed_gripper_collides_at_all_sampled_endpoints'
    assert 0 < row['sampled_pose_count'] <= 280
    assert row['checked_pose_count'] == row['sampled_pose_count']
    assert evidence['whole_destination_infeasible'] is False


def test_empty_footprint_domain_rejects_without_query_or_global_claim():
    small = reference() | {'region_polygon_world_xy_m': [[0, 0], [.01, 0], [.01, .01], [0, .01]]}
    kept, evidence = screen_grasp_placement([candidate()], geometry(), small, fixed_gripper_spheres('B'),
        lambda world: pytest.fail('no endpoint exists'), scene_token='scene',
        settings=PlanningSettings().model_dump(), uncertainty_m=.005)
    assert kept == [] and evidence['queries'] == []
    assert evidence['candidate_evidence'][0]['sampled_pose_count'] == 0
    assert evidence['whole_destination_infeasible'] is False


def test_clear_endpoint_in_later_query_chunk_retains_candidate(monkeypatch):
    monkeypatch.setattr('luna_mujoco.grasp_placement_screen.MAX_QUERY_SPHERES', 22000)
    calls = []
    def query(world):
        calls.append(len(world))
        result = query_response(world, free=False)
        if len(calls) == 2:
            result['pose_collision_free'][-1] = True
            result['maximum_world_collision_cost'][-1] = 0.
        return result
    candidates = [candidate()]
    kept, evidence = screen_grasp_placement(candidates, geometry(), reference(), fixed_gripper_spheres('B'),
        query, scene_token='scene', settings=PlanningSettings().model_dump(), uncertainty_m=.005)
    assert kept == candidates and len(calls) == 2
    assert evidence['candidate_evidence'][0]['checked_pose_count'] == 8
    assert evidence['candidate_evidence'][0]['accepted'] is True


def test_prefix_queries_preserve_first_clear_witness_across_capacity_boundaries(monkeypatch):
    monkeypatch.setattr('luna_mujoco.grasp_placement_screen.MAX_QUERY_SPHERES', 550)
    item, points, spheres = candidate(), geometry(), fixed_gripper_spheres('B')
    settings = PlanningSettings().model_dump()
    poses, _ = placement_endpoint_poses(item['T_world_tcp'], points, reference(), settings, uncertainty_m=.005)
    expected = spheres[:, :3] @ poses[13, :3, :3].T + poses[13, :3, 3]
    seen = []

    def query(world):
        seen.extend(world.copy())
        free = np.all(np.isclose(world[:, :, :3], expected, atol=1e-12, rtol=0), axis=(1, 2))
        result = query_response(world, free=False)
        result.update(pose_collision_free=free.tolist(),
                      maximum_world_collision_cost=np.where(free, 0., .01).tolist())
        return result

    kept, evidence = screen_grasp_placement([item], points, reference(), spheres, query,
        scene_token='scene', settings=settings, uncertainty_m=.005)
    assert kept == [item]
    np.testing.assert_array_equal(evidence['candidate_evidence'][0]['selected_endpoint'], poses[13])
    assert len(seen) == evidence['candidate_evidence'][0]['checked_pose_count'] < len(poses)
    expected_prefix = spheres[None, :, :3] @ poses[0, :3, :3].T + poses[:len(seen), None, :3, 3]
    np.testing.assert_array_equal(np.asarray(seen)[:, :, :3], expected_prefix)


def test_release_gap_cannot_be_extended_by_higher_configuration():
    poses, evidence = placement_endpoint_poses(candidate()['T_world_tcp'], geometry(), reference(),
        PlanningSettings(place_max_release_gap_m=.1).model_dump(), uncertainty_m=.005)
    assert len(poses)
    assert evidence['object_floor_gaps_m'][-1] == pytest.approx(.05)


def test_small_usable_region_refines_xy_samples_before_declaring_limited_coverage():
    patch = reference() | {'region_polygon_world_xy_m': [[.2, -.5], [.36, -.5], [.36, -.36], [.2, -.36]]}
    _, evidence = placement_endpoint_poses(candidate()['T_world_tcp'], geometry(), patch,
        PlanningSettings().model_dump(), uncertainty_m=.005)
    assert 1 <= evidence['xy_sample_count'] <= 28


@pytest.mark.parametrize('value', ['0', False])
def test_cost_values_must_be_json_numbers(value):
    def query(world):
        result = query_response(world)
        result['maximum_world_collision_cost'][0] = value
        return result
    with pytest.raises(ValueError, match='grasp_placement_query'):
        screen_grasp_placement([candidate()], geometry(), reference(), fixed_gripper_spheres('B'),
            query, scene_token='scene', settings=PlanningSettings().model_dump(), uncertainty_m=.005)


@pytest.mark.parametrize('field,value', [
    ('ok', False), ('scene_token', 'old'), ('attached', True), ('query_sha256', '0' * 64),
    ('required_clearance_m', .001), ('pose_collision_free', []),
    ('maximum_world_collision_cost', []),
])
def test_query_response_must_match_exact_input_and_safety_settings(field, value):
    def query(world):
        return query_response(world) | {field: value}
    with pytest.raises(ValueError, match='grasp_placement_query'):
        screen_grasp_placement([candidate()], geometry(), reference(), fixed_gripper_spheres('B'),
            query, scene_token='scene', settings=PlanningSettings().model_dump(), uncertainty_m=.005)


@pytest.mark.parametrize('bad_cost', [float('nan'), -.1, .1])
def test_query_flags_cannot_claim_clear_despite_invalid_or_positive_cost(bad_cost):
    def query(world):
        result = query_response(world)
        result['maximum_world_collision_cost'][0] = bad_cost
        return result
    with pytest.raises(ValueError, match='grasp_placement_query'):
        screen_grasp_placement([candidate()], geometry(), reference(), fixed_gripper_spheres('B'),
            query, scene_token='scene', settings=PlanningSettings().model_dump(), uncertainty_m=.005)


def demo_inputs():
    points = geometry()
    snapshot = {'observation_id': 'capture', 'generation': 4, 'objects': [
        {'id': 'target', 'points_world_m': points, 'uncertainty_m': .005},
        {'id': 'destination', 'kind': 'bin', 'placement_reference': reference()}],
        'arms': {arm: {'joints_rad': [.1] * 7, 'gripper_position_rad': .2} for arm in ('A', 'B')}}
    scene = {'scene_token': 'scene', 'observation_id': 'capture', 'generation': 4, 'target_id': 'target'}
    return snapshot, scene


def test_demo_override_uses_obb_volume_and_bound_read_only_request(monkeypatch):
    snapshot, scene = demo_inputs()
    # A surface cloud alone would falsely admit the region if OBB corners were omitted.
    snapshot['objects'][0]['points_world_m'] = np.array([[.37, .17, .63], [.43, .23, .65]])
    calls = []
    def request(path, body):
        assert path == '/tools/query_world_spheres'
        assert body['observation_id'] == 'capture' and body['generation'] == 4
        assert body['arm'] == 'B' and body['scene_token'] == 'scene'
        assert body['gripper_joint_positions_rad'] == {'A': .2, 'B': .2}
        assert 'carrying_binding' not in body
        np.testing.assert_allclose(body['start_joints_deg'], np.rad2deg([.1] * 7))
        calls.append(body)
        return query_response(body['spheres_world_m'])
    algorithm = DemoAlgorithms(curobo_request=request)
    candidates = [candidate()]
    kept, evidence = algorithm._grasp_destination_candidates(candidates, snapshot, 'target', 'B', scene,
        PlanningSettings().model_dump(), 'destination')
    assert kept == candidates and calls
    assert evidence['object_geometry_source'] == 'observed_obb'
    assert evidence['object_geometry_point_count'] == 8
    assert evidence['destination_id'] == 'destination'


@pytest.mark.parametrize('destination_id,stale', [(None, False), ('destination', True)])
def test_demo_without_current_destination_region_is_explicitly_deferred(destination_id, stale):
    snapshot, scene = demo_inputs()
    if stale:
        snapshot['objects'][1]['placement_reference']['observation_id'] = 'old'
    algorithm = DemoAlgorithms(curobo_request=lambda *_: pytest.fail('no current region may be queried'))
    candidates = [candidate()]
    kept, evidence = algorithm._grasp_destination_candidates(candidates, snapshot, 'target', 'B', scene,
        PlanningSettings().model_dump(), destination_id)
    assert kept == candidates and evidence['applied'] is False


def test_demo_cad_uses_validated_complete_mesh_and_keeps_library_pose(monkeypatch):
    snapshot, scene = demo_inputs()
    target = snapshot['objects'][0]
    target['cad'] = {'instance_id': 'target', 'observation_id': 'capture', 'generation': 4,
        'asset_id': 'black', 'source_sha256': 'a' * 64, 'mesh_sha256': 'b' * 64,
        'mesh_key': 'black_core', 'pose_validation': {'accepted': True}, 'T_world_object': np.eye(4).tolist()}
    monkeypatch.setattr('luna_mujoco.demo_algorithms.registered_grasp_geometry', lambda *_: geometry())
    algorithm = DemoAlgorithms(curobo_request=lambda path, body: query_response(body['spheres_world_m']))
    candidates = [candidate()]
    kept, evidence = algorithm._grasp_destination_candidates(candidates, snapshot, 'target', 'B', scene,
        PlanningSettings().model_dump(), 'destination')
    assert kept == candidates
    assert evidence['object_geometry_source'] == 'registered_cad'
    assert evidence['object_geometry_point_count'] == len(geometry())
    assert evidence['source_sha256'] == 'a' * 64 and evidence['mesh_sha256'] == 'b' * 64
    target['cad']['instance_id'] = 'wrong'
    with pytest.raises(ValueError, match='cad_pose_binding_stale'):
        algorithm._grasp_destination_candidates(candidates, snapshot, 'target', 'B', scene,
            PlanningSettings().model_dump(), 'destination')


@pytest.mark.parametrize('label,index', [('A4', 116), ('B1', 75)])
def test_frozen_cad_fixed_gripper_witness_survives_current_sampling(label, index):
    import json

    import trimesh

    from luna_mujoco.cad_perception import CadCatalog

    archive = Path(__file__).parents[1] / 'data/cad-obb-iteration-20261030'
    folder = archive / ('black-' + label + '-placement')
    if not folder.exists() or not (archive / 'grasp-placement-cpu-replay.json').exists():
        pytest.skip('local frozen CAD experiment archive is unavailable')
    def load(name):
        return json.loads((folder / name).read_text(encoding='utf-8'))
    if label == 'A4':
        event = load('event-00082.json')
        observation, target, _, scene = event['args']
        arm, destination = event['kwargs']['requested_arm'], event['kwargs']['destination_id']
        grasp = next(row for row in load('event-00087.json')['result'][arm]['valid_candidates']
            if row['candidate_index'] == index)
    else:
        event = load('event-00149.json')
        observation, target, arm, scene, holding, destination = event['args']
        grasp = {'T_world_tcp': holding['grasp_support_reference']['T_world_tcp_at_grasp']}
    settings = event['kwargs']['planning_settings']
    objects = {obj['id']: obj for obj in observation['objects']}
    # Retest the archived physical poses against the current low-poly library.
    for obj in objects.values():
        if obj.get('cad'):
            current = CadCatalog.from_file().geometry(obj['cad']['mesh_key'])
            obj['cad'].update({key: current[key] for key in ('source_sha256', 'mesh_sha256')})
    points = DemoAlgorithms._completed_object_geometry(observation, objects[target])
    poses, evidence = placement_endpoint_poses(grasp['T_world_tcp'], points,
        objects[destination]['placement_reference'], settings,
        uncertainty_m=objects[target].get('uncertainty_m', .005))
    assert 1 <= evidence['xy_sample_count'] <= 28
    frozen = next(row for row in json.loads((archive / 'grasp-placement-cpu-replay.json').read_text())
        if row['label'] == label)
    witness = np.asarray(frozen['selected_endpoint'])
    pose = poses[np.linalg.norm(poses[:, :3, 3] - witness[:3, 3], axis=1).argmin()]
    spheres = fixed_gripper_spheres(arm)
    world = spheres[:, :3] @ pose[:3, :3].T + pose[:3, 3]
    catalog = CadCatalog.from_file()
    for obj in objects.values():
        if obj['id'] == target or obj.get('cad') is None:
            continue
        cad = obj['cad']
        geometry = catalog.geometry(cad['mesh_key'])
        assert geometry['mesh_sha256'] == cad['mesh_sha256']
        mesh = trimesh.Trimesh(vertices=geometry['vertices_m'], faces=geometry['faces'], process=True)
        transform = np.asarray(cad['T_world_object'])
        local = (world - transform[:3, 3]) @ transform[:3, :3]
        _, distance, _ = trimesh.proximity.closest_point(mesh, local)
        separation = np.where(mesh.contains(local), -distance, distance) - spheres[:, 3]
        assert np.min(separation) >= settings['trajectory_collision_clearance_m']
    table = scene['table']
    rotation = Rotation.from_quat(table['pose'][3:], scalar_first=True).as_matrix()
    q = np.abs((world - table['pose'][:3]) @ rotation) - np.asarray(table['dims']) / 2
    distance = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
    assert np.min(distance - spheres[:, 3]) >= settings['trajectory_collision_clearance_m']
