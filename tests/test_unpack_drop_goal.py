import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.demo_unpack_workflow import UnpackWorkflow, right_bin_transport_goal
from luna_mujoco.geometry import front_tcp_pose


@pytest.fixture
def frozen():
    return json.loads((Path(__file__).parent / 'fixtures/unpack_bin_occupied_bounds.json').read_text())


def test_v17_cad_drop_uses_complete_registered_cavity_not_visible_floor_fragment():
    value = json.loads((Path(__file__).parent / 'fixtures/cad_drop_v17.json').read_text())
    before = copy.deepcopy(value)
    goal, proof = right_bin_transport_goal(value['lift_tcp'], value['holding'],
        value['destination'], value['scene'], value['observation'])
    assert front_tcp_pose(goal, 'B')[0, 3] < .481
    np.testing.assert_array_equal(goal[:3, :3], np.asarray(value['lift_tcp'])[:3, :3])
    assert proof['region_source'] == 'registered_cad_cavity_bottom_section'
    assert 'carried_projection' not in proof
    held = np.asarray(value['holding']['points_tool_m']) @ goal[:3, :3].T + goal[:3, 3]
    assert held[:, 2].min() >= proof['occupied_top_z_m'] + .10 - 1e-9
    assert proof['z_policy'] == 'shortest_shoulder_wrist_reach_above_minimum_bin_clearance'
    assert value == before


@pytest.mark.parametrize('damage', ['disabled_shell', 'bad_pose'])
def test_cad_drop_still_rejects_oversized_carry_or_invalid_registration(damage):
    value = json.loads((Path(__file__).parent / 'fixtures/cad_drop_v17.json').read_text())
    identifier = value['destination']['id']
    if damage == 'disabled_shell':
        value['scene']['disabled_obstacle_names'].append(f'semantic_instance::{identifier}::bin_shell::cad_part_00')
    else:
        value['scene']['mapping_diagnostics']['bin_shell_completions'][identifier]['T_world_cad'][0][0] = 10
    with pytest.raises(ValueError):
        right_bin_transport_goal(value['lift_tcp'], value['holding'],
            value['destination'], value['scene'], value['observation'])


def test_frozen_robot_points_cannot_raise_bin_transport_goal(frozen):
    original = copy.deepcopy(frozen)
    goal, evidence = right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
        frozen['destination'], frozen['scene'], frozen['observation'])
    assert evidence['minimum_tcp_z_m'] == pytest.approx(.9584201725382142, abs=1e-12)
    assert goal[2, 3] >= evidence['minimum_tcp_z_m']
    assert goal[2, 3] < 1.15
    assert frozen['old_goal'][2][3] == pytest.approx(1.2078796683061652)
    polygon = np.asarray(frozen['destination']['placement_reference']['region_polygon_world_xy_m'])
    assert front_tcp_pose(goal, 'B')[0, 3] < (polygon[:, 0].min() + polygon[:, 0].max()) / 2
    np.testing.assert_array_equal(goal[:3, :3], np.array(frozen['lift_tcp'])[:3, :3])
    held_world = np.asarray(frozen['holding']['points_tool_m']) @ goal[:3, :3].T + goal[:3, 3]
    assert held_world[:, 2].min() >= .6319991443306208 + .10 - 1e-9
    assert evidence['source'] == 'published_inventory_container_occupancy_bounds'
    assert evidence['occupied_top_z_m'] == .6319991443306208
    assert evidence['above_occupancy_allowance_m'] == .10
    assert evidence['layer_name'] == 'semantic_instance::' + frozen['destination']['id']
    assert evidence['cell_count'] == 651
    assert evidence['scene_token'] == frozen['scene']['scene_token']
    assert frozen == original, 'goal derivation must not alter retained voxels or input geometry'


def test_drop_uses_front_target_without_held_projection_gate(frozen):
    frozen['lift_tcp'] = np.eye(4).tolist()
    frozen['holding'] = {'points_tool_m': [[.06, .02, -.09], [.08, .04, -.05]],
        'measurement_provenance': {'measured_reference_tool_m': [.07, .03, -.07]},
        'uncertainty_m': .005, 'margin_m': .001}
    goal, proof = right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
        frozen['destination'], frozen['scene'], frozen['observation'])
    polygon = np.asarray(frozen['destination']['placement_reference']['region_polygon_world_xy_m'])
    center = (polygon.min(0) + polygon.max(0)) / 2
    center[0] -= .25 * np.ptp(polygon[:, 0])
    np.testing.assert_allclose(goal[:2, 3], center)
    assert proof['xy_policy'] == 'finger_front_at_robot_near_bin_half'
    assert 'carried_projection' not in proof
    # Held extent cannot create an extra XY containment requirement.
    frozen['holding']['points_tool_m'][1][0] = np.ptp(polygon[:, 0]) * 2
    wide_goal, _ = right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
        frozen['destination'], frozen['scene'], frozen['observation'])
    np.testing.assert_allclose(wide_goal[:2, 3], goal[:2, 3])


def test_transport_never_reads_raw_bin_points(frozen):
    original, _ = right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
        frozen['destination'], frozen['scene'], frozen['observation'])
    frozen['destination'].pop('points_world_m')
    goal, _ = right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
        frozen['destination'], frozen['scene'], frozen['observation'])
    np.testing.assert_array_equal(goal, original)


@pytest.mark.parametrize('location,key,value', [
    ('header', 'scene_token', 'stale'), ('header', 'observation_id', 'stale'),
    ('header', 'generation', 99), ('observation', 'observation_id', 'stale'),
    ('observation', 'generation', 99), ('scene', 'scene_token', None),
    ('scene', 'generation', None), ('region', 'observation_id', 'stale'),
])
def test_bin_bounds_reject_stale_or_missing_binding(frozen, location, key, value):
    locations = {'header': frozen['scene']['collision_layer_bounds'],
        'scene': frozen['scene'], 'observation': frozen['observation'],
        'region': frozen['destination']['placement_reference']}
    locations[location][key] = value
    with pytest.raises(ValueError, match='drop_requires_current_bin_occupancy'):
        right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
            frozen['destination'], frozen['scene'], frozen['observation'])


@pytest.mark.parametrize('field,value', [
    ('enabled', False), ('kind', 'cuboid'), ('source', 'published_cuboid_bounds'),
    ('bounds_source', 'preview_cells'), ('bounds_source', None),
    ('cell_count', 0), ('cell_count', True), ('voxel_size_m', float('nan')),
    ('minimum_world_m', [0, 0]), ('maximum_world_m', [0, 0, float('inf')]),
    ('maximum_world_m', [0, 0, 0]),
])
def test_bin_bounds_reject_partial_disabled_or_invalid_native_layer(frozen, field, value):
    name = 'semantic_instance::' + frozen['destination']['id']
    frozen['scene']['collision_layer_bounds']['layers'][name][field] = value
    with pytest.raises(ValueError, match='drop_requires_complete_enabled_bin_occupancy'):
        right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
            frozen['destination'], frozen['scene'], frozen['observation'])


@pytest.mark.parametrize('missing', ['header', 'exact_layer', 'disabled_names'])
def test_bounds_never_fall_back_to_raw_bin_or_another_layer(frozen, missing):
    scene = frozen['scene']
    name = 'semantic_instance::' + frozen['destination']['id']
    if missing == 'header':
        scene.pop('collision_layer_bounds')
    elif missing == 'exact_layer':
        scene['collision_layer_bounds']['layers'].pop(name)
    else:
        scene['disabled_obstacle_names'] = [name]
    with pytest.raises(ValueError, match='drop_requires_'):
        right_bin_transport_goal(frozen['lift_tcp'], frozen['holding'],
            frozen['destination'], scene, frozen['observation'])


class WorkflowHarness(UnpackWorkflow):
    def __init__(self, frozen):
        self.observation = frozen['observation'] | {'arms': {'B': {'tcp': frozen['lift_tcp']}},
            'objects': [frozen['destination']]}
        self.state = {'target_id': 'item', 'arm': 'B', 'facts': {}}
        self.demo_task = {'right_bin_id': frozen['destination']['id'], 'transport_ids': ['item', 'next']}
        self.lifecycle = {'appearance': 'unpacked', 'drop_action': {'action_id': 'drop'}}
        self.task_progress = SimpleNamespace(records={'item': {'visual_release': {}}},
            completed_ids=['item'], confirm_visual_transfer=lambda *args: True)
        self.algorithms = SimpleNamespace(plan_attached_pose=self.plan, publish_action_envelope=self.publish)
        self.run_id = 'owner'
        self.calls, self.events = [], []

    def current_lifecycle(self):
        return self.lifecycle

    def measured_state(self, context):
        return self.observation

    async def verify_scene(self, context):
        pass

    async def prepare_attachment(self, context, observation):
        pass

    async def call_algorithm(self, function, *args, **kwargs):
        return function(*args, **kwargs)

    def plan(self, *args, **kwargs):
        self.calls.append(('plan', args, kwargs))
        return {'goal': args[5]}

    def publish(self, scene, target, bounds, action, **kwargs):
        self.calls.append(('envelope', bounds, kwargs))
        return scene | {'scene_token': 'after-envelope'}

    def verify_right_bin_drop(self, *args, **kwargs):
        self.calls.append(('visual', kwargs['reference']['drop_bin_region']))
        return {'success': True, 'visual_evidence': {}}

    async def capture_observation(self, context):
        return {'observation_id': 'post-drop'}

    def record_observation(self, *args):
        pass

    def sync_lifecycle(self, context):
        pass

    def emit(self, *args):
        self.events.append(args)


async def test_transport_visual_region_and_success_envelope_share_native_bounds(frozen):
    runner = WorkflowHarness(frozen)
    context = {'holding': frozen['holding'], 'scene': frozen['scene'],
        'planning_settings': {'place_timeout_s': 180}, 'scene_observation': runner.observation,
        'initial_capture': frozen['observation'], 'pending_transfer': {'verified': False}}
    before = copy.deepcopy(context['scene'])
    plan = await runner._plan_transport(context)
    assert plan['transport_goal_evidence']['occupied_top_z_m'] == .6319991443306208
    assert any(event[0] == 'transport_goal_derived' for event in runner.events)
    await runner._verify_drop(context)
    region = next(call[1] for call in runner.calls if call[0] == 'visual')
    envelope = next(call[1] for call in runner.calls if call[0] == 'envelope')
    assert region['source'] == 'published_inventory_container_occupancy_bounds'
    assert region['scene_token'] == before['scene_token']
    assert region['maximum_world_m'][2] == .6319991443306208
    np.testing.assert_array_equal(envelope['minimum_world_m'], region['minimum_world_m'])
    np.testing.assert_allclose(envelope['maximum_world_m'], np.array(region['maximum_world_m']) + [0, 0, .02])
    assert [call[0] for call in runner.calls] == ['plan', 'visual', 'envelope']
    assert before == frozen['scene'], 'existing map remains unchanged; only the action envelope is published'
