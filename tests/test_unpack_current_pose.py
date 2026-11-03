"""Processing uses verified action evidence and never a task phase or future arm."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from processing_fixtures import before_kwargs, capture_and_action, client, release_evidence, work_area

from luna_mujoco.robot_skills.operations import RobotOperations
from luna_mujoco.robot_skills.processing import ProcessingSkills, create_unpack_product
from luna_mujoco.robot_skills.processing_evidence import verify_unpacking
from luna_mujoco.vla_configuration import vla_unpack_configuration


class ProcessingHarness(ProcessingSkills):
    def __init__(self):
        self.state = {'target_id': 'pink', 'generation': 3, 'facts': {}}
        self.task_inventory = {'work_area_id': 'work'}
        self.run_id = 'run'
        self.task_progress = SimpleNamespace(scene_entities={'pink': {'color': 'pink'}})
        self.simulation = SimpleNamespace(robot_snapshot=lambda: {'generation': 3, 'arms': {
            arm: {'joints_rad': np.deg2rad(values).tolist()}
            for arm, values in vla_unpack_configuration()['preparation_joints_deg'].items()}})
        self.check_generation = Mock()
        self.emit = Mock()
        self.unpack_effect = SimpleNamespace(execute=Mock(return_value={
            'action_id': 'action', 'completed': True, 'target_id': 'pink', 'owner_id': 'run',
            'generation': 3, 'source': 'mock_vla'}))


def released_context():
    release = {**release_evidence(), 'destination_id': 'work'}
    return {'planning_settings': {'vla_unpack': {'input_x_m': .36, 'input_y_m': 0.}},
        'place_plan': {'goal': {'release_reference': {'object_id': 'pink', 'object_reference_world_xy_m': [.36, 0.]}}},
        'release_endpoint_evidence': {'at_release_goal': True, 'actual_object_reference_world_xy_m': [.36, 0.], 'rotation_error_rad': 0.}, 'place_verification': {'success': True, 'visual_evidence': release},
        'pending_transfer': {'target_id': 'pink', 'destination_id': 'work', 'verified': True},
        'post_place_capture': {'generation': 3},
        'vla_pose_execution': {'target_id': 'pink', 'generation': 3, 'release_feedback_id': 'table-release'},
        'destination': {'id': 'work', 'placement_reference': {**work_area(), 'region_polygon_world_xy_m': [[.2, -.16], [.52, -.16], [.52, .16], [.2, .16]], 'observation_id': 'measured'}},
        'scene': {'scene_token': 'retained'}}


def test_effect_receipt_never_injects_geometry_or_completes_product():
    runner, context = ProcessingHarness(), released_context()
    before = deepcopy(context['scene'])
    result = asyncio.run(runner._vla_unpack(context))
    assert result['success']
    assert context['processing_action']['processing_skill'] == 'vla_unpack'
    assert context['processing_action']['table_release_feedback_id'] == 'table-release'
    assert context['processing_action']['process_id'].startswith('process-')
    assert context['scene'] == before
    assert runner.state['target_id'] == 'pink'
    with pytest.raises(ValueError, match='already_dispatched'):
        asyncio.run(runner._vla_unpack(context))
    runner.unpack_effect.execute.assert_called_once()


@pytest.mark.parametrize('receipt', [None, 'wrong_delivery', 'wrong_generation', 'valid'])
def test_processing_honors_required_intermediate_return_and_inventory_observation(receipt):
    runner, context = ProcessingHarness(), released_context()
    runner.environment_assumptions = {'completion_requirements': {'after_each_object': {
        'robot_pose': 'observation', 'observe_inventory': True}}}
    release = context['place_verification']['visual_evidence']
    proof = {'status': 'verified', 'generation': 3, 'inventory_observed': True,
        'delivery': {'entity_id': 'pink', 'confirmed_observation_id': release['observation_id']}}
    if receipt == 'wrong_delivery':
        proof['delivery']['confirmed_observation_id'] = 'older-release'
    if receipt == 'wrong_generation':
        proof['generation'] = 2
    runner.task_progress.records = {'pink': {'post_delivery_observation': proof if receipt else {}}}
    if receipt == 'valid':
        assert runner._verified_processing_release(context) == release
    else:
        with pytest.raises(ValueError, match='processing_requires_post_delivery_observation'):
            runner._verified_processing_release(context)
    runner.unpack_effect.execute.assert_not_called()


@pytest.mark.parametrize('damage', [None, 'owner', 'image', 'generation', 'feedback'])
def test_verified_release_survives_rebuild_only_with_unchanged_capture_proof(damage):
    from luna_mujoco.robot_skills.evidence import table_release_reference

    runner, context = ProcessingHarness(), released_context()
    capture, _ = capture_and_action()
    capture['generation'] = runner.state['generation']
    release = context['place_verification']['visual_evidence']
    release['observation_id'] = capture['observation_id']
    context['post_place_capture'] = capture
    context['post_place_verification_input'] = table_release_reference(capture, release, runner.run_id)
    context.pop('pending_transfer')
    if damage == 'owner':
        runner.run_id = 'another-run'
    elif damage == 'image':
        capture['views']['1']['rgb'][0, 0, 0] ^= 1
    elif damage == 'generation':
        capture['generation'] -= 1
    elif damage == 'feedback':
        release['release_feedback_id'] = 'another-release'
    if damage:
        with pytest.raises(ValueError):
            runner._verified_processing_release(context)
    else:
        assert runner._verified_processing_release(context) == release


@pytest.mark.parametrize('mutation', ['held', 'unverified', 'other_target', 'other_destination', 'stale', 'old_pose'])
def test_effect_rejects_unbound_release_and_preparation(mutation):
    runner, context = ProcessingHarness(), released_context()
    if mutation == 'held':
        context['holding'] = {'verified': True}
    elif mutation == 'unverified':
        context['place_verification']['success'] = False
    elif mutation == 'other_target':
        context['pending_transfer']['target_id'] = 'other'
    elif mutation == 'other_destination':
        context['place_verification']['visual_evidence']['destination_id'] = 'other'
    elif mutation == 'stale':
        context['post_place_capture']['generation'] = 2
    else:
        context['vla_pose_execution']['release_feedback_id'] = 'older-release'
    with pytest.raises(ValueError):
        asyncio.run(runner._vla_unpack(context))
    runner.unpack_effect.execute.assert_not_called()


def verified_result():
    capture, action = capture_and_action()
    action.update(process_id='process-test', processing_skill='vla_unpack')
    visual, _ = client()
    result = verify_unpacking(visual, capture, target_id='pink', release_evidence=release_evidence(),
        action_evidence=action, owner_id='run', work_area=work_area(), **before_kwargs())
    return capture, action, result


def test_product_has_distinct_id_and_verified_process_lineage():
    capture, action, result = verified_result()
    product = create_unpack_product('pink', action, result, capture, owner_id='run')
    assert product['id'] != 'pink'
    assert product['input_entity_id'] == 'pink'
    assert product['processing_skill'] == 'vla_unpack'
    assert product['process_id'] == action['process_id']
    assert product['creation_evidence']['observation_id'] == capture['observation_id']
    assert 'geometry_prior' not in product


@pytest.mark.parametrize('mutation', ['uncertain', 'wrong_input', 'wrong_action', 'stale', 'wrong_owner', 'low_confidence'])
def test_product_creation_rejects_broken_visual_action_chain(mutation):
    capture, action, result = verified_result()
    if mutation == 'uncertain':
        result['success'] = False
    elif mutation == 'wrong_input':
        action['target_id'] = 'other'
    elif mutation == 'wrong_action':
        action['action_id'] = 'other'
    elif mutation == 'stale':
        action['generation'] = 2
    elif mutation == 'wrong_owner':
        action['owner_id'] = 'other'
    else:
        result['visual_evidence']['confidence'] = .79
    with pytest.raises(ValueError, match='confirmed_visible_result'):
        create_unpack_product('pink', action, result, capture, owner_id='run')


def test_result_obstacle_verification_precedes_explicit_observation_motion():
    move = AsyncMock()
    runner = SimpleNamespace(move_to_observe=move)
    with pytest.raises(ValueError, match='observation_motion_requires_current_map'):
        asyncio.run(RobotOperations.execute(runner, 'move_to_observe', {
            'scene_refresh_required': True, 'processing_action': {'action_id': 'pending'}, 'pending_transfer': {'verified': True}}))
    move.assert_not_awaited()


def test_grasp_planning_rejects_robot_away_from_observation_pose():
    runner = SimpleNamespace(require_grasp_view=Mock(side_effect=ValueError('mapping_requires_observation_pose')))
    with pytest.raises(ValueError, match='mapping_requires_observation_pose'):
        asyncio.run(RobotOperations.plan_grasp(runner, {}, 'pink', 'A', {}))


def test_grasp_planning_rejects_map_from_different_observation():
    runner = SimpleNamespace(require_grasp_view=Mock())
    context = {'observation_pose_capture_id': 'fresh', 'scene': {'observation_id': 'old'},
               'observation': {'observation_id': 'fresh'}}
    with pytest.raises(ValueError, match='grasp_requires_current_observation_scene'):
        asyncio.run(RobotOperations.plan_grasp(runner, {'observation_id': 'fresh'}, 'pink', 'A', context))


class VerificationHarness(ProcessingHarness):
    def __init__(self, capture, result, *, at_observation_pose):
        super().__init__()
        self.state['facts']['observation_pose_ready'] = at_observation_pose
        self.capture = capture
        self.result = result
        self.checked_arms = []
        self.task_progress = SimpleNamespace(register_product=Mock())
        self.sync_inventory = Mock()
        self.record_observation = Mock()
        self.algorithms = SimpleNamespace(measure_local_object=self.measure,
            measured_object_workspace=self.workspace, publish_local_object=self.publish,
            omit_unpack_obstacle=Mock(side_effect=AssertionError('visible result needs measured obstacle')))

    def _processing_support(self, context):
        return work_area()

    async def capture_observation(self, context):
        return self.capture

    async def call_algorithm(self, function, *args, **kwargs):
        return function(*args, **kwargs)

    def verify_unpacking(self, *args, **kwargs):
        assert kwargs['release_evidence']['target_id'] == 'pink'
        assert 'lifecycle' not in kwargs
        return deepcopy(self.result)

    def measure(self, *args):
        return {'observation_id': self.capture['observation_id']}

    def workspace(self, measured, arm):
        self.checked_arms.append(arm)
        return {'reachable': arm == 'A'}

    def publish(self, scene, *args):
        return {'scene': scene, 'measurement': self.measure()}

    def reset_transfer(self, context):
        self.state['target_id'] = None
        self.state['facts'].clear()


@pytest.mark.parametrize('at_observation_pose', [False, True])
def test_verification_uses_current_evidence_without_changing_pose(at_observation_pose):
    capture, action, result = verified_result()
    runner = VerificationHarness(capture, result, at_observation_pose=at_observation_pose)
    context = released_context()
    reference = before_kwargs()
    context.update(processing_action=action, post_place_capture=reference['table_release_capture'],
        post_place_verification_input=reference['table_release_reference'], visual_inventory_reference={})
    result = asyncio.run(runner._verify_unpack(context))
    assert result['success']
    assert runner.checked_arms == ['A', 'B']
    product = runner.task_progress.register_product.call_args.args[0]
    assert result['product_id'] == result['output_entity_id'] == product['id']
    assert runner.state['target_id'] == product['id']
    assert result['workspace_evidence']['arm'] is None
    assert result['workspace_evidence']['feasible_arms'] == ['A']
    assert context['unpacked_grasp_capture'] is capture
    assert runner.state['facts']['observation_pose_ready'] is at_observation_pose
    assert not runner.state['facts']['observed']


def test_processing_verification_does_not_choose_or_publish_a_map():
    capture, action, result = verified_result()
    runner = VerificationHarness(capture, result, at_observation_pose=False)
    runner.algorithms.publish_local_object = Mock(side_effect=AssertionError('mapping belongs to the graph'))
    context = released_context()
    reference = before_kwargs()
    context.update(processing_action=action, post_place_capture=reference['table_release_capture'],
        post_place_verification_input=reference['table_release_reference'], visual_inventory_reference={})
    result = asyncio.run(runner._verify_unpack(context))
    assert result['success']
    runner.algorithms.publish_local_object.assert_not_called()
    assert runner.state['facts']['scene_available'] is False
    assert context['scene_refresh_required'] is True


@pytest.mark.parametrize('fault', [None, 'generation', 'joints', 'missing_capture'])
def test_current_grasp_view_uses_measured_capture_not_configured_pose(fault):
    capture = {'observation_id': 'measured', 'generation': 3,
               'arms': {arm: {'joints_rad': [.3] * 7} for arm in ('A', 'B')}}
    live = deepcopy(capture)
    from test_demo_runner import TEST_SETTINGS
    runner = object.__new__(RobotOperations)
    runner.state = {'generation': 3, 'target_id': 'pink'}
    runner.simulation = SimpleNamespace(robot_snapshot=lambda: live)
    context = {'observation': capture, 'planning_settings': TEST_SETTINGS.model_dump()}
    if fault == 'generation':
        capture['generation'] = 2
    elif fault == 'joints':
        live['arms']['B']['joints_rad'][0] += .1
    elif fault == 'missing_capture':
        context.pop('observation')
    if fault:
        with pytest.raises(ValueError, match='grasp_view'):
            RobotOperations.require_grasp_view(runner, context)
    else:
        RobotOperations.require_grasp_view(runner, context)


@pytest.mark.parametrize('at_observation_pose', [False, True])
def test_missing_current_depth_keeps_processing_unknown_and_blocks_return(at_observation_pose):
    capture, action, result = verified_result()
    runner = VerificationHarness(capture, result, at_observation_pose=at_observation_pose)
    runner.algorithms.measure_local_object = Mock(side_effect=ValueError('local_object_depth_missing'))
    context = released_context()
    reference = before_kwargs()
    context.update(processing_action=action, post_place_capture=reference['table_release_capture'],
        post_place_verification_input=reference['table_release_reference'], visual_inventory_reference={})
    result = asyncio.run(runner._verify_unpack(context))
    assert result['status'] == 'unknown' and not result['success']
    assert result['observation_assessment']['missing_evidence'] == ['current_object_depth']
    assert context['scene_refresh_required'] is True
    assert 'product_id' not in result
