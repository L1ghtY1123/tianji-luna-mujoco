import numpy as np
import pytest

from luna_mujoco.verification import verify_gripper_position


def feedback(position, target=.5):
    return {'gripper_closed_command': False, 'gripper_closing_command': True,
            'gripper_command_position_rad': target, 'gripper_limits_rad': [0., .8],
            'gripper_position_samples': [[0., position], [.1, position], [.3, position]]}


def test_empty_hand_at_guarded_stop_is_not_false_holding():
    assert verify_gripper_position(feedback(.5))['status'] == 'empty'


def test_object_stopping_before_guarded_target_is_holding():
    result = verify_gripper_position(feedback(.35))
    assert result['status'] == 'held'
    assert result['empty_closed_rad'] == .5


@pytest.mark.parametrize('position', [-.0613184682, .8002])
def test_stable_joint_limit_violation_never_proves_holding(position):
    result = verify_gripper_position(feedback(position, .8))
    assert result['status'] == 'unknown'
    assert result['reason'] == 'gripper_position_outside_authored_limits'


@pytest.mark.parametrize('target', [0., -.1, .9, float('nan'), True])
def test_invalid_guarded_target_never_proves_holding(target):
    assert verify_gripper_position(feedback(.3, target))['status'] == 'unknown'


def test_safe_prefix_stops_before_first_failure_even_if_later_samples_recover():
    from luna_mujoco.gripper_sweep import safe_closure_prefix
    assert safe_closure_prefix(np.array([0., .2, .4, .6]), [True, True, False, True]) == .2
    assert safe_closure_prefix(np.array([0., .2]), [False, True]) is None
    assert safe_closure_prefix(np.array([0., .2]), [True, True]) == .2


def test_free_container_orientation_cannot_be_rejected_by_retained_grasp_attitude():
    import json
    from pathlib import Path

    from luna_mujoco.demo_grasp import filter_transport_reach
    frozen = json.loads((Path(__file__).parent / 'fixtures/cad_drop_v17.json').read_text())
    pose = np.asarray(frozen['lift_tcp'])
    points = np.asarray(frozen['holding']['points_tool_m']) @ pose[:3, :3].T + pose[:3, 3]
    candidate = {'index': 7, 'score': .9, 'T_world_tcp': pose.tolist()}
    request = {'orientation_mode': 'free', 'transport_region': {
        'center_world_xy_m': [.480993, -.625003], 'occupied_top_z_m': .6635, 'observation_id': 'current'}}
    accepted, proof = filter_transport_reach([candidate], points, 'B', request)
    assert accepted == [candidate]
    assert proof['applied'] is False
    assert proof['destination_feasibility'] == 'deferred_to_attached_pose_planning'


def test_real_empty_hand_stops_at_bounded_target_without_false_grasp():
    from luna_mujoco.simulation import Simulation
    simulation = Simulation()
    simulation.start()
    try:
        before = simulation.robot_snapshot()
        generation = before['generation']
        simulation.set_gripper('B', True, generation, maximum_position_rad=.5)
        simulation.dwell(.3, generation)
        feedback = simulation.robot_snapshot()['arms']['B']
        assert feedback['gripper_command_position_rad'] == .5
        assert feedback['gripper_closed_command'] is False
        assert verify_gripper_position(feedback)['status'] == 'empty'
        assert feedback['gripper_position_rad'] == pytest.approx(.5, abs=.016)
        np.testing.assert_allclose(feedback['joints_rad'], before['arms']['B']['joints_rad'], atol=.001)
    finally:
        simulation.close()


async def test_bounded_close_reaches_real_motor_through_policy_controller(tmp_path):
    from unittest.mock import AsyncMock

    from luna_mujoco.runtime.controller import PolicyController
    from luna_mujoco.simulation import Simulation
    simulation = Simulation()
    simulation.start()
    try:
        controller = PolicyController(simulation, None, None, tmp_path)
        controller.state.update(generation=simulation.robot_snapshot()['generation'], arm='B',
            target_id='target', current_step={'skill': 'execute_grasp'})
        controller.measured_state = lambda context: simulation.robot_snapshot()
        controller.validate_motion = AsyncMock(return_value={'valid': True})
        context = {'plan': {'command_checks': {'closure': {
            'full_range_rejection': {'safe_driver_limit_rad': .55},
            'grasp_closure_preflight': {'safe_driver_limit_rad': .5}}}}}
        await controller.gripper(context, True)
        assert simulation.robot_snapshot()['arms']['B']['gripper_command_position_rad'] == .5
        assert controller.state['current_step']['motor_commands'] == ['set_gripper']
        assert controller.observation_epoch == 1
        assert controller.validate_motion.call_args.kwargs['closure_driver_limit_rad'] == .5
    finally:
        simulation.close()


