import pytest
from test_runner import Algorithms, Sim

from jev_mujoco.runner import Runner
from jev_mujoco.task_progress import TaskProgress


def release_case(tmp_path):
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    runner.run_id = 'release-test'
    runner.state.update(generation=1, arm='A', target_id='target', destination_id='configured_region',
                        current_step={'skill': 'verify_place'})
    runner.simulation.state['arms']['A'].update(gripper_position_rad=0., gripper_closed_fraction=0.)
    runner.state['facts'].update(released=True, place_execution_completed=True)
    runner.task_progress = TaskProgress('放置目标')
    runner.task_progress.select({'id': 'target'}, 'configured_region')
    context = {'post_place_capture': {'observation_id': 'after-place', 'arms': {'A': {
        'gripper_position_rad': 0., 'gripper_limits_rad': [0., .8], 'gripper_closed_command': False}}},
        'observation': {'objects': [], 'visual_inventory': {'progress': {}}},
        'pending_transfer': {'target_id': 'target'},
        'visual_plan': {'post_place': 'verify_transfer'}}

    async def visual_confirmation(context, *, checkpoint, purpose):
        assert checkpoint == 'post_place' and purpose == 'scene'
        result = {'objects': [], 'observation_id': 'visual-after-place', 'visual_inventory': {'progress': {
            'target': {'status': 'at_destination', 'destination_id': 'configured_region',
                       'observation_id': 'visual-after-place', 'source': 'openai_decisions'}}}}
        context['observation'] = result
        return result

    runner.observe = visual_confirmation
    return runner, context


async def test_completed_place_requires_open_gripper_and_visual_destination_confirmation(tmp_path):
    runner, context = release_case(tmp_path)
    result = await runner.execute('verify_place', context, {})
    assert result['success'] and result['status'] == 'confirmed'
    assert result['source'] == 'gripper_position'
    assert result['verification_scope'] == 'release_and_visual_destination'
    assert result['destination_pose_verified'] is False
    assert result['destination_region_verified'] is True
    assert context['observation']['observation_id'] == 'visual-after-place'
    assert runner.task_progress.completed_ids == ['target']
    assert runner.state['facts']['place_evidence_pending'] is False
    assert (await runner.verify_place(context))['success']


@pytest.mark.parametrize('fact', ['released', 'place_execution_completed'])
async def test_open_gripper_cannot_confirm_an_unfinished_place(tmp_path, fact):
    runner, context = release_case(tmp_path)
    runner.state['facts'][fact] = False
    result = await runner.verify_place(context)
    assert not result['success'] and result['status'] == 'unknown'


@pytest.mark.parametrize('feedback', [
    {},
    {'gripper_position_rad': .4, 'gripper_limits_rad': [0., .8], 'gripper_closed_command': False},
    {'gripper_position_rad': 0., 'gripper_limits_rad': [0., .8], 'gripper_closed_command': True},
])
async def test_visual_confirmation_cannot_replace_missing_release_feedback(tmp_path, feedback):
    runner, context = release_case(tmp_path)
    runner.simulation.robot_snapshot = lambda: {'generation': 1, 'time': 2., 'arms': {'A': feedback}}
    context['visual_result'] = {'observation_id': 'after-place', 'status': 'confirmed'}
    result = await runner.verify_place(context)
    assert not result['success'] and result['status'] == 'unknown'
    assert result['reason'] == 'open_position_not_confirmed'


def test_success_conclusion_describes_release_without_claiming_final_pose():
    from jev_mujoco.run_conclusion import run_conclusion

    result = run_conclusion({'status': 'succeeded', 'verification': {
        'status': 'confirmed', 'verification_scope': 'release_only'}}, [])
    assert '释放' in result['summary'] and '未核验' in result['summary']
