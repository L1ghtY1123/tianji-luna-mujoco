"""Action verdicts depend on fresh robot feedback, never cameras or attachment."""
import copy

import pytest
from test_current_grasp_attachment import fixture
from test_release_verification import release_case


async def test_grasp_feedback_succeeds_without_result_capture_or_attachment_service(tmp_path):
    runner, context, _, prepared, _, _ = fixture(tmp_path)
    context.pop('post_grasp_capture')
    result = await runner.execute('verify_grasp', context)
    assert result['status'] == 'held'
    assert result['source'] == 'gripper_position'
    assert runner.state['facts']['grasp_held'] is True
    assert context['holding']['verified'] is True
    assert context['holding']['attached'] is False
    assert prepared == []


async def test_grasp_visual_confirmation_cannot_override_missing_feedback(tmp_path):
    runner, context, current, prepared, _, _ = fixture(tmp_path)
    current['arms']['A']['gripper_position_samples'] = []
    context['visual_result'] = {'observation_id': 'post-grasp', 'status': 'confirmed'}
    result = await runner.verify_grasp(context)
    assert result['status'] == 'unknown'
    assert not context['holding']['verified']
    assert prepared == []


async def test_attachment_is_prepared_separately_after_feedback_verification(tmp_path):
    runner, context, _, prepared, _, _ = fixture(tmp_path)
    await runner.verify_grasp(context)
    assert prepared == []
    await runner.prepare_attachment(context)
    assert len(prepared) == 1 and context['holding']['attached']
    await runner.prepare_attachment(context)
    assert len(prepared) == 1


async def test_release_reads_live_feedback_without_post_place_capture(tmp_path):
    runner, context = release_case(tmp_path)
    feedback = copy.deepcopy(context.pop('post_place_capture')['arms'])
    runner.state['generation'] = 1
    runner.simulation.robot_snapshot = lambda: {'generation': 1, 'time': 4., 'arms': feedback}
    result = await runner.verify_place(context)
    assert result['success'] and result['source'] == 'gripper_position'
    assert 'observation_id' not in result
    assert result['feedback_id']


@pytest.mark.parametrize('status', ['confirmed', 'contradicted'])
async def test_grasp_uses_current_feedback_instead_of_stale_frame(tmp_path, status):
    runner, context, current, prepared, _, _ = fixture(tmp_path)
    context['post_grasp_capture'] = copy.deepcopy(current)
    current['arms']['A']['gripper_position_samples'] = [[t, .8] for t in (0., .1, .2, .3)]
    context['visual_result'] = {'observation_id': 'post-grasp', 'status': status}
    result = await runner.verify_grasp(context)
    assert result['status'] == 'empty'
    assert prepared == []
