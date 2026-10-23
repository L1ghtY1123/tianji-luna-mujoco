import pytest
from pydantic import ValidationError

from luna_mujoco.demo_program import DemoProgram


def skill(name):
    return {'type': 'skill', 'skill': name, 'arguments': []}


def test_program_accepts_generated_conditional_skills_and_target_loop():
    value = {'version': 1, 'steps': [{'type': 'for_each_target',
        'between_targets': [skill('move_to_observe'), skill('observe_scene')],
        'body': [skill('identify_target'), {'type': 'if_result', 'condition': 'success',
            'then_steps': [skill('generate_grasps')], 'else_steps': [{'type': 'stop', 'reason': 'target unknown'}]}]}]}
    assert DemoProgram.model_validate(value).steps[0].body[1].condition == 'success'


def test_program_rejects_unregistered_motion_and_nested_unbounded_loops():
    for name in ('set_joint_angles', 'bootstrap_scene'):
        with pytest.raises(ValidationError):
            DemoProgram.model_validate({'version': 1, 'steps': [skill(name)]})
    loop = {'type': 'for_each_target', 'between_targets': [], 'body': [skill('identify_target')]}
    with pytest.raises(ValidationError, match='nested_target_loop'):
        DemoProgram.model_validate({'version': 1, 'steps': [dict(loop, body=[loop])]})
