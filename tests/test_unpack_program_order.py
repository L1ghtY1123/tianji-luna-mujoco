import pytest

from luna_mujoco.demo_program import DemoProgram, order_unpack_observation


def call(name):
    return {'type': 'skill', 'skill': name, 'arguments': []}


def program(sequence):
    return DemoProgram.model_validate({'version': 1, 'steps': [{'type': 'for_each_target',
        'body': [call('verify_unpack'), {'type': 'if_result', 'condition': 'success', 'then_steps': [],
            'else_steps': [{'type': 'stop', 'reason': 'unconfirmed'}]}, *[call(name) for name in sequence]],
        'between_targets': []}]})


def test_only_reorders_explicit_observation_after_verification_guard():
    source = program(['identify_target', 'identify_destination', 'choose_arm', 'observe_scene', 'build_scene'])
    result, changes = order_unpack_observation(source)
    body = result.steps[0].body
    assert [n.skill for n in body if n.type == 'skill'] == [
        'verify_unpack', 'observe_scene', 'identify_target', 'identify_destination', 'choose_arm', 'build_scene']
    assert body[1].type == 'if_result'
    assert source.steps[0].body[2].skill == 'identify_target'
    assert len(changes) == 1
    assert len(body) == len(source.steps[0].body)


def test_correct_order_is_unchanged():
    source = program(['observe_scene', 'identify_target', 'build_scene'])
    result, changes = order_unpack_observation(source)
    assert result == source and changes == []


@pytest.mark.parametrize('sequence', [
    ['identify_target', 'build_scene'], ['identify_target', 'build_scene', 'observe_scene'],
    ['identify_target', 'move_to_observe', 'observe_scene']])
def test_missing_or_motion_delayed_observation_is_rejected(sequence):
    with pytest.raises(ValueError, match='program_unpack_observation_required'):
        order_unpack_observation(program(sequence))
