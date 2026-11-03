"""Model-owned sensing/contact decisions, with evidence checks before execution."""

import pytest
from test_runner import Algorithms, Sim

from luna_mujoco.runner import Runner
from luna_mujoco.skills import Registry, SkillSelectionError


@pytest.mark.parametrize('policy', ['ignore_walls', 'held', None])
async def test_unverified_hold_cannot_build_scene_regardless_of_legacy_policy_argument(tmp_path, policy):
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    runner.state.update(target_id='source')
    runner.state['facts']['views_ready'] = True
    async def forbidden(*args):
        pytest.fail('unsubstantiated collision policy reached mapper')
    runner.build_scene = forbidden
    with pytest.raises(ValueError, match='unknown_skill:build_scene'):
        await runner.execute('build_scene', {'observation': {}, 'holding': {'verified': False}}, {'collision_policy': policy})


def test_stepwise_decision_snapshot_entrypoint_is_removed(tmp_path):
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    assert not hasattr(runner, 'snapshot_for_decision')


@pytest.mark.parametrize('skill,facts,pending', [
    ('verify_grasp', {'lifted': True, 'verified': False}, 'grasp_evidence_pending'),
    ('verify_place', {'needs_place': True, 'released': True, 'place_execution_completed': True,
                      'place_verification_completed': False}, 'place_evidence_pending'),
])
def test_verification_reads_fresh_feedback_without_visual_pending_flags(skill, facts, pending):
    registry = Registry()
    registry.validate_selection(skill, facts | {pending: False})
    registry.validate_selection(skill, facts | {pending: True})


def test_retired_visual_result_skill_cannot_be_selected():
    facts = {'observed': True, 'visual_available': True, 'result_capture_ready': True}
    registry = Registry()
    for inspected in (False, True):
        with pytest.raises(SkillSelectionError, match='unknown_skill'):
            registry.validate_selection('inspect_visual_result', facts | {'result_inspected': inspected})
