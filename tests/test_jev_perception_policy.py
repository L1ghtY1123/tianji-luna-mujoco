"""Model-owned sensing/contact decisions, with evidence checks before execution."""
from copy import deepcopy

import pytest
from test_runner import Algorithms, Sim

from jev_mujoco.runner import Runner
from jev_mujoco.skills import Registry, SkillSelectionError


async def test_atomic_scene_preparation_uses_fixed_policy_without_nested_jev_choice(tmp_path):
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    runner.state.update(target_id='source')
    runner.state['facts'].update(observed=True, views_ready=True)
    context = {'observation': {'observation_id': 'fresh', 'source': 'rgbd', 'objects': []}}
    seen = []

    class Judgment:
        async def choose_visual_plan(self, state, skill, requests):
            assert skill == 'build_scene'
            return requests

    async def prepare(current):
        seen.append('selected')
        return {'exclusion_state': 'selected'}

    runner.prepare_grasp_scene = prepare
    arguments = await runner.select_arguments('build_scene', context, Judgment())
    result = await runner.execute('build_scene', context, arguments)
    assert arguments == {'visual_plan': {'fine': 'initialize_inventory'}}
    assert result['exclusion_state'] == 'selected'
    assert seen == ['selected']


@pytest.mark.parametrize('policy', ['ignore_walls', 'held', None])
async def test_unverified_hold_cannot_build_scene_regardless_of_legacy_policy_argument(tmp_path, policy):
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    runner.state.update(target_id='source')
    runner.state['facts']['views_ready'] = True
    async def forbidden(*args):
        pytest.fail('unsubstantiated collision policy reached mapper')
    runner.build_scene = forbidden
    with pytest.raises(ValueError, match='grasp_scene_requires_empty_hands'):
        await runner.execute('build_scene', {'observation': {}, 'holding': {'verified': False}}, {'collision_policy': policy})


def test_jev_sees_capture_freshness_visual_reuse_and_coverage_limit(tmp_path):
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    runner.state['facts']['observed'] = True
    # Minimal real public-observation shape, no hidden environment geometry.
    context = {'observation': {'observation_id': 'fresh', 'source': 'rgbd', 'objects': []},
               'scene': {'observation_id': 'old', 'scene_token': 'map'},
               'observation_checkpoint': 'fine', 'scene_refresh_required': True,
               'visual_scene': {'observation_id': 'fresh', 'status': 'unknown'}}
    original = deepcopy(context)
    evidence = runner.snapshot_for_decision(context)['perception_state']
    assert evidence['observation_id'] == 'fresh'
    assert evidence['scene_observation_id'] == 'old'
    assert evidence['scene_matches_observation'] is False
    assert evidence['scene_refresh_required'] is True
    assert evidence['visual_scene_matches_observation'] is True
    assert evidence['unobserved_space_collision_checked'] is False
    assert context == original


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
