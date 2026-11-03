"""Held placement never rebuilds the pregrasp collision world."""

from copy import deepcopy

import pytest
from test_current_grasp_attachment import fixture

from luna_mujoco.runner import Runner
from luna_mujoco.skills import Registry, SkillSelectionError
from luna_mujoco.task_progress import TaskProgress


def test_build_scene_contract_rejects_held_grasp_and_allows_verified_empty_hands():
    registry = Registry()
    facts = {'observed': True, 'inventory_finalized': True,
             'recovery_needed': False, 'grasp_completed': True, 'grasp_held': True}
    with pytest.raises(SkillSelectionError) as caught:
        registry.validate_selection('build_scene', facts)
    assert caught.value.details['unmet_requirements'] == {
        'grasp_held': {'expected': False, 'actual': True}}
    # A historical completed grasp does not mean that an object is still held.
    facts['grasp_held'] = False
    registry.validate_selection('build_scene', facts)


@pytest.mark.parametrize('entry', ['skill', 'internal'])
async def test_held_build_scene_is_rejected_before_mapping_or_motion(tmp_path, entry):
    runner, context, _, _, captures, _ = fixture(tmp_path)
    await runner.verify_grasp(context)
    calls = []

    def build_scene(*args, **kwargs):
        calls.append(kwargs['exclusion_state'])
        return context['scene']

    runner.algorithms.build_scene = build_scene
    runner.build_scene = Runner.build_scene.__get__(runner)
    scene = deepcopy(context['scene'])
    with pytest.raises(ValueError, match='unknown_skill:build_scene' if entry == 'skill' else 'grasp_scene_requires_empty_hands'):
        if entry == 'skill':
            await runner.execute('build_scene', context)
        else:
            await runner.build_scene(context, 'held')
    assert calls == [] and captures == []
    assert context['scene'] == scene
    assert context['holding']['verified']


@pytest.mark.parametrize('invalid_scene', ['missing', 'refresh_required', 'not_ready'])
async def test_place_retry_rejects_unusable_saved_scene_without_requesting_rebuild(tmp_path, invalid_scene):
    runner, context, _, prepared, captures, maps = fixture(tmp_path)
    await runner.verify_grasp(context)
    runner.task_progress = TaskProgress('place the held object')
    runner.task_progress.fail('source', 'plan_place', 'place_no_valid_candidate', {})
    runner.state['facts'].update(observed=True, scene_ready=True, recovery_needed=True)
    if invalid_scene == 'missing':
        context.pop('scene')
    elif invalid_scene == 'refresh_required':
        context['scene_refresh_required'] = True
    else:
        runner.state['facts']['scene_ready'] = False
    facts = deepcopy(runner.state['facts'])
    failure = deepcopy(runner.task_progress.failure)
    with pytest.raises(ValueError, match='unknown_skill:recover_task'):
        await runner.execute('recover_task', context, {'action': 'retry_place'})
    assert runner.state['facts'] == facts
    assert runner.task_progress.failure == failure
    assert prepared == [] and captures == [] and maps == []


async def test_removed_place_retry_preserves_failure_and_current_scene(tmp_path):
    runner, context, _, prepared, captures, maps = fixture(tmp_path)
    await runner.verify_grasp(context)
    runner.task_progress = TaskProgress('place the held object')
    runner.task_progress.fail('source', 'plan_place', 'place_no_valid_candidate', {})
    scene = deepcopy(context['scene'])
    failure = deepcopy(runner.task_progress.failure)
    with pytest.raises(ValueError, match='unknown_skill:recover_task'):
        await runner.execute('recover_task', context, {'action': 'retry_place'})
    assert runner.task_progress.failure == failure
    assert context['scene'] == scene
    assert prepared == [] and captures == [] and maps == []
