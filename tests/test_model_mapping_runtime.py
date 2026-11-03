from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from luna_mujoco.runner import Runner


@pytest.mark.parametrize('scope', ['global', 'local'])
async def test_model_map_uses_all_available_cameras_and_can_retain_unseen_placed_entity(scope):
    observation = {'observation_id': 'fresh', 'objects': [{'id': 'current'}],
                   'views': {key: {} for key in ('1', '2', '3')}}
    scene = {'scene_token': 'new', 'observation_id': 'fresh'}
    runner = SimpleNamespace(holding_state=None,
        state={'target_id': 'current', 'facts': {}},
        algorithms=SimpleNamespace(build_scene=object()),
        call_algorithm=AsyncMock(return_value=scene),
        task_progress=SimpleNamespace(membership={'current': 'required'},
            records={'unseen': {'id': 'unseen', 'status': 'completed'}}))
    context = {'observation': observation, 'planning_settings': {},
        'mapping_lifecycle': {'requested_scope': scope, 'strategy_source': 'decisions',
            'scope_reason': 'Task evidence', 'mapping_entity_ids': ['current']}}
    await Runner.build_scene(runner, context, 'observation')
    actual = runner.call_algorithm.call_args.args[1]
    assert set(actual['views']) == {'1', '2', '3'}
    assert actual['mapping_lifecycle'] == context['mapping_lifecycle']
    assert context['scene'] == scene


async def test_incomplete_update_keeps_measured_map_without_authorizing_motion():
    scene = {'scene_token': 'partial', 'observation_id': 'fresh',
             'inventory_map': {'entities': {'fresh': {'status': 'observed'},
                                          'moved': {'status': 'stale'}}},
             'coverage': {'inventory_complete': False, 'stale_entity_ids': ['moved']}}
    runner = SimpleNamespace(holding_state=None, state={'target_id': None, 'facts': {}},
        algorithms=SimpleNamespace(build_scene=object()), call_algorithm=AsyncMock(return_value=scene),
        task_progress=SimpleNamespace(membership={}, records={}, bound_arm=lambda: None))
    context = {'observation': {'observation_id': 'fresh', 'objects': [], 'views': {}},
        'planning_settings': {}, 'scene': {'scene_token': 'old'},
        'scene_refresh_required': True, 'plan': {'obsolete': True},
        'mapping_lifecycle': {'requested_scope': 'local',
            'invalidated_entity_ids': ['fresh', 'moved', 'consumed'], 'retired_entity_ids': ['consumed']}}
    result = await Runner.build_scene(runner, context, 'observation')
    assert result['status'] == 'unknown'
    assert result['reason'] == 'inventory_map_incomplete'
    assert result['details']['stale_entity_ids'] == ['moved']
    assert result['details']['diagnostic_checks'][0]['actual']['stale'] == ['moved']
    assert context['scene'] == scene
    assert context['scene_refresh_required']
    assert context['mapping_lifecycle']['invalidated_entity_ids'] == ['moved']
    assert not any(runner.state['facts'][key] for key in
                   ('scene_available', 'scene_ready', 'observation_map_ready'))
    assert 'plan' not in context


def test_target_reset_cannot_restore_readiness_of_dirty_scene(tmp_path):
    from policy_processing_fixture import runner_and_context

    runner, context, _ = runner_and_context(tmp_path)
    context['scene']['exclusion_state'] = 'observation'
    context['scene_refresh_required'] = True
    context['mapping_lifecycle'] = {'invalidated_entity_ids': ['pink']}
    runner.reset_transfer(context)
    assert not runner.state['facts']['scene_available']
    assert not runner.state['facts']['observation_map_ready']
    assert context['mapping_lifecycle']['invalidated_entity_ids'] == ['pink']
