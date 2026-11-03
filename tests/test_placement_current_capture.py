"""Candidate semantics may use a new camera pose over a verified static scene."""
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from test_placement_candidate_grounding import projection_inputs, projector
from test_unified_placement_controller import controller

from luna_mujoco.runtime.placement import candidate_capture_rejection, capture_candidate_view


def context_and_runner(tmp_path):
    runner = controller(tmp_path)
    capture, resolution, held = projection_inputs()
    zone = resolution['placement_zone']
    zone['semantic_scope'] = 'candidate_selection'
    capture['observation_id'] = 'after-grasp'
    context = {'placement_zone': zone, 'latest_capture': capture,
        'capture_epochs': {capture['observation_id']: runner.observation_epoch},
        'scene': {'scene_token': 'verified-scene', 'observation_id': zone['observation_id'],
                  'generation': zone['generation']}, 'holding': held}
    runner.capture_observation = AsyncMock(return_value=capture)
    runner.emit = Mock()
    return runner, context, resolution


async def test_new_capture_keeps_geometry_and_camera_provenance_distinct(tmp_path):
    runner, context, resolution = context_and_runner(tmp_path)
    before = deepcopy(resolution)
    assert candidate_capture_rejection(runner, context, resolution['placement_zone'])['status'] == 'unknown'
    result = await capture_candidate_view(runner, context, resolution['placement_zone'])
    assert result['status'] == 'success'
    assert candidate_capture_rejection(runner, context, resolution['placement_zone']) is None
    projected = projector()(context['latest_capture'], resolution, context['holding'],
                            geometry_binding=context['placement_candidate_view'])
    assert projected['observation_id'] == 'after-grasp'
    assert projected['geometry_observation_id'] == 'now'
    assert resolution == before
    runner.capture_observation.assert_awaited_once()


@pytest.mark.parametrize('change', ['scene', 'geometry_observation', 'generation', 'epoch', 'zone'])
async def test_new_capture_cannot_rebind_changed_geometry_or_robot_evidence(tmp_path, change):
    runner, context, resolution = context_and_runner(tmp_path)
    await capture_candidate_view(runner, context, resolution['placement_zone'])
    if change == 'scene':
        context['scene']['scene_token'] = 'replaced-scene'
    elif change == 'geometry_observation':
        context['scene']['observation_id'] = 'another-measurement'
    elif change == 'generation':
        context['latest_capture']['generation'] += 1
    elif change == 'epoch':
        runner.observation_epoch += 1
    else:
        resolution['placement_zone']['floor_z_m'] += .01
    assert candidate_capture_rejection(runner, context, resolution['placement_zone'])['status'] == 'unknown'


async def test_explicit_fresh_resolve_acquires_without_full_observation(tmp_path, monkeypatch):
    runner, context, _ = context_and_runner(tmp_path)
    runner.verify_scene = AsyncMock()
    runner.measured_state = Mock(return_value={'observation_id': 'now'})
    runner.execution_stage = AsyncMock()
    runner._bind_region = Mock(return_value={})
    runner.algorithms = Mock()
    runner.call_algorithm = AsyncMock(side_effect=RuntimeError('reached_endpoint_search'))
    monkeypatch.setattr('luna_mujoco.runtime.controller.resolve_xy_then_z',
        AsyncMock(side_effect=RuntimeError('reached_endpoint_search')))
    runner.observe = AsyncMock(side_effect=AssertionError('full observation is not allowed here'))
    context.update(planning_settings={}, observation={'observation_id': 'now'})
    with pytest.raises(RuntimeError, match='reached_endpoint_search'):
        await runner.execute('resolve_placement', context, {'capture_mode': 'fresh'})
    runner.capture_observation.assert_awaited_once()
    runner.verify_scene.assert_awaited_once()
    runner.observe.assert_not_awaited()
    assert context['placement_zone']['observation_id'] == 'now'


def test_resolution_schema_offers_explicit_fresh_capture(tmp_path):
    catalog = controller(tmp_path).registry.describe()
    assert catalog['resolve_placement']['arguments']['properties']['capture_mode']['enum'] == ['reuse_current', 'fresh']
