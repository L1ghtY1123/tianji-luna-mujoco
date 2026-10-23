from copy import deepcopy

import pytest
from geometry_perception_fixture import GeometryPerception
from test_broad_perception import PromptSegmenter
from test_inventory_image_order import head_observation, progress_for
from test_perception import capture_fixture
from test_route_observation import route_context

from luna_mujoco.perception import PerceptionError
from luna_mujoco.task_progress import TaskProgress


def test_scene_inventory_fixes_entities_roles_queries_and_table_before_motion():
    observation = head_observation()
    observation['table'] = {'surface_height_m': .6, 'position': [.5, 0., .6]}
    for obj in observation['objects']:
        obj['color'] = 'red' if obj['kind'] != 'bin' else 'blue'
    progress = progress_for(observation)
    progress.freeze_inventory(observation)
    initial = deepcopy(progress.scene_entities)
    assert initial['tl']['roles'] == ['transport', 'collision']
    assert initial['obstacle']['roles'] == ['collision']
    assert initial['bin']['roles'] == ['container', 'collision']
    assert initial['region']['roles'] == ['target_region']
    assert initial['table']['roles'] == ['table', 'collision']
    assert initial['table']['observation_id'] == 'head-1'
    assert initial['tl']['segmentation_query'] == {'kind': 'cube', 'color': 'red', 'prompt': 'red cube'}

    progress.observe([{'id': 'tl', 'kind': 'object', 'color': 'red', 'position': [1., 2., 3.]},
                      {'id': 'new', 'kind': 'cube', 'color': 'blue'}], 'later')
    assert set(progress.scene_entities) == set(initial)
    assert progress.scene_entities['tl']['kind'] == 'cube'
    assert progress.scene_entities['tl']['position'] == [1., 2., 3.]
    assert progress.scene_entities['bin']['visibility'] == 'unseen'
    assert progress.scene_entities['table'] == initial['table']


def test_fixed_queries_do_not_discover_shapes_or_apply_cube_synonym_fallback():
    capture, masks = capture_fixture()
    detector = PromptSegmenter({'red sphere': masks, 'cube': masks})
    queries = [{'kind': 'sphere', 'color': 'red', 'prompt': 'red sphere', 'entity_ids': ['ball-1', 'ball-2']},
               {'kind': 'bin', 'color': 'blue', 'prompt': 'blue open box', 'entity_ids': ['box']}]
    result = GeometryPerception(detector, identities=('ball-1', 'ball-2')).observe(capture, checkpoint='fine', source_hint={
        'kind': 'any', 'include_containers': True, 'scene_queries': queries})
    assert detector.calls == ['red sphere', 'blue open box'] * 3
    assert len(result['objects']) == 2
    assert {obj['id'] for obj in result['objects']} == {'ball-1', 'ball-2'}
    assert all('candidate_entity_ids' not in obj for obj in result['objects'])


def test_alias_queries_preserve_visual_ids_from_both_initial_descriptions():
    capture, masks = capture_fixture()
    first = GeometryPerception(PromptSegmenter({'cube': [masks[0]], 'block': [masks[1]]}),
        prompts={'first': 'cube', 'second': 'block'}).observe(
        capture, source_hint={'kind': 'cube', 'camera_ids': ['3']})
    progress = TaskProgress('move both')
    progress.initialize_scene_inventory(first)

    class Aliases:
        def segment(self, rgb, text, timestamp_us):
            return [{'mask': mask, 'score': .95 if text.endswith('block') else .9} for mask in masks]

    current = GeometryPerception(Aliases(), identities=('first', 'second', 'first', 'second')).observe(capture, source_hint={
        'scene_queries': progress.scene_queries(), 'camera_ids': ['3'], 'table': first['table']})
    assert len(first['objects']) == len(current['objects']) == 2
    assert {obj['id'] for obj in current['objects']} == {obj['id'] for obj in first['objects']}


def test_fixed_inventory_queries_do_not_drop_geometry_at_discrete_color_boundary():
    capture, masks = capture_fixture()
    for view in capture['views'].values():
        for mask in masks:
            view['rgb'][mask] = [191, 191, 191]
    first = GeometryPerception(PromptSegmenter({'cube': masks})).observe(
        capture, source_hint={'kind': 'cube', 'camera_ids': ['3']})
    progress = TaskProgress('move both')
    progress.initialize_scene_inventory(first)
    for view in capture['views'].values():
        for mask in masks:
            view['rgb'][mask] = [192, 192, 192]
    detector = PromptSegmenter({query['prompt']: masks for query in progress.scene_queries()})
    current = GeometryPerception(detector).observe(capture, checkpoint='fine', source_hint={
        'environment_queries': [{'kind': 'object', 'color': 'gray', 'prompt': 'gray object'}],
        'scene_queries': progress.scene_queries(), 'camera_ids': ['3'], 'table': first['table']})
    assert {obj['id'] for obj in current['objects']} == {obj['id'] for obj in first['objects']}


async def test_runner_uses_first_scene_inventory_for_all_later_cameras(tmp_path, monkeypatch):
    runner, context, algorithms = route_context(tmp_path)
    hints = []
    original = algorithms.observe

    def observe(*args, **kwargs):
        hints.append(deepcopy(kwargs['source_hint']))
        return original(*args, **kwargs)

    monkeypatch.setattr(algorithms, 'observe', observe)
    context['visual_plan'] = {'initial': 'initialize_inventory'}
    await runner.observe(context, purpose='scene')
    initial_ids = set(runner.task_progress.scene_entities)
    context['scene'] = {'observation_id': context['observation']['observation_id']}
    context['visual_plan'] = {'initial': 'associate_inventory'}
    await runner.observe(context, purpose='scene')
    hint = hints[-1]
    assert hint['camera_ids'] == ['1', '2', '3']
    assert hint['scene_queries']
    assert set(runner.task_progress.scene_entities) == initial_ids
    assert {key for query in hint['scene_queries'] for key in query['entity_ids']} == initial_ids - {'table'}


async def test_unmatched_detection_cannot_enter_frozen_scene_or_map(tmp_path, monkeypatch):
    runner, context, algorithms = route_context(tmp_path)
    context['visual_plan'] = {'initial': 'initialize_inventory'}
    await runner.observe(context, purpose='scene')
    initial_ids = set(runner.task_progress.scene_entities)
    original = algorithms.observe

    def extra(*args, **kwargs):
        result = original(*args, **kwargs)
        result['objects'].append({**deepcopy(result['objects'][0]), 'id': 'invented-cube'})
        return result

    monkeypatch.setattr(algorithms, 'observe', extra)
    context['visual_plan'] = {'initial': 'associate_inventory'}
    with pytest.raises(PerceptionError, match='visual_inventory_invalid_ids'):
        await runner.observe(context, purpose='scene')
    assert set(runner.task_progress.scene_entities) == initial_ids
    assert algorithms.map_calls == []
