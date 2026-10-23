"""Exercise the actual runner/perception boundary without sensors or model calls."""
import asyncio
import hashlib
import json
import threading
from copy import deepcopy

import numpy as np
import pytest
from test_demo_lifecycle_perception import (
    before_kwargs,
    capture_and_action,
    client,
    drop_inputs,
    record,
    work_area,
)
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision
from test_experiments import begin

from luna_mujoco.demo_process import ProcessTaskProgress as TaskProgress
from luna_mujoco.demo_process import initialize_processes
from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.experiments import ExperimentStore


def runner_and_context(tmp_path):
    robot, algorithms = DemoRobot(), DemoVision(1)
    visual, calls = client()
    algorithms.visual_inventory = visual
    algorithms.publish_local_object = lambda scene, *args, **kwargs: {
        'scene': deepcopy(scene), 'measurement': {'source': 'test_head_rgbd'}}
    algorithms.publish_action_envelope = lambda scene, *args, **kwargs: deepcopy(scene)
    runner = DemoRunner(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.run_id = 'run'
    runner.state.update(status='running', generation=1, task='grasp_place', target_id='pink',
                        facts={'unpack_pending': True, 'vla_pose_ready': True, 'observation_pose_ready': False})
    runner.task_progress = TaskProgress('unwrap', route='box_table_box')
    runner.task_progress.records = {'pink': {'id': 'pink', 'status': 'pending', 'stage_index': 1, 'display_id': 1, 'destination_visual_evidence': {'status': 'at_destination'}}}
    runner.task_progress.membership = {'pink': 'required'}
    runner.task_progress.inventory_finalized = True
    capture = robot.capture_rgbd()
    _, action = capture_and_action()
    action.update(generation=1, completed_time=0., completed_perf_counter_ns=1)
    lifecycle = record('unpack_pending') | {'mock_action': action}
    runner.demo_task = {'workflow': 'unpack_transfer', 'destination_table_id': 'main', 'item_lifecycle': {'pink': lifecycle},
                        'source_object_ids': ['pink'], 'transport_ids': ['pink']}
    initialize_processes(runner.demo_task, runner.task_progress.records)
    runner.task_progress.process_task = runner.demo_task
    runner.task_progress.scene_entities = {'pink': {'id': 'pink', 'kind': 'object', 'color': 'pink', 'display_id': 1}}
    runner.task_progress.next_display_id = 2
    reference = {'entities': {'pink': {'color': 'pink'}}, 'demo_task': deepcopy(runner.demo_task)}
    context = {'planning_settings': TEST_SETTINGS.model_dump(), 'fine_capture': capture,
               'observation': capture | {'table': {'surface_height_m': .6}}, 'visual_inventory_reference': reference,
               'scene': {'scene_token': 'retained', 'exclusion_state': 'released'},
               'analysis_cache': {'old_phase': 'must not survive'}}
    context['observation']['demo_tables'] = {'primary_table_id': 'main',
        'source_observation_id': 'initial', 'tables': {'main': {
            'model': {'pose': [.46, 0, .45, 1, 0, 0, 0], 'dims': [.6, .8, .3]},
            'surface_height_m': .6, 'source': 'initial_inventory_ransac_table'}}}
    context['scene_observation'] = {**capture, 'objects': [{'id': 'right-bin', 'kind': 'bin',
        'identity_source': 'measured_container_correspondence',
        'points_world_m': np.array([[.05, -.8, .4], [.35, -.4, .9]])}]}
    context['scene'].update(observation_id=capture['observation_id'], generation=capture['generation'])
    context['scene']['collision_layer_bounds'] = {
        'scene_token': 'retained', 'observation_id': capture['observation_id'], 'generation': capture['generation'],
        'layers': {'semantic_instance::right-bin': {'kind': 'voxel', 'enabled': True,
            'source': 'native_surface_voxel_cell_bounds',
            'bounds_source': 'all_native_surface_cells_before_preview_sampling',
            'minimum_world_m': [.05, -.8, .4], 'maximum_world_m': [.35, -.4, .6],
            'cell_count': 180, 'voxel_size_m': .02}}}
    before = before_kwargs(capture, lifecycle, action=action)
    context.update(post_place_capture=before['table_release_capture'],
                   post_place_verification_input=before['table_release_reference'])
    async def move(ctx):
        ctx['fine_capture'] = await runner.capture_observation(ctx)
        return {'success': True}
    runner.move_to_observe = move
    return runner, context, calls


async def test_verified_unpack_preserves_capture_and_syncs_stage_before_geometry(tmp_path):
    runner, context, calls = runner_and_context(tmp_path)
    capture = context['fine_capture']
    result = await runner._step('verify_unpack', context)
    assert result['success']
    assert 'fine_capture' not in context
    assert context['post_unpack_capture']['observation_id'] != capture['observation_id']
    assert context['latest_capture'] is context['post_unpack_capture']
    assert context['visual_inventory_reference']['entities']['pink']['color'] == 'pink'
    product_id = result['output_entity_id']
    assert product_id != 'pink'
    assert runner.demo_task['item_lifecycle']['pink']['phase'] == 'consumed'
    lifecycle = runner.demo_task['item_lifecycle'][product_id]
    assert lifecycle['phase'] == 'unpacked_confirmed' and lifecycle['appearance_revision'] == 1
    assert context['visual_inventory_reference']['demo_task']['item_lifecycle'][product_id] == lifecycle
    assert runner.task_progress.records[product_id]['lifecycle'] == lifecycle
    assert 'analysis_cache' not in context
    assert runner.state['target_id'] == product_id
    assert runner.state['facts']['unpack_verified'] is True
    assert runner.state['facts']['grasp_completed'] is False
    assert runner.state['facts']['observation_pose_ready'] is True
    assert runner.state['facts']['observed'] is False
    runner.registry.validate_selection('move_to_observe', runner.state['facts'])
    from luna_mujoco.skills import SkillSelectionError
    runner.registry.validate_selection('observe_scene', runner.state['facts'])
    for name in ('identify_target', 'generate_grasps', 'plan_grasp'):
        with pytest.raises(SkillSelectionError):
            runner.registry.validate_selection(name, runner.state['facts'])
    reference = context['visual_inventory_reference']['product_references'][product_id]
    assert reference['verification_capture'] == result['verification_capture']
    assert reference['observation_id'] == context['post_unpack_capture']['observation_id']
    assert runner.transfer_routes is False
    assert runner.task_progress.route == 'box_table_box'
    assert len(calls) == 2 and runner.algorithms.checkpoints == []
    # The unpacked active ID need not remain in the left wrist source view.
    runner._check_source_view({'objects': [{'id': product_id, 'coverage': {'camera_ids': ['1']}}]})


@pytest.mark.parametrize('key', ['post_place_capture', 'post_place_verification_input'])
async def test_runner_requires_private_confirmed_before_reference_before_model_call(tmp_path, key):
    runner, context, calls = runner_and_context(tmp_path)
    context.pop(key)
    with pytest.raises(ValueError, match='table_release_reference'):
        await runner._verify_unpack(context)
    assert not calls


async def test_runner_rejects_another_active_lifecycle_before_any_visual_call(tmp_path):
    runner, context, calls = runner_and_context(tmp_path)
    runner.demo_task['item_lifecycle']['other'] = record('table_released')
    with pytest.raises(ValueError, match='single_active'):
        await runner._verify_unpack(context)
    assert not calls


@pytest.mark.parametrize('answer,completed', [('yes', True), ('unknown', False)])
async def test_current_drop_helper_and_runner_agree_on_before_frame_and_final_stage(tmp_path, answer, completed):
    runner, context, _ = runner_and_context(tmp_path)
    visual, calls = client({'current_item_added': (answer, .95)})
    runner.algorithms.visual_inventory = visual
    before = context['fine_capture']
    _, action, _ = drop_inputs()
    action.update(generation=1, before_observation_id=before['observation_id'], completed_time=.02,
                  completed_perf_counter_ns=1)
    lifecycle = record('right_bin_released') | {'drop_action': action}
    runner.demo_task.update(right_bin_id='right-bin', item_lifecycle={'pink': lifecycle})
    runner.task_progress.records['pink'].update(destination_id='right-bin', status='selected')
    runner.task_progress.current_id = 'pink'
    runner.task_progress.record_release('pink', feedback_id=action['release_feedback_id'])
    runner.state['facts'].update(drop_completed=True, grasp_held=False)
    runner.simulation.time = .03
    context.update(initial_capture=before, pending_transfer={'target_id': 'pink', 'verified': False})
    result = await runner._step('verify_drop', context)
    assert result['success'] is completed
    assert runner.task_progress.completed_ids == (['pink'] if completed else [])
    assert runner.current_lifecycle()['phase'] == ('completed' if completed else 'right_bin_released')
    assert context['pending_transfer']['verified'] is completed
    assert result['visual_evidence']['before_observation_id'] == before['observation_id']
    assert result['observation_id'] != before['observation_id']
    assert len(calls) == 1 and runner.algorithms.checkpoints == []


@pytest.mark.parametrize('key,value', [('observation_id', 'stale'), ('generation', 9)])
async def test_drop_region_must_come_from_the_actual_before_observation(tmp_path, key, value):
    runner, context, calls = runner_and_context(tmp_path)
    before = context['fine_capture']
    _, action, _ = drop_inputs()
    action.update(generation=1, before_observation_id=before['observation_id'], completed_time=.02,
                  completed_perf_counter_ns=1)
    runner.demo_task.update(right_bin_id='right-bin',
        item_lifecycle={'pink': record('right_bin_released') | {'drop_action': action}})
    runner.simulation.time = .03
    context.update(initial_capture=before)
    context['scene_observation'][key] = value
    with pytest.raises(ValueError, match='drop_bin_measurement_frame_mismatch'):
        await runner._verify_drop(context)
    assert not calls


@pytest.mark.parametrize('kind', ['unpack', 'drop'])
async def test_lifecycle_visual_calls_retain_archive_until_actual_call_finishes(tmp_path, monkeypatch, kind):
    runner, _, _ = runner_and_context(tmp_path)
    leases = []

    def retain():
        done = threading.Event()
        leases.append(done)
        return lambda event: None, done

    monkeypatch.setattr(runner, 'visual_audit_sink', retain)
    if kind == 'unpack':
        capture, action = capture_and_action()
        function, lifecycle, extra = runner.verify_unpacking, record('unpack_pending'), {'work_area': work_area(), **before_kwargs()}
    else:
        capture, action, reference = drop_inputs()
        function, lifecycle, extra = runner.verify_right_bin_drop, record('right_bin_released'), {'reference': reference}
    result = await runner.call_algorithm(function,
        capture, target_id='pink', lifecycle=lifecycle, action_evidence=action, owner_id='run', **extra)
    assert result['success']
    assert len(leases) == 1, 'new lifecycle request must own an in-flight archive lease'
    assert leases[0].is_set()


@pytest.mark.parametrize('kind,image_count', [('unpack', 2), ('drop', 5)])
async def test_real_experiment_archives_lifecycle_call_without_client_or_credentials(tmp_path, kind, image_count):
    runner, context, _ = runner_and_context(tmp_path / 'logs')
    store = ExperimentStore(tmp_path / 'experiments')
    session = begin(store)
    runner.run_id, runner.experiment = session.id, session
    runner.visual_inventory.api_key = 'archive-regression-secret'
    if kind == 'unpack':
        runner.current_lifecycle()['mock_action']['owner_id'] = session.id
        context['post_place_verification_input']['owner_id'] = session.id
    else:
        before = context['fine_capture']
        _, action, _ = drop_inputs()
        action.update(owner_id=session.id, generation=1, before_observation_id=before['observation_id'],
                      completed_time=.02, completed_perf_counter_ns=1)
        lifecycle = record('right_bin_released') | {'drop_action': action}
        runner.demo_task.update(right_bin_id='right-bin', item_lifecycle={'pink': lifecycle})
        runner.task_progress.records['pink'].update(destination_id='right-bin', status='selected')
        runner.task_progress.current_id = 'pink'
        runner.task_progress.record_release('pink', feedback_id=action['release_feedback_id'])
        runner.state['facts'].update(drop_completed=True, grasp_held=False)
        runner.simulation.time = .03
        context.update(initial_capture=before, pending_transfer={'target_id': 'pink', 'verified': False})
    try:
        result = await runner._step('verify_' + kind, context)
        assert result['success']
        json.dumps(runner.events, allow_nan=False)
    finally:
        session.finish({'status': 'needs_attention'})
        store.close()
    manifest = store.get(session.id)
    assert manifest['errors'] == [] and not runner.state.get('log_error')
    events = store.events(session.id)
    start = next(event for event in events if event['kind'] == 'algorithm_start')
    archived_start = json.loads(store.file(session.id, start['artifact']).read_text(encoding='utf-8'))
    archived_args = archived_start['args']
    assert len(archived_args) == 1 and 'views' in archived_args[0]
    if kind == 'unpack':
        area = archived_start['kwargs']['work_area']
        assert area['center_world_xy_m'] == [.46, 0.]
        np.testing.assert_allclose(area['region_polygon_world_xy_m'],
            [[.16, -.4], [.76, -.4], [.76, .4], [.16, .4]])
        assert area['floor_z_m'] == .6 and area['source'] == 'measured_cached_main_table_support'
        assert area['table_id'] == 'main' and area['source_observation_id'] == 'initial'
    else:
        region = archived_start['kwargs']['reference']['drop_bin_region']
        assert region['source'] == 'published_inventory_container_occupancy_bounds'
        assert region['entity_id'] == 'right-bin'
        assert region['observation_id'] == before['observation_id']
        assert region['minimum_world_m'] == [.05, -.8, .4]
        assert region['maximum_world_m'] == [.35, -.4, .6]
    visual_event = next(event for event in events if event['kind'] == 'visual_decisions_call')
    archived_call = json.loads(store.file(session.id, visual_event['data']['artifact']).read_text(encoding='utf-8'))
    images = [part['image_url'] for part in archived_call['request']['input'][0]['content']
              if part['type'] == 'input_image']
    assert len(images) == image_count
    for image in images:
        raw = store.file(session.id, image['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == image['sha256']
    assert archived_call['response']['answers']
    for path in session.directory.rglob('*.json'):
        text = path.read_text(encoding='utf-8')
        assert 'archive-regression-secret' not in text
        assert 'DemoVisualInventory' not in text and 'api_key' not in text


@pytest.mark.parametrize('kind,image_count', [('unpack', 2), ('drop', 1)])
async def test_bound_lifecycle_call_keeps_real_archive_after_cancellation(tmp_path, monkeypatch, kind, image_count):
    runner, _, _ = runner_and_context(tmp_path / 'logs')
    store = ExperimentStore(tmp_path / 'experiments')
    session = begin(store)
    runner.run_id, runner.experiment = session.id, session
    started, release = threading.Event(), threading.Event()
    post = runner.visual_inventory.client.post

    def blocked_post(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return post(*args, **kwargs)

    monkeypatch.setattr(runner.visual_inventory.client, 'post', blocked_post)
    if kind == 'unpack':
        capture, action = capture_and_action()
        function, lifecycle, extra = runner.verify_unpacking, record('unpack_pending'), {'work_area': work_area(), **before_kwargs()}
    else:
        capture, action, reference = drop_inputs()
        function, lifecycle, extra = runner.verify_right_bin_drop, record('right_bin_released'), {'reference': reference}
    action['owner_id'] = session.id
    if kind == 'unpack':
        extra['table_release_reference']['owner_id'] = session.id
    task = asyncio.create_task(runner.call_algorithm(function, capture, target_id='pink', lifecycle=lifecycle,
        action_evidence=action, owner_id=session.id, **extra))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        session.stop({'status': 'cancelled'})
        runner.clear()
        assert session.busy
    finally:
        release.set()
        store.close()
        await runner.planning_worker
    assert not runner.events, 'late response must not enter a reset run'
    manifest = store.get(session.id)
    assert manifest['status'] == 'cancelled' and manifest['recording_complete']
    assert manifest['errors'] == []
    archived_calls = list((session.directory / 'decisions').glob('*.json'))
    assert len(archived_calls) == 1
    saved = json.loads(archived_calls[0].read_text(encoding='utf-8'))
    assert saved['response']['answers']
    images = [part['image_url'] for part in saved['request']['input'][0]['content'] if part['type'] == 'input_image']
    assert len(images) == image_count
    for image in images:
        assert hashlib.sha256(store.file(session.id, image['path']).read_bytes()).hexdigest() == image['sha256']
