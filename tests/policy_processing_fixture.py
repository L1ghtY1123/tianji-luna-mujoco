"""Measured release and processing receipts with the generic entity ledger."""
from copy import deepcopy

from processing_fixtures import before_kwargs, capture_and_action, client, release_evidence, work_area
from test_agent_model_layer import task_data
from test_demo_runner import TEST_SETTINGS, DemoRobot, DemoVision

from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.runtime.controller import PolicyController
from luna_mujoco.runtime.inventory import PolicyTaskProgress


def runner_and_context(tmp_path):
    robot, algorithms = DemoRobot(), DemoVision(1)
    visual, calls = client()
    algorithms.visual_inventory = visual
    algorithms.publish_local_object = lambda scene, *args, **kwargs: {
        'scene': deepcopy(scene), 'measurement': {'source': 'test_head_rgbd'}}
    algorithms.measure_local_object = lambda capture, *args: {'observation_id': capture['observation_id']}
    algorithms.measured_object_workspace = lambda measured, arm: {
        'reachable': True, 'arm': arm, 'observation_id': measured['observation_id']}
    runner = PolicyController(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    runner.run_id = 'run'
    runner.state.update(status='running', generation=1, task='grasp_place', target_id='pink',
        facts={'unpack_pending': True, 'vla_pose_ready': True, 'observation_pose_ready': False,
               'table_release_verified': True, 'grasp_held': False})
    runner.task_spec = TaskSpec.model_validate(task_data() | {
        'target_refs': ['pink'], 'processing': ['vla_unpack'], 'work_area_ref': 'work'})
    runner.task_inventory = {'source_object_ids': ['pink'], 'selected_ids': ['pink'],
        'source_id': runner.task_spec.source_ref, 'destination_id': runner.task_spec.destination_ref,
        'work_area_id': 'work', 'support_id': None}
    runner.task_progress = PolicyTaskProgress('unwrap')
    runner.task_progress.records = {'pink': {'id': 'pink', 'status': 'observed', 'display_id': 1}}
    runner.task_progress.membership = {'pink': 'required'}
    runner.task_progress.inventory_finalized = True
    runner.task_progress.scene_entities = {'pink': {'id': 'pink', 'kind': 'object', 'color': 'pink',
        'display_id': 1, 'visual_membership': 'required'}}
    runner.task_progress.next_display_id = 2
    runner.task_progress.configure_task(runner.task_spec, ['pink'])
    capture = robot.capture_rgbd()
    _, action = capture_and_action()
    action.update(generation=1, completed_time=0., completed_perf_counter_ns=1,
                  process_id='process-fixture', processing_skill='vla_unpack')
    release = {**release_evidence(), 'destination_id': 'work'}
    before = before_kwargs(capture, release, action=action)
    context = {'planning_settings': {**TEST_SETTINGS.model_dump(), 'vla_unpack': {'input_x_m': .36, 'input_y_m': 0.}},
        'place_plan': {'goal': {'release_reference': {'object_id': 'pink', 'object_reference_world_xy_m': [.36, 0.]}}},
        'release_endpoint_evidence': {'at_release_goal': True, 'actual_object_reference_world_xy_m': [.36, 0.], 'rotation_error_rad': 0.}, 'fine_capture': capture,
        'observation': capture | {'objects': [], 'table': {'surface_height_m': .6},
            'demo_tables': {'primary_table_id': 'table', 'source_observation_id': 'table-capture',
                'tables': {'table': {'source': 'initial_inventory_ransac_table', 'surface_height_m': .56,
                    'model': {'pose': [.36, 0., .52, 1, 0, 0, 0], 'dims': [.6, .8, .08]}}}}},
        'visual_inventory_reference': {'entities': deepcopy(runner.task_progress.scene_entities),
                                      'task_spec': runner.task_spec.model_dump(mode='json')},
        'scene': {'scene_token': 'retained', 'exclusion_state': 'released'},
        'analysis_cache': {'obsolete_capture': True}, 'processing_action': action,
        'place_verification': {'success': True, 'visual_evidence': release},
        'pending_transfer': {'target_id': 'pink', 'destination_id': 'work', 'verified': True},
        'destination': {'id': 'work', 'placement_reference': {**work_area(), 'region_polygon_world_xy_m': [[.2, -.16], [.52, -.16], [.52, .16], [.2, .16]], 'observation_id': 'measured'}},
        'vla_pose_execution': {'target_id': 'pink', 'generation': 1, 'release_feedback_id': 'table-release'},
        'post_place_capture': before['table_release_capture'],
        'post_place_verification_input': before['table_release_reference']}
    runner.sync_inventory(context)

    async def move(ctx):
        ctx['fine_capture'] = await runner.capture_observation(ctx)
        return {'success': True}
    runner.move_to_observe = move
    return runner, context, calls
