import asyncio
import copy
import time

import numpy as np
import pytest
from test_rgbd_workflow import SensorRobot, VisionAlgorithms

from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.geometry import TOOL_PROFILE
from luna_mujoco.planning_settings import PlanningSettings

TEST_SETTINGS = PlanningSettings(observation={"A": [0.] * 7, "B": [0.] * 7})


def program_fixture():
    def skill(name):
        return {'type': 'skill', 'skill': name, 'arguments': []}
    return {'version': 1, 'steps': [{'type': 'for_each_target',
        'between_targets': [skill(name) for name in ('move_to_observe', 'observe_scene')],
        'body': [skill(name) for name in ('identify_target', 'identify_destination', 'choose_arm',
            'build_scene', 'generate_grasps', 'plan_grasp', 'execute_grasp', 'verify_grasp')] + [
            {'type': 'if_result', 'condition': 'success',
             'then_steps': [skill(name) for name in ('plan_place', 'execute_place', 'verify_place')],
             'else_steps': [{'type': 'stop', 'reason': 'grasp_not_confirmed'}]}]}]}


class DemoRobot(SensorRobot):
    def __init__(self, empty_grasp=False):
        super().__init__()
        self.empty_grasp = empty_grasp
        self.moves = []

    def capture_rgbd(self):
        capture = super().capture_rgbd()
        capture["captured_monotonic_s"] = time.monotonic()
        capture["captured_perf_counter_ns"] = time.perf_counter_ns()
        for view in capture["views"].values():
            view.update(T_world_camera=np.eye(4).tolist(),
                        intrinsics={"fx": .5, "fy": .5, "cx": 0., "cy": 1.})
        return capture

    def robot_snapshot(self):
        result = super().robot_snapshot()
        if self.empty_grasp and self.closed:
            state = result['arms']['A']
            state.update(gripper_position_rad=.8, gripper_closed_fraction=1.,
                         gripper_position_samples=[[t, .8] for t in (0., .05, .1, .15, .2, .25)])
        return result

    def execute_dual_trajectory(self, paths, dt, generation):
        self.moves.append('observation')
        return super().execute_dual_trajectory(paths, dt, generation)


class DemoVision(VisionAlgorithms):
    def __init__(self, count=4, requested=None, progress_status='at_destination', inventory_check='none'):
        super().__init__()
        self.perception = self
        self.count, self.requested = count, requested
        self.progress_status, self.inventory_check = progress_status, inventory_check
        self.discoveries = 0
        self.map_order = []
        self.target_order = []
        self.requested_arms = []
        self.grasp_placement_requests = []
        self.grasp_planning_settings = []
        self.grasp_completed_ids = []
        self.release_calls = 0

    def bootstrap(self, capture):
        self.map_order.append('bootstrap')
        return copy.deepcopy(capture) | {'objects': [], 'demo_background_mapping': True,
            'table': {'position': [.55, 0., .595], 'size': [.7, 1., .01], 'surface_height_m': .6}}

    def observe(self, capture, **kwargs):
        result = super().observe(capture, **kwargs)
        hint = kwargs.get('source_hint') or {}
        if not hint.get('scene_entities'):
            self.discoveries += 1
        ids = [f'pink-{i}' for i in range(1, self.count + 1)]
        selected = ids[:self.requested] if self.requested else ids
        source = result['objects'][0]
        result['objects'] = []
        entities = {'table': {'id': 'table', 'kind': 'table', 'roles': ['table', 'collision']}}
        for index, identifier in enumerate(ids):
            obj = copy.deepcopy(source)
            obj.update(id=identifier, color='pink', visual_membership='required' if identifier in selected else 'excluded')
            result['objects'].append(obj)
            entities[identifier] = {'id': identifier, 'kind': 'cylinder', 'color': 'pink',
                'bbox': [.1 + .2 * (index % 2), .1 + .3 * (index // 2), .25 + .2 * (index % 2), .3 + .3 * (index // 2)],
                'roles': ['collision'], 'visual_membership': obj['visual_membership'],
                'reference_camera_id': '2', 'initial_parent_id': 'source-bin',
                'segmentation_query': {'kind': 'cylinder', 'color': 'pink', 'prompt': 'pink cup'}}
        result['visual_inventory']['membership'] = {i: entities[i]['visual_membership'] for i in ids}
        result['visual_inventory']['progress'] = {i: {'source': 'openai_decisions',
            'status': self.progress_status, 'observation_id': capture['observation_id'],
            **{key: item[key] for key in ('destination_id', 'region_id', 'slot_id') if key in item}}
            for i, item in hint.get('visual_progress', {}).items()}
        result['scene_entities'] = entities
        result['demo_tables'] = {'source_observation_id': capture['observation_id']}
        result['demo_task'] = {'scope': 'count' if self.requested else 'all', 'requested_count': self.requested,
            'transport_ids': selected, 'source_object_ids': ids, 'source_bin_id': 'source-bin',
            'destination_table_id': 'table', 'ordering_camera_id': '2', 'source_inventory_complete': 'confirmed'}
        result['visual_reference'] = {'observation_id': capture['observation_id'],
            'views': copy.deepcopy(capture['views']), 'entities': entities, 'demo_task': result['demo_task']}
        result['demo_program'] = program_fixture()
        result['demo_inventory_check'] = {'status': self.inventory_check, 'observation_id': capture['observation_id'],
            'source': 'openai_decisions', 'confidence': .99, 'request_hash': 'decisions-' + capture['observation_id']}
        result['visual_discovery'] = {'source': 'openai_responses', 'observation_id': capture['observation_id'],
                                      'request_hash': 'responses-' + capture['observation_id']}
        result['visual_inventory']['evidence'] = {
            'source': 'openai_decisions', 'observation_id': capture['observation_id'],
            'request_hash': result['demo_inventory_check']['request_hash']}
        result['visual_inventory']['identity_coverage'] = {
            **result['visual_inventory']['evidence'], 'purpose': 'initialize_inventory',
            'minimum_confidence': .8, 'required_entity_ids': list(ids), 'accepted_entity_ids': list(ids),
            'accepted_left_entity_ids': list(ids), 'source_object_ids': list(ids), 'remaining_source_ids': list(ids)}
        for view in result['views'].values():
            view['instance_masks'] = {i: np.ones((2, 2), dtype=bool) for i in ids}
        return result

    def build_scene(self, observation, target_id, **kwargs):
        self.map_order.append(('map', target_id, tuple(observation['views'])))
        return super().build_scene(observation, target_id, **kwargs) | {
            "table": {"pose": [.55, 0., .45, 1., 0., 0., 0.], "dims": [.6, .8, .3]},
            "mapping_diagnostics": {"table_cache": {"table_revision": "inventory-table-v1",
                "source_observation_id": observation["observation_id"], "table_ids": ["table"],
                "source": "initial_inventory_ransac_table"}}}

    def publish_released_object(self, scene, target, grasp_tcp, release_tcp, after_capture_id, **kwargs):
        self.map_order.append(("relocate", target, after_capture_id))
        return scene | {"scene_token": scene["scene_token"] + ":released", "exclusion_state": "released"}

    def generate_grasps(self, observation, target, **kwargs):
        self.target_order.append(target)
        self.grasp_placement_requests.append(copy.deepcopy(kwargs.get('placement_request')))
        self.grasp_planning_settings.append(copy.deepcopy(kwargs.get('planning_settings')))
        return super().generate_grasps(observation, target, **kwargs) | {
            "sampling_round": kwargs.get("sampling_round", 1),
            "requested_candidate_count": {1: 256, 2: 512, 3: 1000}[kwargs.get("sampling_round", 1)]}

    def evaluate_arms(self, snapshot, target, grasps, scene, **kwargs):
        values = super().evaluate_arms(snapshot, target, grasps, scene, **kwargs)
        for arm, evaluation in values.items():
            peer = "B" if arm == "A" else "A"
            evaluation.update(arm=arm, target_id=target, generation=snapshot["generation"],
                scene_token=scene["scene_token"], planning_settings=copy.deepcopy(kwargs["planning_settings"]),
                tool_profile_revision=TOOL_PROFILE["revision"], sampling_round=grasps["sampling_round"],
                tested_count=len(grasps["candidates"]), destination_id=kwargs.get("destination_id"),
                start_joints_deg=np.rad2deg(snapshot["arms"][arm]["joints_rad"]).tolist(),
                other_joints_deg=np.rad2deg(snapshot["arms"][peer]["joints_rad"]).tolist())
        return values

    def evaluate_reachability(self, *args, **kwargs):
        self.requested_arms.append(kwargs.get('requested_arm'))
        return super().evaluate_reachability(*args, **kwargs)

    def plan_grasp(self, snapshot, *args, **kwargs):
        self.grasp_completed_ids.append(copy.deepcopy(snapshot.get("completed_target_ids")))
        return super().plan_grasp(snapshot, *args, **kwargs)

    def plan_place(self, observation, target, arm, scene, holding, destination, **kwargs):
        return super().plan_place(observation, target, arm, scene, holding, destination, **kwargs) | {
            "target_id": target, "destination_id": destination,
            "placement_request": copy.deepcopy(kwargs["placement_request"])}

    def validate_execution(self, *args, **kwargs):
        binding = args[5]
        return super().validate_execution(*args, **kwargs) | {
            "attached": binding is not None, "scene_token": args[2]["scene_token"],
            "carrying_owner_id": binding.get("owner_id") if binding else None}

    def verify_release(self, capture, binding, reference, instruction=""):
        self.release_calls += 1
        self.checkpoints.append("post_place")
        common = {"source": "openai_decisions", "observation_id": capture["observation_id"],
                  "confidence": .99, "request_hash": "release-" + capture["observation_id"]}
        return {"binding_id": binding["binding_id"], "observation_id": capture["observation_id"],
            "visual_evidence": {**common, "identity_source": "execution_chain",
                "status": self.progress_status, **{key: binding[key] for key in (
                    "binding_id", "target_id", "destination_id", "region_id", "slot_id",
                    "layout_revision", "release_feedback_id")}},
            "demo_inventory_check": {**common, "status": self.inventory_check
                if binding["needs_visible_inventory_check"] else "not_requested"},
            "geometry_updated": False}


async def run_demo(tmp_path, *, count=4, requested=None, empty=False, progress='at_destination', extra='none'):
    robot = DemoRobot(empty)
    algorithms = DemoVision(count, requested, progress, extra)
    runner = DemoRunner(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('把盒子中的所有粉色物体放到桌面指定位置')
    await asyncio.wait_for(runner.worker, 10)
    return runner, robot, algorithms


@pytest.mark.parametrize('count', [1, 2, 3, 4])
async def test_demo_once_inventory_completes_fixed_ids_without_jev(tmp_path, count):
    runner, robot, algorithms = await run_demo(tmp_path, count=count)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert algorithms.discoveries == 1
    assert runner.task_progress.completed_ids == [f'pink-{i}' for i in range(1, count + 1)]
    assert algorithms.target_order == runner.task_progress.completed_ids
    assert algorithms.grasp_completed_ids == [
        [f"pink-{index}" for index in range(1, cycle)] for cycle in range(1, count + 1)
    ]
    assert algorithms.requested_arms == [None] * count
    assert len(robot.moves) == count - 1
    assert not any(item == 'bootstrap' for item in algorithms.map_order)
    assert len([item for item in algorithms.map_order if isinstance(item, tuple) and item[0] == 'map']) == count
    assert len(algorithms.checkpoints) == count * 2
    assert all(item[2] == ('1', '2', '3') for item in algorithms.map_order if isinstance(item, tuple) and item[0] == 'map')
    assert len([item for item in algorithms.map_order if isinstance(item, tuple) and item[0] == 'relocate']) == count - 1
    assert not any(event['kind'] == 'jev' for event in runner.events)
    assert 'post_grasp' not in algorithms.checkpoints
    layout = runner.task_progress.placement_request('demo-destination')['layout']
    assert layout['target_count'] == count
    assert layout['method'] == 'equal_area_radial_partitions'
    assert layout['section_count'] == 2 * ((count + 1) // 2)


async def test_start_away_from_observation_pose_cannot_capture_or_publish_map(tmp_path):
    robot, algorithms = DemoRobot(), DemoVision(1)
    robot.joints = [.1] * 7
    runner = DemoRunner(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把盒子中的粉色物体放到目标区域")
    await runner.worker
    assert runner.state["error"] == "mapping_requires_observation_pose"
    assert algorithms.checkpoints == []
    assert algorithms.map_order == []


async def test_post_place_capture_is_never_accepted_for_mapping(tmp_path):
    runner = DemoRunner(DemoRobot(), DemoVision(1), None, tmp_path, planning_settings=TEST_SETTINGS)
    observation = runner.simulation.robot_snapshot() | {"checkpoint": "post_place"}
    with pytest.raises(ValueError, match="mapping_requires_observation_capture"):
        await runner.build_scene({"planning_settings": TEST_SETTINGS.model_dump(),
                                 "observation": observation}, "selected")


async def test_demo_count_stops_after_requested_subset(tmp_path):
    runner, robot, algorithms = await run_demo(tmp_path, count=4, requested=2)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert algorithms.target_order == ['pink-1', 'pink-2']
    assert runner.task_progress.records['pink-3']['status'] != 'completed'
    assert len(robot.moves) == 1
    assert runner.task_progress.placement_request('demo-destination')['layout']['target_count'] == 2


async def test_grasp_filter_receives_each_targets_frozen_placement_slot(tmp_path):
    runner, _, algorithms = await run_demo(tmp_path, count=2)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    requests = algorithms.grasp_placement_requests
    assert len(requests) == 2 and all(request is not None for request in requests)
    assert [request['object_id'] for request in requests] == ['pink-1', 'pink-2']
    assert len({request['slot_id'] for request in requests}) == 2
    assert len({request['layout_revision'] for request in requests}) == 1
    assert all(request['target_count'] == 2 for request in requests)
    assert algorithms.grasp_planning_settings == [runner.state['planning_settings']] * 2


async def test_empty_grasp_stops_before_attachment_and_place(tmp_path):
    runner, robot, algorithms = await run_demo(tmp_path, empty=True)
    assert runner.state['status'] == 'needs_attention'
    assert runner.task_progress.completed_ids == []
    assert algorithms.prepare_count == 0
    assert len(robot.moves) == 0


async def test_visual_and_feedback_checkpoints_record_actual_ledger(tmp_path):
    runner, _, _ = await run_demo(tmp_path, count=1)
    snapshots = [event for event in runner.events if event['kind'] == 'inventory_snapshot']
    assert {event['data']['checkpoint'] for event in snapshots} >= {
        'bind_inventory', 'verify_grasp', 'verify_place'}
    feedback = next(event['data'] for event in snapshots if event['data']['checkpoint'] == 'verify_grasp')
    assert feedback['evidence_source'] == 'gripper_feedback'
    assert feedback['model_event_ids'] == []
    assert snapshots[-1]['data']['task_progress']['completed']


async def test_unknown_place_does_not_continue_or_complete(tmp_path):
    runner, _robot, algorithms = await run_demo(tmp_path, progress='unknown')
    assert runner.state['status'] == 'needs_attention'
    assert runner.task_progress.completed_ids == []
    assert algorithms.target_order == ['pink-1']


@pytest.mark.parametrize('extra', ['unknown', 'extra'])
async def test_all_request_cannot_claim_completion_with_uncertain_inventory(tmp_path, extra):
    runner, _, _ = await run_demo(tmp_path, count=1, extra=extra)
    assert runner.state['status'] == 'needs_attention'
    assert runner.state['error'] in {'source_inventory_unknown', 'unlisted_source_object'}


async def test_responses_program_can_stop_without_running_grasp(tmp_path):
    class StopVision(DemoVision):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            observation['demo_program'] = {'version': 1, 'steps': [{'type': 'stop', 'reason': 'insufficient_evidence'}]}
            return observation
    algorithms = StopVision(1)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('把粉色物体放到指定位置')
    await runner.worker
    assert runner.state['status'] == 'needs_attention'
    assert algorithms.target_order == []
    assert runner.state['error'] == 'program_stopped:insufficient_evidence'


@pytest.mark.parametrize('skip', ['build_scene', 'verify_grasp', 'between_targets'])
async def test_generated_program_cannot_bypass_measured_preconditions(tmp_path, skip):
    class IncompleteVision(DemoVision):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            loop = observation['demo_program']['steps'][0]
            if skip == 'between_targets':
                loop['between_targets'] = []
            else:
                loop['body'] = [node for node in loop['body'] if node.get('skill') != skip]
            return observation
    algorithms = IncompleteVision(2)
    runner = DemoRunner(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('把两个粉色物体放到指定位置')
    await runner.worker
    assert runner.state['status'] == 'needs_attention'
    if skip == 'between_targets':
        assert algorithms.target_order == ['pink-1']
    else:
        assert runner.task_progress.completed_ids == []
        assert algorithms.prepare_count == 0
