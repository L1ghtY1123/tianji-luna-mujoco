import asyncio
import copy
import time
from typing import ClassVar

import numpy as np
import pytest
from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms

from luna_mujoco.agent.responses import ResponsesBudget
from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.geometry import TOOL_PROFILE
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.runtime.controller import PolicyController

TEST_SETTINGS = PlanningSettings(observation={"A": [0.] * 7, "B": [0.] * 7})


def observation_return_steps(prefix='between'):
    """The mock model explicitly measures a map before requesting return motion."""
    return [
        {'type': 'skill', 'id': prefix + '_measure', 'skill': 'observe_scene', 'arguments': {}},
        {'type': 'skill', 'id': prefix + '_map', 'skill': 'build_scene', 'arguments': {
            'scope': 'global', 'reason': 'Measure obstacles before observation motion',
            'entity_ids': [], 'exclusion_state': 'observation'}},
        {'type': 'skill', 'id': prefix + '_return', 'skill': 'move_to_observe', 'arguments': {}},
        {'type': 'skill', 'id': prefix + '_observe', 'skill': 'observe_scene', 'arguments': {}},
    ]


def program_fixture(identifiers):
    from graph_contract_fixtures import call, linear_graph

    scopes, calls = [], []
    for index, identifier in enumerate(identifiers):
        prefix = f'object{index}_'
        arguments = {'identify_target': {'target_id': '$target'},
            'identify_destination': {'destination_id': '$destination', 'destination_kind': 'support_region'},
            'choose_arm': {'arm': 'auto'},
            'build_scene': {'scope': 'global', 'reason': 'fixture task evidence',
                            'entity_ids': [], 'exclusion_state': 'selected'}}
        skills = ('identify_target', 'identify_destination', 'perceive_placement_zone', 'choose_arm',
                  'build_scene', 'generate_grasps', 'plan_grasp', 'execute_grasp', 'verify_grasp',
                  'resolve_placement', 'plan_place', 'execute_place', 'verify_place')
        nodes = [call(prefix + name, name, **arguments.get(name, {})) for name in skills]
        nodes += observation_return_steps(prefix + 'return')
        scope = linear_graph(nodes, identifier=prefix+'task', finish=prefix+'done')
        scope.pop('version')
        scope.pop('subgraphs')
        scope.update(input_id=identifier, task=None)
        scopes.append(scope)
        calls.append({'type': 'subgraph', 'id': prefix+'call', 'subgraph_id': scope['id']})
    graph = linear_graph(calls)
    graph['subgraphs'] = scopes
    return graph


class FixtureCompiler:
    """Typed planning fixture; physical regressions do not depend on a network model."""
    configured_regions: ClassVar[dict] = {'demo-destination': {}}

    def __init__(self, algorithms):
        self.algorithms = algorithms
        self.configured_regions = copy.deepcopy(type(self).configured_regions)
        self.scene_context = {}
        self.reset()

    def reset(self):
        self.budget = ResponsesBudget()
        self.original_task = None

    def discover(self, capture, *, instruction):
        self.budget.reserve()
        start = len(self.algorithms.checkpoints)
        result = self.algorithms.observe(capture, instruction=instruction, checkpoint='proposal', source_hint={})
        del self.algorithms.checkpoints[start:]
        task = copy.deepcopy(result['demo_task'])
        # Match the real compiler's frozen initial numbering, before count selection.
        ordered = [*task['source_object_ids'],
                   *[key for key in result['scene_entities'] if key not in task['source_object_ids']]]
        for number, identifier in enumerate(ordered, 1):
            result['scene_entities'][identifier]['display_id'] = number
        task['destination_id'] = 'demo-destination'
        task['work_area_id'] = 'demo-destination'
        self.original_task = TaskSpec(instruction=instruction, supported=True, reason='',
            scope=task['scope'], target_selector='pink objects', target_refs=tuple(task['source_object_ids']),
            quantity=task['requested_count'], source_ref=task['source_bin_id'], destination_ref='demo-destination',
            work_area_ref=None, selection_order='nearest', processing=(),
            completion_conditions=('Objects verified at the destination',), constraints=(),
            source_inventory_complete=task['source_inventory_complete'])
        spec = self.original_task.model_dump(mode='json')
        inventory = {'source_object_ids': task['source_object_ids'], 'selected_ids': task['transport_ids'],
                     'source_id': task['source_bin_id'], 'destination_id': 'demo-destination',
                     'work_area_id': None, 'support_id': task.get('destination_table_id', 'table')}
        from graph_contract_fixtures import call, linear_graph
        from luna_mujoco.inventory_order import robot_distance_order

        graph = result['demo_program']
        for scope in graph['subgraphs']:
            scope['task'] = {**spec, 'scope': 'ids', 'quantity': 1,
                'target_refs': [scope['input_id']], 'selection_order': 'explicit', 'completion_pose': None}
        nodes = graph['nodes'][:-1]
        try:
            rows = robot_distance_order([obj for obj in result['objects'] if obj['id'] in task['source_object_ids']])
            rank = {row['id']: i for i, row in enumerate(rows)}
            scope_ids = {scope['id']: scope['input_id'] for scope in graph['subgraphs']}
            nodes.sort(key=lambda node: rank[scope_ids[node['subgraph_id']]])
        except (ValueError, KeyError):
            pass  # Malformed measurement is deliberately checked by confirm_inventory.
        nodes = [call('initial_view', 'observe_scene'), call('initial_inventory', 'confirm_inventory')] + nodes
        end = graph['nodes'][-1]
        if end['status'] == 'success':
            nodes.append(call('whole_task_check', 'verify_task_completion'))
        current = linear_graph(nodes, identifier=graph['id'], finish=end['id'])
        current['nodes'][-1] = end
        current['subgraphs'] = graph['subgraphs']
        result['demo_program'] = current
        reference = {**result['visual_reference'], 'task_spec': spec,
                     'task_inventory': inventory, 'policy_graph': result['demo_program'],
                     'inventory_state': {'entities': result['scene_entities'], 'retired_entity_ids': [],
                                         'transformations': {}}}
        reference.pop('demo_task', None)
        reference.pop('demo_program', None)
        return {'task_spec': spec, 'task_inventory': inventory, 'policy_graph': result['demo_program'],
                'entities': result['scene_entities'], 'reference': reference}


class FixtureJudgments:
    def __init__(self, algorithms):
        self.algorithms = algorithms

    def judge(self, capture, *, requests, context, reference=None):
        judgments = {row.name: {'status': 'yes', 'probability': .99} for row in requests}
        for row in requests:
            if getattr(row, 'choices', None):
                value = 'stop' if 'stop' in row.choices else row.choices[0]
                judgments[row.name] = {'status': 'proposed', 'value': value, 'confidence': .99}
        # These fixture tasks allow any location in their bound destination.
        if 'placement_additional_spatial_constraint' in judgments:
            judgments['placement_additional_spatial_constraint'] = {'status': 'no', 'probability': .01}
        if context.get('destination_candidates'):
            destination = context['object_task']['destination_ref']
            value = next(row['value'] for row in context['destination_candidates']
                         if row['instance_id'] == destination)
            judgments['destination_requirement_matches'] = {'status': 'known', 'value': value, 'confidence': .99}
        if 'goal_satisfied' in judgments and getattr(self.algorithms, 'final_source_status', None) == 'unknown':
            judgments['source_complete'] = {'status': 'unknown', 'probability': .5}
        return {'judgments': judgments, 'evidence': {'source': 'openai_decisions',
                'observation_id': capture['observation_id'], 'request_hash': 'task-judgment'}}


class TestPolicyController(PolicyController):
    __test__ = False

    def __init__(self, robot, algorithms, decisions_factory, directory, **kwargs):
        super().__init__(robot, algorithms, decisions_factory, directory,
            compiler=FixtureCompiler(algorithms), judgments=FixtureJudgments(algorithms), **kwargs)


class DemoRobot(SensorRobot):
    def __init__(self, empty_grasp=False):
        super().__init__()
        self.empty_grasp = empty_grasp
        self.moves = []

    def capture_rgbd(self):
        capture = copy.deepcopy(super().capture_rgbd())
        capture["captured_monotonic_s"] = time.monotonic()
        capture["captured_perf_counter_ns"] = time.perf_counter_ns()
        for camera, view in capture["views"].items():
            view.update(camera_id=camera, frame_id=capture['observation_id'],
                        simulation_generation=capture['generation'], calibration_id='fixture-camera-' + camera,
                        T_world_camera=np.eye(4).tolist(),
                        intrinsics={"fx": .5, "fy": .5, "cx": 0., "cy": 1.})
        return capture

    def robot_snapshot(self):
        result = super().robot_snapshot()
        result.update(manifest_revision=1, command_revision=0, active_commands=[])
        for arm in result['arms'].values():
            arm['gripper_command_position_rad'] = .8 if arm['gripper_closed_command'] else 0.
        if self.empty_grasp and self.closed:
            state = result['arms']['A']
            state.update(gripper_position_rad=.8, gripper_closed_fraction=1.,
                         gripper_position_samples=[[t, .8] for t in (0., .05, .1, .15, .2, .25)])
        return result

    def execute_dual_trajectory(self, paths, dt, generation, *, execution_settings=None):
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
        result['demo_program'] = program_fixture(ids)
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
                tool_profile_revision=TOOL_PROFILE["revision"], tcp_frame="gripper_front_" + arm, sampling_round=grasps["sampling_round"],
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

    def placement_xy_candidates(self, observation, target, holding, request, zone, settings, **kwargs):
        # Lifecycle fixture: match its scripted planner. Real numeric subdivision,
        # held-footprint and XY-before-Z behavior are tested with DemoAlgorithms.
        cell = next(row for row in request['layout']['cells'] if row['slot_id'] == request['slot_id'])
        return [dict(request, selected_reference_world_xy_m=cell['center_world_xy_m'])], {'generated_count': 1}

    def resolve_placement(self, observation, target, arm, scene, holding, destination, **kwargs):
        from luna_mujoco.algorithms import Algorithms

        # This lifecycle double uses the same scripted endpoint as its planner;
        # runtime retry recording still receives the complete candidate inputs.
        plan = super().plan_place(observation, target, arm, scene, holding, destination, **kwargs)
        settings = PlanningSettings.model_validate(kwargs['planning_settings']).model_dump()
        payload = Algorithms._motion_request(observation, arm, scene, plan['goal']['T_world_tcp'], settings)
        payload.update(carrying_binding=copy.deepcopy(plan['carrying_binding']),
            approach_axis=settings['place_approach_axis'],
            approach_offset_m=settings['place_approach_offset_m'],
            approach_in_tool_frame=settings['place_approach_in_tool_frame'],
            lift_axis=settings['place_retreat_axis'], lift_offset_m=settings['place_retreat_offset_m'],
            lift_in_tool_frame=settings['place_retreat_in_tool_frame'],
            trajectory_collision_clearance_m=settings['trajectory_collision_clearance_m'],
            optimizer_collision_activation_distance_m=settings['optimizer_collision_activation_distance_m'])
        candidate = {'candidate_id': 'placement-' + str(kwargs.get('candidate_indices', [0])[0]),
            'request': copy.deepcopy(kwargs['placement_request']),
            'settings': settings, 'payload': payload, 'goal': copy.deepcopy(plan['goal']),
            'binding': copy.deepcopy(plan['carrying_binding'])}
        return {'status': 'success', 'placement_zone': copy.deepcopy(kwargs['placement_zone']),
                'candidates': [candidate], 'selection_evidence': {}, 'resolution_id': 'fixture-resolution'}

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
            "demo_inventory_check": {**common, "status": "not_requested"},
            "geometry_updated": False}


async def run_demo(tmp_path, *, count=4, requested=None, empty=False, progress='at_destination', extra='none'):
    robot = DemoRobot(empty)
    algorithms = DemoVision(count, requested, progress, extra)
    runner = TestPolicyController(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
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
    assert len(robot.moves) == count
    assert not any(item == 'bootstrap' for item in algorithms.map_order)
    assert len([item for item in algorithms.map_order if isinstance(item, tuple) and item[0] == 'map']) == count * 2
    assert len(algorithms.checkpoints) == count * 3 + 1
    assert all(item[2] == ('1', '2', '3') for item in algorithms.map_order if isinstance(item, tuple) and item[0] == 'map')
    assert len([item for item in algorithms.map_order if isinstance(item, tuple) and item[0] == 'relocate']) == count
    assert not any(event['kind'] == 'jev' for event in runner.events)
    assert 'post_grasp' not in algorithms.checkpoints
    layout = runner.task_progress.placement_request('demo-destination')['layout']
    assert layout['target_count'] == 1
    assert layout['method'] == 'shared_work_area'
    assert layout['section_count'] == 1


async def test_start_away_from_observation_pose_uses_current_capture_and_map(tmp_path):
    robot, algorithms = DemoRobot(), DemoVision(1)
    robot.joints = [.1] * 7
    runner = TestPolicyController(robot, algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start("把盒子中的粉色物体放到目标区域")
    await runner.worker
    assert runner.state.get("error") != "mapping_requires_observation_pose"
    assert algorithms.checkpoints
    assert any(isinstance(item, tuple) and item[0] == 'map' for item in algorithms.map_order)


async def test_post_place_capture_is_never_accepted_for_mapping(tmp_path):
    runner = TestPolicyController(DemoRobot(), DemoVision(1), None, tmp_path, planning_settings=TEST_SETTINGS)
    observation = runner.simulation.robot_snapshot() | {"checkpoint": "post_place"}
    with pytest.raises(ValueError, match="mapping_requires_observation_capture"):
        await runner.build_scene({"planning_settings": TEST_SETTINGS.model_dump(),
                                 "observation": observation}, "selected")


async def test_demo_count_stops_after_requested_subset(tmp_path):
    runner, robot, algorithms = await run_demo(tmp_path, count=4, requested=2)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert algorithms.target_order == ['pink-1', 'pink-2']
    assert runner.task_progress.records['pink-3']['status'] != 'completed'
    assert len(robot.moves) == 2
    assert runner.task_progress.placement_request('demo-destination')['layout']['method'] == 'shared_work_area'


async def test_grasp_filter_receives_individual_ids_and_shared_free_region(tmp_path):
    runner, _, algorithms = await run_demo(tmp_path, count=2)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    requests = algorithms.grasp_placement_requests
    assert len(requests) == 2 and all(request is not None for request in requests)
    assert [request['object_id'] for request in requests] == ['pink-1', 'pink-2']
    assert {request['slot_id'] for request in requests} == {1}
    assert requests[0]['layout']['region_polygon_world_xy_m'] == requests[1]['layout']['region_polygon_world_xy_m']
    assert len({request['layout']['observation_id'] for request in requests}) == 2
    assert all(request['free_placement'] and request['layout']['method'] == 'shared_work_area'
               for request in requests)
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
    assert runner.state['error'].endswith('inventory_order_unconfirmed'
        if extra == 'unknown' else 'unlisted_source_object')


async def test_responses_program_can_stop_without_running_grasp(tmp_path):
    class StopVision(DemoVision):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            from graph_contract_fixtures import linear_graph
            observation['demo_program'] = linear_graph([])
            observation['demo_program']['nodes'][0].update(status='failure', reason='insufficient_evidence')
            return observation
    algorithms = StopVision(1)
    runner = TestPolicyController(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('把粉色物体放到指定位置')
    await runner.worker
    assert runner.state['status'] == 'needs_attention'
    assert algorithms.target_order == []
    assert runner.state['error'].startswith('policy_stopped:insufficient_evidence')


@pytest.mark.parametrize('skip', ['build_scene', 'verify_grasp', 'between_targets'])
async def test_generated_program_cannot_bypass_measured_preconditions(tmp_path, skip):
    class IncompleteVision(DemoVision):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            from graph_contract_fixtures import linear_graph
            for scope in observation['demo_program']['subgraphs']:
                nodes = scope['nodes'][:-1]
                nodes = [node for node in nodes if (not node['id'].startswith(scope['id'].replace('task', 'return'))
                    if skip == 'between_targets' else node.get('skill') != skip)]
                rebuilt = linear_graph(nodes, identifier=scope['id'], finish=scope['nodes'][-1]['id'])
                scope.update(nodes=rebuilt['nodes'], edges=rebuilt['edges'], entry=rebuilt['entry'])
            return observation
    algorithms = IncompleteVision(2)
    runner = TestPolicyController(DemoRobot(), algorithms, None, tmp_path, planning_settings=TEST_SETTINGS)
    await runner.start('把两个粉色物体放到指定位置')
    await runner.worker
    assert runner.state['status'] == 'needs_attention'
    if skip == 'between_targets':
        assert algorithms.target_order == ['pink-1']
    else:
        assert runner.task_progress.completed_ids == []
        assert not any(event['kind'] == 'policy_node' and event['data'].get('node_id', '').endswith('execute_place')
                       and event['data'].get('status') == 'success'
                       for event in runner.events)
