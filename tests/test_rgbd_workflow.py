import asyncio
import copy

import numpy as np
from test_runner import visual_inventory_result

from jev_mujoco.runner import Runner


def transform(position):
    pose = np.eye(4)
    pose[:3, 3] = position
    return pose.tolist()


class SensorRobot:
    """Robot/physics boundary fixture; no environment snapshot is available to Runner."""
    def __init__(self):
        self.generation = 1
        self.time = 0.
        self.stage = 0
        self.joints = [0.] * 7
        self.peer_joints = [0.] * 7
        self.closed = False
        self.tcp = [.5, .1, .8]

    def snapshot(self):
        raise AssertionError("runner read simulation truth")

    def robot_snapshot(self):
        return {"generation": self.generation, "confirmed": True, "time": self.time,
                "arms": {arm: {"joints_rad": self.joints if arm == "A" else self.peer_joints,
                                "tcp": transform(self.tcp if arm == "A" else [0., 0., 1.]),
                                "gripper_closed_fraction": .6 if self.closed and arm == "A" else 0.,
                                "gripper_position_rad": .48 if self.closed and arm == "A" else 0.,
                                "gripper_limits_rad": [0., .8],
                                "gripper_closed_command": bool(self.closed and arm == "A"),
                                "gripper_position_samples": [[t, .48 if self.closed and arm == "A" else 0.]
                                                              for t in (0., .05, .1, .15, .2, .25)]} for arm in ("A", "B")}}

    def capture_rgbd(self):
        self.time += .01
        return {**self.robot_snapshot(), "source": "rgbd", "observation_id": f"frame-{self.time}",
                "views": {camera: {"rgb": np.full((2, 2, 3), self.stage, dtype=np.uint8)}
                          for camera in ("1", "2", "3")}}

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        assert generation == self.generation
        self.stage = int(positions_rad[-1][0])
        self.joints = list(positions_rad[-1])
        self.tcp = {0: [.5, .1, .8], 1: [.5, .1, .8], 2: [.5, .1, .9],
                    3: [.6, -.2, .95], 4: [.6, -.2, .817], 5: [.6, -.2, .95]}[self.stage]
        self.time += .1
        return {"completed": True}

    def execute_dual_trajectory(self, paths, dt, generation):
        assert generation == self.generation
        self.joints, self.peer_joints = list(paths['A'][-1]), list(paths['B'][-1])
        self.time += .1
        return {'completed': True}

    def set_gripper(self, arm, closed, generation):
        assert generation == self.generation
        self.closed = closed
        return {"completed": True, "closed_command": closed}

    def dwell(self, duration_s, generation):
        assert generation == self.generation
        self.time += duration_s
        return {"completed": True}


class VisionAlgorithms:
    def __init__(self, *, fail_after_release=False, membership_status='required'):
        self.fail_after_release = fail_after_release
        self.membership_status = membership_status
        self.clear_count = 0
        self.prepare_count = 0
        self.checkpoints = []
        self.source_hints = []

    def observe(self, capture, instruction="", checkpoint=None, source_hint=None):
        self.checkpoints.append(checkpoint)
        self.source_hints.append(copy.deepcopy(source_hint))
        stage = int(capture["views"]["3"]["rgb"][0, 0, 0])
        closed = capture["arms"]["A"]["gripper_closed_fraction"] > 0
        source = [.5, .1, .62]
        if closed:
            source = np.asarray(capture["arms"]["A"]["tcp"])[:3, 3] - [0., 0., .18]
        elif stage >= 4:
            source = [.6, -.2, .62]
        points = np.array([np.asarray(source) + [x, y, .02] for x in np.linspace(-.02, .02, 5)
                           for y in np.linspace(-.02, .02, 5)])
        coverage = {"camera_ids": ["1", "2", "3"], "observed_surface_only": True}
        obj = {"id": "source", "kind": "cube", "color": "red", "confidence": .95,
               "position": points.mean(axis=0).tolist(), "visible_centroid_world": points.mean(axis=0).tolist(),
               "points_world_m": points, "coverage": coverage}
        bin_points = np.array([[x, y, .6] for x in np.linspace(.52, .68, 5)
                               for y in np.linspace(-.26, -.14, 5)])
        container = {"id": "destination", "kind": "bin", "color": "blue", "position": [.6, -.2, .6],
                     "visible_centroid_world": [.6, -.2, .6], "points_world_m": bin_points,
                     "confidence": .95, "coverage": coverage,
                     "placement_reference": {"position_world_m": [.6, -.2, .6], "floor_z_m": .6,
                                             "region_polygon_world_xy_m": [[.52, -.26], [.68, -.26], [.68, -.14], [.52, -.14]],
                                             "source": "rgbd_visible_floor"}}
        views = copy.deepcopy(capture["views"])
        for view in views.values():
            view.update(T_world_camera=transform([0., 0., 0.]),
                        intrinsics={"fx": .5, "fy": .5, "cx": 0., "cy": 1.},
                        depth_m=np.full((2, 2), float(points[0, 2]), dtype=np.float32),
                        valid_depth=np.ones((2, 2), dtype=bool),
                        instance_masks={"destination": np.ones((2, 2), dtype=bool),
                                        "source": np.ones((2, 2), dtype=bool)})
        return {key: copy.deepcopy(capture[key]) for key in
                ("generation", "time", "observation_id", "source", "arms")} | {
                    "objects": [obj, container], "views": views,
                    "visual_inventory": visual_inventory_result(
                        capture, source_hint, [obj, container],
                        membership_status=self.membership_status,
                        progress_status='at_destination' if stage >= 4 and not closed else 'elsewhere'),
                    "table": {"position": [.5, 0., .595], "size": [.7, 1., .01], "surface_height_m": .6}}

    def build_scene(self, observation, target_id, *, planning_settings=None, exclusion_state="selected"):
        if self.fail_after_release and exclusion_state == "released":
            raise RuntimeError("released_map_unavailable")
        return {"scene_token": observation["observation_id"], "observation_id": observation["observation_id"],
                "generation": observation["generation"], "target_id": target_id, "exclusion_state": exclusion_state}

    def generate_grasps(self, *args, **kwargs):
        return {"candidates": [{"index": 1}]}

    def evaluate_arms(self, *args, **kwargs):
        return {"A": {"valid_candidates": [{"candidate_index": 1}]}, "B": {"valid_candidates": []}}

    def evaluate_reachability(self, *args, **kwargs):
        return {'A': {'reachable': True, 'feasibility_scope': 'offline_workspace'},
                'B': {'reachable': False, 'feasibility_scope': 'offline_workspace'}}

    def plan_dual_observation_motion(self, observation, scene, goals, **kwargs):
        return {'paths': {arm: [observation['arms'][arm]['joints_rad'], goals[arm]]
                          for arm in ('A', 'B')}, 'dt': .02}

    def validate_dual_observation(self, *args, **kwargs):
        return {'collision_free': True}

    @staticmethod
    def command(stage):
        return {"positions_rad": [[float(max(0, stage - 1))] * 7, [float(stage)] * 7], "dt": .02}

    def plan_grasp(self, *args, **kwargs):
        phases = {name: self.command(stage) for name, stage in (("approach", 0), ("grasp", 1), ("lift", 2))}
        phases['approach']['positions_rad'][0] = list(args[0]['arms'][args[2]]['joints_rad'])
        return {"T_world_tcp": transform([.5, .1, .8]), "candidate_index": 1, "phases": phases}

    def validate_execution(self, *args, **kwargs):
        return {"collision_free": True}

    def prepare_carrying(self, observation, arm, scene, holding, **kwargs):
        assert holding["verified"] and not holding["attached"]
        self.prepare_count += 1
        return {"attached": True, "scene_token": scene["scene_token"]}

    def detach_carrying(self):
        self.clear_count += 1
        return {"attached": False}

    def plan_place(self, observation, target, arm, scene, holding, destination, **kwargs):
        from jev_mujoco.algorithms import Algorithms

        return {"arm": arm, "scene_token": scene["scene_token"], "observation_id": observation["observation_id"],
                "generation": observation["generation"],
                "carrying_binding": Algorithms._held_binding(observation, target, arm, scene, holding),
                "holding_reference": copy.deepcopy(holding["points_tool_m"]),
                "goal": {"T_world_tcp": transform([.6, -.2, .817]),
                         "release_reference": {"floor_z_m": .6, "release_height_m": .012,
                                               "tool_reference_point_m": [0., 0., .185],
                                               "target_reference_world_xy_m": [.6, -.2],
                                               "required_clearance_m": .01,
                                               "attachment_spheres_tool_m": [[0., 0., -.195, .01]],
                                               "nominal_object_floor_gap_m": .017}},
                "phases": {name: self.command(stage) for name, stage in (("approach", 3), ("place", 4), ("retreat", 5))}}

    def plan_retreat(self, *args, **kwargs):
        return {"command": self.command(5), "check": {"attached": False}}

    def clear_carrying(self):
        self.clear_count += 1
        return {"ok": True}


class Choices:
    def __init__(self, task="grasp_place"):
        self.task = task

    async def choose_visual_plan(self, state, skill, requests):
        return dict(requests)

    async def bind_request(self, instruction):
        return {'task': self.task, **await self.bind_attributes(instruction, self.task),
                'arm': 'auto', 'scope': 'single',
                **{'destination_' + key: value for key, value in
                   (await self.bind_destination_attributes(instruction)).items()}}

    async def bind_task(self, instruction):
        return self.task

    async def bind_attributes(self, instruction, task="grasp"):
        return {"kind": "cube", "color": "red", "position": "any"}

    async def bind_destination_attributes(self, instruction):
        return {"kind": "bin", "color": "blue", "position": "any"}

    async def choose_skill(self, state, registry):
        if state.get('verification', {}).get('status') == 'unknown':
            return 'stop'
        if state['facts'].get('recovery_needed'):
            return 'recover_task'
        from test_runner import legacy_workflow_choice
        return legacy_workflow_choice(state, registry.eligible(state['facts']))

    async def choose_recovery(self, state):
        return 'stop'

    async def choose_target(self, state, objects):
        return "source"

    async def choose_destination(self, state, objects):
        return "destination"

    async def choose_arm(self, state, evaluations):
        return "A"


async def test_direct_single_confirms_only_the_jev_selected_measured_member(tmp_path):
    from test_collection_runner import CollectionVision

    class InspectingChoices(Choices):
        async def choose_target(self, state, objects):
            assert state['task_progress']['objects'] == []
            assert set(state['task_progress']['membership_pending_ids']) == {'source', 'second'}
            return 'source'

    runner = Runner(SensorRobot(), CollectionVision(), lambda record: InspectingChoices(), tmp_path)
    await runner.start('仅抓取指定的一个物体放进盒子')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert [obj['id'] for obj in runner.task_progress.snapshot()['objects']] == ['source']
    assert runner.task_progress.membership['source'] == 'required'
    assert runner.task_progress.records['source']['display_id'] == 1
    assert runner.task_progress.records['second']['display_id'] > 1
    assert runner.task_progress.scene_entities['second']['display_id'] == runner.task_progress.records['second']['display_id']
    assert all(event['title'] != 'assess_task' for event in runner.events if event['kind'] == 'skill_complete')


async def test_grasp_place_continues_after_grasp_verification_and_finishes_only_after_place(tmp_path):
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: Choices(), tmp_path)
    await runner.start("抓取红色方块到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["status"] == "succeeded", runner.state
    assert runner.state["facts"]["released"] is True
    assert runner.state["facts"]["grasp_held"] is False
    assert runner.state["facts"]["place_verified"] is True
    skills = [event["title"] for event in runner.events if event["kind"] == "skill_complete"]
    assert skills[-1] == "verify_place"
    assert skills.index("verify_grasp") < skills.index("plan_place")
    assert "plan_grasp" in skills and "plan_grasp_and_place" not in skills
    assert runner.algorithms.checkpoints == ["initial", "fine", "post_place"]
    assert runner.algorithms.prepare_count == 1
    assert runner.algorithms.clear_count == 1
    assert runner.holding_state is None


async def test_place_reuses_map_without_release_rebuild(tmp_path):
    runner = Runner(SensorRobot(), VisionAlgorithms(fail_after_release=True), lambda record: Choices(), tmp_path)
    await runner.start("pick red cube into blue bin")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    assert runner.state["facts"]["released"] is True
    assert runner.algorithms.clear_count == 1
    assert runner.algorithms.checkpoints == ["initial", "fine", "post_place"]


async def test_rejected_skill_is_returned_to_jev_without_physical_recovery(tmp_path):
    class CorrectedChoices(Choices):
        def __init__(self):
            super().__init__()
            self.calls = 0
        async def choose_skill(self, state, registry):
            self.calls += 1
            if self.calls == 1:
                return 'execute_grasp'
            if self.calls == 2:
                assert state['skill_rejection']['selected_skill'] == 'execute_grasp'
                assert not state['facts'].get('recovery_needed')
                assert not state['task_progress']['attempts']
            return await super().choose_skill(state, registry)
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: CorrectedChoices(), tmp_path)
    await runner.start('把红色方块放进蓝盒')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert not any(e['kind'] == 'skill_start' and e['title'] == 'recover_task' for e in runner.events)


async def test_place_without_session_holding_is_rejected(tmp_path):
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: Choices("place"), tmp_path)
    await runner.start("放到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["error"] == "place_requires_verified_holding"


async def test_repeated_invalid_jev_selection_is_bounded_without_physical_attempts(tmp_path):
    class InvalidChoices(Choices):
        async def choose_skill(self, state, registry):
            return 'execute_grasp'
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: InvalidChoices(), tmp_path)
    await runner.start('把红色方块放进蓝盒')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'needs_attention'
    assert runner.state['error'] == 'skill_selection_budget_exhausted'
    assert not runner.task_progress.snapshot()['attempts']
    errors = [event for event in runner.events if event['kind'] == 'skill_error']
    assert len(errors) == 8
    assert not any(event['kind'] == 'skill_complete' for event in runner.events)




async def test_standalone_grasp_retains_geometry_and_defers_attachment_until_place(tmp_path):
    class PronounChoices(Choices):
        async def bind_attributes(self, instruction, task="grasp"):
            if task == "place":
                return {"kind": "any", "color": "any", "position": "any"}
            return await super().bind_attributes(instruction, task)

    choices = PronounChoices("grasp")
    algorithms = VisionAlgorithms()
    runner = Runner(SensorRobot(), algorithms, lambda record: choices, tmp_path)
    await runner.start("抓取红色方块")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["status"] == "succeeded", runner.state
    assert runner.holding_state["verified"] and not runner.holding_state["attached"]
    assert algorithms.prepare_count == algorithms.clear_count == 0
    observe = algorithms.observe

    def held_source_occluded(capture, **kwargs):
        result = observe(capture, **kwargs)
        if kwargs.get("checkpoint") == "standalone_place":
            result["objects"] = [obj for obj in result["objects"] if obj["id"] != "source"]
            for view in result["views"].values():
                view["instance_masks"].pop("source", None)
        return result

    algorithms.observe = held_source_occluded
    choices.task = "place"
    await runner.start("放到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["status"] == "succeeded", runner.state
    assert algorithms.prepare_count == 1 and algorithms.clear_count == 1
    assert algorithms.checkpoints == ["initial", "fine", "post_place"]
    assert all(hint["kind"] == "cube" and hint["color"] == "red" for hint in algorithms.source_hints)
    assert runner.holding_state is None


async def test_standalone_place_reuses_saved_scene_then_observes_final_destination(tmp_path):
    class RecordedChoices(Choices):
        def __init__(self):
            super().__init__("grasp")
            self.states = []

        async def choose_skill(self, state, registry):
            self.states.append(copy.deepcopy(state))
            return await super().choose_skill(state, registry)

    choices = RecordedChoices()
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: choices, tmp_path)
    await runner.start("抓取红色方块")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["status"] == "succeeded", runner.state
    previous_observation_id = runner.holding_state["observation"]["observation_id"]
    choices.task = "place"
    choices.states.clear()
    await runner.start("放到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["status"] == "succeeded", runner.state
    initial = choices.states[0]
    assert initial['facts']['observed'] and initial['facts']['scene_ready']
    assert initial['facts']['arm_selected'] and initial['facts']['inventory_ready']
    assert not initial['facts']['attached'] and not initial['facts']['place_plan_ready']
    assert initial['observation']['observation_id'] == previous_observation_id
    assert runner.algorithms.checkpoints == ['initial', 'fine', 'post_place']


async def test_failed_place_planning_keeps_verified_attachment(tmp_path):
    class FailedPlace(VisionAlgorithms):
        def plan_place(self, *args, **kwargs):
            raise RuntimeError("place_path_unavailable")

    algorithms = FailedPlace()
    robot = SensorRobot()
    runner = Runner(robot, algorithms, lambda record: Choices(), tmp_path)
    await runner.start("抓取红色方块到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["error"] == "place_path_unavailable"
    assert runner.state['failed_step']['stage'] == 'placement_candidates'
    assert runner.holding_state["verified"] and runner.holding_state["attached"]
    assert robot.stage == 2 and robot.closed
    assert algorithms.prepare_count == 1 and algorithms.clear_count == 0


async def test_destination_visible_to_one_registered_camera_is_sufficient(tmp_path):
    class WristOnlyDestination(VisionAlgorithms):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            observation["objects"][1]["coverage"] = {"camera_ids": ["1", "2"]}
            return observation

    robot = SensorRobot()
    runner = Runner(robot, WristOnlyDestination(), lambda record: Choices(), tmp_path)
    await runner.start("抓取红色方块到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert robot.stage == 5 and not robot.closed


async def test_measured_open_detaches_even_if_motor_reports_error_after_release(tmp_path):
    class ErrorAfterRelease(SensorRobot):
        def set_gripper(self, arm, closed, generation):
            result = super().set_gripper(arm, closed, generation)
            if not closed and self.stage == 4:
                raise RuntimeError("release_command_reply_lost")
            return result

    robot = ErrorAfterRelease()
    algorithms = VisionAlgorithms()
    choices = Choices()
    runner = Runner(robot, algorithms, lambda record: choices, tmp_path)
    await runner.start("抓取红色方块到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["error"] == "release_command_reply_lost"
    assert runner.state["facts"]["released"] is True
    assert runner.state["facts"]["grasp_held"] is False
    assert runner.state["facts"]["attached"] is False
    assert algorithms.clear_count == 1 and not robot.closed
    assert runner.holding_state["state"] == "release_pending"
    assert runner.holding_state["verified"] is False
    choices.task = "grasp"
    await runner.start("再次抓取红色方块")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["error"] == "robot_already_holding_object"
    assert robot.stage == 4 and algorithms.prepare_count == 1
