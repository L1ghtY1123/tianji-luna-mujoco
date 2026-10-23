import copy

import numpy as np
from test_runner import visual_inventory_result


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
        from luna_mujoco.algorithms import Algorithms

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
