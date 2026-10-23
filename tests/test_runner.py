import asyncio
import copy
import json

import numpy as np
import pytest

from luna_mujoco.runner import Runner, public_summary


def visual_inventory_result(capture, source_hint, objects, *, membership_status='required',
                            progress_status='at_destination'):
    """Explicit Decisions evidence for synthetic runner observations."""
    progress = {}
    for identifier, record in (source_hint or {}).get('visual_progress', {}).items():
        progress[identifier] = {
            'status': progress_status, 'source': 'openai_decisions',
            'observation_id': capture['observation_id'],
            **{key: record[key] for key in ('destination_id', 'region_id', 'slot_id') if key in record},
        }
    return {
        'membership': {obj['id']: 'excluded' if obj['kind'] in {'bin', 'marker', 'table'} else membership_status
                       for obj in objects},
        'progress': progress,
        'evidence': {'source': 'openai_decisions', 'observation_id': capture['observation_id']},
    }


class Sim:
    def __init__(self):
        self.frame = 0
        self.state = {"generation": 1, "time": 1., "confirmed": True,
                      "objects": [{"id": "black", "kind": "cube", "color": "black",
                                   "T_world_object": np.eye(4).tolist()}],
                      "arms": {arm: {"joints_rad": [0.] * 7, "tcp": np.eye(4).tolist(),
                                     "gripper_closed_fraction": .6, "gripper_position_rad": .48,
                                     "gripper_limits_rad": [0., .8]} for arm in ("A", "B")}}

    def snapshot(self):
        return copy.deepcopy(self.state)

    def robot_snapshot(self):
        result = {key: copy.deepcopy(self.state[key]) for key in ("generation", "time", "confirmed", "arms")}
        for arm in result["arms"].values():
            q = arm["gripper_position_rad"]
            arm.update(gripper_limits_rad=[0., .8], gripper_closed_command=q > 0.,
                       gripper_position_samples=[[t, q] for t in (0., .05, .1, .15, .2, .25)])
        return result

    def capture_rgbd(self):
        self.frame += 1
        frame_id = f"rgbd-{self.state['generation']}-{self.frame}"
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        # This synthetic camera pixel encodes the observed height, without object metadata.
        rgb[0, 0, 0] = round(self.state["objects"][0]["T_world_object"][2][3] * 1000)
        return {"source": "rgbd", "generation": self.state["generation"], "time": self.state["time"],
                "observation_id": frame_id, "arms": self.robot_snapshot()["arms"],
                "views": {str(i): {"rgb": rgb.copy(), "depth_m": np.ones((4, 4), dtype=np.float32)}
                          for i in (1, 2, 3)}}

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        assert generation == self.state["generation"]
        if positions_rad[-1][0] == 1:
            self.state["objects"][0]["T_world_object"][2][3] = .1
            self.state["arms"][arm]["tcp"][2][3] = .1
        self.state["arms"][arm]["joints_rad"] = list(positions_rad[-1])
        return {"completed": True}

    def execute_dual_trajectory(self, paths, dt, generation):
        assert generation == self.state['generation']
        for arm in ('A', 'B'):
            self.state['arms'][arm]['joints_rad'] = list(paths[arm][-1])
        return {'completed': True}

    def dwell(self, duration_s, generation):
        assert generation == self.state["generation"]
        return {"completed": True, "duration_s": duration_s}

    def set_gripper(self, arm, closed, generation):
        assert generation == self.state["generation"]
        self.state["arms"][arm].update(gripper_closed_fraction=.6 if closed else 0.,
                                       gripper_position_rad=.48 if closed else 0.)
        return {"completed": True}


def test_batch_candidate_ids_remain_complete_in_trace_but_trajectories_are_summarized():
    data = public_summary({"candidate_indices": list(range(100)), "positions_rad": [[0.] * 7] * 100})
    assert data["candidate_indices"] == list(range(100))
    assert data["positions_rad"]["count"] == 100


@pytest.mark.parametrize("numpy_values", [False, True])
def test_algorithm_event_summarizes_numpy_evidence_without_mutating_input(tmp_path, numpy_values):
    runner = Runner(Sim(), None, None, tmp_path)
    runner.run_id = "full_evaluation"
    expected = {"A": {"valid_candidates": [
        {"candidate_index": index, "score": .8123456789012345,
         "T_world_tcp": [[.1234567890123456] * 4 for _ in range(4)],
         "ik": {"position_error_m": 6.664001972467304e-8,
                "rotation_error_rad": 3.2784054582180033e-8}}
        for index in range(12)]}}
    result = copy.deepcopy(expected)
    if numpy_values:
        for candidate in result["A"]["valid_candidates"]:
            candidate["candidate_index"] = np.int64(candidate["candidate_index"])
            candidate["score"] = np.float64(candidate["score"])
            candidate["T_world_tcp"] = np.asarray(candidate["T_world_tcp"])
    data = {"result": result, "duration_s": 1.25}
    original = copy.deepcopy(data)
    runner.emit("compute_ik", "抓姿批量 IK", data)
    event = runner.events[0]
    assert event["data"]["result"]["A"]["valid_candidates"]["count"] == 12
    assert "full_result" not in event["data"]
    logged = json.loads((tmp_path / "full_evaluation.jsonl").read_text(encoding="utf-8"))
    assert "full_result" not in logged["data"]
    assert logged["data"]["result"] == event["data"]["result"]
    for current, before in zip(result["A"]["valid_candidates"], original["result"]["A"]["valid_candidates"]):
        np.testing.assert_array_equal(current["T_world_tcp"], before["T_world_tcp"])
        assert type(current["T_world_tcp"]) is type(before["T_world_tcp"])
        assert current["score"] == before["score"]
        assert current["ik"] == before["ik"]
    assert "full_result" not in data


def test_algorithm_event_log_failure_keeps_existing_fail_closed_behavior(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.write_text("file", encoding="utf-8")
    runner = Runner(Sim(), None, None, occupied)
    runner.run_id = "full_evaluation"
    runner.emit("compute_ik", "抓姿批量 IK", {"result": {"A": {"valid_candidates": list(range(12))}}})
    assert runner.state["log_error"].startswith("run_log_write_failed:")
    assert "full_result" not in runner.events[0]["data"]
    with pytest.raises(RuntimeError, match="run_log_write_failed:"):
        runner.check_log()


class Algorithms:
    def clear_carrying(self):
        return {"ok": True, "attached": False}

    def observe(self, capture, *, instruction="", checkpoint=None, source_hint=None):
        pose = np.eye(4)
        pose[2, 3] = float(capture["views"]["1"]["rgb"][0, 0, 0]) / 1000
        local = np.asarray([[x, y, .02] for x in np.linspace(-.02, .02, 5)
                            for y in np.linspace(-.02, .02, 5)], dtype=np.float32)
        points = local + pose[:3, 3]
        obj = {"id": "black", "kind": "cube", "color": "black",
               "position": points.mean(axis=0).tolist(), "visible_centroid_world": points.mean(axis=0).tolist(),
               "points_world_m": points,
               "coverage": {"observed_surface_only": True, "camera_ids": ["1", "2", "3"],
                            "limitations": ["measured_cube_top"]},
               "uncertainty_m": .002, "source": "mock_registered_rgbd", "confidence": 1.}
        return {"source": "rgbd", "observation_id": capture["observation_id"],
                "generation": capture["generation"], "time": capture["time"],
                "arms": copy.deepcopy(capture["arms"]), "objects": [obj],
                "visual_inventory": visual_inventory_result(capture, source_hint, [obj]),
                "views": {camera: {**view, 'instance_masks': {'black': np.ones(view['depth_m'].shape, dtype=bool)}}
                          for camera, view in capture['views'].items()},
                "table": {"position": [.55, 0., -.04], "size": [.7, 1., .08],
                          "surface_height_m": 0., "source": "rgbd"}}

    def build_scene(self, observation, target, *, planning_settings=None, exclusion_state="selected"):
        return {"scene_token": "x", "observation_id": observation["observation_id"],
                "generation": observation["generation"], "target_id": target, "exclusion_state": exclusion_state}

    def generate_grasps(self, *args, **kwargs):
        return {"candidates": [{"index": 1}]}

    def evaluate_arms(self, *args, planning_settings=None, progress=None, cancelled=None, requested_arm=None):
        return {"A": {"valid_candidates": [{"index": 1}]}, "B": {"valid_candidates": []}}

    def evaluate_reachability(self, *args, **kwargs):
        return {'A': {'reachable': True, 'feasibility_scope': 'offline_workspace'},
                'B': {'reachable': False, 'feasibility_scope': 'offline_workspace'}}

    def plan_dual_observation_motion(self, observation, scene, goals, **kwargs):
        return {'paths': {arm: [observation['arms'][arm]['joints_rad'], goals[arm]]
                          for arm in ('A', 'B')}, 'dt': .02}

    def validate_dual_observation(self, *args, **kwargs):
        return {'collision_free': True}

    def plan_grasp(self, *args, planning_settings=None, progress=None, cancelled=None):
        observation = args[0]
        arm = args[2]
        start = observation["arms"][arm]["joints_rad"]
        return {"T_world_tcp": np.eye(4).tolist(),
                "phases": {phase: {"positions_rad": [list(start) if phase == 'approach' else [0.] * 7,
                                                      [float(phase == "lift")] * 7], "dt": .1}
                           for phase in ("approach", "grasp", "lift")}}

    def validate_execution(self, *args, **kwargs):
        return {"collision_free": True}

    def prepare_carrying(self, observation, arm, scene, holding, *, planning_settings=None):
        assert holding["verified"] and not holding["attached"]
        return {"attached": True, "scene_token": scene["scene_token"]}

    def detach_carrying(self):
        return {"attached": False}


def legacy_workflow_choice(state, eligible):
    """Choose the accepted atomic scene cycle from the public eligible catalog."""
    order = ('recover_task', 'observe_scene', 'assess_task', 'identify_target', 'identify_destination',
             'evaluate_reachability', 'choose_arm', 'build_scene', 'generate_grasps', 'plan_grasp',
             'execute_grasp', 'verify_grasp', 'plan_place', 'execute_place', 'verify_place')
    return next(skill for skill in order if skill in eligible)


class Decisions:
    async def choose_visual_plan(self, state, skill, requests):
        return dict(requests)

    async def bind_request(self, instruction):
        task = await self.bind_task(instruction)
        attributes = await self.bind_attributes(instruction, task=task)
        destination = await self.bind_destination_attributes(instruction) if task != 'grasp' else {}
        return {'task': task, **attributes, 'arm': 'auto', 'scope': 'single',
                **{'destination_' + key: value for key, value in destination.items()}}

    async def bind_destination_attributes(self, instruction):
        return {'kind': 'bin', 'color': 'any', 'position': 'any'}

    async def bind_task(self, instruction):
        return "grasp"

    async def bind_attributes(self, instruction, task="grasp"):
        return {"kind": "any", "color": "black", "position": "any"}

    async def choose_skill(self, state, registry):
        if state['facts'].get('recovery_needed'):
            return 'recover_task'
        return legacy_workflow_choice(state, registry.eligible(state['facts']))

    async def choose_recovery(self, state):
        return 'stop'

    async def choose_target(self, state, objects):
        return "black"

    async def choose_arm(self, state, evaluations):
        return "A"


async def wait(runner):
    await asyncio.wait_for(runner.worker, 5)








async def test_unconfirmed_scene_rejected(tmp_path):
    sim = Sim()
    sim.state["confirmed"] = False
    runner = Runner(sim, Algorithms(), lambda record: Decisions(), tmp_path)
    with pytest.raises(ValueError, match="confirm_scene_first"):
        await runner.start("抓取黑色物体")
