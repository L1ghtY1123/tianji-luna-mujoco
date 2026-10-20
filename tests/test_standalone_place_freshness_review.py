"""Independent checks for fresh place observations without pruning JEV choices."""

import asyncio
import copy

import numpy as np
import pytest
from test_decisions import install_agent
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

from jev_mujoco.decisions import Decisions
from jev_mujoco.runner import Runner


def held_runner(tmp_path, choices=None):
    robot = SensorRobot()
    robot.closed = True
    robot.stage = 2
    robot.tcp = [.5, .1, .9]
    algorithms = VisionAlgorithms()
    observation = algorithms.observe(robot.capture_rgbd(), checkpoint="post_grasp")
    choices = choices or Choices("place")
    runner = Runner(robot, algorithms, lambda record: choices, tmp_path)
    points = np.asarray(observation["objects"][0]["points_world_m"])
    runner.holding_state = {"state": "held", "verified": True, "attached": True,
                            "generation": 1, "instance_id": "source", "arm": "A", "owner_id": "earlier-run",
                            "scene": {"scene_token": "shared", "observation_id": observation["observation_id"]},
                            "scene_observation": copy.deepcopy(observation),
                            "grasp_support_reference": {"T_world_tcp_at_grasp": np.eye(4).tolist()},
                            "observation": observation, "points_tool_m": (points - robot.tcp).tolist()}
    runner.state.update(task="place", target_id="source", arm="A", planning_attempts=[],
                        facts={"observed": False, "attached": True, "grasp_held": True,
                               "grasp_completed": True, "target_selected": True})
    return runner, observation


def test_unobserved_snapshot_hides_old_scene_but_keeps_holding_facts_and_robot_feedback(tmp_path):
    runner, observation = held_runner(tmp_path)
    state = runner.snapshot_for_decision({"observation": observation})
    assert state["observation"] is None
    assert state["facts"]["attached"] is True and state["target_id"] == "source"
    assert state["simulation"]["arms"]["A"]["gripper_closed_fraction"] == .6
    assert "objects" not in state["simulation"]


def test_observed_snapshot_publishes_current_scene_and_is_frozen_for_all_decision_callers(tmp_path):
    runner, retained = held_runner(tmp_path)
    current = runner.algorithms.observe(runner.simulation.capture_rgbd(), checkpoint="standalone_place")
    runner.state["facts"]["observed"] = True
    state = runner.snapshot_for_decision({"observation": current})
    assert state["observation"]["observation_id"] != retained["observation_id"]
    assert len(state["observation"]["objects"]) == 2
    assert all("points_world_m" not in obj for obj in state["observation"]["objects"])
    saved = copy.deepcopy(state)
    current["objects"][0]["color"] = "yellow"
    runner.state["facts"]["observed"] = False
    assert state == saved
    assert runner.snapshot_for_decision()["observation"] is None


@pytest.mark.parametrize("observed", [False, True])
async def test_place_freshness_preserves_complete_skill_catalog_and_no_eligibility_hints(
        monkeypatch, tmp_path, observed):
    runner, retained = held_runner(tmp_path)
    runner.state["facts"]["observed"] = observed
    state = runner.snapshot_for_decision({"observation": retained})
    captured, records = [], []
    install_agent(monkeypatch, {"skill": "plan_place"}, {"skill": .9}, captured)
    model = Decisions(None, lambda kind, title, data: records.append(data))
    assert await model.choose_skill(state, runner.registry) == "plan_place"
    assert set(records[0]["questions"]["skill"]["criteria"]) == (set(runner.registry.contracts) - {"observe"}) | {"stop"}
    supplied = captured[0]["state"]
    assert supplied["skill_catalog"] == runner.registry.describe()
    assert supplied["facts"]["observed"] is observed
    assert all("eligible" not in item and "unmet" not in item for item in supplied["skill_catalog"].values())


async def test_place_destination_choice_uses_saved_scene_without_hidden_observation_or_motion(tmp_path):
    class SkipObservation(Choices):
        async def choose_skill(self, state, registry):
            if state['facts'].get('destination_selected'):
                return 'stop'
            assert state['observation']['observation_id'] == retained['observation_id']
            assert state['facts']['observed'] and state['facts']['arm_selected']
            return "identify_destination"

    runner, retained = held_runner(tmp_path, SkipObservation("place"))
    await runner.start("放到蓝色盒子里")
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state["error"] == 'jev_selected_stop'
    assert runner.algorithms.checkpoints == ["post_grasp"]
    assert runner.holding_state["observation"]["observation_id"] == retained["observation_id"]
    assert runner.holding_state["verified"] and runner.holding_state["attached"]
    assert runner.simulation.stage == 2 and runner.simulation.closed
    assert [event['title'] for event in runner.events if event['kind'] == 'skill_start'] == ['identify_destination']
