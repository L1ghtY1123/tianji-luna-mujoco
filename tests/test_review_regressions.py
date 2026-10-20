import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms
from test_runner import Algorithms, Decisions, Sim

from jev_mujoco.runner import Runner


class VisualAlgorithms(Algorithms):
    """Controlled perception estimates keep drift tests off editor truth."""

    def __init__(self):
        self.visual_poses = [np.eye(4), np.eye(4)]
        self.visual_poses[1][1, 3] = .15

    def observe(self, capture, *, instruction="", checkpoint=None, source_hint=None):
        observation = super().observe(capture, instruction=instruction, checkpoint=checkpoint, source_hint=source_hint)
        template = observation["objects"][0]
        objects = []
        for identifier, pose in zip(("black", "obstacle"), self.visual_poses, strict=True):
            obj = copy.deepcopy(template)
            local = np.asarray(obj["points_world_m"])
            points = local @ pose[:3, :3].T + pose[:3, 3]
            obj.update(id=identifier, color="black" if identifier == "black" else "gray",
                       position=points.mean(axis=0).tolist(), visible_centroid_world=points.mean(axis=0).tolist(),
                       points_world_m=points)
            objects.append(obj)
        observation["objects"] = objects
        observation['visual_inventory']['membership']['obstacle'] = 'excluded'
        return observation


async def prepare_runner(tmp_path):
    simulation = Sim()
    algorithms = VisualAlgorithms()
    runner = Runner(simulation, algorithms, lambda record: Decisions(), tmp_path)
    from jev_mujoco.task_progress import TaskProgress
    runner.task_progress = TaskProgress('grasp the selected object', requested_arm='A')
    runner.task_progress.bind_arm('A')
    runner.state.update(generation=1, target_id="black", arm="A", task="grasp")
    context = {"initial": simulation.robot_snapshot(), "planning_settings": runner.planning_settings.model_dump()}
    runner.state['current_step'] = {'skill': 'observe_scene'}
    arguments = await runner.select_arguments('observe_scene', context, Decisions())
    await runner.execute("observe_scene", context, arguments)
    await runner.execute("identify_target", context, {"target_id": "black"})
    arguments = await runner.select_arguments('build_scene', context, Decisions())
    runner.state['current_step'] = {'skill': 'build_scene'}
    await runner.execute("build_scene", context, arguments)
    await runner.execute("generate_grasps", context, Decisions())
    return runner, algorithms, context


def move_object(algorithms, object_index, change):
    matrix = algorithms.visual_poses[object_index].copy()
    if change == "translation":
        matrix[0, 3] += .2
    else:
        matrix[:3, :3] = Rotation.from_euler("z", 10, degrees=True).as_matrix()
    algorithms.visual_poses[object_index] = matrix


@pytest.mark.parametrize("object_index", [0, 1])
@pytest.mark.parametrize("change", ["translation", "rotation"])
@pytest.mark.parametrize("when", ["before_evaluation", "before_planning"])
async def test_explicit_new_observation_invalidates_old_scene_before_planning(tmp_path, object_index, change, when):
    runner, algorithms, context = await prepare_runner(tmp_path)
    skill = "plan_grasp"
    if when == "before_planning":
        await runner.verify_scene(context)
        skill = "plan_grasp"
    move_object(algorithms, object_index, change)
    context["visual_plan"] = {"fine": "associate_inventory"}
    await runner.observe(context, checkpoint="fine", recovery=True)
    with pytest.raises(ValueError, match="scene_observation_stale"):
        if when == 'before_evaluation':
            await runner.verify_scene(context)
        else:
            await runner.execute(skill, context, Decisions())


async def test_execution_rejects_new_observation_after_planning(tmp_path):
    runner, algorithms, context = await prepare_runner(tmp_path)
    await runner.verify_scene(context)
    await runner.execute("plan_grasp", context, Decisions())
    move_object(algorithms, 0, "rotation")
    context["visual_plan"] = {"fine": "associate_inventory"}
    await runner.observe(context, checkpoint="fine", recovery=True)
    with pytest.raises(ValueError, match="scene_observation_stale"):
        await runner.execute("execute_grasp", context, Decisions())


async def test_planning_does_not_implicitly_reobserve_between_declared_checkpoints(tmp_path):
    runner, algorithms, context = await prepare_runner(tmp_path)
    frame_count = runner.simulation.frame
    await runner.verify_scene(context)
    algorithms.visual_poses[0][0, 3] = .02
    await runner.execute("plan_grasp", context, Decisions())
    assert runner.simulation.frame == frame_count
    assert context["observation"]["checkpoint"] == "fine"




class RecordingSensorRobot(SensorRobot):
    def __init__(self):
        super().__init__()
        self.motor_calls = []

    def execute_trajectory(self, arm, positions_rad, dt, generation):
        self.motor_calls.append(int(positions_rad[-1][0]))
        return super().execute_trajectory(arm, positions_rad, dt, generation)


class PlaceVisualAlgorithms(VisionAlgorithms):
    def __init__(self):
        super().__init__()
        self.destination_shift_m = 0.
        self.source_shift_m = 0.
        self.source_yaw_deg = 0.
        self.source_surface_half_width_m = None
        self.place_calls = 0

    def observe(self, capture, instruction="", checkpoint=None, source_hint=None):
        observation = super().observe(capture, instruction=instruction, checkpoint=checkpoint, source_hint=source_hint)
        source, destination = observation["objects"]
        points = np.asarray(source["points_world_m"]).copy()
        center = points.mean(axis=0)
        if self.source_surface_half_width_m is not None:
            width = self.source_surface_half_width_m
            points = center + np.array([[x, y, 0.] for x in np.linspace(-width, width, 5)
                                       for y in np.linspace(-width, width, 5)])
        if self.source_yaw_deg:
            rotation = Rotation.from_euler("z", self.source_yaw_deg, degrees=True).as_matrix()
            points = (points - center) @ rotation.T + center
        points += [self.source_shift_m, 0., 0.]
        source.update(points_world_m=points, position=points.mean(axis=0).tolist(),
                      visible_centroid_world=points.mean(axis=0).tolist())
        destination["points_world_m"] += [0., self.destination_shift_m, 0.]
        destination["position"][1] += self.destination_shift_m
        destination["placement_reference"]["position_world_m"][1] += self.destination_shift_m
        return observation

    def plan_place(self, observation, target, arm, scene, holding, destination, **kwargs):
        self.place_calls += 1
        return super().plan_place(observation, target, arm, scene, holding, destination, **kwargs)


async def verified_visual_holding(tmp_path, *, surface_half_width_m=None):
    from test_runner import wait

    robot = RecordingSensorRobot()
    algorithms = PlaceVisualAlgorithms()
    algorithms.source_surface_half_width_m = surface_half_width_m
    runner = Runner(robot, algorithms, lambda record: Choices("grasp"), tmp_path)
    await runner.start("抓取红色方块")
    await wait(runner)
    assert runner.state["status"] == "succeeded", runner.state["error"]
    assert runner.holding_state["verified"] is True
    runner.state.update(status="running", task="place", destination_id="destination",
                        current_step={"skill": "execute_place"})
    context = {"initial": robot.robot_snapshot(), "planning_settings": runner.planning_settings.model_dump(),
               "observation": copy.deepcopy(runner.holding_state["observation"]),
               "scene": copy.deepcopy(runner.holding_state["scene"]),
               "scene_observation": copy.deepcopy(runner.holding_state["scene_observation"]),
               "holding": copy.deepcopy(runner.holding_state)}
    return runner, algorithms, context


async def test_accepted_small_visual_change_does_not_refit_measured_attachment(tmp_path):
    runner, algorithms, context = await verified_visual_holding(tmp_path)
    original = copy.deepcopy(context["holding"])
    motors = runner.simulation.motor_calls.copy()
    algorithms.source_shift_m = .019
    context["visual_plan"] = {"standalone_place": "associate_inventory"}
    observation = await runner.observe(context, checkpoint="standalone_place")
    holding = context["holding"]
    np.testing.assert_array_equal(holding["points_tool_m"], original["points_tool_m"])
    assert holding["observation"]["observation_id"] == observation["observation_id"]
    assert "estimated_bounds_tool_m" not in holding and "T_tcp_object" not in holding
    assert runner.holding_state["verified"] is True
    assert runner.simulation.motor_calls == motors


@pytest.mark.parametrize("surface_width", [.004, .02])
async def test_standalone_view_change_does_not_change_position_verdict_or_attachment(tmp_path, surface_width):
    runner, algorithms, context = await verified_visual_holding(tmp_path, surface_half_width_m=surface_width)
    anchor = copy.deepcopy(context["holding"]["points_tool_m"])
    motors = runner.simulation.motor_calls.copy()
    algorithms.source_shift_m = .012
    context["visual_plan"] = {"standalone_place": "associate_inventory"}
    await runner.observe(context, checkpoint="standalone_place")
    assert runner.holding_state["verified"] is True
    np.testing.assert_array_equal(context["holding"]["points_tool_m"], anchor)
    algorithms.source_shift_m = .06
    context["visual_plan"] = {"standalone_place": "associate_inventory"}
    await runner.observe(context, checkpoint="standalone_place")
    assert runner.holding_state["verified"] is True
    np.testing.assert_array_equal(runner.holding_state["points_tool_m"], anchor)
    assert runner.state["facts"]["grasp_held"]
    assert runner.simulation.motor_calls == motors


@pytest.mark.parametrize("change", ["translation", "new_measured_point"])
async def test_place_plan_rejects_changed_attachment_geometry_before_any_motor(tmp_path, change):
    runner, _, context = await verified_visual_holding(tmp_path)
    await runner.execute("plan_place", context, Choices("place"))
    reference = copy.deepcopy(context["place_plan"]["holding_reference"])
    motors = runner.simulation.motor_calls.copy()
    if change == "translation":
        context["holding"]["points_tool_m"] = (np.asarray(reference) + [.01, 0., 0.]).tolist()
    else:
        context["holding"]["points_tool_m"].append([0., 0., -.1])
    with pytest.raises(ValueError, match="holding_geometry_changed_since_place_plan"):
        await runner.execute("execute_place", context, Choices("place"))
    assert context["place_plan"]["holding_reference"] == reference
    assert runner.simulation.motor_calls == motors
    assert not runner.state["facts"].get("released")


async def test_partial_surface_binding_never_synthesizes_unseen_shape_corners(tmp_path):
    runner, _, context = await verified_visual_holding(tmp_path, surface_half_width_m=.004)
    points = np.asarray(context["holding"]["points_tool_m"])
    assert np.ptp(points[:, 0]) <= .00800001
    assert np.ptp(points[:, 1]) <= .00800001
    assert "estimated_bounds_tool_m" not in context["holding"]
    assert "shape_prior_provenance" not in context["holding"]
    await runner.execute("plan_place", context, Choices("place"))
    np.testing.assert_array_equal(context["place_plan"]["holding_reference"], points)


async def test_place_execution_rejects_new_checkpoint_before_any_motor(tmp_path):
    runner, algorithms, context = await verified_visual_holding(tmp_path)
    await runner.execute("plan_place", context, Choices("place"))
    motors = runner.simulation.motor_calls.copy()
    algorithms.destination_shift_m = .2
    context["visual_plan"] = {"standalone_place": "associate_inventory"}
    await runner.observe(context, checkpoint="standalone_place")
    with pytest.raises(ValueError, match="scene_observation_stale"):
        await runner.execute("execute_place", context, Choices("place"))
    assert runner.simulation.motor_calls == motors
    assert not runner.state["facts"].get("released")
    assert runner.holding_state["verified"] is True


@pytest.mark.parametrize("lost_evidence", ["open_gripper"])
async def test_standalone_place_rechecks_held_feedback_before_planning_or_movement(tmp_path, lost_evidence):
    from test_runner import wait

    class MissingSurface(PlaceVisualAlgorithms):
        missing = False
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            if self.missing:
                observation["objects"][0]["points_world_m"] = np.zeros((0, 3))
            return observation
    robot = RecordingSensorRobot()
    algorithms = MissingSurface()
    choices = Choices("grasp")
    runner = Runner(robot, algorithms, lambda record: choices, tmp_path)
    await runner.start("抓取红色方块")
    await wait(runner)
    assert runner.state["status"] == "succeeded", runner.state["error"]
    grasp_motors = robot.motor_calls.copy()
    if lost_evidence == "open_gripper":
        robot.closed = False
    elif lost_evidence == "surface_shift":
        algorithms.source_shift_m = .06
    else:
        algorithms.missing = True
    choices.task = "place"
    await runner.start("放到蓝色盒子里")
    await wait(runner)
    assert runner.state["status"] == "needs_attention", runner.state
    assert algorithms.place_calls == 0
    assert robot.motor_calls == grasp_motors
    assert not runner.state["facts"].get("place_execution_completed")


@pytest.mark.parametrize("status", ["IK_FAIL", "TRAJOPT_FAIL", "COLLISION", "LINEAR_CONSTRAINT_FAIL"])
def test_native_batch_candidate_failure_tries_next_scored_candidate(status):
    from test_planning_batch import batch_response, evaluated

    algorithms, observation, scene, evaluations, _ = evaluated(2, scores=[.9, .8])
    evaluations["A"]["destination_id"] = "blue-bin"
    def request(path, body):
        response = batch_response(body, successful=[1])
        response["items"][0]["status"] = status
        return response
    algorithms.request = request
    result = algorithms.plan_grasp(observation, "red", "A", evaluations, scene)
    assert result["candidate_index"] == 101
    assert result["prior_failures"][0]["status"] == status


@pytest.mark.parametrize("code", ["CUROBO_GRASP_FAILED", "CUROBO_TRAJOPT_FAILED", "CUROBO_INVALID_RESULT",
                                "CUROBO_RESOURCE_EXHAUSTED", "algorithm_service_unavailable"])
def test_native_service_internal_or_transport_failure_remains_fatal(code):
    from test_planning_batch import evaluated

    from jev_mujoco.algorithms import AlgorithmError

    algorithms, observation, scene, evaluations, _ = evaluated(2, scores=[.9, .8])
    evaluations["A"]["destination_id"] = "blue-bin"
    calls = []
    def request(path, body):
        calls.append(path)
        raise AlgorithmError("unexpected SDK failure", code=code, details={"arm": "A"})
    algorithms.request = request
    with pytest.raises(AlgorithmError, match="unexpected SDK failure"):
        algorithms.plan_grasp(observation, "red", "A", evaluations, scene)
    assert calls == ["/tools/plan_grasp_batch"]


async def test_excessive_live_joint_drift_stops_before_the_first_trajectory(tmp_path, monkeypatch):
    runner, _, context = await prepare_runner(tmp_path)
    await runner.verify_scene(context)
    await runner.execute("plan_grasp", context, Decisions())
    runner.state.update(status="running", current_step={"skill": "execute_grasp"})
    runner.simulation.state["arms"]["A"]["joints_rad"][0] = .04
    motors = []
    original = runner.simulation.execute_trajectory

    def execute_trajectory(*args):
        motors.append(args)
        return original(*args)

    monkeypatch.setattr(runner.simulation, "execute_trajectory", execute_trajectory)
    with pytest.raises(ValueError, match="start|command"):
        await runner.execute("execute_grasp", context, Decisions())
    assert motors == []


def test_collision_proof_cannot_omit_the_live_active_joint_feedback():
    import hashlib
    import json

    from test_rgbd_algorithms import observed_scene

    from jev_mujoco.algorithms import AlgorithmError
    from jev_mujoco.algorithms import Algorithms as RealAlgorithms

    observation = observed_scene()
    observation["arms"]["A"]["joints_rad"] = [.005] * 7
    scene = {"target_id": "seen-red", "generation": 1, "observation_id": "capture-4", "scene_token": "map-1"}

    def request(path, payload):
        legacy_payload = {key: value for key, value in payload.items() if key != "current_joints_rad"}
        digest = hashlib.sha256(json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        return {"scene_token": "map-1", "binding_sha256": digest, "collision_free": True, "attached": False}

    algorithms = RealAlgorithms(curobo_request=request)
    with pytest.raises(AlgorithmError, match="proof differs"):
        algorithms.validate_execution(observation, "A", scene, "retreat",
                                      {"positions_rad": [[.005] * 7, [0.] * 7], "dt": .02})
