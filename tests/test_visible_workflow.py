import copy

import numpy as np
from test_runner import Algorithms as FakeAlgorithms
from test_runner import Decisions, Sim, wait

from jev_mujoco.algorithms import Algorithms
from jev_mujoco.planning_settings import PlanningSettings
from jev_mujoco.runner import Runner


class VisibleAlgorithms(FakeAlgorithms):
    def __init__(self):
        self.observations = []
        self.prepared = []

    def observe(self, capture, *, instruction="", checkpoint=None, source_hint=None):
        observation = super().observe(capture, instruction=instruction)
        for obj in observation["objects"]:
            for key in ("size", "T_world_object", "quaternion", "points_object_m"):
                obj.pop(key, None)
            obj["visible_centroid_world"] = np.mean(obj["points_world_m"], axis=0).tolist()
            obj["position"] = obj["visible_centroid_world"]
            obj["coverage"]["camera_ids"] = ["3", "1", "2"]
        observation["table"]["surface_height_m"] = 0.
        self.observations.append(capture["observation_id"])
        return observation

    def plan_carried_lift(self, *args, **kwargs):
        raise AssertionError("grasp must execute its original lift without attachment")

    def prepare_carrying(self, observation, arm, scene, holding, *, planning_settings=None):
        assert observation["arms"][arm]["tcp"][2][3] == .1
        self.prepared.append(copy.deepcopy(holding))
        return {"attached": True, "scene_token": scene["scene_token"]}


async def test_visible_grasp_keeps_binding_without_preparing_attachment(tmp_path):
    algorithms = VisibleAlgorithms()
    simulation = Sim()
    settings = PlanningSettings(attachment={"voxel_size_m": .007, "margin_m": .002, "sphere_budget": 12})
    runner = Runner(simulation, algorithms, lambda record: Decisions(), tmp_path, planning_settings=settings)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state["status"] == "succeeded", runner.state.get("error")
    assert runner.holding_state["attached"] is False
    assert runner.holding_state["verified"] is True
    assert algorithms.prepared == []
    for name, value in settings.attachment.model_dump().items():
        assert runner.holding_state[name] == value
    assert len(algorithms.observations) == 2  # Initial discovery and atomic pregrasp preparation.
    assert "estimated_bounds_tool_m" not in runner.holding_state
    assert "T_tcp_object" not in runner.holding_state
    assert runner.holding_state["grasp_support_reference"]["table_height_m"] == 0.


async def test_target_geometry_from_registered_wrist_camera_is_sufficient(tmp_path):
    class WristOnly(VisibleAlgorithms):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            observation["objects"][0]["coverage"]["camera_ids"] = ["1", "2"]
            return observation

    simulation = Sim()
    runner = Runner(simulation, WristOnly(), lambda record: Decisions(), tmp_path)
    await runner.start("抓取黑色物体")
    await wait(runner)
    assert runner.state['status'] == 'succeeded', runner.state.get('error')
    assert simulation.state["arms"]["A"]["joints_rad"] == [1.] * 7


def test_grasp_cloud_uses_measured_points_without_object_pose():
    from jev_mujoco.cameras import deproject_view

    class Graspgen:
        def generate(self, request):
            np.testing.assert_allclose(request["object_points_camera_m"].mean(axis=0), 0., atol=1e-6)
            return {"model": "test", "candidates": [{"index": 0, "score": 1., "T_camera_grasp": np.eye(4)}]}

    view = {"depth_m": np.ones((5, 5)), "valid_depth": np.ones((5, 5), bool),
            "intrinsics": {"fx": 100., "fy": 100., "cx": 2., "cy": 2., "width": 5, "height": 5},
            "T_world_camera": np.eye(4)}
    points = deproject_view(view)
    source = {"id": "source", "points_world_m": points, "visible_centroid_world": points.mean(axis=0)}
    observation = {"source": "rgbd", "generation": 1, "observation_id": "test", "objects": [source],
                   "views": {"3": view}}
    result = Algorithms(graspgen_client=Graspgen()).generate_grasps(observation, "source")
    assert result["frame"].startswith("cloud:")
    np.testing.assert_allclose(np.asarray(result["T_world_cloud"])[:3, 3], points.mean(axis=0))


def test_grasp_validation_keeps_open_model_without_support_departure():
    import hashlib
    import json

    requests = []
    def request(path, payload):
        requests.append(payload)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {"scene_token": "s", "binding_sha256": digest, "collision_free": True, "attached": False}

    observation = {"generation": 1, "observation_id": "o", "arms": Sim().robot_snapshot()["arms"]}
    scene = {"generation": 1, "observation_id": "o", "target_id": "source", "scene_token": "s"}
    Algorithms(curobo_request=request).validate_execution(observation, "A", scene, "grasp",
        {"positions_rad": [[0.] * 7]}, planning_settings=PlanningSettings().model_dump())
    assert requests[0]["gripper_geometry_mode"] == "fixed_open"
    assert "support_contact_policy" not in requests[0]


def test_prepare_carrying_validates_union_point_provenance():
    from jev_mujoco.carrying import CarryingBinding

    simulation = Sim()
    observation = VisibleAlgorithms().observe(simulation.capture_rgbd())
    target = observation["objects"][0]["id"]
    scene = {"generation": observation["generation"], "observation_id": observation["observation_id"],
             "scene_token": "surface-scene", "target_id": target}
    points = np.asarray(observation["objects"][0]["points_world_m"])
    held_points = np.vstack((points, points + [0., 0., .001])).tolist()
    holding = {"instance_id": target, "owner_id": "owner", "points_tool_m": held_points,
               "voxel_size_m": .007, "margin_m": .002, "sphere_budget": 12,
               "measurement_provenance": {"source": "registered_rgbd_visible_surface", "point_count": len(held_points),
                                          "observation_ids": ["pre-grasp", observation["observation_id"]]}}
    def request(path, payload):
        binding = CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm="A")
        assert binding.measurement_provenance["observation_ids"][0] == "pre-grasp"
        assert (binding.voxel_size_m, binding.margin_m, binding.sphere_budget) == (.007, .002, 12)
        return {"attached": True, "scene_token": scene["scene_token"]}

    Algorithms(curobo_request=request).prepare_carrying(observation, "A", scene, holding)


def test_held_scene_removes_predicted_surface_without_target_segmentation():
    from jev_mujoco.cameras import deproject_view
    from jev_mujoco.perception_mapping import decode_mapping_payload
    from jev_mujoco.simulation import Simulation
    simulation = Simulation()
    simulation.start()
    try:
        capture = simulation.capture_rgbd()
    finally:
        simulation.close()
    for view in capture['views'].values(): view['instance_masks'] = {}
    points = deproject_view(capture['views']['3'])
    capture.update(objects=[], held_surface_world_m=points[::30].tolist())
    def request(path, body):
        decoded, identity = decode_mapping_payload(body)
        assert identity['target_id'] == 'held-target'
        assert decoded['mapping_mode'] == 'background'
        assert body['selected_surface_world_m'] == []
        assert not any(view['instance_masks'] for view in decoded['views'].values())
        return identity
    result = Algorithms(curobo_request=request).build_scene(capture, 'held-target', exclusion_state='held')
    assert result['target_id'] == 'held-target'
    assert capture['objects'] == []
