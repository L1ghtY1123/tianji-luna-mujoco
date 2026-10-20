"""Attachment retains the pregrasp visible surface across verification captures."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_mujoco.algorithms import Algorithms
from jev_mujoco.carrying import CarryingBinding
from jev_mujoco.placement import carrying_binding, observed_object
from jev_mujoco.planning_settings import PlanningSettings
from jev_mujoco.runner import Runner


def fixture(tmp_path, *, closed_fraction=.6, shift=.012, stationary=False, duplicate=False):
    before_points = np.array([[.42, y, z] for y in np.linspace(.13, .17, 5)
                              for z in np.linspace(.61, .65, 5)])
    current_points = before_points.copy() if stationary else before_points + [shift, 0., .1]
    if duplicate:
        current_points = np.vstack((current_points, current_points[-1]))
    grasp = np.eye(4)
    grasp[:3, :3] = Rotation.from_euler("xyz", [20, -10, 35], degrees=True).as_matrix()
    grasp[:3, 3] = [.4, .15, .8]
    tcp = grasp.copy()
    tcp[2, 3] += .1
    feedback = {"joints_rad": [0.] * 7, "tcp": tcp.tolist(),
                "gripper_closed_fraction": closed_fraction, "gripper_position_rad": .8*closed_fraction,
                "gripper_limits_rad": [0., .8], "gripper_closed_command": True,
                "gripper_position_samples": [[t,.8*closed_fraction] for t in (0.,.05,.1,.15,.2,.25)]}
    source = {"id": "source", "kind": "cube", "color": "red",
              "position": before_points.mean(0).tolist(), "points_world_m": before_points, "uncertainty_m": .005}
    before = {"observation_id": "before-closure", "generation": 1, "source": "rgbd", "views": {},
              "arms": {"A": feedback | {"tcp": grasp.tolist()}, "B": feedback}, "objects": [source],
              "table": {"surface_height_m": .6}}
    current = {**before, "observation_id": "post-grasp", "arms": {"A": feedback, "B": feedback},
               "objects": [source | {"points_world_m": current_points, "position": current_points.mean(0).tolist()}]}
    robot = SimpleNamespace(robot_snapshot=lambda: {"generation": 1, "time": 2., "arms": current["arms"]})
    prepared, checkpoints, scene_calls = [], [], []
    scene = {"observation_id": "before-closure", "generation": 1, "scene_token": "old-map", "target_id": "source"}

    def request(path, payload):
        assert path == "/carrying/prepare"
        prepared.append(CarryingBinding.from_mapping(payload["carrying_binding"], scene=scene, arm="A"))
        return {"attached": True, "scene_token": scene["scene_token"],
                "attachment_spheres_tool_m": [[0., 0., -.15, .01]]}

    runner = Runner(robot, Algorithms(curobo_request=request), None, tmp_path)
    runner.state.update(generation=1, arm="A", target_id="source", status="running", task="grasp")
    holding = carrying_binding(before, "source", "A", "old-map", owner_id="owned-session")
    holding.update(state="provisional", verified=False, attached=False, observation=copy.deepcopy(before),
                   voxel_size_m=.007, margin_m=.002, sphere_budget=12,
                   grasp_support_reference={"T_world_tcp_at_grasp": grasp.tolist(), "table_height_m": .6,
                                            "source": "rgbd_table_and_joint_feedback"})
    runner.holding_state = copy.deepcopy(holding)
    context = {"scene": scene, "scene_observation": copy.deepcopy(before), "holding": holding, "observation": before, "points_before_lift": before_points.copy(),
               "tcp_before_lift": grasp.tolist(), "planning_settings": PlanningSettings().model_dump(), "post_grasp_capture": current}

    async def observe(context, *, checkpoint, capture=None):
        checkpoints.append(checkpoint)
        context["observation"] = current
        return current

    async def build_scene(context, exclusion_state):
        scene_calls.append(exclusion_state)
        context["scene"] = scene
        return scene

    runner.observe, runner.build_scene = observe, build_scene
    return runner, context, current, prepared, checkpoints, scene_calls


async def test_small_closure_shift_does_not_replace_pregrasp_geometry(tmp_path):
    runner, context, _current, prepared, checkpoints, scene_calls = fixture(tmp_path)
    original = copy.deepcopy(context["holding"])
    result = await runner.verify_grasp(context)
    assert result["success"]
    assert checkpoints == [] and scene_calls == [] and prepared == []
    assert not context['holding']['attached']
    await runner.prepare_attachment(context)
    assert len(prepared) == 1
    np.testing.assert_array_equal(prepared[0].points_tool_m, original["points_tool_m"])
    assert prepared[0].measurement_provenance == original["measurement_provenance"]
    assert context["holding"]["attached"] and runner.holding_state["verified"]


async def test_postgrasp_sampling_does_not_change_attachment_point_count(tmp_path):
    runner, context, _current, prepared, _, _ = fixture(tmp_path, duplicate=True)
    await runner.verify_grasp(context)
    await runner.prepare_attachment(context)
    assert len(prepared[0].points_tool_m) == 25


async def test_grasp_reference_records_actual_closure_tcp_and_feedback_source(tmp_path):
    from jev_mujoco.geometry import TOOL_PROFILE

    runner, context, current, _, _, _ = fixture(tmp_path)
    expected = copy.deepcopy(current['arms']['A']['tcp'])
    await runner.capture_holding_reference(context)
    reference = context['holding']['grasp_support_reference']
    np.testing.assert_array_equal(reference['T_world_tcp_at_grasp'], expected)
    assert reference['reference_phase'] == 'after_close_before_lift'
    assert reference['arm'] == 'A'
    assert reference['tool_profile_revision'] == TOOL_PROFILE['revision']
    assert reference['feedback_id'] == 'feedback-1-2.0'
    assert reference['feedback_time'] == 2.


@pytest.mark.parametrize("options", [{"closed_fraction": 1.}])
async def test_failed_hold_verification_never_replaces_reference_or_prepares_attachment(tmp_path, options):
    runner, context, _, prepared, _, scene_calls = fixture(tmp_path, **options)
    context["observation_checkpoint"] = "post_grasp_semantic"
    reference = copy.deepcopy(context["holding"]["points_tool_m"])
    result = await runner.verify_grasp(context)
    assert not result["success"]
    assert prepared == [] and scene_calls == []
    assert context["holding"]["points_tool_m"] == reference
    assert not runner.holding_state["verified"] and not runner.holding_state["attached"]


@pytest.mark.parametrize('independent_motion', [False, True], ids=['rigid-held-motion', 'independent-object-motion'])
async def test_future_standalone_observation_keeps_current_only_binding_frozen(tmp_path, independent_motion):
    runner, context, current, _, _, _ = fixture(tmp_path)
    await runner.verify_grasp(context)
    frozen = copy.deepcopy(context["holding"])
    future = copy.deepcopy(current)
    future["observation_id"] = "standalone-place"
    future["arms"]["A"]["tcp"][0][3] += .05
    tcp = np.asarray(future['arms']['A']['tcp'])
    points = np.asarray(frozen['points_tool_m']) @ tcp[:3, :3].T + tcp[:3, 3]
    if independent_motion:
        points += [.05, 0., 0.]
    source = future["objects"][0]
    source.update(points_world_m=points)
    source["position"] = source["points_world_m"].mean(0).tolist()
    future["objects"].append({"id": "bin", "kind": "bin", "color": "blue",
        "position": [.7, .2, .6], "points_world_m": np.array([[.7, .2, .6]]), "coverage": {"camera_ids": ["1"]}})
    future["views"] = {"1": {"frame_id": "standalone-place", "camera_id": "1",
        "intrinsics": {"fx": 1., "fy": 1., "cx": 0., "cy": 0.}, "T_world_camera": np.eye(4),
        "depth_m": np.ones((1, 1)), "valid_depth": np.ones((1, 1), bool),
        "instance_masks": {"bin": np.ones((1, 1), bool)}}}

    future['visual_inventory'] = {'source': 'openai_decisions', 'membership': {}, 'progress': {},
                                  'evidence': {'observation_id': future['observation_id']}}
    class Vision:
        def observe(self, *args, **kwargs):
            return copy.deepcopy(future)

    runner.algorithms.perception = Vision()
    runner.simulation.capture_rgbd = lambda: future
    runner.state.update(task="place", destination_attributes={"kind": "bin", "color": "blue"})
    runner.observe = Runner.observe.__get__(runner)
    context['visual_plan'] = {'standalone_place': 'initialize_inventory'}
    result = await runner.observe(context, checkpoint="standalone_place")
    np.testing.assert_array_equal(observed_object(result, 'source')['points_world_m'], points)
    assert result['visual_inventory']['source'] == 'openai_decisions'
    np.testing.assert_array_equal(context["holding"]["points_tool_m"], frozen["points_tool_m"])
    assert context["holding"]["measurement_provenance"] == frozen["measurement_provenance"]
    assert context["holding"]["last_position_evidence"]["status"] == "held"
    fresh_scene = {"scene_token": "fresh-scene", "generation": 1, "observation_id": result["observation_id"]}
    binding = runner.algorithms._held_binding(result, "source", "A", fresh_scene, context["holding"])
    np.testing.assert_array_equal(binding["points_tool_m"], frozen["points_tool_m"])
    assert binding["measurement_provenance"]["reference_observation_id"] == "before-closure"


async def test_rebuilt_held_scene_needs_no_current_source_object(tmp_path):
    runner, context, current, _, _, _ = fixture(tmp_path)
    await runner.verify_grasp(context)
    await runner.prepare_attachment(context)
    current['objects'] = []
    scene = {**context['scene'], 'scene_token': 'new-map', 'observation_id': current['observation_id']}
    bound = runner.algorithms._held_binding(current, 'source', 'A', scene, context['holding'])
    parsed = CarryingBinding.from_mapping(bound, scene=scene, arm='A')
    assert parsed.measurement_provenance == context['holding']['measurement_provenance']
    assert bound['points_tool_m'] == context['holding']['points_tool_m']
    assert parsed.observation_id == current['observation_id']
    assert parsed.measurement_provenance['reference_observation_id'] == 'before-closure'

    current['objects'] = [{'id': 'bin', 'kind': 'bin', 'placement_reference': {
        'position_world_m': [.6, .2, .61], 'floor_z_m': .61}}]
    def planning_request(path, body):
        assert path == '/tools/solve_ik_batch'
        assert body['carrying_binding']['points_tool_m'] == bound['points_tool_m']
        raise RuntimeError('planner_reached_without_current_source')
    runner.algorithms.request = planning_request
    with pytest.raises(RuntimeError, match='planner_reached_without_current_source'):
        runner.algorithms.plan_place(current, 'source', 'A', scene, context['holding'], 'bin')
