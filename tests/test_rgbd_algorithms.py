import numpy as np
import pytest

from jev_mujoco.algorithms import AlgorithmError, Algorithms


def observed_scene():
    points = np.array([[x, y, .64] for x in np.linspace(.48, .52, 5)
                       for y in np.linspace(.08, .12, 5)], dtype=np.float32)
    view = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8), "depth_m": np.ones((8, 8), dtype=np.float32),
            "valid_depth": np.ones((8, 8), dtype=bool),
            "intrinsics": {"fx": 10., "fy": 10., "cx": 4., "cy": 4., "width": 8, "height": 8},
            "T_world_camera": np.eye(4).tolist(), "robot_joints_deg": {"A": [0.] * 7, "B": [0.] * 7},
            "instance_masks": {"seen-red": np.ones((8, 8), dtype=bool)}}
    return {"source": "rgbd", "observation_id": "capture-4", "generation": 1, "time": 2.,
            "objects": [{"id": "seen-red", "kind": "cube", "color": "red", "points_world_m": points,
                         "position": points.mean(axis=0).tolist(),
                         "visible_centroid_world": points.mean(axis=0).tolist(),
                         "coverage": {"camera_ids": ["1", "2", "3"], "observed_surface_only": True}}],
            "table": {"surface_height_m": .6},
            "views": {camera: {**view, "camera_id": camera} for camera in ("1", "2", "3")},
            "arms": {arm: {"joints_rad": [0.] * 7, "tcp": np.eye(4).tolist(),
                            "gripper_position_rad": .3} for arm in ("A", "B")}}


def test_build_scene_uploads_rgbd_and_masks_not_primitive_truth():
    calls = []

    def request(path, body):
        calls.append((path, body))
        return {"ok": True, "scene_token": "measured-map"}

    result = Algorithms(curobo_request=request).build_scene(observed_scene(), "seen-red")
    path, body = calls[0]
    assert path == "/perception/build"
    assert "objects" not in body and "table" not in body
    assert body["observation_set_id"] == "capture-4"
    assert set(body["views"]) == {"1", "2", "3"}
    assert body["views"]["3"]["depth_left_wire"]["encoding"] == "zlib+base64"
    assert result["observation_id"] == "capture-4"


@pytest.mark.parametrize('head_only', [True, False])
def test_observation_map_keeps_all_requested_camera_instances_without_target_exclusion(head_only):
    from jev_mujoco.perception_mapping import decode_mapping_payload

    calls = []
    def request(path, body):
        calls.append(body)
        return {"ok": True, "scene_token": "measured-map"}
    algorithms = Algorithms(curobo_request=request)
    observation = observed_scene()
    algorithms.build_scene(observation, 'seen-red')
    selected, _ = decode_mapping_payload(calls[-1])
    result = algorithms.build_scene(observation, 'seen-red', exclusion_state='observation', head_only=head_only)
    decoded, identity = decode_mapping_payload(calls[-1])
    assert decoded.get('mapping_mode', 'semantic') == 'semantic'
    assert identity['disabled_obstacle_names'] == ()
    assert not result['target_id']
    assert set(decoded['views']) == ({'3'} if head_only else {'1', '2', '3'})
    for camera, view in decoded['views'].items():
        assert set(view['instance_masks']) == {'seen-red'}
        np.testing.assert_array_equal(view['depth_left_m'], selected['views'][camera]['depth_left_m'])
        np.testing.assert_array_equal(view['valid_left_mask'], selected['views'][camera]['valid_left_mask'])
    assert observation['views']['3']['instance_masks']['seen-red'].any()


def test_graspgen_receives_the_measured_partial_cloud_without_surface_completion():
    class Client:
        def __init__(self):
            self.points = []

        def generate(self, payload):
            self.points.append(np.asarray(payload["object_points_camera_m"]).copy())
            return {"ok": True, "model": "real-format", "candidates": [
                {"index": 0, "score": .8, "T_camera_grasp": np.eye(4).tolist()}]}

    client = Client()
    observation = observed_scene()
    result = Algorithms(graspgen_client=client).generate_grasps(observation, "seen-red")
    assert result["candidate_count"] == 2
    assert result["cloud"]["source"] == "rgbd_visible_surface"
    assert len(client.points[0]) == 25
    points = observation["objects"][0]["points_world_m"]
    np.testing.assert_allclose(client.points[0], points.astype(float) - points.astype(float).mean(axis=0), atol=1e-7)
    np.testing.assert_array_equal(client.points[0], client.points[1])


def test_place_uses_attached_source_native_phases_and_place_specific_parameters():
    observation = observed_scene()
    with_bin(observation)
    scene = {"target_id": "seen-red", "generation": 1, "observation_id": "capture-4", "scene_token": "map-1"}
    calls = []

    def request(path, body):
        calls.append((path, body))
        if path == "/carrying/validate":
            import hashlib
            import json
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                              allow_nan=False).encode()).hexdigest()
            return {"ok": True, "collision_free": True, "scene_token": body["scene_token"],
                    "binding_sha256": digest, "attached": body.get("carrying_binding") is not None}
        phase = {"positions_rad": [[0.] * 7, [.01] * 7], "joint_names": [f"Joint{i}_L" for i in range(1, 8)],
                 "velocities_rad_s": [[0.] * 7] * 2, "accelerations_rad_s2": [[0.] * 7] * 2,
                 "interpolation_dt_s": .02}
        if path == "/tools/plan_to_pose":
            assert body["gripper_joint_positions_rad"] == {"A": 0., "B": .3}
            assert "carrying_binding" not in body
            return {"scene_token": "map-1", "attached": False, "trajectory": phase}
        if path == "/tools/solve_ik_batch":
            return {"scene_token": "map-1", "batch_size": len(body["goal_matrices_world"]),
                    "items": [{"candidate_index": index, "success": True}
                              for index in range(len(body["goal_matrices_world"]))],
                    "attached": True, "collision_evidence": {"optimizer_attachment": True}}
        return {"ok": True, "scene_token": "map-1",
                "collision_evidence": {"optimizer_attachment": True},
                "items": [{"index": index, "success": True, "approach_success": True, "place_success": True,
                           "phases": {name: phase for name in ("approach", "place")}} for index in range(16)]}

    holding = {"arm": "A", "owner_id": "owner-1", "verified": True, "attached": True,
               "attachment_evidence": {"attachment_spheres_tool_m": [[.5, .1, .64, .02]]},
               "points_tool_m": observation["objects"][0]["points_world_m"].tolist(),
               "grasp_support_reference": {"T_world_tcp_at_grasp": np.eye(4).tolist(),
                                           "table_height_m": .6,
                                           "source": "rgbd_table_and_joint_feedback"}}
    result = Algorithms(curobo_request=request).plan_place(observation, "seen-red", "A", scene,
                                                          holding, "blue-bin")
    assert calls[0][0] == "/tools/solve_ik_batch"
    path, payload = calls[1]
    assert path == "/tools/plan_place_batch"
    assert len(payload["goal_matrices_world"]) == 16
    assert payload["approach_offset_m"] == .10 and payload["approach_in_tool_frame"] is False
    assert payload["lift_offset_m"] == .10 and payload["lift_in_tool_frame"] is False
    assert payload["carrying_binding"]["owner_id"] == "owner-1"
    assert payload["gripper_joint_positions_rad"] == {"A": .3, "B": .3}
    assert set(result["phases"]) == {"approach", "place", "retreat"}
    assert result["retreat_requires_replan"] is False


def test_actual_command_check_rejects_a_proof_for_another_request():
    observation = observed_scene()
    scene = {"target_id": "seen-red", "generation": 1, "observation_id": "capture-4", "scene_token": "map-1"}
    algorithms = Algorithms(curobo_request=lambda path, body: {
        "ok": True, "collision_free": True, "scene_token": "map-1", "binding_sha256": "wrong"})
    with pytest.raises(AlgorithmError, match="proof"):
        algorithms.validate_execution(observation, "A", scene, "lift", {"positions_rad": [[0.] * 7]},
                                      {"owner_id": "owner-1"})


def with_bin(observation):
    points = np.array([[x, y, .6] for x in np.linspace(.52, .68, 5)
                       for y in np.linspace(-.26, -.14, 5)])
    observation["objects"].append({"id": "blue-bin", "kind": "bin", "points_world_m": points,
        "position": points.mean(axis=0).tolist(),
        "placement_reference": {"position_world_m": [.6, -.2, .6], "floor_z_m": .6,
                                "source": "rgbd_visible_floor"}})
    return observation


def test_arm_evaluation_defers_destination_checks_until_actual_holding():
    observation = with_bin(observed_scene())
    scene = {"target_id": "seen-red", "generation": 1, "observation_id": "capture-4", "scene_token": "map-1"}
    calls = []

    def request(path, body):
        calls.append(body)
        return {"scene_token": "map-1", "batch_size": 2, "items": [
            {"candidate_index": i, "success": True} for i in range(2)]}

    grasps = {"candidates": [{"index": i + 10, "score": .9 - i * .1,
                               "T_object_grasp": np.eye(4).tolist()} for i in range(2)]}
    result = Algorithms(curobo_request=request).evaluate_arms(
        observation, "seen-red", grasps, scene, destination_id="blue-bin")
    assert len(calls) == 2
    assert [candidate["candidate_index"] for candidate in result["A"]["valid_candidates"]] == [10, 11]
    assert result["A"]["feasibility_scope"] == "grasp_ik_only"
    assert result["A"]["place_planning_stage"] == "after_grasp_attachment"
    assert "destination_ik" not in result["A"]["valid_candidates"][0]


def test_grasp_plan_defers_attachment_and_formal_place_until_after_grasp(monkeypatch):
    from test_planning_batch import batch_response, evaluated

    algorithms, observation, scene, evaluations, _ = evaluated(2, scores=[.9, .8])
    calls = []
    def request(path, body):
        calls.append((path, body))
        return batch_response(body)
    algorithms.request = request
    evaluations["A"]["destination_id"] = "blue-bin"
    result = algorithms.plan_grasp(observation, "red", "A", evaluations, scene,
                                             destination_id="blue-bin")
    assert result["candidate_index"] == 100
    assert [path for path, _ in calls] == ["/tools/plan_grasp_batch"]
    assert all("carrying_binding" not in body for _, body in calls)
    assert result["place_preflight"]["scope"] == "deferred_until_attachment"


def test_attachment_uses_actual_postlift_tcp_feedback_and_measured_surface():
    from jev_mujoco.placement import carrying_binding

    observation = observed_scene()
    actual = np.eye(4)
    actual[:3, 3] = [.3, .2, .9]
    observation["arms"]["A"]["tcp"] = actual.tolist()
    scene = {"scene_token": "map-1", "target_id": "seen-red", "generation": 1, "observation_id": "capture-4"}
    holding = carrying_binding(observation, "seen-red", "A", "map-1", owner_id="owner-1")
    calls = []
    def request(path, body):
        calls.append((path, body))
        return {"attached": True, "scene_token": "map-1"}
    algorithms = Algorithms(curobo_request=request)
    algorithms.prepare_carrying(observation, "A", scene, holding)
    path, payload = calls[0]
    assert path == "/carrying/prepare"
    np.testing.assert_allclose(payload["goal_position_m"], actual[:3, 3])
    np.testing.assert_allclose(payload["carrying_binding"]["points_tool_m"],
                               observation["objects"][0]["points_world_m"] - actual[:3, 3])
    assert "estimated_bounds_tool_m" not in payload["carrying_binding"]
