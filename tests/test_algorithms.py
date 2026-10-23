import numpy as np
import pytest

from luna_mujoco.algorithms import AlgorithmError, Algorithms
from luna_mujoco.geometry import GeometryCache, grasp_targets, primitive_sdf


def snapshot():
    from test_rgbd_algorithms import observed_scene
    state = observed_scene()
    source = state["objects"][0]
    points = np.array([[x, y, .655] for x in np.linspace(.375, .425, 5)
                       for y in np.linspace(.175, .225, 5)], dtype=np.float32)
    source.update(id="red", position=points.mean(axis=0).tolist(),
                  visible_centroid_world=points.mean(axis=0).tolist(), points_world_m=points)
    for view in state["views"].values():
        view["instance_masks"] = {"red": view["instance_masks"]["seen-red"]}
    state.update(confirmed=True, table={"position":[.5,0,.58],"size":[.6,.9,.04], "surface_height_m": .6})
    return state


def test_cloud_cache_is_local_and_keeps_surface():
    cache = GeometryCache()
    # Offline primitive geometry utility, outside the RGBD control fixture.
    obj = {"id": "primitive", "kind": "cube", "size": [.05] * 3,
           "position": [.4, .2, .63], "quaternion": [1., 0., 0., 0.]}
    cloud = cache.object_cloud(obj)
    assert cloud.dtype == np.float32
    assert cache.object_cloud({**obj, "position": [8,9,10]}) is cloud
    np.testing.assert_allclose(primitive_sdf(cloud, "cube", obj["size"]), 0, atol=1e-7)


def test_primitive_signed_distance():
    points = np.array([[0,0,0],[.1,0,0],[0,0,.2]])
    for kind in ("cube", "sphere", "cylinder"):
        values = primitive_sdf(points, kind, [.1,.1,.1])
        np.testing.assert_allclose(values, [-.05,.05,.15])


def test_arm_orientation_and_tcp_offsets():
    world = np.eye(4)[None]
    a = grasp_targets(world, "A")
    b = grasp_targets(world, "B")
    np.testing.assert_allclose(a[0,:3,:3], np.diag([-1,-1,1]))
    np.testing.assert_allclose(b[0,:3,:3], np.eye(3))
    assert b[0,2,3] == pytest.approx(.010)


def test_evaluate_keeps_original_candidate_indices_and_arms():
    calls = []
    def request(path, body=None):
        calls.append((path,body))
        return {"ok":True,"scene_token":"test", "batch_size":1,
                "items":[{"candidate_index":0,"success":True}]}
    algorithms = Algorithms(curobo_request=request)
    grasps = {"candidates":[{"index":17,"score":.9,"T_object_grasp":np.eye(4).tolist()}]}
    result = algorithms.evaluate_arms(snapshot(), "red", grasps, {"scene_token":"test","target_id":"red","generation":1,"observation_id":"capture-4"})
    assert [call[1]["arm"] for call in calls] == ["A", "B"]
    assert result["A"]["valid_candidates"][0]["candidate_index"] == 17
    assert result["B"]["valid_candidates"][0]["candidate_index"] == 17
    assert calls[0][1]["start_joints_deg"] == [0]*7


def test_reject_stale_scene_and_unknown_target():
    algorithms = Algorithms(curobo_request=lambda *args: {})
    with pytest.raises(AlgorithmError, match="target"):
        algorithms.build_scene(snapshot(), "missing")
    with pytest.raises(AlgorithmError, match="generation"):
        algorithms.evaluate_arms(snapshot(), "red", {}, {"generation":0,"target_id":"red"})


def test_grasp_pool_combines_two_real_batches_and_preserves_source_identity():
    class GraspGen:
        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            batch = len(self.requests)
            return {"model": {"name": "test"}, "candidates": [
                {"index": index, "score": batch / 10 + index / 1000,
                 "T_camera_grasp": np.eye(4).tolist()} for index in range(100)]}

    client = GraspGen()
    result = Algorithms(graspgen_client=client).generate_grasps(snapshot(), "red")
    candidates = result["candidates"]
    assert len(candidates) == 200
    assert len(client.requests) == 2
    assert client.requests[0]["object_points_camera_m"] is client.requests[1]["object_points_camera_m"]
    assert [item["index"] for item in candidates] == list(range(200))
    assert {(item["source_batch"], item["source_index"]) for item in candidates} == {
        (batch, index) for batch in (1, 2) for index in range(100)}
    assert candidates[0]["source_batch"] == 2 and candidates[0]["source_index"] == 99
    assert result["candidate_count"] == 200


def test_grasp_pool_reports_actual_count_and_propagates_second_request_failure():
    class GraspGen:
        def __init__(self, fail):
            self.count = 0
            self.fail = fail

        def generate(self, request):
            self.count += 1
            if self.fail and self.count == 2:
                return {"ok": False, "error": "service unavailable"}
            return {"model": {}, "candidates": [
                {"index": 7, "score": .9, "T_camera_grasp": np.eye(4).tolist()}]}

    result = Algorithms(graspgen_client=GraspGen(False)).generate_grasps(snapshot(), "red")
    assert result["candidate_count"] == 2
    assert result["requested_candidate_count"] == 200
    with pytest.raises(AlgorithmError, match="service unavailable"):
        Algorithms(graspgen_client=GraspGen(True)).generate_grasps(snapshot(), "red")


def test_scene_layers_disable_only_target_for_entire_operation(monkeypatch):
    from unittest.mock import MagicMock

    from luna_mujoco import geometry_service
    from luna_mujoco.geometry import geometry_payload

    monkeypatch.setattr(geometry_service, "make_grid",
                        lambda name, kind, size, pose: (name, {"dims": size, "voxel_size": .01}))
    state = snapshot()
    # Legacy geometry utility is tested separately from online RGBD build_scene.
    state["objects"][0].update(size=[.05] * 3, quaternion=[1., 0., 0., 0.])
    state["objects"].append({**state["objects"][0], "id": "blue", "position": [.7,0,.63]})
    runtime = geometry_service.GeometryRuntime()
    result = runtime.build_geometry(geometry_payload(state, "red"))
    assert result["excluded_object_ids"] == ["red"]
    assert result["collision_objects"] == ["blue"]
    assert runtime.geometry_grids == ["geometry_table", "semantic_instance::red", "semantic_instance::blue"]
    calls = []
    class Backend:
        motion_planner = MagicMock()

        def update_voxel_scene(self, grids, model, *, disabled_obstacle_names):
            calls.append(disabled_obstacle_names)
    for phase in ("approach", "grasp", "lift"):
        runtime._refresh_scene_unlocked(arm="A", backend=Backend())
    assert calls == [("semantic_instance::red",)] * 3
