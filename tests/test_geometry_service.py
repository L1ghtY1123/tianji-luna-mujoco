from dataclasses import replace
from types import SimpleNamespace

import pytest

from jev_mujoco import geometry_service
from jev_mujoco.planning_settings import PlanningSettings


def test_profile_change_destroys_cached_solvers_but_time_scale_does_not():
    runtime = geometry_service.GeometryRuntime()
    calls = []
    backend = SimpleNamespace(destroy=lambda: calls.append("destroyed"))
    runtime._ik_batch_backends[("A", .01, 50)] = backend
    runtime.configure_planning(PlanningSettings(time_scale=4))
    assert not calls
    runtime.configure_planning(PlanningSettings(ik_num_seeds=24))
    assert calls == ["destroyed"]
    assert not runtime._ik_batch_backends


@pytest.mark.parametrize("arm", ["A", "B"])
def test_batch_peer_refresh_uses_fk_without_constructing_other_planner(monkeypatch, arm):
    runtime = geometry_service.GeometryRuntime()
    calls = []

    def reject_planner(_arm):
        raise AssertionError("peer FK must not construct a MotionPlanner")

    monkeypatch.setattr(runtime, "_backend_unlocked", reject_planner)
    backend = SimpleNamespace(update_peer_collision_model=lambda *values: calls.append(values))
    active = [10.] * 7
    other = [20.] * 7
    assert runtime._refresh_peer_collision_unlocked(
        arm=arm, backend=backend, active_joints_deg=active, other_joints_deg=other,
        cache_snapshot=False,
    ) is True
    assert calls == [(None, active, other)]
    runtime._refresh_peer_collision_unlocked(
        arm=arm, backend=backend, active_joints_deg=active, other_joints_deg=[30.] * 7,
    )
    assert calls[-1] == (None, active, [30.] * 7)


def test_scene_refresh_excludes_only_the_selected_target():
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene", "target_id": "target"}
    runtime.geometry_grids = []
    runtime.geometry_model = {"cuboid": {}}
    calls = []
    backend = SimpleNamespace(update_voxel_scene=lambda *args, **kwargs: calls.append((args, kwargs)))
    runtime._refresh_scene_unlocked(arm="A", backend=backend)
    assert calls == [(([], {"cuboid": {}}), {"disabled_obstacle_names": ("semantic_instance::target",)})]


@pytest.mark.parametrize("arm", ["A", "B", "AB"])
def test_scene_upload_preserves_integer_voxel_strides(arm):
    import numpy as np

    class Tensor(np.ndarray):
        """CPU stand-in for the public GPU metadata tensor."""
        def round_(self):
            np.rint(self, out=self)
            return self

    # Recorded Mapper dimensions: float32 division yields 119.999992 in Y.
    voxel_size = np.float32(.01)
    dimensions = np.array([1.099999975413084, 1.1999999731779099, 1.5999999642372131], dtype=np.float32)
    raw_counts = dimensions / voxel_size
    assert raw_counts.astype(int).tolist() == [110, 119, 160]
    params = np.zeros((1, 2, 4), dtype=np.float32).view(Tensor)
    features = np.arange(110 * 120 * 160, dtype=np.float32)
    voxels = SimpleNamespace(params=params)
    planner = SimpleNamespace(scene_collision_checker=SimpleNamespace(data=SimpleNamespace(voxels=voxels)))

    def load(*args, **kwargs):
        params[0, 0] = [*raw_counts, voxel_size]

    backend = SimpleNamespace(update_voxel_scene=load, motion_planner=planner)
    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene", "target_id": "target"}
    runtime.geometry_grids = [object()]
    runtime.geometry_model = {"voxel": {}}
    for _ in range(2):
        assert runtime._refresh_scene_unlocked(arm=arm, backend=backend) == "scene"
        nx, ny, nz = params[0, 0, :3].astype(int)
        # Match the pinned collision kernel's X-slowest / Z-fastest addressing.
        assert features[(49 * ny + 92) * nz + 68] == features.reshape(110, 120, 160)[49, 92, 68]
        assert (nx, ny, nz) == (110, 120, 160)
        assert params[0, 0, 3] == voxel_size
        assert not params[0, 1].any()


@pytest.mark.parametrize("arm", ["A", "B"])
def test_small_batches_reuse_one_backend_without_exposing_padding(monkeypatch, arm):
    from test_backend_builder import native_batch

    runtime = geometry_service.GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene", "target_id": "target"}
    runtime.geometry_grids = []
    runtime.geometry_model = {"cuboid": {}}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    runtime.effective_config = {arm: {"plan": {"ik_num_seeds": 16}}}
    created_capacities = []
    submitted_goals = []
    sample = native_batch()

    def create_backend(arm, *, batch_size, **kwargs):
        created_capacities.append(batch_size)
        runtime.effective_config[arm] = {"plan": {"ik_num_seeds": 16}}

        def plan(request):
            submitted_goals.append(request.goal_matrices_world)
            items = tuple(replace(sample.items[0], candidate_index=index)
                          for index in range(len(request.goal_matrices_world)))
            failed = replace(items[1], success=False, lift_success=False, lift=None, status="collision")
            items = (items[0], failed, *items[2:])
            return replace(sample, items=items, batch_size=batch_size, batch_capacity=batch_size)

        return SimpleNamespace(
            update_voxel_scene=lambda *args, **kwargs: None,
            update_peer_collision_model=lambda *args: None,
            plan_grasp_batch_native=plan,
        )

    monkeypatch.setattr(runtime, "_create_real_backend_unlocked", create_backend)
    for count in (1, 5, 9, 16, 17):
        payload = request_payload(PlanningSettings())
        payload["arm"] = arm
        goals = []
        for index in range(count):
            goal = [row[:] for row in payload["goal_matrices_world"][0]]
            goal[0][3] += index * .001
            goals.append(goal)
        payload["goal_matrices_world"] = goals
        result = geometry_service.run_operation(runtime, "/tools/plan_grasp_batch", payload)
        assert result["batch_size"] == count
        assert result["batch_capacity"] == (16 if count <= 16 else 50)
        assert [item["candidate_index"] for item in result["items"]] == list(range(count))
        assert [item["success"] for item in result["items"]] == [index != 1 for index in range(count)]
        assert len(submitted_goals[-1]) == result["batch_capacity"]
        assert [list(map(list, goal)) for goal in submitted_goals[-1][:count]] == goals
        assert all(goal == submitted_goals[-1][count - 1] for goal in submitted_goals[-1][count:])
    assert created_capacities == [16, 50]


def request_payload(settings):
    return {"arm": "A", "start_joints_deg": [0] * 7, "other_joints_deg": [0] * 7,
            "goal_matrices_world": [[[1, 0, 0, .4], [0, 1, 0, .1], [0, 0, 1, .7], [0, 0, 0, 1]]],
            "scene_token": "scene", "planning_settings": settings.model_dump()}


class Runtime:
    def __init__(self):
        self.geometry_scene = {"scene_token": "scene"}
        self.geometry_solver_profile = PlanningSettings().solver_payload()
        self.effective_config = {"A": {"ik": {"ik_num_seeds": 16}, "plan": {"ik_num_seeds": 16}}}

    def _scene_snapshot_unlocked(self):
        return self.geometry_scene

    def configure_planning(self, settings):
        self.settings = settings

    def solve_ik_batch_native(self, request, lock_timeout_s):
        assert request.trajectory_collision_clearance_m == self.settings.trajectory_collision_clearance_m
        return SimpleNamespace(to_payload=lambda: {"ok": True, "scene_token": "scene", "items": []})


def test_ik_operation_uses_profile_and_reports_effective_values():
    result = geometry_service.run_operation(Runtime(), "/tools/solve_ik_batch",
                                            request_payload(PlanningSettings()))
    assert result["effective_config"]["ik_num_seeds"] == 16


def test_reject_conflicting_legacy_values_instead_of_ignoring_them():
    payload = request_payload(PlanningSettings()) | {"trajectory_collision_clearance_m": .03}
    with pytest.raises(ValueError, match="planning_settings"):
        geometry_service.run_operation(Runtime(), "/tools/solve_ik_batch", payload)


def test_reject_solver_profile_different_from_scene_profile():
    payload = request_payload(PlanningSettings(ik_num_seeds=24))
    with pytest.raises(ValueError, match="scene.*settings"):
        geometry_service.run_operation(Runtime(), "/tools/solve_ik_batch", payload)


def test_batch_service_preserves_flags_and_nested_phase_payload():
    from test_backend_builder import native_batch
    native = native_batch()
    native = replace(native, items=(native.items[0],), batch_size=1)

    class BatchRuntime(Runtime):
        def plan_grasp_batch_native(self, request, lock_timeout_s):
            assert request.freq_hz == 50
            assert request.lift_offset_m == .1 and request.lift_in_tool_frame is False
            return native

    result = geometry_service.run_operation(BatchRuntime(), "/tools/plan_grasp_batch",
                                            request_payload(PlanningSettings()))
    assert result["batch_size"] == 1 and result["batch_capacity"] == 2
    assert result["items"][0]["success"] is True
    assert len(result["items"][0]["phases"]["grasp"]["positions_rad"]) == 2


def test_large_collision_profile_expands_esdf_grid_padding(monkeypatch):
    from test_algorithms import snapshot

    from jev_mujoco.geometry import geometry_payload

    calls = []

    def grid(name, kind, size, pose, *, padding_m=.3):
        calls.append(padding_m)
        return name, {"dims": size, "voxel_size": .01}

    monkeypatch.setattr(geometry_service, "make_grid", grid)
    monkeypatch.setattr(geometry_service, "maximum_world_sphere_radius", lambda _: .2)
    settings = PlanningSettings(trajectory_collision_clearance_m=.08, place_release_height_m=.09,
                                optimizer_collision_activation_distance_m=.5)
    state = snapshot()
    # This offline primitive utility requires explicit shapes; RGBD fixtures do not expose them.
    state["objects"][0].update(size=[.05] * 3, quaternion=[1., 0., 0., 0.])
    payload = geometry_payload(state, "red") | {"planning_settings": settings.model_dump()}
    runtime = geometry_service.GeometryRuntime()
    result = runtime.build_geometry(payload)
    assert calls == pytest.approx([.72, .72])
    assert result["esdf_padding_m"] == pytest.approx(.72)
    assert runtime.geometry_solver_profile == settings.solver_payload()


def set_geometry_layers(runtime, count, *, size=(1.0, 1.0, 1.0)):
    names = ["geometry_table", *[f"semantic_instance::{index}" for index in range(count - 1)]]
    runtime.geometry_scene = {"scene_token": f"scene-{count}", "target_id": "0"}
    runtime.geometry_grids = names
    runtime.geometry_model = {
        "voxel": {name: {"dims": list(size), "voxel_size": .01} for name in names},
    }


def test_geometry_cache_reserves_eight_layers_without_adding_obstacles():
    runtime = geometry_service.GeometryRuntime()
    set_geometry_layers(runtime, 4)

    snapshot = runtime._prepare_scene_cache_unlocked()

    assert runtime._collision_cache["voxel"]["layers"] == 8
    assert snapshot == {"scene_token": "scene-4", "target_id": "0"}
    assert list(runtime._scene_model["voxel"]) == [
        "geometry_table", "semantic_instance::0", "semantic_instance::1", "semantic_instance::2",
    ]


def test_geometry_cache_grows_once_and_reuses_planners_after_scene_shrinks():
    runtime = geometry_service.GeometryRuntime()
    set_geometry_layers(runtime, 4)
    runtime._prepare_scene_cache_unlocked()
    destroyed = []
    backend = SimpleNamespace(destroy=lambda: destroyed.append("old"))
    runtime._batch_backends["A"] = backend
    runtime._ik_batch_backends["A"] = backend
    runtime._scene_snapshot_key_by_arm["A"] = (backend, "scene-4")
    runtime._peer_collision_snapshot_by_arm["A"] = (backend, "peer")
    runtime.effective_config["A"] = {"plan": {"max_batch_size": 16}}

    set_geometry_layers(runtime, 9)
    runtime._prepare_scene_cache_unlocked()

    assert runtime._collision_cache["voxel"]["layers"] == 16
    assert destroyed == ["old"]
    assert not runtime._batch_backends and not runtime._ik_batch_backends
    assert not runtime._scene_snapshot_key_by_arm and not runtime._peer_collision_snapshot_by_arm
    assert not runtime.effective_config

    retained = SimpleNamespace(destroy=lambda: destroyed.append("retained"))
    runtime._batch_backends["A"] = retained
    for count in (16, 8, 4, 2, 4):
        set_geometry_layers(runtime, count)
        runtime._prepare_scene_cache_unlocked()
        assert runtime._collision_cache["voxel"]["layers"] == 16
        assert runtime._batch_backends["A"] is retained
        assert len(runtime._scene_model["voxel"]) == count
    assert destroyed == ["old"]


def test_geometry_cache_expands_for_larger_grids_and_retains_that_capacity():
    runtime = geometry_service.GeometryRuntime()
    set_geometry_layers(runtime, 4)
    runtime._prepare_scene_cache_unlocked()
    destroyed = []
    runtime._batch_backends["A"] = SimpleNamespace(destroy=lambda: destroyed.append("old"))

    set_geometry_layers(runtime, 4, size=(4.0, 4.0, 4.0))
    runtime._prepare_scene_cache_unlocked()

    assert destroyed == ["old"]
    assert runtime._collision_cache["voxel"]["layers"] == 8
    dimensions = runtime._collision_cache["voxel"]["dims"].copy()
    assert dimensions[0] * dimensions[1] * dimensions[2] >= 64
    retained = SimpleNamespace(destroy=lambda: destroyed.append("retained"))
    runtime._batch_backends["A"] = retained
    set_geometry_layers(runtime, 4)
    runtime._prepare_scene_cache_unlocked()
    assert runtime._collision_cache["voxel"]["dims"] == dimensions
    assert runtime._batch_backends["A"] is retained


def test_geometry_cache_cleanup_failure_drops_all_stale_planners_and_reports_error():
    from tianji_curobo.contracts import CuroboToolError

    runtime = geometry_service.GeometryRuntime()
    set_geometry_layers(runtime, 4)
    runtime._prepare_scene_cache_unlocked()
    destroyed = []

    def broken_destroy():
        destroyed.append("broken")
        raise RuntimeError("release failed")

    runtime._batch_backends["A"] = SimpleNamespace(destroy=broken_destroy)
    runtime._ik_batch_backends["B"] = SimpleNamespace(destroy=lambda: destroyed.append("other"))
    runtime._scene_snapshot_key_by_arm["A"] = "old"
    runtime.effective_config["A"] = {"plan": {"max_batch_size": 16}}
    set_geometry_layers(runtime, 9)

    with pytest.raises(CuroboToolError, match="cleanup") as caught:
        runtime._prepare_scene_cache_unlocked()

    assert caught.value.code == "CUROBO_BACKEND_CLEANUP_FAILED"
    assert destroyed == ["broken", "other"]
    assert not runtime._batch_backends and not runtime._ik_batch_backends
    assert not runtime._scene_snapshot_key_by_arm and not runtime.effective_config


def test_extra_observed_layer_reuses_reserved_planner_capacity():
    runtime = geometry_service.GeometryRuntime()
    set_geometry_layers(runtime, 4)
    runtime._prepare_scene_cache_unlocked()
    backend = SimpleNamespace(destroy=lambda: pytest.fail("common 4-to-5 layer growth must reuse solver"))
    runtime._backends['A'] = backend
    set_geometry_layers(runtime, 5)
    runtime._prepare_scene_cache_unlocked()
    assert runtime._backends['A'] is backend
    assert len(runtime._scene_model['voxel']) == 5
