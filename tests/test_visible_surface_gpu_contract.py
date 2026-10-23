from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.carrying import CarryingBinding, carrying_spheres, create_carrying_robot_config
from luna_mujoco.geometry_service import GeometryRuntime


def measured_binding():
    return CarryingBinding.from_mapping({
        "arm": "A", "owner_id": "hold", "observation_id": "obs", "generation": 1,
        "scene_token": "scene", "instance_id": "object", "frame": "tool", "state": "held",
        "points_tool_m": [[-.03, -.03, .04], [.03, .03, .04]],
        "uncertainty_m": .004, "coverage": .5,
    }, scene={"scene_token": "scene", "observation_id": "obs", "generation": 1, "target_id": "object"}, arm="A")


def test_only_measured_surface_is_covered():
    binding = measured_binding()
    spheres = np.asarray(carrying_spheres(binding))
    assert np.allclose(spheres[:, 2], .04)
    assert np.min(np.linalg.norm(spheres[:, :3] - [0, 0, -.04], axis=1) - spheres[:, 3]) > 0
    assert np.all(np.min(np.linalg.norm(np.asarray(binding.points_tool_m)[:, None] - spheres[None, :, :3], axis=2) - spheres[None, :, 3], axis=1) <= -.004)


def test_surface_budget_keeps_last_observed_voxel():
    binding = measured_binding()
    from dataclasses import replace
    points = np.array([[x, y, .04] for x in np.linspace(-.15, .15, 40) for y in np.linspace(-.15, .15, 40)])
    spheres = np.asarray(carrying_spheres(replace(binding, points_tool_m=tuple(map(tuple, points))), maximum_spheres=31))
    assert len(spheres) <= 31
    assert np.all(np.min(np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=2) - spheres[None, :, 3], axis=1) <= -.004)


def test_attachment_exempts_exact_owner_hand_without_removing_world_or_other_self_checks():
    touch_links = ["gripL_Base", *(f"gripL_Link{part}_{side}"
                                 for part in range(1, 5) for side in "LR")]
    checked_links = ["TCP_Link_L", "Link7_L", "gripR_Base", "gripR_Link4_L", "gripL_new_camera"]
    names = touch_links + checked_links
    config = {"robot_cfg": {"kinematics": {
        "collision_link_names": names,
        "collision_spheres": {name: [{"center": [0., 0., 0.], "radius": .01}] for name in names},
        "self_collision_ignore": {"gripL_Link1_L": ["gripL_Link2_L"]},
    }}}
    original = deepcopy(config)
    carried, link = create_carrying_robot_config(config, measured_binding())
    kin = carried["robot_cfg"]["kinematics"]
    assert set(kin["self_collision_ignore"][link]) == set(touch_links)
    for name in names:
        expected = set(original["robot_cfg"]["kinematics"]["self_collision_ignore"].get(name, []))
        if name in touch_links:
            expected.add(link)
        assert set(kin["self_collision_ignore"].get(name, [])) == expected
        assert name in kin["collision_link_names"]
        assert kin["collision_spheres"][name] == original["robot_cfg"]["kinematics"]["collision_spheres"][name]
    assert kin["collision_spheres"][link]
    assert config == original


def test_scope_reuses_exact_configuration_and_preserves_logical_binding():
    runtime = GeometryRuntime()
    binding = measured_binding()
    runtime.logical_carrying_binding = binding
    destroyed = []
    with runtime.carrying_scope(binding, {"A": .4, "B": 0.}):
        from collections import OrderedDict

        assert isinstance(runtime._compound_backends, OrderedDict)
        assert isinstance(runtime._batch_backends, OrderedDict)
        assert isinstance(runtime._ik_batch_backends, OrderedDict)
        runtime._backends["dummy"] = SimpleNamespace(destroy=lambda: destroyed.append(True))
    with runtime.carrying_scope(None, {"A": 0., "B": 0.}):
        assert "dummy" in runtime._backends
    with pytest.raises(RuntimeError), runtime.carrying_scope(binding, {"A": .4, "B": 0.}):
        raise RuntimeError("planner failed")
    assert destroyed == []
    with runtime.carrying_scope(binding, {"A": .4, "B": 0.}):
        assert "dummy" in runtime._backends
    assert runtime.logical_carrying_binding == binding
    with runtime.carrying_scope(binding, {"A": .40001, "B": 0.}):
        pass
    assert destroyed == []


def test_support_departure_does_not_replace_open_grasp_and_observation_solvers():
    runtime = GeometryRuntime()
    destroyed = []
    feedback = {'A': 0., 'B': 0.}
    ordinary = SimpleNamespace(destroy=lambda: destroyed.append('ordinary'))
    departure = SimpleNamespace(destroy=lambda: destroyed.append('departure'))
    with runtime.carrying_scope(None, feedback):
        runtime._backends['open'] = ordinary
    frame = {'arm': 'A', 'link_name': 'support_axis'}
    with runtime.carrying_scope(None, feedback, support_frame=frame):
        assert 'open' not in runtime._backends
        runtime._backends['departure'] = departure
    with runtime.carrying_scope(None, feedback):
        assert runtime._backends['open'] is ordinary
        assert 'departure' not in runtime._backends
    assert destroyed == []
    runtime.detach_carrying()
    with runtime.carrying_scope(None, feedback):
        assert runtime._backends['open'] is ordinary
    assert destroyed == []


def test_carrying_metadata_and_new_spheres_retain_fixed_capacity_solver():
    from dataclasses import replace
    runtime = GeometryRuntime()
    binding = measured_binding()
    feedback = {'A': .4, 'B': 0.}
    destroyed = []
    backend = SimpleNamespace(destroy=lambda: destroyed.append(True))
    with runtime.carrying_scope(binding, feedback):
        runtime._ik_batch_backends['ik'] = backend
    later = replace(binding, owner_id='new-owner', scene_token='new-scene', generation=2)
    with runtime.carrying_scope(later, feedback):
        assert runtime._ik_batch_backends['ik'] is backend
        assert runtime.carrying_binding is later
    moved = replace(later, points_tool_m=tuple((x, y, z + .01) for x, y, z in later.points_tool_m))
    with runtime.carrying_scope(moved, feedback):
        assert runtime._ik_batch_backends['ik'] is backend
    assert destroyed == []


def test_mapping_surface_identity_and_observation_retains_target():
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload
    _, identity = decode_mapping_payload(wire() | {"exclusion_state": "observation", "selected_surface_world_m": [[0, 0, 1], [.01, 0, 1]]})
    assert identity["disabled_obstacle_names"] == ()
    assert "selected_object_estimate" not in identity
    assert len(identity["selected_surface_world_m"]) == 2


def test_table_roi_does_not_certify_outside_unknown_space():
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    view = {"depth_left_m": np.full((21, 21), .5), "valid_left_mask": np.ones((21, 21)),
            "left_intrinsics": [[10, 0, 10], [0, 10, 10], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {}}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": view["depth_left_m"]}, "scene")
    guard.support_plane = {"observed_polygon_xy_m": [[.3, -.3], [.8, -.3], [.8, .3], [.3, .3]],
                           "normal_xyz": [0, 0, 1], "contact_offset_m": -.7}
    accepted, proof = guard.check(np.array([[[0, 0, .8, .04], [.5, 0, .8, .04]]]))
    assert not accepted[0]
    assert proof["outside_observation_scope_sphere_count"] == 1
    assert proof["unobserved_sphere_count"] == 1
    assert proof["outside_observation_scope_status"] == "unknown_not_certified_free"


def test_rejected_mapper_depth_cannot_carve_free_space(monkeypatch):
    from test_perception_mapping import wire

    from luna_mujoco import pointcloud_filter
    from luna_mujoco.perception_mapping import decode_mapping_payload, filter_mapping_views
    decoded, _ = decode_mapping_payload(wire())
    def reject_first(points, sources, *, settings):
        rejected = np.zeros(len(points), dtype=bool)
        rejected[0] = True
        assert sources["camera_ids"][0] == "1"
        return {"rejected_mask": rejected, "counts": {"input": len(points), "kept": len(points) - 1, "rejected": 1}}
    monkeypatch.setattr(pointcloud_filter, "filter_observed_points", reject_first)
    proof = filter_mapping_views(decoded, {"enabled": True})
    assert decoded["views"]["1"]["valid_left_mask"][0, 0] == 0
    assert decoded["views"]["1"]["depth_left_m"][0, 0] == 0
    assert proof["counts"]["rejected"] == 1


def test_detach_retains_scene_and_releases_native_slots():
    runtime = GeometryRuntime()
    runtime.geometry_scene = {"scene_token": "scene"}
    runtime.logical_carrying_binding = measured_binding()
    detached = []
    runtime._backends["dummy"] = SimpleNamespace(destroy=lambda: detached.append(True))
    assert runtime.detach_carrying() == {"ok": True, "attached": False}
    assert runtime.geometry_scene == {"scene_token": "scene"}
    assert runtime.logical_carrying_binding is None
    assert detached == [] and 'dummy' in runtime._backends


def test_native_planner_can_acquire_gpu_lock_inside_scope():
    runtime = GeometryRuntime()
    with runtime.carrying_scope(measured_binding(), {"A": .4, "B": 0.}):
        acquired = runtime.gpu_lock.acquire(blocking=False)
        if acquired:
            runtime.gpu_lock.release()
        assert acquired, "native planner must acquire its own GPU lock"


@pytest.mark.parametrize("scene_state", ["selected", "held", "released"])
def test_retreat_validation_uses_shared_manipulation_scene(monkeypatch, scene_state):
    from luna_mujoco import geometry_service

    runtime = GeometryRuntime()
    runtime.geometry_scene = {"exclusion_state": scene_state, "scene_token": "scene"}
    planner = object()
    monkeypatch.setattr(runtime, "_prepare_scene_cache_unlocked", lambda: None)
    monkeypatch.setattr(runtime, "_compound_backend_unlocked", lambda *args: SimpleNamespace(motion_planner=planner))
    monkeypatch.setattr(runtime, "_refresh_scene_unlocked", lambda **kwargs: None)
    checked = []

    def check(runtime_arg, planner_arg, payload, binding):
        assert runtime_arg is runtime and planner_arg is planner and binding is None
        checked.append(payload["phase"])
        return {"collision_free": True}

    monkeypatch.setattr(geometry_service, "check_command_path", check)
    assert runtime.validate_execution({"arm": "A", "phase": "retreat", "scene_token": "scene"}, None)["collision_free"]
    assert checked == ["retreat"]


def test_selected_pixels_have_contact_intent_evidence_only():
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    view = {"depth_left_m": np.ones((21, 21)), "valid_left_mask": np.ones((21, 21)),
            "left_intrinsics": [[10, 0, 10], [0, 10, 10], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {"selected": np.ones((21, 21))}}
    guard = ObservedFreeSpaceGuard({"1": view}, {"1": view["depth_left_m"]}, "scene")
    guard.selected_instance_id = "selected"
    spheres = np.array([[[0, 0, 1, .02]]])
    rejected, _ = guard.check(spheres)
    accepted, proof = guard.check(spheres, source_indices=[0], source_contact_samples=[True])
    assert not rejected[0] and accepted[0]
    assert proof["source_contact_intent_sphere_count"] == 1
    assert proof["source_contact_intent_certifies_free_space"] is False
    view["instance_masks"]["selected"] = np.zeros((21, 21))
    non_target = ObservedFreeSpaceGuard({"1": view}, {"1": view["depth_left_m"]}, "scene")
    non_target.selected_instance_id = "selected"
    rejected, _ = non_target.check(spheres, source_indices=[0], source_contact_samples=[True])
    assert not rejected[0]


def test_observation_phase_is_detached_actual_geometry():
    from luna_mujoco.carrying import validate_phase_request
    assert validate_phase_request({"phase": "observation", "gripper_geometry_mode": "actual"}) == "observation"
    with pytest.raises(ValueError, match="detached"):
        validate_phase_request({"phase": "observation", "carrying_binding": {}})


def test_mapping_token_binds_effective_nested_filter_settings():
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload
    payload = wire()
    _, original = decode_mapping_payload(payload | {"planning_settings": {"pointcloud_filter": {"enabled": False}}})
    _, changed = decode_mapping_payload(payload | {"planning_settings": {"pointcloud_filter": {"enabled": True}}})
    assert original["scene_token"] != changed["scene_token"]
    _, top = decode_mapping_payload(payload | {"pointcloud_filter": {"enabled": True},
                                               "planning_settings": {"pointcloud_filter": {"enabled": False}}})
    assert top["scene_token"] == original["scene_token"]


def held_source_guard():
    from dataclasses import replace

    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    view = {"depth_left_m": np.ones((41, 41)), "valid_left_mask": np.ones((41, 41)),
            "left_intrinsics": [[100, 0, 20], [0, 100, 20], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {"object": np.ones((41, 41))}}
    guard = ObservedFreeSpaceGuard({"1": view}, {"1": view["depth_left_m"]}, "scene")
    guard.selected_instance_id = "object"
    guard.observation_id, guard.generation = "obs", 1
    binding = replace(measured_binding(), points_tool_m=((-.002, 0., 1.), (.002, 0., 1.)))
    guard.selected_surface_world_m = list(binding.points_tool_m)
    return guard, binding, view


def test_held_source_return_contact_is_not_free_space():
    guard, binding, _ = held_source_guard()
    known, proof = guard.check(np.array([[[0, 0, 1, .015]], [[0, 0, 1.01, .015]]]),
        source_indices=[0], held_binding=binding, held_capture_points_world_m=binding.points_tool_m)
    assert known.all()
    assert proof["held_source_contact_sphere_count"] == 2
    assert proof["source_contact_intent_certifies_free_space"] is False


@pytest.mark.parametrize("change", ["far_behind", "other_pixel", "invalid_pixel", "stale_binding", "wrong_fk", "opened_pad"])
def test_held_source_certificate_rejects_unmeasured_space(change):
    from dataclasses import replace

    guard, binding, _view = held_source_guard()
    sphere = [0, 0, 1, .015]
    capture = np.asarray(binding.points_tool_m)
    if change in ("far_behind", "opened_pad"):
        sphere[2] = 1.1
    elif change == "other_pixel":
        guard.views[0][4]["object"][20, 20] = False
    elif change == "invalid_pixel":
        guard.views[0][3][20, 20] = 0
        guard.views[0][0][20, 20] = 0
    elif change == "stale_binding":
        binding = replace(binding, observation_id="older")
    elif change == "wrong_fk":
        capture = capture + [0., 0., .1]
    known, proof = guard.check(np.array([[sphere]]), source_indices=[0],
        held_binding=binding, held_capture_points_world_m=capture)
    assert not known[0], change
    assert proof["unobserved_sphere_count"] == 1


def test_observation_head_only_uses_native_background_without_fabricated_views():
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload, mapping_builder_payload
    decoded, _ = decode_mapping_payload(wire() | {"exclusion_state": "observation"})
    for camera in ("1", "2"):
        decoded["views"][camera]["depth_left_m"][:] = 0
        decoded["views"][camera]["valid_left_mask"][:] = 0
    native = mapping_builder_payload(decoded)
    assert native["mapping_mode"] == "background"
    assert native["confirmed_instance_id"] == ""
    assert list(native["views"]) == ["3"]
    assert native["views"]["3"]["instance_masks"] == {}
    assert decoded["views"]["3"]["instance_masks"]
    assert np.all(native["views"]["3"]["depth_left_m"] == 1)
    assert list(decoded["views"]) == ["1", "2", "3"]


def test_blank_views_do_not_weaken_selected_object_mapping():
    from test_perception_mapping import wire

    from luna_mujoco.perception_mapping import decode_mapping_payload, mapping_builder_payload
    decoded, _ = decode_mapping_payload(wire())
    decoded["views"]["1"]["valid_left_mask"][:] = 0
    with pytest.raises(ValueError, match="observation"):
        mapping_builder_payload(decoded)


def test_table_roi_requires_camera_frustum_but_invalid_inside_stays_unknown():
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    view = {"depth_left_m": np.ones((21, 21)), "valid_left_mask": np.ones((21, 21)),
            "left_intrinsics": [[100, 0, 10], [0, 100, 10], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {}}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": view["depth_left_m"]}, "scene")
    guard.support_plane = {"observed_polygon_xy_m": [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]],
                           "normal_xyz": [0, 0, 1], "contact_offset_m": -.3}
    accepted, proof = guard.check(np.array([[[.4, .4, .6, .02]]]))
    assert accepted[0]
    assert proof["outside_observation_scope_sphere_count"] == 1
    assert proof["outside_observation_scope_status"] == "unknown_not_certified_free"
    guard.views[0][0][10, 10] = 0
    rejected, proof = guard.check(np.array([[[0, 0, .6, .02]]]))
    assert not rejected[0]
    assert proof["outside_observation_scope_sphere_count"] == 0


def test_guard_registers_rgb_masks_to_depth_using_capture_calibration():
    from luna_mujoco.perception_mapping import registered_source_masks
    depth = np.ones((5, 5))
    rgb_mask = np.zeros((3, 3), dtype=bool)
    rgb_mask[1, 1] = True
    view = {"left_intrinsics": [[10, 0, 2], [0, 10, 2], [0, 0, 1]],
            "rgb_intrinsics": [[10, 0, 1], [0, 10, 1], [0, 0, 1]],
            "left_to_rgb": np.eye(4), "instance_masks": {"source": rgb_mask}}
    registered = registered_source_masks(view, depth)["source"]
    assert registered.shape == depth.shape
    assert registered[2, 2]
    assert not registered[1, 1]


def test_camera_scope_applies_without_fitted_support_plane():
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    view = {"depth_left_m": np.ones((21, 21)), "valid_left_mask": np.ones((21, 21)),
            "left_intrinsics": [[100, 0, 10], [0, 100, 10], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {}}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": view["depth_left_m"]}, "scene")
    assert guard.support_plane is None
    accepted, proof = guard.check(np.array([[[0, 0, -.5, .02], [.4, .4, .6, .02]]]))
    assert accepted[0]
    assert proof["outside_observation_scope_sphere_count"] == 2
    assert proof["outside_observation_scope_status"] == "unknown_not_certified_free"
    guard.views[0][0][10, 10] = 0
    rejected, proof = guard.check(np.array([[[0, 0, .6, .02]]]))
    assert not rejected[0]
    assert proof["outside_observation_scope_sphere_count"] == 0




def test_partial_camera_footprint_checks_visible_invalid_pixels():
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    view = {"depth_left_m": np.ones((21, 21)), "valid_left_mask": np.ones((21, 21)),
            "left_intrinsics": [[100, 0, 10], [0, 100, 10], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {}}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": view["depth_left_m"]}, "scene")
    sphere = np.array([[[.054, 0, .6, .02]]])
    guard.views[0][0][10, 19] = 0
    rejected, proof = guard.check(sphere)
    assert not rejected[0]
    assert proof["outside_observation_scope_sphere_count"] == 0
    assert proof["partial_observation_scope_sphere_count"] == 1
    guard.views[0][0][10, 19] = 1
    accepted, proof = guard.check(sphere)
    assert accepted[0]
    assert proof["partial_footprint_outside_status"] == "unknown_not_certified_free"


def test_partial_clear_sliver_cannot_override_available_full_footprint():
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    partial = {"depth_left_m": np.ones((21, 21)), "valid_left_mask": np.ones((21, 21)),
               "left_intrinsics": [[100, 0, 10], [0, 100, 10], [0, 0, 1]],
               "T_base_left_camera": np.eye(4), "instance_masks": {}}
    full = partial | {"left_intrinsics": [[50, 0, 10], [0, 50, 10], [0, 0, 1]], "valid_left_mask": np.ones((21, 21))}
    full["valid_left_mask"][10, 15] = 0
    filtered = np.ones((21, 21))
    filtered[10, 15] = 0
    guard = ObservedFreeSpaceGuard({"1": partial, "3": full}, {"1": partial["depth_left_m"], "3": filtered}, "scene")
    rejected, _ = guard.check(np.array([[[.054, 0, .6, .02]]]))
    assert not rejected[0]


def test_attachment_knobs_change_budget_margin_and_report_effective_cells():
    from dataclasses import asdict

    from luna_mujoco.carrying import carrying_geometry
    original = measured_binding()
    scene = {"scene_token": "scene", "observation_id": "obs", "generation": 1, "target_id": "object"}
    value = CarryingBinding.from_mapping(asdict(original) | {"voxel_size_m": .004, "margin_m": .003, "sphere_budget": 1}, scene=scene, arm="A")
    result = carrying_geometry(value)
    assert len(result["spheres"]) == result["sphere_count"] == 1
    assert result["sphere_budget"] == 1
    assert result["minimum_partition_size_m"] == .004
    assert result["maximum_group_extent_m"] >= .004
    assert result["margin_m"] == .003
    for changes in ({"voxel_size_m": 0}, {"voxel_size_m": .051}, {"margin_m": -.001},
                    {"margin_m": .031}, {"sphere_budget": 32}, {"sphere_budget": True}):
        with pytest.raises(ValueError, match="attachment"):
            CarryingBinding.from_mapping(asdict(original) | changes, scene=scene, arm="A")


def test_robot_removed_return_still_certifies_only_raw_front_interval():
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    view = {"depth_left_m": np.ones((21, 21)), "valid_left_mask": np.ones((21, 21)),
            "left_intrinsics": [[20, 0, 10], [0, 20, 10], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {}}
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": np.zeros((21, 21))}, "scene")
    spheres = np.array([[[0, 0, .7, .02]], [[0, 0, 1.1, .02]], [[0, 0, .99, .02]]])
    accepted, _ = guard.check(spheres)
    assert accepted.tolist() == [True, False, False]
    view["valid_left_mask"][10, 10] = 0
    invalid = ObservedFreeSpaceGuard({"3": view}, {"3": np.zeros((21, 21))}, "scene")
    accepted, _ = invalid.check(spheres[:1])
    assert not accepted[0]




def task_height_guard(*, tall=False, rejected=False, robot_removed=False):
    from luna_mujoco.perception_mapping import ObservedFreeSpaceGuard
    depth = np.full((21, 21), .7)
    valid = np.ones((21, 21))
    depth[2, 2] = .95 if tall else .7
    if rejected:
        valid[2, 2] = 0
    view = {"depth_left_m": depth, "valid_left_mask": valid,
            "left_intrinsics": [[100, 0, 10], [0, 100, 10], [0, 0, 1]],
            "T_base_left_camera": np.eye(4), "instance_masks": {"object": np.zeros((21, 21))}}
    filtered = depth.copy()
    if robot_removed:
        filtered[2, 2] = 0
    guard = ObservedFreeSpaceGuard({"3": view}, {"3": filtered}, "scene")
    guard.selected_instance_id = "object"
    table = {"pose": [0, 0, .3, 1, 0, 0, 0], "dims": [.4, .4, .6]}
    guard.set_measured_table_task_scope(table, {"3": view}, {"3": filtered})
    return guard, view, filtered, table


def test_measured_task_height_excludes_only_above_and_checks_intersection():
    guard, _view, _filtered, _table = task_height_guard()
    assert guard.measured_table_task_scope["measured_max_z_m"] == .7
    accepted, proof = guard.check(np.array([[[0, 0, .9, .02]]]))
    assert accepted[0]
    assert proof["outside_above_measured_task_sphere_count"] == 1
    assert proof["outside_observation_scope_status"] == "unknown_not_certified_free"
    rejected, proof = guard.check(np.array([[[0, 0, .72, .02]]]))
    assert not rejected[0]
    assert proof["outside_above_measured_task_sphere_count"] == 0


def test_tall_nonsemantic_return_expands_scope_and_invalid_inside_rejects():
    guard, _view, _filtered, _table = task_height_guard(tall=True)
    assert guard.measured_table_task_scope["upper_z_m"] == pytest.approx(.96)
    guard.views[0][0][10, 10] = 0
    rejected, proof = guard.check(np.array([[[0, 0, .85, .02]]]))
    assert not rejected[0]
    assert proof["outside_above_measured_task_sphere_count"] == 0


def test_rejected_and_robot_capture_returns_do_not_expand_task_height():
    for changes in ({"rejected": True}, {"robot_removed": True}):
        guard, _view, _filtered, _table = task_height_guard(tall=True, **changes)
        assert guard.measured_table_task_scope["measured_max_z_m"] == .7


def test_current_selected_returns_restore_robot_masked_source_only():
    guard, view, filtered, table = task_height_guard(tall=True, robot_removed=True)
    view["instance_masks"]["object"][2, 2] = 1
    guard.selected_surface_world_m = [[-.076, -.076, .95], [-.077, -.076, .95]]
    guard.set_measured_table_task_scope(table, {"3": view}, {"3": filtered})
    assert guard.measured_table_task_scope["measured_max_z_m"] == .95
    view["valid_left_mask"][2, 2] = 0
    guard.set_measured_table_task_scope(table, {"3": view}, {"3": filtered})
    assert guard.measured_table_task_scope["measured_max_z_m"] == .7

    view["instance_masks"]["object"][10, 10] = 1
    guard.selected_surface_world_m = [[0, 0, .705], [0, 0, .707]]
    guard.set_measured_table_task_scope(table, {"3": view}, {"3": filtered})
    assert guard.measured_table_task_scope["measured_max_z_m"] == .7

def test_configured_attachment_is_disabled_even_before_manager_upload(monkeypatch):
    from luna_mujoco.backend_builder import ValidatedSingleBackend, _NativeCuroboBackend
    calls = []
    backend = object.__new__(ValidatedSingleBackend)
    backend.configured_carrying_link = "jev_carried_object_A"
    backend.motion_planner = SimpleNamespace(
        attachment_manager=SimpleNamespace(detach=lambda **values: calls.append(("detach", values))),
        disable_link_collision=lambda links: calls.append(("disable", links)),
    )
    monkeypatch.setattr(_NativeCuroboBackend, "destroy", lambda _: calls.append(("destroy", None)))
    backend.destroy()
    assert calls == [("detach", {"link_name": "jev_carried_object_A"}),
                     ("disable", ["jev_carried_object_A"]), ("destroy", None)]

