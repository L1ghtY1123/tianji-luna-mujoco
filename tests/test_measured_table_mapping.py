import importlib.util

import numpy as np
import pytest


def table_cache():
    assert importlib.util.find_spec("luna_mujoco.measured_table_mapping"), "project table adapter missing"
    from tianji_curobo.mixed_esdf_scene import TableCompletion

    from luna_mujoco.measured_table_mapping import MeasuredTableCache
    calls = []
    def fit(groups, spheres, **kwargs):
        calls.append(kwargs)
        return TableCompletion({"pose": [.5, 0, .4525, 1, 0, 0, 0], "dims": [1, 1, .305]},
                               .6, (0, 0, 1), 20, kwargs["robot_clearance_m"], None)
    return MeasuredTableCache(table_builder=fit), calls


def payload():
    return {"planning_settings": {}, "camera_capture_metadata": {"3": {"calibration_id": "cal-a"}},
            "views": {"3": {"depth_left_m": np.ones((2, 3)), "left_intrinsics": np.eye(3),
              "rgb_intrinsics": np.eye(3), "left_to_rgb": np.eye(4)}}}


def test_measured_table_top_and_bottom_do_not_include_height_inflation():
    cache, calls = table_cache()
    table = cache.build([], np.array([[0, 0, 0, .1]]), payload(), hard_clearance_m=.01)
    assert table.model["pose"][2] + table.model["dims"][2] / 2 == pytest.approx(.6)
    assert table.model["pose"][2] - table.model["dims"][2] / 2 == pytest.approx(.3)
    assert table.model["dims"][:2] == [1, 1]
    assert calls[0]["robot_clearance_m"] >= .01


def test_table_cache_reuses_cameras_and_unrelated_settings_but_rejects_geometry_changes():
    cache, calls = table_cache()
    capture = payload()
    first = cache.build([], [], capture, hard_clearance_m=.01)
    again = cache.build([], [], capture | {"observation_set_id": "next", "selection_generation": 99}, hard_clearance_m=.01)
    assert again is first and len(calls) == 1
    assert cache.diagnostics["cache_hit"] is True
    capture['views']['1'] = dict(capture['views']['3'])
    capture['planning_settings'] = {'dwell_release_s': 1.}
    assert cache.build([], [], capture, hard_clearance_m=.01) is first
    assert len(calls) == 1
    with pytest.raises(ValueError, match='table_configuration_changed'):
        cache.build([], [], capture, hard_clearance_m=.04)
    assert cache.table is first
    capture["camera_capture_metadata"]["3"]["calibration_id"] = "cal-b"
    cache.build([], [], capture, hard_clearance_m=.01)
    assert len(calls) == 2
    cache.reset()
    cache.build([], [], capture, hard_clearance_m=.01)
    assert len(calls) == 3


def test_initial_downward_extension_is_bounded_by_robot_clearance():
    cache, calls = table_cache()
    table = cache.build([], [[.5, 0., .4, .05]], payload(), hard_clearance_m=.01)
    assert .001 < table.model['dims'][2] < .3
    assert table.model['pose'][2] + table.model['dims'][2] / 2 == pytest.approx(.6)
    assert cache.diagnostics['current_robot_clearance_m'] >= .01 - 1e-8
    assert len(calls) == 1


def test_crop_model_uses_plane_height_without_changing_collision_cuboid():
    cache, _ = table_cache()
    table = cache.build([], [], payload(), hard_clearance_m=.01)
    from luna_mujoco.measured_table_mapping import table_surface_crop_model
    crop = table_surface_crop_model(table)
    assert crop["pose"][2] - crop["dims"][2] / 2 == pytest.approx(.6)
    assert crop["pose"][:2] == table.model["pose"][:2]
    assert crop["dims"][:2] == table.model["dims"][:2]
    assert table.model["dims"][2] == pytest.approx(.3)


def test_cached_table_is_never_refitted_to_hide_robot_collision():
    cache, calls = table_cache()
    original = cache.build([], [[0, 0, 0, .1]], payload(), hard_clearance_m=.01)
    with pytest.raises(ValueError, match="tabletop_robot_clearance"):
        cache.build([], [[.5, 0, .4, .1]], payload(), hard_clearance_m=.01)
    assert len(calls) == 1
    assert cache.table is original
