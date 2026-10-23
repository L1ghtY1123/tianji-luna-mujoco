from dataclasses import replace

import numpy as np
import pytest

from luna_mujoco import carrying
from luna_mujoco.carrying import CarryingBinding, carrying_geometry


def measured_binding(points, **settings):
    payload = {
        "arm": "A", "owner_id": "tight-fit", "observation_id": "obs", "generation": 1,
        "scene_token": "scene", "instance_id": "object", "frame": "tool", "state": "held",
        "points_tool_m": np.asarray(points).tolist(), "uncertainty_m": .005, "margin_m": .001,
    } | settings
    scene = {"scene_token": "scene", "observation_id": "obs", "generation": 1, "target_id": "object"}
    return CarryingBinding.from_mapping(payload, scene=scene, arm="A")


def visible_box_surface():
    x, y = np.meshgrid(np.linspace(-.06, .06, 81), np.linspace(-.025, .025, 41))
    top = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .02)))
    x, z = np.meshgrid(np.linspace(-.06, .06, 81), np.linspace(-.02, .02, 31))
    front = np.column_stack((x.ravel(), np.full(x.size, -.025), z.ravel()))
    y, z = np.meshgrid(np.linspace(-.025, .025, 41), np.linspace(-.02, .02, 31))
    side = np.column_stack((np.full(y.size, .06), y.ravel(), z.ravel()))
    return np.concatenate((top, front, side))


def assert_covered(points, spheres, padding):
    gaps = np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=2) - spheres[None, :, 3]
    assert np.all(gaps.min(axis=1) <= -padding + 1e-12)


@pytest.mark.parametrize("angle", [0., .37, 1.1])
def test_asymmetric_surface_gets_tighter_center_without_losing_padding(angle):
    points = np.array([[-.03, 0., 0.], [.03, 0., 0.], [0., .05, 0.]])
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    points = points @ rotation.T
    value = measured_binding(points, sphere_budget=1)
    spheres = np.asarray(carrying_geometry(value)["spheres"])
    # The original axis-aligned box center needs 39.05 mm at zero rotation;
    # this measured triangle admits a covering radius below 35 mm.
    assert spheres[0, 3] < .035 + .006
    assert_covered(points, spheres, .006)


@pytest.mark.parametrize("angle", [0., .37, 1.1])
def test_rotated_three_face_cloud_keeps_every_padded_point_within_budget(angle):
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    points = visible_box_surface() @ rotation.T
    value = measured_binding(points)
    geometry = carrying_geometry(value)
    spheres = np.asarray(geometry["spheres"])
    assert len(spheres) == geometry["sphere_count"] <= 24
    assert sum(row["measured"] for row in geometry["source_counts_by_sphere"]) == len(points)
    assert_covered(points, spheres, .006)


def test_cached_geometry_does_not_reread_cloud_and_returns_defensive_copies(monkeypatch):
    value = measured_binding(visible_box_surface())
    expected = carrying_geometry(value)
    first = carrying_geometry(value)
    first["spheres"][0][0] = 99.
    first.pop("margin_m")
    original_asarray = np.asarray

    def forbid_cloud_conversion(candidate, *args, **kwargs):
        if candidate is value.points_tool_m:
            pytest.fail("cached geometry must not reread or rehash the full cloud")
        return original_asarray(candidate, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", forbid_cloud_conversion)
    assert carrying_geometry(value) == expected


def test_cache_respects_changed_points_and_all_fit_settings():
    points = visible_box_surface()
    original = measured_binding(points)
    baseline = np.asarray(carrying_geometry(original)["spheres"])
    for field in ("margin_m", "uncertainty_m"):
        changed = replace(original, **{field: getattr(original, field) + .002})
        result = np.asarray(carrying_geometry(changed)["spheres"])
        assert np.allclose(result[:, :3], baseline[:, :3])
        assert np.allclose(result[:, 3], baseline[:, 3] + .002)
    translated = replace(original, points_tool_m=tuple(map(tuple, points + [0., 0., .1])))
    translated_spheres = np.asarray(carrying_geometry(translated)["spheres"])
    assert np.allclose(translated_spheres[:, :3], baseline[:, :3] + [0., 0., .1])
    assert np.allclose(translated_spheres[:, 3], baseline[:, 3])
    assert (carrying_geometry(translated)["source_counts_by_sphere"]
            == carrying_geometry(original)["source_counts_by_sphere"])
    assert len(carrying_geometry(replace(original, sphere_budget=2))["spheres"]) == 2
    assert len(carrying_geometry(original, maximum_spheres=3)["spheres"]) == 3
    coarse = carrying_geometry(replace(original, voxel_size_m=.05))
    assert len(coarse["spheres"]) < len(baseline)
    assert coarse["minimum_partition_size_m"] == .05


def test_equal_cloud_from_new_binding_reuses_fit_across_scene_metadata(monkeypatch):
    value = measured_binding([[-.037, .003, 0.], [.023, .003, 0.], [0., .057, 0.]], sphere_budget=1)
    expected = carrying_geometry(value)
    new_binding = replace(value, owner_id="another-owner", observation_id="later", scene_token="new-scene")

    def forbid_refit(*args):
        pytest.fail("equal point content and fit settings must reuse the fitted geometry")

    monkeypatch.setattr(carrying, "_fit_carrying_geometry", forbid_refit)
    assert carrying_geometry(new_binding) == expected


def test_geometry_cache_evicts_old_entries_without_changing_results():
    value = measured_binding([[-.031, .001, 0.], [.021, .001, 0.], [0., .051, 0.]], sphere_budget=1)
    expected = carrying_geometry(value)
    for index in range(40):
        carrying_geometry(replace(value, margin_m=.0001 * index))
    assert len(carrying._geometry_cache) <= 32
    assert carrying_geometry(value) == expected
