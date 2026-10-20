"""Regression for RGB mask spill that Mapper already returned to background."""

from dataclasses import replace

import numpy as np
import pytest
from test_visible_surface_gpu_contract import measured_binding

from jev_mujoco.perception_mapping import ObservedFreeSpaceGuard


def guard_with_limits(limits):
    raw = np.ones((41, 41))
    view = {"depth_left_m": raw, "valid_left_mask": np.ones_like(raw),
        "left_intrinsics": [[100, 0, 20], [0, 100, 20], [0, 0, 1]],
        "T_base_left_camera": np.eye(4), "instance_masks": {"object": np.ones_like(raw)}}
    guard = ObservedFreeSpaceGuard({"1": view}, {"1": raw}, "scene", registered_source_limits=limits)
    # These are actual current z=1 pixel returns. The broad RGB mask also covers
    # unsupported background 15 cm to the right, which must remain an obstacle.
    points = tuple((column / 100, row / 100, 1.) for row in range(-2, 3) for column in range(-2, 3))
    binding = replace(measured_binding(), points_tool_m=points)
    guard.selected_instance_id = "object"
    guard.observation_id, guard.generation = "obs", 1
    guard.selected_surface_world_m = points
    return guard, binding


def current_limits():
    limit = np.zeros((41, 41), dtype=bool)
    limit[18:23, 18:23] = True
    return {"1": {"object": limit}}


def check(guard, binding, *, x=0., held=True):
    return guard.check(np.array([[[x, 0., 1., .015]]]), source_indices=[0],
        held_binding=binding if held else None,
        held_capture_points_world_m=binding.points_tool_m if held else None,
        source_contact_samples=[not held])


@pytest.mark.parametrize("held", [True, False])
def test_unsupported_selected_rgb_spill_cannot_certify_held_or_grasp_contact(held):
    guard, binding = guard_with_limits(current_limits())
    accepted, proof = check(guard, binding, x=.15, held=held)
    assert not accepted[0]
    assert proof["source_contact_intent_sphere_count"] == 0
    assert proof["owned_source_occupied_pixel_count"] == 0


def test_current_supported_measured_source_contact_still_passes():
    guard, binding = guard_with_limits(current_limits())
    accepted, proof = check(guard, binding)
    assert accepted[0]
    assert proof["held_source_contact_sphere_count"] == 1
    assert proof["owned_source_occupied_pixel_count"] > 0
    assert proof["source_contact_intent_certifies_free_space"] is False


@pytest.mark.parametrize("limits", [{}, {"2": {"object": np.ones((41, 41), dtype=bool)}},
                                  {"1": {"foreign": np.ones((41, 41), dtype=bool)}}])
def test_missing_current_camera_or_instance_support_fails_contact_closed(limits):
    guard, binding = guard_with_limits(limits)
    assert not check(guard, binding)[0][0]


def test_guard_owns_capture_limit_copy_and_does_not_reuse_it_for_next_capture():
    limits = current_limits()
    guard, binding = guard_with_limits(limits)
    limits["1"]["object"][:] = False
    assert check(guard, binding)[0][0]
    next_guard, next_binding = guard_with_limits({})
    assert not check(next_guard, next_binding)[0][0]
