"""Larger bounded fits retain every measured point and its uncertainty."""
import json
from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError
from test_tight_carrying import assert_covered, measured_binding, visible_box_surface

from luna_mujoco.carrying import CarryingBinding, carrying_geometry, carrying_spheres
from luna_mujoco.planning_settings import PlanningSettings, load_settings, save_settings


@pytest.mark.parametrize("budget", [1, 16, 24, 31])
def test_bounded_fit_preserves_all_measured_points_and_padding(budget):
    points = visible_box_surface()
    original = points.copy()
    binding = measured_binding(points, sphere_budget=budget)
    geometry = carrying_geometry(binding)
    spheres = np.asarray(carrying_spheres(binding))
    assert geometry["sphere_budget"] == budget
    assert 1 <= geometry["sphere_count"] == len(spheres) <= budget
    assert_covered(points, spheres, .005 + .001)
    assert geometry["measurement_uncertainty_m"] == .005
    assert geometry["margin_m"] == .001
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(binding.points_tool_m, original)
    # Explicit fits must not contaminate the cached default fit.
    legacy = measured_binding(points)
    assert legacy.sphere_budget == 24
    assert len(carrying_spheres(legacy)) == 24


@pytest.mark.parametrize("budget", [1, 16, 24, 31])
def test_valid_saved_budget_is_preserved(tmp_path, budget):
    settings = PlanningSettings(attachment={"sphere_budget": budget})
    path = tmp_path / "settings.json"
    save_settings(path, settings)
    assert load_settings(path) == settings
    assert PlanningSettings().attachment.sphere_budget == 24
    schema = PlanningSettings.model_json_schema()["$defs"]["AttachmentSettings"]["properties"]["sphere_budget"]
    assert (schema["minimum"], schema["maximum"], schema["default"]) == (1, 31, 24)


@pytest.mark.parametrize('budget', [32, 256, 257, True])
def test_invalid_new_budget_rejected_by_settings_and_binding(budget):
    with pytest.raises(ValidationError):
        PlanningSettings(attachment={"sphere_budget": budget})
    payload = asdict(measured_binding(visible_box_surface()))
    payload["sphere_budget"] = budget
    scene = {"scene_token": payload["scene_token"], "observation_id": payload["observation_id"],
             "generation": payload["generation"], "target_id": payload["instance_id"]}
    with pytest.raises(ValueError, match="sphere budget"):
        CarryingBinding.from_mapping(payload, scene=scene, arm="A")


@pytest.mark.parametrize("budget", [32, 128, 256])
def test_historical_saved_budget_migration_is_explicit_and_does_not_rewrite_file(tmp_path, budget):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"attachment": {"sphere_budget": budget}}), encoding="utf8")
    original = path.read_bytes()
    with pytest.warns(UserWarning, match=f'sphere_budget {budget}.*24'):
        assert load_settings(path).attachment.sphere_budget == 24
    assert path.read_bytes() == original


def test_saved_out_of_previous_range_is_not_silently_migrated(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'attachment': {'sphere_budget': 257}}), encoding='utf8')
    with pytest.raises(ValidationError):
        load_settings(path)
