import json

import pytest
from pydantic import ValidationError

from jev_mujoco.planning_settings import PlanningSettings, load_settings, save_settings


def test_place_defaults_are_independent_of_existing_grasp_settings():
    settings = PlanningSettings()
    assert (settings.approach_axis, settings.approach_offset_m, settings.approach_in_tool_frame) == ("z", -.1, True)
    assert (settings.lift_axis, settings.lift_offset_m, settings.lift_in_tool_frame) == ("z", .1, False)
    assert (settings.place_approach_axis, settings.place_approach_offset_m, settings.place_approach_in_tool_frame) == ("z", .1, False)
    assert (settings.place_retreat_axis, settings.place_retreat_offset_m, settings.place_retreat_in_tool_frame) == ("z", .1, False)
    assert settings.place_release_height_m == .012
    assert settings.place_timeout_s == 180
    assert "place_max_candidates" not in settings.model_dump()
    assert "place_max_candidates" not in settings.model_json_schema()["properties"]
    assert [settings.dwell_place_approach_s, settings.dwell_place_s, settings.dwell_release_s, settings.dwell_retreat_s] == [1, 2, 1, 1]


@pytest.mark.parametrize("changes", [
    {"place_approach_offset_m": 0}, {"place_retreat_offset_m": 0},
    {"place_approach_axis": "w"},
    {"place_timeout_s": 0},
    {"dwell_release_s": -1}, {"dwell_place_s": float("nan")},
])
def test_invalid_place_settings_are_rejected(changes):
    with pytest.raises(ValidationError):
        PlanningSettings(**changes)


def test_old_saved_values_gain_place_defaults_and_new_values_change_revision(tmp_path):
    path = tmp_path / "planning-settings.json"
    path.write_text(json.dumps({"approach_offset_m": -.08, "time_scale": 4, "place_max_candidates": 200}), encoding="utf-8")
    migrated = load_settings(path)
    assert migrated.approach_offset_m == -.08
    assert migrated.place_approach_offset_m == .1
    changed = PlanningSettings(**{**migrated.model_dump(), "place_approach_offset_m": .15})
    assert changed.revision() != migrated.revision()
    save_settings(path, changed)
    assert load_settings(path) == changed
    assert "place_max_candidates" not in json.loads(path.read_text(encoding="utf-8"))
    assert PlanningSettings.model_json_schema()["properties"]["place_release_height_m"]["group"] == "place"
    assert PlanningSettings.model_json_schema()["properties"]["dwell_release_s"]["group"] == "execution"


@pytest.mark.parametrize("legacy_name", ["release_margin_m", "place_release_margin_m"])
def test_legacy_saved_release_margin_is_replaced_by_collision_threshold(tmp_path, legacy_name):
    path = tmp_path / "planning-settings.json"
    path.write_text(json.dumps({legacy_name: .02, "approach_axis": "x", "approach_offset_m": -.08,
                                "lift_axis": "y", "lift_offset_m": .12, "lift_in_tool_frame": True}), encoding="utf-8")
    migrated = load_settings(path)
    assert migrated.place_release_height_m == .012
    assert (migrated.approach_axis, migrated.approach_offset_m) == ("x", -.08)
    assert (migrated.lift_axis, migrated.lift_offset_m, migrated.lift_in_tool_frame) == ("y", .12, True)
    save_settings(path, migrated)
    assert legacy_name not in json.loads(path.read_text(encoding="utf-8"))


def test_optional_filter_defaults_off_and_validates_measured_filter_parameters():
    settings = PlanningSettings()
    assert settings.pointcloud_filter.enabled is False
    assert settings.pointcloud_filter.dbscan_preview is False
    for changes in [{"method": "custom"}, {"radius_m": 0}, {"nb_neighbors": 1}, {"std_ratio": float("nan")}]:
        with pytest.raises(ValidationError):
            PlanningSettings(pointcloud_filter=changes)
