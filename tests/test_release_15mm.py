import json

import pytest

from luna_mujoco.planning_settings import PlanningSettings, load_settings


@pytest.mark.parametrize('clearance', [.01, .012, .02, .03])
def test_release_height_is_independent_of_collision_threshold(clearance):
    settings = PlanningSettings(trajectory_collision_clearance_m=clearance, place_release_height_m=.05)
    assert settings.place_release_height_m == .05


@pytest.mark.parametrize('gap', [.007, .01, .011, .02, .05])
def test_saved_release_height_is_preserved_without_changing_collision_threshold(tmp_path, gap):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'place_release_height_m': gap, 'time_scale': 4}), encoding='utf-8')
    settings = load_settings(path)
    assert settings.place_release_height_m == gap
    assert settings.trajectory_collision_clearance_m == .005
    assert settings.time_scale == 4
    assert json.loads(path.read_text(encoding='utf-8'))['place_release_height_m'] == gap


@pytest.mark.parametrize('legacy_name', ['release_margin_m', 'place_release_margin_m'])
def test_legacy_release_margin_is_retired(legacy_name):
    settings = PlanningSettings.model_validate({legacy_name: .007, 'trajectory_collision_clearance_m': .012})
    assert settings.place_release_height_m == 0.
