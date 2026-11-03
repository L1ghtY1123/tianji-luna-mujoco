import pytest
from pydantic import ValidationError

from luna_mujoco.demo_scene import demo_region
from luna_mujoco.planning_settings import PlanningSettings, load_settings, save_settings
from luna_mujoco.vla_configuration import vla_unpack_configuration


def test_input_xy_persists_without_changing_solver_or_preparation_orientation(tmp_path):
    original = PlanningSettings()
    changed = PlanningSettings(vla_unpack={'input_x_m': .48, 'input_y_m': .16})
    path = tmp_path / 'settings.json'
    save_settings(path, changed)
    assert load_settings(path) == changed
    assert original.solver_payload() == changed.solver_payload()
    before, after = vla_unpack_configuration(original), vla_unpack_configuration(changed)
    assert before['input_xy_m'] == demo_region()['center_world_xy_m']
    assert after['input_xy_m'] == [.48, .16]
    assert after['input_region']['center_world_xy_m'] == [.48, .16]
    assert after['preparation_joints_deg'] == before['preparation_joints_deg']
    assert after['required_arms'] == ['A', 'B']
    assert after['output_position'] == 'unknown_until_current_visual_measurement'
    assert 'output_pose' not in after
    assert after['input_region_id'] != 'demo-destination'


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_vla_input_is_rejected(value):
    with pytest.raises(ValidationError):
        PlanningSettings(vla_unpack={'input_x_m': value})


def test_web_schema_exposes_xy_without_editable_pose():
    fields = PlanningSettings.model_json_schema()['$defs']['VLAUnpackSettings']['properties']
    assert set(fields) == {'input_x_m', 'input_y_m', 'input_size_m'}
    assert all(field['group'] == 'vla' and field['unit'] == 'm' for field in fields.values())


def test_vla_xy_defines_square_region_not_a_two_millimeter_center_target():
    config = vla_unpack_configuration()
    region = config['input_region']
    assert region['precise_xy'] is False
    assert config['input_position_constraint'] == 'object_within_configured_square'
    assert region['size_m'] == demo_region()['size_m']
    assert region['physical_marker'] is False
