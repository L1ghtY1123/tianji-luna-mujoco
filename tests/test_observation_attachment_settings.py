import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pydantic import ValidationError

from luna_mujoco.observation_motion import observation_poses
from luna_mujoco.planning_settings import PlanningSettings, load_settings, save_settings

ROOT = Path(__file__).resolve().parents[1]


def test_calibrated_observation_defaults_and_attachment_settings_roundtrip(tmp_path):
    settings = PlanningSettings()
    calibrated = json.loads((ROOT / 'resources/observation_poses.json').read_text())['joints_rad']
    assert observation_poses(settings.model_dump()) == calibrated
    assert settings.attachment.model_dump() == {'minimum_partition_size_m': .01, 'margin_m': .001, 'sphere_budget': 24}
    values = settings.model_dump()
    values['observation']['A'] = list(values['observation']['A'])
    values['observation']['A'][0] += .01
    values['attachment']['margin_m'] = .002
    changed = PlanningSettings.model_validate(values)
    assert changed.revision() != settings.revision()
    assert changed.solver_payload() == settings.solver_payload()
    path = tmp_path / 'settings.json'
    save_settings(path, changed)
    assert load_settings(path) == changed
    poses = observation_poses(changed.model_dump())
    poses['A'][0] = 99
    assert changed.observation.A[0] != 99
    assert observation_poses(settings.model_dump()) == calibrated


@pytest.mark.parametrize('changes', [
    {'voxel_size_m': .0009}, {'voxel_size_m': .051}, {'margin_m': -.001},
    {'margin_m': .031}, {'sphere_budget': 0}, {'sphere_budget': 32}, {'sphere_budget': 1.5},
])
def test_attachment_settings_reject_invalid_geometry(changes):
    with pytest.raises(ValidationError):
        PlanningSettings(attachment=changes)


def test_observation_joint_bounds_match_robot_and_schema():
    resource = ET.parse(ROOT / 'assets/marvin_urdf/marvin_with_grippers.urdf').getroot()
    schema = PlanningSettings.model_json_schema()
    observation = schema['$defs']['ObservationSettings']['properties']
    values = PlanningSettings().model_dump()
    for arm, suffix in [('A', 'L'), ('B', 'R')]:
        for index in range(7):
            limit = resource.find(f"joint[@name='Joint{index + 1}_{suffix}']/limit")
            lower, upper = float(limit.attrib['lower']), float(limit.attrib['upper'])
            field = observation[arm]['prefixItems'][index]
            assert (field['minimum'], field['maximum']) == (lower, upper)
            assert field['group'] == 'observation' and field['unit'] == '°'
            outside = copy.deepcopy(values)
            outside['observation'][arm] = list(outside['observation'][arm])
            outside['observation'][arm][index] = upper + .001
            with pytest.raises(ValidationError, match='observation_joint_out_of_range'):
                PlanningSettings.model_validate(outside)


@pytest.mark.parametrize('joints', [[0.] * 6, [0.] * 8, [float('nan')] * 7, [True] * 7])
def test_observation_rejects_invalid_joint_vectors(joints):
    with pytest.raises(ValidationError):
        PlanningSettings(observation={'A': joints})


def test_observation_execution_boundary_rejects_unvalidated_dictionary():
    with pytest.raises(ValueError, match='observation_joint_out_of_range'):
        observation_poses({'observation': {'A': [99.] * 7, 'B': [0.] * 7}})


def test_new_settings_preserve_legacy_release_migration():
    settings = PlanningSettings.model_validate({'release_margin_m': .02})
    assert settings.place_release_height_m == settings.trajectory_collision_clearance_m
    assert settings.attachment.sphere_budget == 24
