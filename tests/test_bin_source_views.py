"""A box wall can hide one wrist view without hiding the measured grasp surface."""
import copy

import numpy as np
import pytest
from geometry_perception_fixture import GeometryPerception
from test_broad_perception import PromptSegmenter
from test_perception import capture_fixture

from luna_mujoco.observation_motion import initial_view_evidence


def measured_bin_source():
    source = {'id': 'cube', 'kind': 'cube', 'color': 'red',
              'points_world_m': [[x, y, .635] for x in [-.02, -.01, .01, .02] for y in [-.02, -.01, .01, .02]],
              'coverage': {'camera_ids': ['3', '2'], 'sam3_camera_ids': ['3', '2']},
              'observation_provenance': {'observation_id': 'now'}}
    container = {'id': 'bin', 'kind': 'bin', 'color': 'blue',
                 'points_world_m': [[x, y, z] for x in [-.08, .08] for y in [-.08, .08] for z in [.61, .64, .67, .70]],
                 'observation_provenance': {'observation_id': 'now'},
                 'placement_reference': {'floor_z_m': .61, 'observation_id': 'now',
                     'region_polygon_world_xy_m': [[-.08, -.08], [.08, -.08], [.08, .08], [-.08, .08]]}}
    return {'source': 'rgbd', 'observation_id': 'now', 'objects': [source, container],
            'table': {'surface_height_m': .60}}


def evidence(observation):
    return initial_view_evidence(observation, 'grasp', {'kind': 'cube', 'color': 'red'}, {})


def test_measured_bin_source_accepts_head_and_one_wrist():
    result = evidence(measured_bin_source())
    assert result['views_ready'] is True
    assert result['source_views'][0]['policy'] == 'current_measured_geometry'


@pytest.mark.parametrize('missing', ['wrist', 'bin', 'floor', 'current_bin', 'current_floor', 'bin_points', 'containment'])
def test_target_geometry_readiness_does_not_depend_on_container_completeness(missing):
    observation = measured_bin_source()
    source, container = observation['objects']
    if missing == 'wrist':
        source['coverage']['camera_ids'] = ['3']
    elif missing == 'bin':
        observation['objects'].remove(container)
    elif missing == 'floor':
        container.pop('placement_reference')
    elif missing == 'current_bin':
        container['observation_provenance']['observation_id'] = 'old'
    elif missing == 'current_floor':
        container['placement_reference']['observation_id'] = 'old'
    elif missing == 'bin_points':
        container.pop('points_world_m')
    else:
        source['points_world_m'] = (np.array(source['points_world_m']) + [.2, 0, 0]).tolist()
    assert evidence(observation)['views_ready'] is True


def test_ordinary_table_source_accepts_available_measured_views():
    observation = measured_bin_source()
    observation['objects'] = observation['objects'][:1]
    assert evidence(observation)['views_ready'] is True
    observation['objects'][0]['coverage']['camera_ids'].append('1')
    assert evidence(observation)['views_ready'] is True


@pytest.mark.parametrize('include_containers,expected', [
    (False, ['red cube', 'cube', 'sphere', 'cylinder', 'object', 'block']),
    (True, ['red cube', 'cube', 'sphere', 'cylinder', 'object', 'open box', 'block'])])
def test_explicit_recovery_observation_refreshes_container_after_grasp(include_containers, expected):
    capture, masks = capture_fixture()
    detector = PromptSegmenter({'red cube': masks})
    hint = {'kind': 'cube', 'color': 'red', 'camera_ids': ['3'], 'include_containers': include_containers,
            'destination': {'kind': 'bin', 'color': 'blue'}}
    GeometryPerception(detector).observe(copy.deepcopy(capture), checkpoint='post_grasp', source_hint=hint)
    assert detector.calls == ['red cube']  # Fixed inventory is not expanded by a legacy flag.
