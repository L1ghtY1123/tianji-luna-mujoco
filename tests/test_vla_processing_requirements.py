from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.robot_skills.processing import ProcessingSkills
from luna_mujoco.vla_configuration import vla_unpack_configuration


def harness(arm='B'):
    runner = ProcessingSkills()
    runner.state = {'target_id': 'pink', 'arm': arm, 'generation': 1}
    runner.task_inventory = {'work_area_id': 'work'}
    runner.task_progress = SimpleNamespace(scene_entities={'pink': {'color': 'pink', 'kind': 'object'}})
    config = vla_unpack_configuration()
    runner.simulation = SimpleNamespace(robot_snapshot=lambda: {'generation': 1, 'arms': {
        key: {'joints_rad': np.deg2rad(value).tolist()} for key, value in config['preparation_joints_deg'].items()}})
    xy = np.array(config['input_xy_m'])
    context = {'planning_settings': {}, 'destination': {'id': 'work', 'placement_reference': {
        'region_polygon_world_xy_m': (xy + np.array([[-.16, -.16], [.16, -.16], [.16, .16], [-.16, .16]])).tolist(),
        'floor_z_m': .572, 'observation_id': 'obs'}},
        'place_plan': {'goal': {'release_reference': {'object_id': 'pink', 'object_reference_world_xy_m': xy.tolist()}}},
        'release_endpoint_evidence': {'at_release_goal': True,
            'actual_object_reference_world_xy_m': xy.tolist(), 'rotation_error_rad': 0.}}
    return runner, context


def test_vla_input_accepts_either_prior_placement_arm_and_retains_preparation():
    for arm in ('A', 'B'):
        runner, context = harness(arm)
        assert runner._vla_input_requirements(context)['center_world_xy_m'] == pytest.approx(vla_unpack_configuration()['input_xy_m'])
        runner._require_vla_preparation(context)


@pytest.mark.parametrize('damage', ['wrong_xy', 'missing_object_reference', 'unverified_endpoint', 'nonpink'])
def test_vla_input_rejects_wrong_or_missing_package_reference(damage):
    runner, context = harness()
    reference = context['place_plan']['goal']['release_reference']
    if damage == 'wrong_xy':
        context['release_endpoint_evidence']['actual_object_reference_world_xy_m'][0] += .17
    elif damage == 'missing_object_reference':
        reference.clear()
    elif damage == 'unverified_endpoint':
        context['release_endpoint_evidence']['at_release_goal'] = False
    else:
        runner.task_progress.scene_entities['pink']['color'] = 'black'
    with pytest.raises(ValueError, match='vla_input'):
        runner._vla_input_requirements(context)


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_vla_preparation_uses_live_joint_feedback(arm):
    runner, context = harness()
    snapshot = runner.simulation.robot_snapshot()
    snapshot['arms'][arm]['joints_rad'][0] += .1
    runner.simulation.robot_snapshot = lambda: snapshot
    with pytest.raises(ValueError, match='vla_preparation'):
        runner._require_vla_preparation(context)


def test_processing_product_search_uses_cached_whole_table_even_without_task_support_id():
    runner, context = harness()
    context['observation'] = {'demo_tables': {'primary_table_id': 'table', 'source_observation_id': 'table-capture',
        'tables': {'table': {'source': 'initial_inventory_ransac_table', 'surface_height_m': .572,
            'model': {'pose': [.572, -.026, .532, 1, 0, 0, 0], 'dims': [.6, .8, .08]}}}}}
    support = runner._processing_support(context)
    assert support['table_id'] == 'table'
    assert np.ptp(support['region_polygon_world_xy_m'], axis=0) == pytest.approx([.6, .8])
    assert support['source'] == 'measured_cached_main_table_support'


@pytest.mark.parametrize('offset', [[.03, 0.], [-.08, .08]])
def test_vla_accepts_measured_object_within_square_away_from_center(offset):
    runner, context = harness()
    actual = np.array(vla_unpack_configuration()['input_xy_m']) + offset
    context['release_endpoint_evidence']['actual_object_reference_world_xy_m'] = actual.tolist()
    context['place_plan']['goal']['release_reference']['object_reference_world_xy_m'] = actual.tolist()
    assert runner._vla_input_requirements(context)


def test_vla_rejects_changed_release_orientation():
    runner, context = harness()
    context['release_endpoint_evidence']['rotation_error_rad'] = .2
    with pytest.raises(ValueError, match='vla_input'):
        runner._vla_input_requirements(context)


def test_map_rebuild_may_discard_plan_without_discarding_verified_release_geometry():
    runner, context = harness()
    context['verified_release_reference'] = context.pop('place_plan')['goal']['release_reference']
    assert runner._vla_input_requirements(context)
