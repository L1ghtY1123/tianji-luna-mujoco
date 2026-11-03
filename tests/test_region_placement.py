"""Release poses use measured regions; collision checks use the three-dimensional scene."""


import numpy as np
import pytest
from test_visible_placement import measured_inputs

from luna_mujoco import placement
from luna_mujoco.perception import measured_bin_reference


def region_inputs(kind="bin"):
    target, destination, holding = measured_inputs()
    destination["kind"] = kind
    destination["placement_reference"]["region_polygon_world_xy_m"] = [
        [.58, -.32], [.82, -.32], [.82, -.08], [.58, -.08]]
    points = np.array([[x, y, .615] for x in np.linspace(.58, .82, 25) for y in np.linspace(-.32, -.08, 25)])
    destination["points_world_m"] = points
    destination["placement_reference"]["measurement_sources"] = {"point_indices": list(range(len(points)))}
    holding["instance_id"] = "source"
    spheres = [[.0275, -.03, -.17, .025]]
    return target, destination, holding, spheres


def test_measured_bin_floor_exposes_a_supported_inner_rectangle():
    floor = np.array([[x, y, .615] for x in np.linspace(.6, .8, 30) for y in np.linspace(-.3, -.1, 30)])
    wall = np.array([[.6, y, z] for y in np.linspace(-.3, -.1, 30) for z in np.linspace(.62, .7, 20)])
    points = np.vstack((floor, wall))
    reference = measured_bin_reference(points, {"surface_height_m": .6, "uncertainty_m": .003})
    polygon = np.asarray(reference["region_polygon_world_xy_m"])
    assert .6 <= polygon[:, 0].min() < .64  # Unsupported wall-adjacent rows are excluded.
    assert polygon[:, 0].max() <= .8
    np.testing.assert_allclose(polygon[:, 1].min(), -.3)
    np.testing.assert_allclose(polygon[:, 1].max(), -.1)
    assert reference['region_measurement']['method'] == 'plane_hull_inscribed_rectangle'


def test_grasp_support_uses_containing_measured_bin_floor_and_records_visible_lowest_point():
    target, destination, holding, _ = region_inputs()
    target["points_world_m"] = np.array([[.68 + x, -.2 + y, .665] for x in [.0, .01] for y in [.0, .01]])
    target["visual_semantics"] = {"inside_id": "destination", "ambiguous": False}
    observation = {"table": {"surface_height_m": .6}, "objects": [target, destination]}
    assert hasattr(placement, "grasp_support_reference"), "Measured grasp support helper missing"
    result = placement.grasp_support_reference(observation, "source", holding["grasp_support_reference"]["T_world_tcp_at_grasp"])
    assert result["support_height_m"] == .615
    assert result["support_container_id"] == "destination"
    assert result["visible_source_lowest_z_m"] == .665
    assert "bottom" not in result


def test_explicit_bin_membership_without_observed_floor_does_not_fall_back_to_table():
    target, destination, holding, _ = region_inputs()
    target["visual_semantics"] = {"inside_id": "destination", "ambiguous": False}
    destination.pop("placement_reference")
    assert hasattr(placement, "grasp_support_reference"), "Measured grasp support helper missing"
    with pytest.raises(ValueError, match="source_support_unobservable"):
        placement.grasp_support_reference({"table": {"surface_height_m": .6}, "objects": [target, destination]},
            "source", holding["grasp_support_reference"]["T_world_tcp_at_grasp"])


def support_from_current_lift():
    target, destination, holding, _ = region_inputs()
    target['points_world_m'] = [[.55, -.2, .62], [.56, -.2, .63]]
    destination['placement_reference']['observation_id'] = 'view'
    observation = {'observation_id': 'view', 'generation': 2,
                   'table': {'surface_height_m': .7}, 'objects': [target, destination]}
    observation['inventory_state'] = {'objects': [{'id': 'source', 'initial_source': {
        'measurement_observation_id': 'view', 'task_source_evidence': {
            'source': 'openai_decisions', 'status': 'yes', 'target_id': 'source',
            'source_id': 'destination', 'source_kind': 'bin', 'observation_id': 'view',
            'request_hash': 'decision-proof'}}}]}
    evidence = {'target_id': 'source', 'source_container_id': 'destination', 'support_kind': 'bin',
                'observation_id': 'view', 'generation': 2, 'scene_token': 'current-map',
                'geometry_source': 'published_retained_cells_and_pregrasp_full_surface_sphere_fit'}
    return observation, holding['grasp_support_reference']['T_world_tcp_at_grasp'], evidence


def test_current_lift_source_uses_measured_floor_without_treating_visible_patch_as_whole_box():
    observation, tcp, evidence = support_from_current_lift()
    result = placement.grasp_support_reference(observation, 'source', tcp, source_reference=evidence, scene_token='current-map',
            source_record=observation['inventory_state']['objects'][0])
    assert result['support_container_id'] == 'destination'
    assert result['support_height_m'] == .615
    assert result['visible_source_lowest_z_m'] == .62
    assert result['source_binding'] == evidence


@pytest.mark.parametrize('field,value', [('target_id', 'other'), ('observation_id', 'old'),
    ('generation', 1), ('geometry_source', 'simulator_truth'), ('source_container_id', 'missing')])
def test_lift_source_reference_must_match_current_measurement(field, value):
    observation, tcp, evidence = support_from_current_lift()
    evidence[field] = value
    with pytest.raises(ValueError, match='source_support_'):
        placement.grasp_support_reference(observation, 'source', tcp, source_reference=evidence, scene_token='current-map',
            source_record=observation['inventory_state']['objects'][0])


def test_bound_source_missing_measured_floor_cannot_fall_back_to_table():
    observation, tcp, evidence = support_from_current_lift()
    observation['objects'][1].pop('placement_reference')
    with pytest.raises(ValueError, match='source_support_'):
        placement.grasp_support_reference(observation, 'source', tcp, source_reference=evidence, scene_token='current-map',
            source_record=observation['inventory_state']['objects'][0])


def test_half_overlap_uses_same_bin_floor_for_membership_and_grasp_support():
    target, destination, holding, _ = region_inputs()
    target['points_world_m'] = [[.7, -.2, .665], [.83, -.2, .665]]
    observation = {'table': {'surface_height_m': .6}, 'objects': [target, destination]}
    result = placement.grasp_support_reference(observation, 'source',
        holding['grasp_support_reference']['T_world_tcp_at_grasp'])
    assert result['support_container_id'] == destination['id']
    assert result['support_height_m'] == .615


def test_source_bin_relationship_is_partial_measured_evidence_without_hidden_membership():
    target, destination, _, _ = region_inputs()
    target["points_world_m"] = np.array([[x, -.2, .65] for x in [.69, .70, .71]])
    assert hasattr(placement, "measured_source_relationships"), "Measured relationship evidence helper missing"
    relationships = placement.measured_source_relationships(target, [target, destination])
    assert relationships == [{"relation": "above_observed_bin_floor", "object_id": "destination",
        "visible_xy_fraction": 1., "floor_z_m": .615, "visible_source_lowest_z_m": .65,
        "source": "registered_rgbd_visible_samples", 'complete_containment_verified': False}]
    target["points_world_m"][:, 0] += 1.
    assert placement.measured_source_relationships(target, [target, destination]) == []


@pytest.mark.parametrize('change', ['missing_record', 'wrong_record', 'stale_floor', 'stale_map', 'invalid_current_support', 'moved'])
def test_support_cannot_reuse_initial_relationship_after_invalid_or_changed_evidence(change):
    observation, tcp, evidence = support_from_current_lift()
    record = observation['inventory_state'].pop('objects')[0]
    if change == 'missing_record':
        record = None
    elif change == 'wrong_record':
        record['id'] = 'other'
    elif change == 'stale_floor':
        observation['objects'][1]['placement_reference']['observation_id'] = 'old'
    elif change == 'stale_map':
        evidence['scene_token'] = 'old'
    elif change == 'invalid_current_support':
        observation['inventory_state']['entities'] = {'source': {'observed_support': {
            'source': 'openai_decisions', 'observation_id': 'old', 'request_hash': 'old',
            'support_entity_id': 'destination'}}}
    elif change == 'moved':
        record['transfers'] = [{'destination_id': 'other', 'observation_id': 'later', 'release_feedback_id': 'release'}]
    with pytest.raises(ValueError, match='source_support_'):
        placement.grasp_support_reference(observation, 'source', tcp, source_reference=evidence,
            scene_token='current-map', source_record=record)
