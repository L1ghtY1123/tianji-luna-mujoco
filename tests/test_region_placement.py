"""Release poses use measured regions; collision checks use the three-dimensional scene."""

from copy import deepcopy

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


def test_marker_release_accepts_measured_region_and_uses_the_grasp_support_height():
    target, destination, holding, _ = region_inputs("marker")
    holding["grasp_support_reference"].update(support_height_m=.65, source="measured_bin_floor")
    result = placement.release_goal(target, destination, holding)
    assert result["T_world_tcp"][2][3] == pytest.approx(.78)
    assert result["release_reference"]["support_height_m"] == .65


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


@pytest.mark.parametrize("kind", ["bin", "marker"])
def test_region_generates_eighteen_fixed_tcp_poses(kind):
    target, destination, holding, spheres = region_inputs(kind)
    result = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01,
        placement_request={'target_count': 3})
    locations = {tuple(np.round(np.asarray(item['T_world_tcp'])[:2, 3], 6)) for item in result}
    assert len(locations) == 1
    assert len(result) == 18
    assert [item["candidate_index"] for item in result] == list(range(len(result)))
    for candidate in result:
        tcp = np.asarray(candidate["T_world_tcp"])
        np.testing.assert_allclose((tcp @ [0., 0., .185, 1.])[:2], candidate['release_reference']['position_world_m'][:2])


@pytest.mark.parametrize('kind', ['bin', 'marker'])
def test_region_obstacles_are_checked_by_three_dimensional_planner(kind):
    target, destination, holding, spheres = region_inputs(kind)
    obstacle = [[.7, -.14, .70]]
    observation = {"objects": [destination, {"id": "pile", "points_world_m": obstacle}]}
    before = deepcopy(holding)
    goals = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres,
        clearance_m=.01, observation=observation)
    assert len(goals) == 18
    assert all(goal['release_reference']['collision_policy'] == 'three_dimensional_attachment_and_scene'
               for goal in goals)
    assert holding == before

def test_bin_rim_is_diagnostic_and_never_raises_release():
    target, destination, holding, spheres = region_inputs()
    rim = np.array([[x, -.32, .70] for x in np.linspace(.58, .82, 80)])
    destination['points_world_m'] = np.vstack((destination['points_world_m'], rim))
    goals = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres,
                                          clearance_m=.01, map_resolution_m=.01)
    for goal in goals:
        tcp = np.asarray(goal['T_world_tcp'])
        bottom = (np.asarray(spheres)[0, :3] @ tcp[:3, :3].T + tcp[:3, 3])[2] - spheres[0][3]
        assert goal['release_reference']['release_height_m'] == pytest.approx(.012)
        assert bottom < .70
        assert goal['release_reference']['bin_rim_source'] == 'current_measured_bin_points'


def test_marker_boundary_does_not_constrain_attachment_projection():
    target, destination, holding, spheres = region_inputs("marker")
    spheres[0][3] = .15
    goals = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01)
    assert len(goals) == 18

def test_voxel_size_does_not_change_fixed_tcp_xy_or_yaw_count():
    target, destination, holding, spheres = region_inputs("marker")
    spheres[0][3] = .15
    goals = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres,
        clearance_m=.01, map_resolution_m=.01, placement_request={'target_count': 2})
    assert len(goals) == 18
    assert all(goal['release_reference']['region_constraint'] == 'tcp_world_xy_at_assigned_slot'
               for goal in goals)

def test_partial_visible_bin_floor_limits_reference_points_not_attachment_envelope():
    target, destination, holding, spheres = region_inputs()
    floor = np.array([[x, y, .615] for x in np.linspace(.69, .73, 15) for y in np.linspace(-.25, -.15, 25)])
    destination["points_world_m"] = floor
    destination["placement_reference"]["region_polygon_world_xy_m"] = [
        [.69, -.25], [.73, -.25], [.73, -.15], [.69, -.15]]
    result = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres,
        clearance_m=.01, map_resolution_m=.01)
    assert result
    for candidate in result:
        reference = candidate["release_reference"]
        np.testing.assert_allclose(reference['position_world_m'], [.71, -.175, .615])
        assert reference['position_source'] == 'centroids_of_measured_support_polygon_partitions'
        assert reference["region_constraint"] == "tcp_world_xy_at_assigned_slot"
        assert reference["required_clearance_m"] == .01
        assert reference["map_resolution_allowance_m"] == .01


def test_measured_rim_does_not_filter_yaw_candidates_in_xy():
    target, destination, holding, spheres = region_inputs()
    rim = np.array([[.734, y, .70] for y in np.linspace(-.3, -.1, 80)])
    destination["points_world_m"] = np.vstack((destination["points_world_m"], rim))
    result = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres,
        clearance_m=.01, map_resolution_m=.01, observation={'objects': [destination]})
    assert len(result) == 18

def test_one_object_has_two_measured_half_region_centers():
    _, destination, _, _ = region_inputs()
    floor = np.array([[x, y, .615] for x in np.linspace(.60, .80, 21) for y in np.linspace(-.3, -.1, 21)])
    wall = np.array([[.80, y, .70] for y in np.linspace(-.3, -.1, 41)])
    destination["points_world_m"] = np.vstack((floor, wall))
    destination["placement_reference"].update(position_world_m=[.78, -.2, .615],
        region_polygon_world_xy_m=[[.6, -.3], [.8, -.3], [.8, -.1], [.6, -.1]])
    positions = placement._region_floor_positions(destination)
    np.testing.assert_allclose(positions, [[.7, -.15, .615], [.7, -.25, .615]])


@pytest.mark.parametrize('kind', ['bin', 'marker'])
def test_obstacle_does_not_shift_assigned_tcp_or_raise_release(kind):
    target, destination, holding, spheres = region_inputs(kind)
    options = {'spheres_tool_m': spheres, 'clearance_m': .01, 'placement_request': {'target_count': 2}}
    before = placement.release_candidates(target, destination, holding, **options)
    result = placement.release_candidates(target, destination, holding, **options,
        observation={"objects": [destination, {"id": "obstacle", "points_world_m": [[.7, -.14, .70]]}]})
    assert [goal['T_world_tcp'] for goal in result] == [goal['T_world_tcp'] for goal in before]

def test_candidate_generation_does_not_use_unsegmented_depth_as_xy_occupancy():
    target, destination, holding, spheres = region_inputs("marker")
    goals = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres,
        clearance_m=.01, observation={"objects": [destination], "views": {"3": {"depth_m": "unused"}}})
    assert len(goals) == 18

def test_single_measured_reference_also_generates_all_eighteen_yaws():
    target, destination, holding, spheres = region_inputs()
    destination['placement_reference'].pop('region_polygon_world_xy_m')
    goals = placement.release_candidates(target, destination, holding, spheres_tool_m=spheres, clearance_m=.01,
        observation={'objects': [destination, {'id': 'placed', 'points_world_m': [[.7, -.2, .68]]}]})
    assert len(goals) == 18

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
