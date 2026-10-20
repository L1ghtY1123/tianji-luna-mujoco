import numpy as np
import pytest

from jev_mujoco.algorithms import AlgorithmError, Algorithms


def test_semantic_map_cannot_omit_an_unobserved_fixed_obstacle():
    from test_rgbd_algorithms import observed_scene

    observation = observed_scene() | {'observation_pose_verified': True,
        'mapping_unresolved': [{'status': 'unseen', 'previous_id': 'other'}]}
    calls = []
    algorithms = Algorithms(curobo_request=lambda *args: calls.append(args) or {})
    with pytest.raises(AlgorithmError, match='semantic_scene_identity_unresolved'):
        algorithms.build_scene(observation, 'seen-red')
    assert not calls


def test_target_cannot_stand_in_for_declared_same_color_environment_obstacle():
    from test_rgbd_algorithms import observed_scene

    observation = observed_scene() | {'observation_pose_verified': True, 'task_member_ids': ['seen-red'],
        'declared_environment': [{'color': 'red', 'observed_ids': ['seen-red'], 'status': 'observed'}]}
    with pytest.raises(AlgorithmError, match='declared_obstacle_not_observed'):
        Algorithms(curobo_request=lambda *args: {}).build_scene(observation, 'seen-red')


def test_combined_target_quantity_is_offered_despite_environment_quantity():
    from jev_mujoco.task_request import quantity_candidates

    assert 5 in quantity_candidates('桌面有四个黑色障碍，把两个红色和三个绿色物体搬到盒子里')


def test_map_radius_bound_covers_rotated_attachment_fits_and_uncertainty():
    from test_rgbd_algorithms import observed_scene

    observation = observed_scene()
    target = observation['objects'][0]
    target['uncertainty_m'] = .02
    sent = []
    Algorithms(curobo_request=lambda path, body: sent.append(body) or {}).build_scene(observation, 'seen-red')
    # Fitting may choose different groups after tool rotation; a group's mean
    # is inside its convex hull, so its maximum distance is bounded by diameter.
    diameter_bound = np.linalg.norm(np.ptp(target['points_world_m'], axis=0))
    assert sent[0]['mapping_max_sphere_radius_m'] >= diameter_bound + .02 + .001 - 1e-8

