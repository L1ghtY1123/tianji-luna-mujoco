from copy import deepcopy

import pytest
from test_place_yaw_selection import setup_plan

import jev_mujoco.algorithms as module


@pytest.mark.parametrize('first_feasible', [True, False])
def test_region_candidates_use_sixteen_pose_batches_and_stop_after_full_plan(monkeypatch, first_feasible):
    algorithms, args, requests, _, _ = setup_plan(monkeypatch)
    original = module.release_candidate_groups

    def region(*values, **kwargs):
        goals = next(original(*values, **kwargs))
        yield goals
        assert len(requests) == 3
        second = deepcopy(goals)
        for index, goal in enumerate(second, 18):
            goal['candidate_index'] = index
            goal['T_world_tcp'][0][3] += .03
        yield second
        raise AssertionError('must stop after successful position')

    monkeypatch.setattr(module, 'release_candidate_groups', region)
    request = algorithms.request

    def bounded(path, body):
        assert len(body['goal_matrices_world']) == (18 if path == '/tools/solve_ik_batch' else 16)
        result = request(path, body)
        if path == '/tools/plan_place_batch' and not first_feasible and len(requests) <= 3:
            for item in result['items']:
                item['success'] = False
        return result

    algorithms.request = bounded
    result = algorithms.plan_place(*args)
    assert len(requests) == (2 if first_feasible else 5)
    assert result['goal']['candidate_index'] == (0 if first_feasible else 18)
    assert result['candidate_count'] == (18 if first_feasible else 36)
