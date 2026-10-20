from copy import deepcopy
from types import SimpleNamespace

import pytest

from jev_mujoco.algorithms import Algorithms
from jev_mujoco.geometry_service import GeometryRuntime, run_operation


def test_arm_prefilter_reads_both_measured_centers_without_gpu():
    calls = []

    def lookup(arm, positions):
        calls.append((arm, deepcopy(positions)))
        return [
            {'reachable': True, 'position_world_m': positions[0]},
            {'reachable': arm == 'A', 'position_world_m': positions[1]}]

    def reject_network(*args):
        pytest.fail('Arm workspace selection must not request a GPU service')

    algorithm = Algorithms(workspace_lookup=lookup, curobo_request=reject_network,
                           graspgen_client=object(), perception=object())
    observation = {'objects': [{'id': 'source', 'position': [.5, .35, .65]},
                               {'id': 'destination', 'position': [.5, 0, .6]}]}
    result = algorithm.evaluate_reachability(observation, 'source', 'destination')
    assert [arm for arm, _ in calls] == ['A', 'B']
    assert all(positions == [[.5, .35, .65], [.5, 0, .6]] for _, positions in calls)
    assert result['A']['reachable'] and not result['B']['reachable']
    assert all(not item['collision_checked'] for item in result.values())
    assert all(item['feasibility_scope'] == 'offline_position_workspace' for item in result.values())


def test_gpu_reachability_is_scene_independent_and_reuses_solver(monkeypatch):
    from jev_mujoco import reachability

    created, queries = [], []

    def factory(*args, **kwargs):
        created.append((args, kwargs))
        return SimpleNamespace(solve=lambda positions: queries.append(positions) or [{'reachable': True}] * len(positions))

    monkeypatch.setattr(reachability, 'PositionReachability', factory)
    runtime = GeometryRuntime()
    payload = {'arm': 'A', 'positions_world_m': [[.5, .35, .65], [.5, 0, .6]]}
    for _ in range(2):
        result = run_operation(runtime, '/tools/evaluate_reachability', payload)
        assert result['arm'] == 'A' and len(result['items']) == 2
    assert len(created) == 1 and len(queries) == 2
    assert runtime.geometry_scene is None


@pytest.mark.parametrize('arm,positions', [('C', [[0, 0, 0]]), ('A', []),
    ('A', [[0, 0, float('nan')]]), ('A', [[0, 0, 0]] * 3)])
def test_invalid_center_queries_are_rejected_before_gpu_allocation(arm, positions):
    with pytest.raises(ValueError, match='reachability'):
        run_operation(GeometryRuntime(), '/tools/evaluate_reachability',
                      {'arm': arm, 'positions_world_m': positions})


@pytest.mark.parametrize('position', [None, [float('nan'), 0, 0]])
def test_missing_measured_center_has_explicit_error_without_gpu_call(position):
    calls = []
    algorithm = Algorithms(curobo_request=lambda *args: calls.append(args),
                           graspgen_client=object(), perception=object())
    with pytest.raises(ValueError, match='reachability_requires_measured_center:source'):
        algorithm.evaluate_reachability({'objects': [{'id': 'source', 'position': position}]}, 'source')
    assert calls == []
