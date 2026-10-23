"""Application contract for fixed workspace lookup without GPU requests."""
import pytest

from luna_mujoco.algorithms import AlgorithmError, Algorithms


def test_requested_arm_and_single_target_do_not_query_other_arm_or_destination():
    calls = []
    def lookup(arm, positions):
        calls.append((arm, positions))
        return [{'reachable': True, 'position_world_m': positions[0]}]
    algorithm = Algorithms(workspace_lookup=lookup, graspgen_client=object(), perception=object())
    result = algorithm.evaluate_reachability({'objects': [{'id': 'source', 'position': [.5, .2, .6]}]},
                                             'source', requested_arm='A')
    assert list(result) == ['A'] and result['A']['destination'] is None
    assert len(calls) == 1 and calls[0][0] == 'A' and len(calls[0][1]) == 1


@pytest.mark.parametrize('response', [[], [{'reachable': True, 'position_world_m': [9, 9, 9]}], [None]])
def test_wrong_arm_or_point_response_cannot_select_an_arm(response):
    algorithm = Algorithms(workspace_lookup=lambda *args: response, graspgen_client=object(), perception=object())
    with pytest.raises(AlgorithmError, match='invalid workspace lookup'):
        algorithm.evaluate_reachability({'objects': [{'id': 'source', 'position': [.5, .2, .6]}]},
                                       'source', requested_arm='A')
