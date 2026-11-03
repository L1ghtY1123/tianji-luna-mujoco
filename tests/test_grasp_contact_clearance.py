"""Only the selected target's gripping pads may enter its clearance shell."""
from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.grasp_contact import combine_grasp_costs, target_disabled_for_environment


def test_closing_contact_keeps_target_clearance_for_every_nonpad_link():
    from luna_mujoco.grasp_contact import combine_closure_costs

    full = np.array([[[.02, .03, .04, .05]]])
    environment = np.array([[[0., .001, 0., 0.]]])
    links = {0: 'gripL_Link3_L', 1: 'gripL_Link3_R',
             2: 'gripL_Link4_L', 3: 'gripR_Link3_L'}
    result = combine_closure_costs(full, environment, links, 'A')
    np.testing.assert_array_equal(result, [[[0., .001, .04, .05]]])


def test_clearance_shell_is_not_penetration_for_intended_gripping_pads():
    # Frozen run18: target clearance cost .0024934; physical and environment 0.
    full = np.array([[[.0024934, .003, .004]]])
    actual = combine_grasp_costs(full, np.zeros_like(full), np.zeros_like(full),
        {0: 'gripL_Link3_R', 1: 'gripL_Base', 2: 'gripR_Link3_L'}, 'A')
    np.testing.assert_array_equal(actual, [[[0., .003, .004]]])
    assert full[0, 0, 0] == .0024934


@pytest.mark.parametrize('environment,penetration,expected', [(.001, 0, .001), (0, .002, .002), (.001,.002,.002)])
def test_finger_still_rejects_other_obstacle_clearance_and_target_penetration(environment, penetration, expected):
    actual = combine_grasp_costs(np.array([[[.003]]]), np.array([[[environment]]]),
        np.array([[[penetration]]]), {0: 'gripR_Link3_L'}, 'B')
    assert actual[0, 0, 0] == expected


@pytest.mark.parametrize('invalid', [np.array([[[float('nan')]]]), np.zeros((1,2,1))])
def test_invalid_costs_cannot_grant_contact_permission(invalid):
    with pytest.raises(ValueError):
        combine_grasp_costs(np.zeros((1,1,1)), invalid, np.zeros((1,1,1)), {0:'gripL_Link3_L'}, 'A')


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('kind,buffer', [('voxel', 'voxels'), ('mesh', 'meshes'), ('cuboid', 'cuboids')])
def test_environment_query_changes_only_selected_target_and_restores_on_failure(fail, kind, buffer):
    states = {'semantic_instance::target':True, 'semantic_instance::other':True}
    def set_enabled(name, *, enabled, env_idx):
        assert env_idx == 0
        states[name] = enabled
    runtime = SimpleNamespace(geometry_scene={'target_id':'target','exclusion_state':'selected'},
        geometry_model={kind:{name:{} for name in states}})
    buffers = SimpleNamespace(**{buffer: SimpleNamespace(set_enabled=set_enabled)})
    planner = SimpleNamespace(scene_collision_checker=SimpleNamespace(data=buffers))
    try:
        with target_disabled_for_environment(runtime, planner):
            assert states == {'semantic_instance::target':False,'semantic_instance::other':True}
            if fail: raise RuntimeError('GPU query failed')
    except RuntimeError:
        assert fail
    assert all(states.values())
