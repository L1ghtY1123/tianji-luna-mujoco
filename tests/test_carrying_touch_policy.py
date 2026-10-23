"""Owned hand contact changes only attachment pairs, not robot/world geometry."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from test_carrying import binding

from luna_mujoco.carrying import CarryingBinding, create_carrying_robot_config


@pytest.mark.parametrize('arm,side', [('A','L'),('B','R')])
def test_explicit_owned_assembly_only_preserves_every_other_pair_and_world_sphere(arm, side):
    root=Path(__file__).resolve().parents[1]
    robot=yaml.safe_load((root/'resources/curobo/marvin_gui_dual.yml').read_bytes())
    # An unreviewed future prefix match must not silently become a touch link.
    robot['robot_cfg']['kinematics']['collision_link_names'].append(f'grip{side}_new_camera')
    original=deepcopy(robot)
    value=CarryingBinding.from_mapping(binding(arm=arm),scene={
        'scene_token':'scene-2','observation_id':'rgbd-2','generation':2,'target_id':'sam-source'},arm=arm)
    result,link=create_carrying_robot_config(robot,value)
    before=original['robot_cfg']['kinematics'];after=result['robot_cfg']['kinematics']
    expected={f'grip{side}_Base'}|{f'grip{side}_Link{n}_{s}' for n in range(1,5) for s in ('L','R')}
    assert set(after['self_collision_ignore'][link])==expected
    for name in before['collision_link_names']:
        assert (link in after['self_collision_ignore'].get(name,[])) == (name in expected)
        assert [item for item in after['self_collision_ignore'].get(name,[]) if item!=link] == before['self_collision_ignore'].get(name,[])
    assert after['collision_link_names']==[*before['collision_link_names'],link]
    assert all(after['collision_spheres'][name]==rows for name,rows in before['collision_spheres'].items())
    assert after['collision_spheres'][link]  # carried body still participates in world checking
    assert after['self_collision_buffer'][link]==0
    assert after['collision_sphere_buffer'][link]==0
    assert robot==original


def test_wrong_arm_cannot_authorize_a_touch_assembly():
    scene={'scene_token':'scene-2','observation_id':'rgbd-2','generation':2,'target_id':'sam-source'}
    with pytest.raises(ValueError,match='arm'):
        CarryingBinding.from_mapping(binding(arm='B'),scene=scene,arm='A')
