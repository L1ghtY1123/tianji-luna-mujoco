"""Public VLA prerequisites inform the graph without changing ordinary placement."""
import asyncio
import json

import pytest
from test_policy_entity_ledger import ledger
from test_policy_integration import FixedInventoryVision
from test_policy_task_geometry import controller_for

from luna_mujoco.demo_scene import demo_region
from luna_mujoco.planning_settings import PlanningSettings
from luna_mujoco.robot_skills.operations import configured_destination
from luna_mujoco.vla_configuration import vla_unpack_configuration


def test_controller_does_not_read_simulation_cad_catalog(tmp_path, monkeypatch):
    def forbidden_catalog():
        raise AssertionError('simulation CAD must not become planning geometry knowledge')

    monkeypatch.setattr('luna_mujoco.cad_assets.cad_catalog', forbidden_catalog)
    controller, _, _ = controller_for(tmp_path, FixedInventoryVision(count=1))
    assert controller.compiler.known_assets == {}


async def test_bootstrap_exposes_current_vla_input_without_rewriting_task_graph(tmp_path):
    controller, http, _ = controller_for(tmp_path, FixedInventoryVision(count=1))
    controller.planning_settings = PlanningSettings(vla_unpack={'input_x_m': .48, 'input_y_m': .16})
    ordinary_region = dict(controller.region)
    await controller.start('Move the red items to the tray')
    await asyncio.wait_for(controller.worker, 10)
    assert controller.state['status'] == 'succeeded', controller.state.get('error')
    prompt = json.loads(http.requests[0][1]['input'][0]['content'][0]['text'])
    capability = vla_unpack_configuration(controller.planning_settings)
    assert prompt['scene_context']['capabilities']['vla_unpack'] == capability
    assert prompt['configured_regions']['vla-unpack-input']['region']['center_world_xy_m'] == [.48, .16]
    region_description = prompt['configured_regions']['vla-unpack-input']['description']
    assert 'choose any feasible point' not in region_description
    assert 'configured_xy' in region_description
    assert controller.region == ordinary_region
    assert controller.state['task_spec']['destination_ref'] == 'demo-destination'
    assert controller.state['task_spec']['processing'] == []
    assert controller.compiler.budget.used == 1


@pytest.mark.parametrize('precise', [False, True])
def test_only_configured_precise_region_fixes_placement_reference(tmp_path, precise):
    controller, _, _ = controller_for(tmp_path, FixedInventoryVision(count=1))
    controller.task_progress = ledger()
    controller.task_progress.current_id = 'item'
    identifier = 'vla-unpack-input' if precise else 'demo-destination'
    region = demo_region()
    if precise:
        region = vla_unpack_configuration()['input_region'] | {'precise_xy': True, 'xy_tolerance_m': .002}
    observation = {'observation_id': 'measured', 'table': {'surface_height_m': .1}}
    destination = configured_destination(observation, region, identifier=identifier)
    observation['objects'] = [destination]
    controller.state['destination_id'] = identifier
    request = controller._bind_region(observation)
    assert request['free_placement'] is not precise
    assert request.get('precise_xy', False) is precise
    assert controller.task_progress.placement_request(identifier) == request
    assert request['layout']['center_world_xy_m'] == pytest.approx(region['center_world_xy_m'])
    if precise:
        assert destination['placement_reference']['precise_xy'] is True
        assert request['xy_tolerance_m'] == .002
        assert request['selected_reference_world_xy_m'] == region['center_world_xy_m']


@pytest.mark.parametrize('mode', ['free_region', 'configured_xy'])
async def test_model_declared_configured_xy_mode_binds_the_workbench_point(tmp_path, mode):
    from types import SimpleNamespace
    controller, _, _ = controller_for(tmp_path, FixedInventoryVision(count=1))
    controller.task_progress = ledger()
    controller.task_progress.current_id = 'item'
    controller.state['target_id'] = 'item'
    controller.task_spec = SimpleNamespace(constraints=[])
    controller.active_object_task = None
    observation = {'observation_id': 'current', 'table': {'surface_height_m': .6}}
    destination = configured_destination(observation, demo_region(), identifier='area')
    observation['objects'] = [destination]
    await controller.execute('identify_destination', {'observation': observation}, {
        'destination_id': 'area', 'destination_kind': 'support_region', 'placement_mode': mode})
    assert destination['placement_reference'].get('fixed_xy', False) is (mode == 'configured_xy')
    assert controller.state['destination_id'] == 'area'
    request = controller.task_progress.placement_request('area')
    assert request['free_placement'] is (mode == 'free_region')
    if mode == 'configured_xy':
        assert request['selected_reference_world_xy_m'] == demo_region()['center_world_xy_m']
