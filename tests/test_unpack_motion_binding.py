from types import SimpleNamespace

import numpy as np
import pytest

from luna_mujoco.demo_unpack_workflow import UnpackWorkflow


class MotionHarness(UnpackWorkflow):
    def __init__(self):
        self.state = {'target_id': 'item', 'arm': 'B', 'facts': {}}
        self.simulation = SimpleNamespace(execute_trajectory=lambda: None)

    def current_lifecycle(self):
        return {'phase': 'black_held'}

    async def verify_scene(self, context):
        pass

    def prepare_command(self, command):
        return command

    async def validate_motion(self, *args, **kwargs):
        return {'collision_free': True}

    async def execution_stage(self, *args):
        return {'completed': True}

    async def motor(self, *args):
        pass

    def measured_state(self, context):
        raise AssertionError('a changed plan must be rejected before reading feedback or opening')


async def test_replanning_after_transport_cannot_reuse_prior_execution_to_drop():
    runner = MotionHarness()
    context = {'planning_settings': {}, 'holding': {'points_tool_m': [[0, 0, 0]]},
        'transport_plan': {'goal': np.eye(4).tolist(), 'planning_settings': {},
        'carrying_binding': {'points_tool_m': [[0, 0, 0]]},
        'command': {'positions_rad': [[0] * 7, [.1] * 7], 'dt': .1}}}
    await runner._execute_transport(context)
    context['transport_plan']['goal'][0][3] += .02
    with pytest.raises(ValueError, match='transport_execution_plan_changed'):
        await runner._drop(context)
