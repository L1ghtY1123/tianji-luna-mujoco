import copy

import pytest
from test_runner import Decisions, Sim
from test_visible_workflow import VisibleAlgorithms

from luna_mujoco.runner import Runner


class RequestDecisions(Decisions):
    async def bind_request(self, instruction):
        return {'task': 'grasp', 'kind': 'any', 'color': 'black', 'position': 'any',
                'arm': 'A', 'scope': 'single', 'visual_semantics': 'optional',
                'destination_kind': 'any', 'destination_color': 'any', 'destination_position': 'any'}




async def test_removed_recovery_dispatch_preserves_held_object(tmp_path):
    runner = Runner(Sim(), VisibleAlgorithms(), None, tmp_path)
    context = {'holding': {'attached': True, 'verified': True, 'arm': 'A'}}
    before = copy.deepcopy(context)
    with pytest.raises(ValueError, match='unknown_skill:recover_task'):
        await runner.execute('recover_task', context, {'action': 'defer'})
    assert context == before
    assert runner.simulation.frame == 0


def test_failure_recording_preserves_unknown_identity_without_selecting_recovery(tmp_path):
    from luna_mujoco.task_progress import TaskProgress
    runner = Runner(Sim(), VisibleAlgorithms(), None, tmp_path)
    runner.task_progress = TaskProgress('grasp selected object', 'single', 'A')
    evidence = {'details': {'status': 'unknown', 'observation_id': 'new-frame'}}
    runner.record_failure({}, 'observe', 'instance association ambiguous', evidence)
    assert runner.task_progress.failure['evidence']['status'] == 'unknown'
    assert runner.state['status'] == 'idle'


@pytest.mark.parametrize('action', ['retry_place', 'retry_grasp', 'retry_plan', 'stop', 'reobserve'])
async def test_removed_recovery_actions_cannot_mutate_state(tmp_path, action):
    runner = Runner(Sim(), VisibleAlgorithms(), None, tmp_path)
    state = copy.deepcopy(runner.state)
    with pytest.raises(ValueError, match='unknown_skill:recover_task'):
        await runner.execute('recover_task', {}, {'action': action})
    assert runner.state == state
    assert runner.simulation.frame == 0
