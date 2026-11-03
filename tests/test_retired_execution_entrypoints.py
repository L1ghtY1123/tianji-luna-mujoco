"""Retired task policies must fail before observation or physical execution."""
import copy

import pytest
from test_runner import Algorithms, Sim

from luna_mujoco.app import create_app
from luna_mujoco.robot_skills.operations import RobotOperations
from luna_mujoco.runner import Runner


@pytest.mark.parametrize("runner_type", [Runner, RobotOperations])
@pytest.mark.parametrize("skill,arguments", [
    ("recover_task", {"action": "reobserve"}),
    ("assess_task", {"judgment": "complete"}),
    ("inspect_scene", {}),
    ("evaluate_reachability", {}),
    ("mock_unpack", {}),
])
async def test_retired_dispatch_rejected_without_state_change(tmp_path, runner_type, skill, arguments):
    runner = runner_type(Sim(), Algorithms(), None, tmp_path)
    context = {}
    state = copy.deepcopy(runner.state)
    with pytest.raises(ValueError, match="unknown_skill:" + skill):
        await runner.execute(skill, context, arguments)
    assert runner.state == state
    assert context == {}
    assert runner.simulation.frame == 0


@pytest.mark.parametrize("method", ["recover_task", "invalidate_recovery_scene", "prepare_grasp_scene",
                                    "inspect_scene", "snapshot_for_decision"])
def test_retired_orchestration_methods_are_absent(method):
    assert not hasattr(Runner, method)
    assert not hasattr(RobotOperations, method)


@pytest.mark.parametrize("mode", [True, False])
def test_removed_app_mode_is_not_silently_accepted(mode):
    with pytest.raises(TypeError, match="demo_mode"):
        create_app(demo_mode=mode)


def test_retired_decisions_factory_is_rejected_before_app_initialization():
    with pytest.raises(ValueError, match="legacy_decisions_factory_not_supported"):
        create_app(decisions_factory=lambda record: None)


@pytest.mark.parametrize("runner_type", [Runner, RobotOperations])
def test_old_decisions_factory_is_rejected_by_python_composition(tmp_path, runner_type):
    with pytest.raises(ValueError, match="legacy_decisions_factory_not_supported"):
        runner_type(Sim(), Algorithms(), lambda record: None, tmp_path)


async def test_algorithm_error_archive_retains_visual_evidence_and_queries(tmp_path):
    from types import SimpleNamespace
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    runner.experiment = SimpleNamespace(event=lambda *args: None)
    error = ValueError('visual identity unknown')
    error.evidence = {'observation_id': 'current', 'status': 'unknown'}
    error.sam3_queries = [{'prompt': 'package', 'status': 'empty'}]
    def fail():
        raise error
    with pytest.raises(ValueError, match='visual identity unknown'):
        await runner.call_algorithm(fail)
    event = next(event for event in runner.events if event['kind'] == 'algorithm_error')
    assert event['data']['details'] == error.evidence
    assert event['data']['sam3_queries'] == error.sam3_queries


async def test_explicit_grasp_retry_does_not_require_retired_recovery_authorization(tmp_path):
    runner = Runner(Sim(), Algorithms(), None, tmp_path)
    calls = []
    async def plan(snapshot, target, arm, context):
        calls.append((target, arm))
        if len(calls) == 1:
            raise ValueError('no_plan')
        return {'success': True}
    runner.plan_grasp_candidates = plan
    with pytest.raises(ValueError, match='no_plan'):
        await runner.plan_grasp({}, 'target', 'A', {})
    assert calls == [('target', 'A')]
    assert (await runner.plan_grasp({}, 'target', 'A', {}))['success']
    assert calls == [('target', 'A'), ('target', 'A')]
    assert [attempt['status'] for attempt in runner.state['planning_attempts']] == ['failed', 'succeeded']
