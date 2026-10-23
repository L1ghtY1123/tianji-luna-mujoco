"""Regression cases found by the 2026-10-23 public-call/geometry audit."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from luna_mujoco.geometry_service import GeometryRuntime, run_operation
from luna_mujoco.planning_settings import PlanningSettings


@pytest.mark.parametrize('arm', ['A', 'B'])
@pytest.mark.parametrize('operation', ['solve_ik_batch', 'plan_grasp_batch'])
def test_grasp_solver_opens_only_active_hand_and_preserves_peer_feedback(arm, operation):
    from test_backend_builder import native_batch
    from test_geometry_service import request_payload

    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene'}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    runtime.effective_config = {arm: {'ik': {}, 'plan': {}}}
    captured = []

    def solve(*args, **kwargs):
        captured.append((deepcopy(runtime.gripper_feedback), runtime.carrying_binding))
        runtime.effective_config = {arm: {'ik': {}, 'plan': {}}}
        if operation == 'solve_ik_batch':
            return SimpleNamespace(to_payload=lambda: {'items': []})
        return native_batch()

    setattr(runtime, operation + '_native', solve)
    feedback = {'A': .2, 'B': .4}
    payload = request_payload(PlanningSettings()) | {
        'arm': arm, 'gripper_joint_positions_rad': feedback}
    run_operation(runtime, '/tools/' + operation, payload)
    assert captured == [({**feedback, arm: 0.}, None)]
    assert runtime.gripper_feedback is None


def test_fixed_open_command_check_keeps_nonexecuting_hand_measured(monkeypatch):
    from test_review_place_coverage_scope import command_boundary

    from luna_mujoco import carrying

    runtime, planner, payload, binding, _ = command_boundary(monkeypatch, 'approach_grasp', None)
    payload['gripper_joint_positions_rad'] = {'A': .25, 'B': .4}
    actual = []
    original = carrying.configure_gripper_feedback

    def configure(robot, feedback, path):
        actual.append(feedback)
        return original(robot, feedback, path)

    monkeypatch.setattr(carrying, 'configure_gripper_feedback', configure)
    carrying.check_command_path(runtime, planner, payload, binding)
    assert actual == [{'A': 0., 'B': .4}]


def test_valid_no_solution_does_not_discard_warmed_place_planner():
    from test_pre_release_place import request, runtime_fixture
    from tianji_curobo.contracts import CuroboToolError

    runtime, backend, _, calls = runtime_fixture()

    def rejected(_):
        raise CuroboToolError('no trajectory', code='CUROBO_GRASP_FAILED', stage='trajopt', status=422,
                             details={'phase': 'grasp'})

    backend.plan_place_native = rejected
    with pytest.raises(CuroboToolError):
        runtime.plan_pre_release_place_native(request())
    assert 'discard' not in calls
    assert not runtime.gpu_lock.locked()


@pytest.mark.parametrize('path', ['/tools/plan_place', '/tools/plan_to_pose', '/tools/plan_observation'])
def test_pose_and_place_never_guess_missing_gripper_feedback(path):
    from test_carrying import binding

    runtime = GeometryRuntime()
    runtime.geometry_scene = {'scene_token': 'scene', 'target_id': 'sam-source',
                              'observation_id': 'rgbd-2', 'generation': 2}
    runtime.geometry_solver_profile = runtime.planning_settings.solver_payload()
    payload = {'arm': 'A', 'scene_token': 'scene', 'planning_settings': PlanningSettings().model_dump()}
    if path == '/tools/plan_place':
        payload['carrying_binding'] = binding(scene_token='scene')
    with pytest.raises(ValueError, match='measured A and B gripper feedback'):
        run_operation(runtime, path, payload)


@pytest.mark.parametrize('change', [None, 'path', 'peer', 'gripper', 'scene'])
def test_reuse_only_exact_command_proof_and_recheck_changed_geometry(change):
    import hashlib
    import json

    from test_rgbd_algorithms import observed_scene

    from luna_mujoco.algorithms import Algorithms

    calls = []

    def request(path, body):
        calls.append(deepcopy(body))
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
        return {'collision_free': True, 'attached': False,
                'scene_token': body['scene_token'], 'binding_sha256': digest}

    observation = observed_scene()
    scene = {'target_id': 'seen-red', 'generation': 1, 'observation_id': 'capture-4', 'scene_token': 'map-1'}
    algorithm = Algorithms(curobo_request=request)
    command = {'positions_rad': [observation['arms']['A']['joints_rad'], [.01] * 7]}
    proof = algorithm.validate_execution(observation, 'A', scene, 'retreat', command)
    if change == 'path':
        command['positions_rad'][-1][0] += .01
    elif change == 'peer':
        observation['arms']['B']['joints_rad'][0] += .01
    elif change == 'gripper':
        observation['arms']['A']['gripper_position_rad'] += .01
    elif change == 'scene':
        scene['scene_token'] = 'map-2'
    result = algorithm.validate_execution(observation, 'A', scene, 'retreat', command,
                                          previous_check=proof)
    assert len(calls) == (1 if change is None else 2)
    assert result['collision_free']


def test_batch_does_not_recheck_raw_paths_before_final_interpolation(monkeypatch):
    from test_backend_builder import native_batch
    from tianji_curobo._curobo_impl import _NativeCuroboBackend

    from luna_mujoco import backend_builder as module

    native = native_batch()
    monkeypatch.setattr(_NativeCuroboBackend, 'plan_grasp_batch_native', lambda *args: native)
    monkeypatch.setattr(module, 'add_trajectory_derivatives', lambda result, raw: result)
    backend = object.__new__(module.ValidatedBatchBackend)
    backend.arm = 'A'
    backend.motion_planner = SimpleNamespace(last_grasp_result=None)
    backend.tool_fk_evidence = lambda *args: {'frame': 'world'}

    def repeated(_):
        raise AssertionError('raw path rechecked before final command check')

    backend.validate_trajectory_native = repeated
    result = backend.plan_grasp_batch_native(SimpleNamespace(scene_token='scene'))
    assert [item.success for item in result.items] == [True, True]


def test_closed_grasp_lift_requires_actual_fingers_but_no_attachment():
    from luna_mujoco.carrying import validate_phase_request

    payload = {'phase': 'grasp_lift', 'gripper_geometry_mode': 'actual'}
    assert validate_phase_request(payload) == 'grasp_lift'
    with pytest.raises(ValueError):
        validate_phase_request(payload | {'carrying_binding': {'state': 'held'}})
    with pytest.raises(ValueError):
        validate_phase_request(payload | {'gripper_geometry_mode': 'fixed_open'})


@pytest.mark.parametrize('reject_lift', [False, True])
def test_execution_checks_closed_geometry_before_original_lift(tmp_path, reject_lift):
    import asyncio
    from unittest.mock import AsyncMock

    from test_runner import Sim

    from luna_mujoco.runner import Runner

    simulation = Sim()
    runner = Runner(simulation, None, None, tmp_path)
    runner.state.update(arm='A', generation=1, target_id='black')
    runner.verify_scene = AsyncMock()
    runner.capture_holding_reference = AsyncMock()
    stages = []

    async def stage(context, name, function, *args, **kwargs):
        stages.append(name)
        if name in {'open_gripper', 'close_gripper'}:
            simulation.set_gripper('A', name == 'close_gripper', 1)
        return {}

    async def check(context, phase, command, binding=None, **kwargs):
        stages.append('check_' + phase)
        assert binding is None
        expected = .48 if phase == 'grasp_lift' else 0.
        assert simulation.robot_snapshot()['arms']['A']['gripper_position_rad'] == expected
        if phase == 'grasp_lift' and reject_lift:
            raise ValueError('closed lift collides')

    runner.execution_stage, runner.validate_motion = stage, check
    context = {'planning_settings': PlanningSettings().model_dump(), 'plan': {
        'T_world_tcp': simulation.state['arms']['A']['tcp'],
        'phases': {name: {'positions_rad': [[0.] * 7, [0.] * 7], 'dt': .02}
                   for name in ('approach', 'grasp', 'lift')}}}
    if reject_lift:
        with pytest.raises(ValueError, match='closed lift collides'):
            asyncio.run(runner.execute_grasp(context))
        assert 'lift' not in stages
    else:
        result = asyncio.run(runner.execute_grasp(context))
        assert result['lift_closed_gripper_collision_checked'] is True
        assert stages.index('close_gripper') < stages.index('check_grasp_lift') < stages.index('lift')
