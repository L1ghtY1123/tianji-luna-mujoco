from typing import ClassVar

import numpy as np

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.planning_settings import PlanningSettings


def test_adapter_reuses_prior_selected_map_for_verified_place_return():
    calls = []
    def request(path, payload):
        calls.append(payload)
        return {'attached': False, 'scene_token': 's', 'sample_dt_s': .02,
                'trajectories': {'A': [[0.] * 7, [1.] * 7], 'B': [[0.] * 7, [-1.] * 7]}}
    algorithms = Algorithms(curobo_request=request)
    scene = {'scene_token': 's', 'target_id': 'one', 'exclusion_state': 'selected',
             'observation_id': 'before', 'generation': 1, 'return_after_place': True}
    feedback = {'observation_id': 'before', 'generation': 1, 'arms': {
        a: {'joints_rad': [0.] * 7, 'gripper_position_rad': 0.} for a in ('A', 'B')}}
    algorithms.plan_dual_observation_motion(feedback, scene, {'A': [.1]*7, 'B': [-.1]*7})
    assert calls[0]['return_after_place'] is True
    assert calls[0]['scene_token'] == 's'


def test_observation_adapter_submits_known_joint_goals_and_joint_validation():
    calls = []
    scene = {'scene_token': 's', 'target_id': 'o', 'exclusion_state': 'observation',
             'observation_id': 'c', 'generation': 1}
    snapshot = {'observation_id': 'c', 'generation': 1, 'arms': {
        a: {'joints_rad': [0.] * 7, 'gripper_position_rad': 0.} for a in ('A', 'B')}}
    def request(path, payload):
        calls.append((path, payload))
        if path.endswith('/validate_dual_observation'):
            return {'collision_free': True, 'scene_token': 's'}
        return {'attached': False, 'scene_token': 's', 'sample_dt_s': .02,
                'trajectories': {'A': [[0.] * 7, [1.] * 7], 'B': [[0.] * 7, [-1.] * 7]}}
    algorithms = Algorithms(curobo_request=request)
    result = algorithms.plan_dual_observation_motion(snapshot, scene, {'A': [.1]*7, 'B': [-.1]*7})
    assert 'goal_transforms_world' not in calls[0][1]
    np.testing.assert_allclose(calls[0][1]['goal_joints_deg']['A'], np.rad2deg([.1] * 7))
    np.testing.assert_allclose(calls[0][1]['goal_joints_deg']['B'], np.rad2deg([-.1] * 7))
    np.testing.assert_allclose(result['paths']['A'][-1], [np.pi/180]*7)
    assert result['dt'] == .02
    algorithms.validate_dual_observation(snapshot, scene, result['paths'])
    assert calls[-1][0] == '/tools/validate_dual_observation'
    np.testing.assert_allclose(calls[-1][1]['points_deg']['B'][-1], [-1.]*7)


def test_service_dual_uses_same_scene_and_measured_grippers():
    from contextlib import nullcontext

    from luna_mujoco.geometry_service import run_operation
    class Runtime:
        geometry_solver_profile = PlanningSettings().solver_payload()
        geometry_scene: ClassVar[dict] = {'exclusion_state': 'observation'}
        def _scene_snapshot_unlocked(self): return {'scene_token': 's'}
        def configure_planning(self, settings): pass
        def carrying_scope(self, binding, feedback):
            assert binding is None and feedback == {'A': 0., 'B': 0.}
            return nullcontext()
        def dual_operation_native(self, request, **kwargs):
            assert set(request.start_joints_deg) == {'A', 'B'}
            assert request.goal_transforms_world['A'][0][3] == 500.
            return {'scene_token': 's'}
    result = run_operation(Runtime(), '/tools/plan_dual_observation', {
        'scene_token': 's', 'planning_settings': PlanningSettings().model_dump(),
        'start_joints_deg': {'A': [0.]*7, 'B': [0.]*7},
        'goal_transforms_world': {'A': [[1.,0.,0.,.5],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]], 'B': np.eye(4).tolist()},
        'gripper_joint_positions_rad': {'A': 0., 'B': 0.}})
    assert result['attached'] is False


def test_geometry_runtime_constructs_joint_backend_not_single_arm(monkeypatch):
    import luna_mujoco.geometry_service as module
    calls = []
    def build(runtime):
        calls.append(runtime)
        return 'joint-backend'
    monkeypatch.setattr(module, 'build_dual_geometry_backend', build, raising=False)
    runtime = object.__new__(module.GeometryRuntime)
    runtime.warmup_status = {'ready': False}
    assert runtime._create_real_backend_unlocked('AB') == 'joint-backend'
    assert calls == [runtime]
