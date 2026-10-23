from types import SimpleNamespace

from luna_mujoco.public_planner import PublicPlanner


def test_compound_call_does_not_intercept_official_internal_plan_pose():
    calls = []
    class Official:
        def plan_pose(self, goal, start, **kwargs):
            calls.append((goal, start.dt, kwargs))
            return object()

        def plan_grasp(self, goal, start, **kwargs):
            return self.plan_pose(goal, start)

    sdk = Official()
    settings = SimpleNamespace(plan_max_attempts=2, finetune_attempts=1, finetune_dt_scale=.55)
    adapter = PublicPlanner(sdk, settings, batch=True)
    state = SimpleNamespace(dt=.02)
    result = adapter.plan_grasp('original-goal', state)
    assert calls == [('original-goal', .02, {})]
    assert adapter.last_grasp_result is result
    assert type(adapter.sdk) is Official
    assert adapter.sdk.plan_pose.__func__ is Official.plan_pose


def test_direct_pose_passes_only_public_parameters_and_retains_output():
    calls = []
    class Official:
        def plan_pose(self, *args, **kwargs):
            calls.append((args, kwargs))
            return object()
    settings = SimpleNamespace(plan_max_attempts=2, finetune_attempts=1, finetune_dt_scale=.55)
    adapter = PublicPlanner(Official(), settings, batch=False)
    result = adapter.plan_pose('goal', 'start', max_attempts=3)
    assert calls == [(('goal', 'start'), {'max_attempts': 3})]
    assert adapter.last_pose_result is result
