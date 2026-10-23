import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from luna_mujoco.backend_builder import linear_retreat_criteria


@pytest.mark.parametrize('fail', [False, True])
def test_official_linear_lift_criteria_are_restored_even_when_planning_fails(fail):
    updates = []
    class Criteria:
        @staticmethod
        def linear_motion(**kwargs): return kwargs
    backend = SimpleNamespace(_tool_pose_criteria_type=Criteria, tool_frame='tcp',
        motion_planner=SimpleNamespace(update_tool_pose_criteria=updates.append))
    try:
        with linear_retreat_criteria(backend, 'z', False):
            assert updates == [{'tcp': {'axis': 'z', 'project_distance_to_goal': False}}]
            if fail:
                raise RuntimeError('planner failed')
    except RuntimeError:
        assert fail
    assert len(updates) == 2 and isinstance(updates[-1]['tcp'], Criteria)


def test_retreat_delegates_without_holding_native_non_reentrant_gpu_lock():
    from luna_mujoco.geometry_service import GeometryRuntime

    lock = threading.Lock()
    backend = SimpleNamespace()
    def native(request, **kwargs):
        assert not lock.locked(), 'the native API acquires this lock itself'
        assert backend.release_departure == ('z', False)
        return 'trajectory'
    runtime = SimpleNamespace(geometry_scene={'exclusion_state': 'released'}, gpu_lock=lock,
        carrying_scope=lambda *args: nullcontext(), _prepare_scene_cache_unlocked=lambda: None,
        _backend_unlocked=lambda arm: backend, plan_to_pose_native=native,
        effective_config={'A': {'plan': {}}})
    result, _ = GeometryRuntime.plan_retreat_scoped(runtime, SimpleNamespace(arm='A'),
        departure_axis='z', departure_in_tool_frame=False)
    assert result == 'trajectory' and backend.release_departure is None
