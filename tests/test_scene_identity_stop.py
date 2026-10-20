import pytest
from test_planning_recovery import Selecting, make_runner
from test_runner import Algorithms, wait

from jev_mujoco.algorithms import AlgorithmError


@pytest.mark.parametrize('code', [
    'semantic_scene_identity_unresolved', 'scene_inventory_identity_unresolved',
    'observed_instance_missing_or_ambiguous:black', 'instance association ambiguous',
    'scene_inventory_unmatched_detection',
])
async def test_scene_identity_failure_stops_without_recover_loop(tmp_path, code):
    details = {'temporal': [{'previous_id': 'black', 'status': 'ambiguous', 'candidates': ['first', 'second']}]}

    class ConflictingScene(Algorithms):
        def build_scene(self, observation, target_id, **kwargs):
            if kwargs.get('exclusion_state') == 'selected':
                raise AlgorithmError(code, code=code, details=details)
            return super().build_scene(observation, target_id, **kwargs)

    class Choices(Selecting):
        recovery_calls = 0

        async def choose_recovery(self, state):
            self.recovery_calls += 1
            return 'stop'

    choices = Choices()
    runner = make_runner(tmp_path, ConflictingScene(), choices)
    await runner.start('抓取黑色物体')
    await wait(runner)
    assert choices.recovery_calls == 0
    assert runner.state['status'] == 'needs_attention'
    assert runner.state['error'] == code
    assert '身份关联未确认，本轮已停止，不再自动 recover' in runner.state['conclusion']['summary']
    assert runner.state['failure']['evidence']['details'] == details
    assert len([event for event in runner.events if event['kind'] == 'task_failure']) == 1
    assert not any(event['title'] in {'execute_grasp', 'recover_task'} for event in runner.events)


async def test_post_release_identity_failure_preserves_release_without_completing_stage(tmp_path):
    from test_atomic_scene_cycle import AtomicChoices, CountedRobot
    from test_route_runner import RouteAlgorithms

    from jev_mujoco.runner import Runner

    class FailedPostPlace(RouteAlgorithms):
        def observe(self, *args, **kwargs):
            if kwargs.get('checkpoint') == 'post_place':
                raise AlgorithmError('scene_inventory_unmatched_detection', code='scene_inventory_unmatched_detection')
            return super().observe(*args, **kwargs)

    runner = Runner(CountedRobot(), FailedPostPlace(), lambda record: AtomicChoices('box_table_box'), tmp_path)
    await runner.start('搬运同一个目标')
    await wait(runner)
    assert runner.state['error'] == 'scene_inventory_unmatched_detection'
    assert not runner.state['facts']['recovery_needed']
    record = runner.task_progress.records['source']
    assert record['release_feedback_id']
    assert record.get('completed_stages', []) == []
    assert runner.task_progress.completed_ids == []
    assert any(e['kind'] == 'scene_identity_blocked' for e in runner.events)
    assert not any(e['kind'] == 'skill_start' and e['title'] == 'recover_task' for e in runner.events)
