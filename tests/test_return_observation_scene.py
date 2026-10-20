import asyncio

import pytest
from test_atomic_scene_cycle import AtomicChoices, CountedRobot
from test_route_runner import RouteAlgorithms

from jev_mujoco.runner import Runner


@pytest.mark.parametrize('invalid_final', ['missing', 'unmeasured', 'original_source'])
async def test_complete_route_destinations_must_be_confirmed_before_first_motion(tmp_path, invalid_final):
    class Algorithms(RouteAlgorithms):
        def observe(self, *args, **kwargs):
            result = super().observe(*args, **kwargs)
            if invalid_final == 'missing':
                result['objects'] = [obj for obj in result['objects'] if obj['id'] != 'destination']
            elif invalid_final == 'unmeasured':
                next(obj for obj in result['objects'] if obj['id'] == 'destination').pop('placement_reference')
            else:
                result['objects'][0]['measured_relationships'][0]['object_id'] = 'destination'
            return result

    class Choices(AtomicChoices):
        async def choose_skill(self, state, registry):
            if state['facts'].get('recovery_needed'):
                return 'stop'
            return await super().choose_skill(state, registry)

    algorithms = Algorithms()
    runner = Runner(CountedRobot(), algorithms, lambda record: Choices('box_table_box'), tmp_path)
    await runner.start('把同一物体经过桌面放入另一盒')
    await asyncio.wait_for(runner.worker, 5)
    assert algorithms.returns == []
    assert not algorithms.maps
    assert runner.state['status'] != 'succeeded'


@pytest.mark.parametrize('missing', [False, True])
async def test_three_camera_update_precedes_return_motion_and_missing_placed_object_blocks_it(tmp_path, missing):
    events = []

    class Algorithms(RouteAlgorithms):
        def observe(self, capture, **kwargs):
            events.append(('observe', kwargs.get('camera_ids', kwargs.get('source_hint', {}).get('camera_ids'))))
            result = super().observe(capture, **kwargs)
            if 'selected' in self.maps and kwargs.get('checkpoint') == 'post_place':
                assert kwargs['source_hint']['camera_ids'] == ['1', '2', '3']
            if missing and 'selected' in self.maps and kwargs.get('checkpoint') == 'post_place':
                result['objects'] = [obj for obj in result['objects'] if obj['id'] != 'source']
                for view in result['views'].values():
                    view['instance_masks'].pop('source', None)
            return result

        def build_scene(self, observation, target_id, **kwargs):
            events.append(('map', kwargs['exclusion_state']))
            if kwargs['exclusion_state'] == 'observation':
                assert target_id is None
                assert 'source' in {obj['id'] for obj in observation['objects']}
            return super().build_scene(observation, target_id, **kwargs)

        def plan_dual_observation_motion(self, *args, **kwargs):
            events.append(('motion', None))
            assert events[-2] == ('map', 'observation')
            return super().plan_dual_observation_motion(*args, **kwargs)

    class Choices(AtomicChoices):
        async def choose_skill(self, state, registry):
            if state['facts'].get('recovery_needed'):
                return 'stop'
            return await super().choose_skill(state, registry)

    algorithms = Algorithms()
    runner = Runner(CountedRobot(), algorithms, lambda record: Choices('box_table_box'), tmp_path)
    await runner.start('把同一物体经过桌面放入另一盒')
    await asyncio.wait_for(runner.worker, 5)
    assert len(algorithms.returns) == (1 if missing else 2)
    if missing:
        assert runner.state['failure']['code'].startswith('placed_object_not_observed')
    else:
        assert runner.state['status'] == 'succeeded'
    # observe_scene has no map before inventory/target selection.
    first_map = next(i for i, e in enumerate(runner.events) if e['kind'] == 'execution_stage_start'
                     and e['data']['stage'] == 'observation_collision_scene')
    assert any(e['kind'] == 'skill_complete' and e['title'] == 'assess_task' for e in runner.events[:first_map])
