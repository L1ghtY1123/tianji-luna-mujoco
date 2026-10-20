import asyncio

import numpy as np
from test_atomic_scene_cycle import AtomicChoices, CountedRobot
from test_route_runner import RouteAlgorithms

from jev_mujoco.measured_regions import supported_floor_rectangle
from jev_mujoco.placement import region_contains_xy
from jev_mujoco.runner import Runner


def test_complementary_cameras_do_not_shrink_floor_to_the_densest_strip():
    coarse_x, coarse_y = np.meshgrid(np.arange(.40, .561, .002), np.arange(.13, .331, .002))
    dense_x, dense_y = np.meshgrid(np.arange(.36, .401, .001), np.arange(.13, .331, .001))
    points = np.vstack((np.c_[coarse_x.ravel(), coarse_y.ravel()],
                        np.c_[dense_x.ravel(), dense_y.ravel()]))
    region = supported_floor_rectangle(points)
    polygon = np.asarray(region['polygon_world_xy_m'])
    assert np.ptp(polygon[:, 0]) >= .19
    assert np.ptp(polygon[:, 1]) >= .19
    assert region_contains_xy([[.46, .23]], polygon)[0]


async def test_placement_layout_uses_three_camera_geometry_before_it_is_frozen(tmp_path):
    class Algorithms(RouteAlgorithms):
        def build_scene(self, observation, target_id, **kwargs):
            if kwargs['exclusion_state'] == 'observation':
                assert all('layout' not in region for region in runner.task_progress.placement_requests.values())
            return super().build_scene(observation, target_id, **kwargs)

        def plan_place(self, observation, *args, **kwargs):
            layout = kwargs['placement_request']['layout']
            assert layout['observation_id'] == observation['observation_id']
            assert set(observation['views']) == {'1', '2', '3'}
            return super().plan_place(observation, *args, **kwargs)

    runner = Runner(CountedRobot(), Algorithms(), lambda record: AtomicChoices('table_box'), tmp_path)
    await runner.start('把桌面物体放入盒子')
    await asyncio.wait_for(runner.worker, 5)
    assert runner.state['status'] == 'succeeded', runner.state.get('failure')
