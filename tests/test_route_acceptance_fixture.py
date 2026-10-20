"""Large acceptance scenes must start with separate objects on measured supports."""

import numpy as np
import pytest
from route_transfer_smoke import fixture


@pytest.mark.parametrize('route', ['box_table', 'table_box', 'box_table_box'])
@pytest.mark.parametrize('count', [1, 3, 4])
def test_count_fixture_keeps_every_cube_on_its_source_without_overlap(route, count):
    objects, _ = fixture(route, count, False)
    cubes = [item for item in objects if item['kind'] == 'cube']
    assert len(cubes) == count
    if route != 'table_box':
        container = next(item for item in objects if item['id'] == 'left-bin')
        for cube in cubes:
            offset = np.abs(np.asarray(cube['position'][:2]) - container['position'][:2])
            assert np.all(offset + np.asarray(cube['size'][:2]) / 2 + .01 <= np.asarray(container['size'][:2]) / 2)
    for index, cube in enumerate(cubes):
        for peer in cubes[index + 1:]:
            assert np.max(np.abs(np.asarray(cube['position'][:2]) - peer['position'][:2])) > .05
