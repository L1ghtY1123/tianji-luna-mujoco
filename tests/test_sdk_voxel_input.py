from dataclasses import dataclass

import numpy as np

from luna_mujoco.sdk_scene import sdk_voxel_grid


@dataclass
class Grid:
    dims: list
    voxel_size: float
    feature_tensor: object


def test_public_voxel_input_keeps_integer_strides_without_editing_sdk_buffers():
    for count in range(1, 513):
        grid = Grid([count * .02] * 3, .02, object())
        converted = sdk_voxel_grid(grid)
        quotient = np.asarray(converted.dims, np.float32) / np.float32(.02)
        assert np.all(quotient.astype(int) == count)
        assert np.allclose(converted.dims, grid.dims, rtol=0, atol=1e-6)
        assert converted.feature_tensor is grid.feature_tensor
        assert grid.dims == [count * .02] * 3
