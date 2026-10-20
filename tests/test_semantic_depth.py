import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def new_tensor(self, values):
        return Tensor(values)


class CameraObservation:
    def __init__(self, *, depth_image, intrinsics, pose, depth_to_meter=1.):
        self.depth_image, self.intrinsics, self.pose = depth_image, intrinsics, pose
        self.depth_to_meter = depth_to_meter

    def get_pointcloud(self, *, project_to_pose):
        depth = self.depth_image.numpy().squeeze(0)
        intrinsics = self.intrinsics.numpy().reshape(3, 3)
        rows, columns = np.indices(depth.shape)
        points = np.stack(((columns-intrinsics[0, 2])*depth/intrinsics[0, 0],
                           (rows-intrinsics[1, 2])*depth/intrinsics[1, 1], depth), axis=-1)
        if project_to_pose:
            points = points @ self.pose[:3, :3].T + self.pose[:3, 3]
        return Tensor(points[None])


def prepared(depth):
    depth = Tensor(np.asarray(depth)[None])
    intrinsics = Tensor(np.eye(3)[None])
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    observation = CameraObservation(depth_image=depth, intrinsics=intrinsics, pose=pose)
    return SimpleNamespace(camera_id="3", stereo_sequence=8, camera_observation=observation,
        filtered_depth_m=depth, base_pointcloud_m=observation.get_pointcloud(project_to_pose=True),
        intrinsics_tensor=intrinsics, pose_matrix_tensor=Tensor(pose), removed_point_count=7)


def projection(masks, *, intrinsics=None, transform=None):
    return SimpleNamespace(rgb_size=[4, 2], rgb_intrinsics=np.eye(3) if intrinsics is None else intrinsics,
        left_to_rgb=np.eye(4) if transform is None else transform, instance_masks=masks)


def partition(value, masks, **kwargs):
    assert importlib.util.find_spec("jev_mujoco.semantic_depth"), "explicit semantic partition is missing"
    from jev_mujoco.semantic_depth import partition_semantic_depth
    return partition_semantic_depth(value, masks, camera_observation_type=CameraObservation, **kwargs)


def test_partition_retains_every_measured_point_and_original_pixel_depth():
    value = prepared([[1, 2, 0, 4], [5, 6, 7, 8]])
    source, other = np.zeros((2, 4), dtype=bool), np.zeros((2, 4), dtype=bool)
    source[0, 0] = True
    other[1, 1] = True
    result = partition(value, projection({"selected": source, "foreign": other}))
    assert result.input_point_count == result.filtered_point_count == 7
    assert result.background_point_count == 5
    assert result.removed_point_count == 7
    assert result.instance_points_base_m["selected"].tolist() == [[1, 2, 4]]
    assert result.instance_points_base_m["foreign"].tolist() == [[7, 8, 9]]
    indices, depths = result.instance_depth_samples["foreign"]
    assert indices.tolist() == [5]
    assert depths.tolist() == [6]
    assert result.camera_observation.depth_image.numpy().tolist() == [[[0, 2, 0, 4], [5, 0, 7, 8]]]
    assert value.filtered_depth_m.numpy().tolist() == [[[1, 2, 0, 4], [5, 6, 7, 8]]]
    assert not result.background_points_base_m.flags.writeable


def test_projection_uses_left_to_rgb_and_rgb_intrinsics_without_pixel_index_alias():
    value = prepared([[1, 1, 1, 1], [1, 1, 1, 1]])
    matrix = np.eye(4)
    matrix[0, 3] = .5
    intrinsics = np.eye(3)
    intrinsics[0, 0] = 2
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, 3] = True
    result = partition(value, projection({"selected": mask}, intrinsics=intrinsics, transform=matrix))
    assert result.instance_depth_samples["selected"][0].tolist() == [1]
    assert result.instance_points_base_m["selected"].tolist() == [[2, 2, 4]]
    assert result.background_point_count == 7


def test_invalid_and_rejected_returns_cannot_enter_layers_or_free_depth():
    value = prepared([[0, np.nan, -1, 1], [1, 1, 1, 1]])
    value.base_pointcloud_m.values[0, 1, 0] = np.nan
    mask = np.ones((2, 4), dtype=bool)
    result = partition(value, projection({"selected": mask}))
    assert result.input_point_count == result.filtered_point_count == 4
    assert result.instance_depth_samples["selected"][0].tolist() == [3, 5, 6, 7]
    assert np.isfinite(result.instance_points_base_m["selected"]).all()
    assert not np.any(result.camera_observation.depth_image.numpy())


def test_empty_head_semantics_and_single_isolated_return_are_preserved():
    value = prepared([[0, 0, 0, 0], [0, 0, 0, 2]])
    result = partition(value, projection({}))
    assert result.input_point_count == result.filtered_point_count == result.background_point_count == 1
    assert result.instance_points_base_m == {}
    assert result.instance_depth_samples == {}
    assert result.background_points_base_m.tolist() == [[7, 4, 5]]
    assert result.camera_observation.depth_image.numpy().tolist() == value.filtered_depth_m.numpy().tolist()


def test_foreign_and_overlapping_masks_do_not_delete_or_relabel_each_other():
    value = prepared([[1, 1, 1, 1], [1, 1, 1, 1]])
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, 0] = True
    result = partition(value, projection({"selected": mask, "foreign": mask.copy()}))
    assert set(result.instance_points_base_m) == {"selected", "foreign"}
    assert len(result.instance_points_base_m["selected"]) == len(result.instance_points_base_m["foreign"]) == 1
    assert result.background_point_count == 7


def test_no_valid_measurements_return_typed_empty_individual_view():
    value = prepared(np.zeros((2, 4)))
    result = partition(value, projection({"selected": np.ones((2, 4), dtype=bool)}))
    assert type(result).__name__ == "PreparedSemanticDepth"
    assert result.input_point_count == result.filtered_point_count == result.background_point_count == 0
    assert result.background_points_base_m.shape == result.instance_points_base_m["selected"].shape == (0, 3)
    assert result.instance_depth_samples["selected"][0].shape == (0,)
    assert result.instance_depth_samples["selected"][1].shape == (0,)
    assert result.camera_id == "3" and result.stereo_sequence == 8
    assert not np.any(result.camera_observation.depth_image.numpy())


def test_registered_source_limit_returns_unsupported_raw_mask_pixels_to_background():
    value = prepared([[1, 1, 1, 1], [1, 1, 1, 0]])
    matrix = np.eye(4)
    matrix[0, 3] = 1
    mask = np.ones((2, 4), dtype=bool)
    foreign = np.zeros((2, 4), dtype=bool)
    foreign[1, 1] = True
    limit = np.zeros((2, 4), dtype=bool)
    limit[0, 1] = limit[1, 3] = True
    result = partition(value, projection({"selected": mask, "foreign": foreign}, transform=matrix),
                       registered_mask_limits={"selected": limit})
    assert result.instance_depth_samples["selected"][0].tolist() == [1]
    assert result.instance_depth_samples["foreign"][0].tolist() == [4]
    assert result.input_point_count == result.filtered_point_count == 7
    assert result.background_point_count == 5
    assert result.camera_observation.depth_image.numpy().tolist() == [[[1, 0, 1, 1], [0, 1, 1, 0]]]
    assert np.array_equal(limit, [[False, True, False, False], [False, False, False, True]])


@pytest.mark.parametrize("limit", [np.ones((4, 2), dtype=bool), np.ones((2, 4), dtype=float)])
def test_registered_limit_requires_depth_space_boolean_shape(limit):
    with pytest.raises(ValueError, match="registered mask limit"):
        partition(prepared(np.ones((2, 4))), projection({"selected": np.ones((2, 4), dtype=bool)}),
                  registered_mask_limits={"selected": limit})
