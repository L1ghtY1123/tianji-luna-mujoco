"""Metric camera snapshots must respect the pose service's public wire range."""
import numpy as np
import pytest

from luna_mujoco.pose_estimation import CadPoseEstimator, snapshot_view
from tianji_foundationpose.contracts import encode_request
from tianji_foundationpose.contracts import PoseError


def view():
    return {'rgb': np.zeros((10, 10, 3), np.uint8), 'depth_m': np.ones((10, 10), np.float32),
            'valid_depth': np.ones((10, 10), bool),
            'intrinsics': {'fx': 100., 'fy': 100., 'cx': 5., 'cy': 5.},
            'frame_id': 'capture', 'camera_id': '3', 'capture_time': 1., 'simulation_generation': 2,
            'calibration_id': 'calibration', 'T_world_camera': np.eye(4)}


class WireClient:
    def estimate(self, rgb, depth, mask, intrinsics, **options):
        encode_request(rgb, depth, mask, intrinsics, options['model_id'],
                       options['capture_id'], options['camera_id'], options['score'])
        return {'T_camera_object': np.eye(4), 'T_source_canonical_m': np.eye(4),
                'capture_id': options['capture_id'], 'camera_id': options['camera_id']}


def test_far_background_is_invalidated_without_changing_original_capture():
    original = view()
    original['depth_m'][0, :3] = [109.81994, 65.535, 0.]
    snapshot = snapshot_view(original)
    assert snapshot['depth_m'][0, 0] == 0.
    assert snapshot['depth_m'][0, 1] == original['depth_m'][0, 1]
    assert snapshot['valid_depth'][0, :3].tolist() == [False, True, False]
    assert original['valid_depth'].all() and original['depth_m'][0, 0] > 100.
    assert snapshot['depth_transport']['out_of_range_depth_pixels'] == 1
    assert snapshot['depth_transport']['zero_depth_pixels'] == 1
    assert snapshot['depth_transport']['transport_max_depth_m'] == 65.535
    result = CadPoseEstimator(pose_client=WireClient()).estimate_instance(
        original, np.ones((10, 10), bool), .9, model_id='blue_box',
        mask_capture_id='capture', mask_camera_id='3')
    assert result['depth_transport'] == snapshot['depth_transport']


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
def test_corrupt_valid_depth_still_fails_closed(value):
    original = view()
    original['depth_m'][0, 0] = value
    with pytest.raises(PoseError, match='invalid_capture'):
        snapshot_view(original)


def test_predeclared_invalid_depth_can_be_nonfinite_and_is_zeroed():
    original = view()
    original['depth_m'][0, 0] = np.nan
    original['valid_depth'][0, 0] = False
    snapshot = snapshot_view(original)
    assert np.isfinite(snapshot['depth_m']).all()
    assert snapshot['depth_m'][0, 0] == 0.
    assert np.isnan(original['depth_m'][0, 0])


def test_segmentation_entry_keeps_original_filter_evidence():
    class Segmenter:
        def segment(self, rgb, text, timestamp_us):
            return [{'mask': np.ones((10, 10), bool), 'score': .9}]

    original = view()
    original['depth_m'][0, 0] = 100.
    result = CadPoseEstimator(pose_client=WireClient(), segmentation_client=Segmenter()).segment_and_estimate(
        original, 'blue box', model_id='blue_box')
    assert result[0]['depth_transport']['out_of_range_depth_pixels'] == 1
