"""The selected cloud preserves every supported pixel from a valid capture."""

import numpy as np

from jev_mujoco.perception_mapping import decode_mapping_payload


def test_selected_surface_above_twenty_thousand_retains_all_measured_returns():
    from test_perception_mapping import wire
    from tianji_curobo.semantic_esdf_transport import (
        decode_semantic_esdf_request,
        encode_semantic_esdf_request,
    )

    request = wire()
    feedback = request.pop("gripper_joint_positions_rad")
    capture = decode_semantic_esdf_request(request)
    view = capture["views"]["1"]
    size = 150
    intrinsics = [[400., 0., 74.5], [0., 400., 74.5], [0., 0., 1.]]
    view.update(depth_left_m=np.ones((size, size), dtype=np.float32),
                valid_left_mask=np.ones((size, size), dtype=np.uint8),
                left_intrinsics=intrinsics, rgb_intrinsics=intrinsics,
                rgb_size=[size, size], instance_masks={"source": np.ones((size, size), dtype=np.uint8)})
    rows, columns = np.indices((size, size))
    measured_points = np.column_stack(((columns.ravel() - 74.5) / 400.,
                                      (rows.ravel() - 74.5) / 400., np.ones(size * size)))
    assert len(measured_points) == 22_500
    payload = encode_semantic_esdf_request(capture) | {
        "gripper_joint_positions_rad": feedback,
        "selected_surface_world_m": measured_points.tolist(), "exclusion_state": "held"}

    decoded, identity = decode_mapping_payload(payload)

    np.testing.assert_array_equal(decoded["selected_surface_world_m"], measured_points)
    np.testing.assert_array_equal(identity["selected_surface_world_m"], measured_points)
    assert decoded["views"]["1"]["depth_left_m"].size == len(measured_points)
