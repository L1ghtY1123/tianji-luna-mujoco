import base64
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))


def payload():
    from test_perception_mapping import wire
    from tianji_curobo.semantic_esdf_transport import decode_semantic_esdf_request
    request = wire()
    request.pop("gripper_joint_positions_rad")
    decoded = decode_semantic_esdf_request(request)
    decoded["views"]["3"]["instance_masks"] = {}
    decoded["views"]["1"]["instance_masks"]["other"] = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    decoded["views"]["2"]["left_intrinsics"][0][0] = 8.
    decoded["views"]["1"]["valid_left_mask"][0, 2] = 0
    return decoded


def codec():
    assert importlib.util.find_spec("jev_mujoco.semantic_transport"), "optional-head transport missing"
    from jev_mujoco import semantic_transport
    return semantic_transport


@pytest.mark.parametrize("state", ["held", "released", "selected", "observation"])
def test_empty_head_roundtrip_restores_actual_semantics_identity_and_calibration(state):
    original = payload()
    packed = codec().encode_project_semantic_request(original, exclusion_state=state)
    restored = codec().decode_project_semantic_request(packed, exclusion_state=state)
    for key in ("observation_set_id", "sam3_generation", "selection_generation", "confirmed_instance_id"):
        assert restored[key] == original[key]
    assert restored.get("mapping_mode", "semantic") == "semantic"
    assert restored["views"]["3"]["instance_masks"] == {}
    for camera in ("1", "2", "3"):
        before, after = original["views"][camera], restored["views"][camera]
        assert before["camera_id"] == after["camera_id"] == camera
        assert before["left_intrinsics"] == after["left_intrinsics"]
        assert before["robot_joints_at_capture"] == after["robot_joints_at_capture"]
        assert np.array_equal(before["valid_left_mask"], after["valid_left_mask"])
        for instance, mask in before["instance_masks"].items():
            assert np.array_equal(mask, after["instance_masks"][instance])
        assert set(before["instance_masks"]) == set(after["instance_masks"])
    assert restored["views"]["1"]["depth_left_m"][0, 2] == 0
    assert original["views"]["1"]["depth_left_m"][0, 2] == 1
    assert packed["depth_request"]["mapping_mode"] == "background"
    assert packed["depth_request"]["confirmed_instance_id"] == ""


def test_ordinary_semantic_wire_is_unchanged():
    from tianji_curobo.semantic_esdf_transport import encode_semantic_esdf_request
    original = payload()
    original["views"]["3"]["instance_masks"] = {"source": np.ones((2, 3), dtype=np.uint8)}
    assert codec().encode_project_semantic_request(original, exclusion_state="held") == encode_semantic_esdf_request(original)


@pytest.mark.parametrize("change", ["state", "selected", "generation", "depth_identity", "camera_id", "duplicate",
                                    "short_bits", "padding_bits", "oversize", "extra_field", "head_not_empty"])
def test_sidecar_rejects_invalid_identity_provenance_sizes_and_bit_encoding(change):
    packed = codec().encode_project_semantic_request(payload(), exclusion_state="held")
    state = "held"
    if change == "state":
        state = "invalid"
    elif change == "selected":
        packed["semantic_identity"]["confirmed_instance_id"] = "unseen"
    elif change == "generation":
        packed["semantic_identity"]["sam3_generation"] += 1
    elif change == "depth_identity":
        packed["depth_request"]["observation_set_id"] = "other-capture"
    elif change == "camera_id":
        packed["depth_request"]["views"]["2"]["camera_id"] = "1"
    elif change == "duplicate":
        packed["semantic_masks"]["1"].append(deepcopy(packed["semantic_masks"]["1"][0]))
    elif change == "short_bits":
        packed["semantic_masks"]["1"][0]["bits_base64"] = ""
    elif change == "padding_bits":
        packed["semantic_masks"]["1"][0]["bits_base64"] = base64.b64encode(bytes([255])).decode()
    elif change == "oversize":
        packed["depth_request"]["views"]["1"]["rgb_size"] = [1_000_000, 1_000_000]
    elif change == "extra_field":
        packed["semantic_masks"]["1"][0]["unknown"] = True
    else:
        packed["semantic_masks"]["3"] = deepcopy(packed["semantic_masks"]["1"])
    with pytest.raises(ValueError):
        codec().decode_project_semantic_request(packed, exclusion_state=state)


def test_encode_rejects_nonbinary_masks_and_missing_selected_before_wire_creation():
    original = payload()
    original["views"]["1"]["instance_masks"]["source"][0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        codec().encode_project_semantic_request(original, exclusion_state="held")
    original = payload()
    original["confirmed_instance_id"] = "unseen"
    with pytest.raises(ValueError, match="selected"):
        codec().encode_project_semantic_request(original, exclusion_state="released")


def test_stereo_capture_sequence_is_shared_but_independent_of_scene_generation():
    original = payload()
    for view in original["views"].values():
        view["stereo_sequence"] = 42
    packed = codec().encode_project_semantic_request(original, exclusion_state="held")
    restored = codec().decode_project_semantic_request(packed, exclusion_state="held")
    assert restored["selection_generation"] == 4
    assert {view["stereo_sequence"] for view in restored["views"].values()} == {42}
    packed["depth_request"]["views"]["1"]["stereo_sequence"] = 43
    with pytest.raises(ValueError, match="sequence"):
        codec().decode_project_semantic_request(packed, exclusion_state="held")
