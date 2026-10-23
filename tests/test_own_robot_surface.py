"""Current robot surface ownership must precede carried point-cloud sampling."""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest
from geometry_perception_fixture import GeometryPerception

from luna_mujoco.cameras import deproject_view
from luna_mujoco.perception import _remove_own_robot_surface
from luna_mujoco.robot_surface import _own_visual_model, first_surface_matches, own_robot_surface_pixels

DATA = Path(__file__).resolve().parents[1] / "data/visible-cloud-work/grasp-place-left-14"
CAPTURE_ID = "rgbd-2-fcb89d19e8984906b74f4a2bf82dd8af"
ROBOT_PIXELS = [(146, 213), (147, 214), (142, 229), (141, 229), (144, 212), (143, 211)]
CONTACT_PIXELS = [(152, 220), (152, 221), (158, 225), (161, 227)]


@pytest.mark.skipif(not (DATA / f"rgbd-{CAPTURE_ID}.npz").exists(), reason="recorded RGBD fixture unavailable")
def test_recorded_own_robot_returns_do_not_enter_source_attachment_geometry():
    capture = np.load(DATA / f"rgbd-{CAPTURE_ID}.npz", allow_pickle=True)["capture"].item()
    saved = np.load(DATA / f"sam3-{CAPTURE_ID}.npz", allow_pickle=True)
    records = list(saved["records"])
    original_capture, original_records = deepcopy(capture), deepcopy(records)

    class RecordedDetector:
        def __init__(self):
            self.index = 0

        def segment(self, rgb, text, timestamp_us):
            # Replay measured masks for geometry testing, not new model inference.
            if text in {'block', 'object'}:
                # This older recording has no synonym inference; contributes no new mask.
                return []
            record = next(item for item in records if item["text"] == text
                          and np.array_equal(rgb, capture["views"][item["camera_id"]]["rgb"]))
            return record["detections"]

    detector = RecordedDetector()
    result = GeometryPerception(detector, identities=('source',)).observe(capture, checkpoint="post_grasp",
                                          source_hint={"kind": "cube", "color": "any"})
    source = next(obj for obj in result["objects"] if obj["kind"] == "cube")
    mask = result["views"]["1"]["instance_masks"][source["id"]]
    assert not any(mask[row, column] for row, column in ROBOT_PIXELS)
    raw_mask = next(item["detections"][0]["mask"] for item in records
                    if item["text"] == "cube" and item["camera_id"] == "1")
    assert all(mask[row, column] for row, column in CONTACT_PIXELS if raw_mask[row, column])
    assert len(source["points_world_m"]) >= 30
    for camera, view in result["views"].items():
        assert not np.any(view["instance_masks"].get(source["id"], False)
                          & view.get("robot_surface_mask", False))
    for camera, view in capture["views"].items():
        for key in ("rgb", "depth_m", "valid_depth"):
            np.testing.assert_array_equal(view[key], original_capture["views"][camera][key])
            np.testing.assert_array_equal(result["views"][camera][key], view[key])
    for record, original in zip(records, original_records, strict=True):
        for detection, old in zip(record["detections"], original["detections"], strict=True):
            np.testing.assert_array_equal(detection["mask"], old["mask"])


def ray_fixture(*, front=True):
    geometry = '<geom type="box" pos="0 0 1.1" size=".15 .15 .1" group="0"/>'
    if front:
        geometry += '<geom type="box" pos="0 0 .95" size=".15 .15 .05" group="0"/>'
    model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{geometry}</worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = {"intrinsics": {"fx": 500., "fy": 500., "cx": 1., "cy": 1., "width": 3, "height": 3},
            "T_world_camera": np.eye(4), "depth_m": np.full((3, 3), .9),
            "valid_depth": np.ones((3, 3), bool), "rgb": np.full((3, 3, 3), 100, np.uint8)}
    return model, data, view


@pytest.mark.parametrize("depth, expected", [(.9, True), (.894, False), (1., False), (1.3, False)])
def test_only_actual_first_surface_matches_not_foreground_or_hidden_robot_surface(depth, expected):
    model, data, view = ray_fixture()
    view["depth_m"][:] = depth
    result = first_surface_matches(model, data, view, [[1, 1]])
    assert bool(result["matched"][0]) is expected


def test_off_axis_euclidean_range_is_not_compared_directly_to_optical_depth():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom type="box" pos=".36 0 .95" size=".03 .03 .05" group="0"/>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, _, view = ray_fixture()
    view["intrinsics"].update(fx=2.5, cx=0.)
    result = first_surface_matches(model, data, view, [[1, 1]])
    assert result["matched"].tolist() == [True]
    assert result["distance_m"][0] < 1e-12


def test_invalid_depth_and_missing_measured_gripper_cannot_justify_removing_pixels():
    model, data, view = ray_fixture()
    view["valid_depth"][1, 1] = False
    assert not first_surface_matches(model, data, view, [[1, 1]])["matched"][0]
    capture = {"arms": {"A": {"joints_rad": [0.] * 7}, "B": {"joints_rad": [0.] * 7}},
               "views": {"1": view}}
    result = own_robot_surface_pixels(capture, {"1": np.ones((3, 3), bool)})
    assert result["status"] == "robot_feedback_unavailable"
    assert result["masks"] == {}


def test_robot_model_excludes_default_table_ground_and_collision_duplicates():
    model, visual, _ = _own_visual_model()
    assert model.geom("table").group[0] == 5
    assert model.geom("ground").group[0] == 5
    assert len(visual) == 45
    assert np.all(model.geom_group[visual] == 0)


@pytest.mark.parametrize("other_depth, other_valid, retained", [(.9, True, True), (.92, True, False),
                                                               (.9, False, False)])
def test_independent_current_same_instance_depth_vetoes_ambiguous_robot_contact(
        other_depth, other_valid, retained):
    _, _, template = ray_fixture()
    views = {camera: deepcopy(template) for camera in ("1", "2")}
    views["2"]["depth_m"][:] = other_depth
    views["2"]["valid_depth"][:] = other_valid
    mask = np.ones((3, 3), bool)
    groups = [[{"camera_id": camera, "mask": mask.copy(), "kind": "cube", "color": "red", "score": .9,
                "points": deproject_view(view, mask), "pixels": view["rgb"][mask],
                "pixels_rc": np.column_stack(np.nonzero(mask)), "view": view}
               for camera, view in views.items()]]
    original = deepcopy(groups)

    def candidate_classifier(capture, masks):
        candidate = np.zeros((3, 3), bool)
        candidate[1, 1] = True
        return {"masks": {"1": candidate}, "status": "measured"}

    changed, evidence = _remove_own_robot_surface(groups, views, {}, candidate_classifier)
    assert bool(changed[0][0]["mask"][1, 1]) is retained
    assert evidence["retained_cross_view_contact_count"] == int(retained)
    np.testing.assert_array_equal(groups[0][0]["mask"], original[0][0]["mask"])
    np.testing.assert_array_equal(views["1"]["depth_m"], template["depth_m"])
