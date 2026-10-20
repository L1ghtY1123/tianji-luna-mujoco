"""Measured geometry regressions with explicit segmentation and visual decision fixtures."""

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import msgpack
import numpy as np
import pytest
from geometry_perception_fixture import GeometryPerception
from PIL import Image


def perception_module():
    assert importlib.util.find_spec("jev_mujoco.perception") is not None, "RGBD perception is missing"
    return importlib.import_module("jev_mujoco.perception")


def capture_fixture():
    rgb = np.full((80, 80, 3), 140, np.uint8)
    depth = np.full((80, 80), .4, np.float32)
    masks = []
    for column in (15, 65):
        mask = np.zeros((80, 80), bool)
        mask[36:45, column - 4:column + 5] = True
        rgb[mask] = [220, 20, 20]
        depth[mask] = .3
        masks.append(mask)
    transform = np.diag([1., -1., -1., 1.])
    transform[2, 3] = 1.
    views = {camera_id: {
        "camera_id": camera_id, "frame_id": "capture-one", "capture_time": 2.,
        "simulation_generation": 3, "calibration_id": "test-k", "registered_to": "rgb",
        "rgb": rgb.copy(), "depth_m": depth.copy(), "valid_depth": np.ones((80, 80), bool),
        "intrinsics": {"fx": 100., "fy": 100., "cx": 40., "cy": 40., "width": 80, "height": 80},
        "T_world_camera": transform.tolist(), "robot_joints_deg": {"A": [0.] * 7, "B": [0.] * 7},
    } for camera_id in ("1", "2", "3")}
    # These are deliberately poisoned privileged fields. None is an input to perception.
    capture = {"observation_id": "capture-one", "generation": 3, "time": 2., "views": views,
               "arms": {"A": {"joints_rad": [0.] * 7}, "B": {"joints_rad": [0.] * 7}},
               "objects": [{"id": "SIM_SECRET", "kind": "sphere", "color": "blue"}],
               "table": {"position": [9., 9., 9.], "size": [8.] * 3}}
    return capture, masks


class FrozenSegmenter:
    def __init__(self, masks):
        self.masks = masks

    def segment(self, rgb, text, timestamp_us):
        if text == "cube":
            return [{"mask": mask.copy(), "score": .92, "label": "cube"} for mask in self.masks]
        if text == "table":
            return [{"mask": ~np.logical_or.reduce(self.masks), "score": .9, "label": "table"}]
        return []


def test_rgb_inventory_prompt_is_used_literally_and_cached():
    capture, masks = capture_fixture()
    calls = []

    class Segmenter:
        def segment(self, rgb, text, timestamp_us):
            calls.append(text)
            return [{'mask': mask, 'score': .9} for mask in masks] if text == 'red block' else []

    perception = GeometryPerception(Segmenter(), prompts={'first': 'red block', 'second': 'red block'})
    result = perception.observe(capture, source_hint={'kind': 'cube', 'color': 'red'})
    assert len(result['objects']) == 2
    assert calls == ['red block'] * 3
    assert all(obj['color'] == 'red' and obj['kind'] == 'cube' for obj in result['objects'])
    perception.observe(capture, source_hint={'kind': 'cube', 'color': 'red'})
    assert len(calls) == 3


def test_two_rgb_descriptions_bind_overlapping_sam_proposals_without_duplicate_instances():
    capture, masks = capture_fixture()
    calls = []

    class Segmenter:
        def segment(self, rgb, text, timestamp_us):
            calls.append(text)
            selected = masks[:1] if text == 'cube' else masks if text == 'block' else []
            return [{'mask': mask, 'score': .9} for mask in selected]

    perception = GeometryPerception(Segmenter(), identities=('first', 'first', 'second'), prompts={'first': 'cube', 'second': 'block'})
    result = perception.observe(capture, source_hint={'kind': 'cube', 'color': 'any'})
    assert calls == ['cube', 'block'] * 3
    assert len(result['objects']) == 2
    assert all(len(view['instance_masks']) == 2 for view in result['views'].values())
    perception.observe(capture, source_hint={'kind': 'cube', 'color': 'any'})
    assert len(calls) == 6


def test_observe_fuses_only_masked_measured_points_under_decisions_ids():
    module = perception_module()
    capture, masks = capture_fixture()
    observation = GeometryPerception(FrozenSegmenter(masks)).observe(capture, instruction="pick red cube")
    assert observation["source"] == "rgbd" and observation["observation_id"] == "capture-one"
    assert len(observation["objects"]) == 2
    identifiers = {obj["id"] for obj in observation["objects"]}
    assert len(identifiers) == 2 and "SIM_SECRET" not in identifiers
    for obj in observation["objects"]:
        assert obj["kind"] == "cube" and obj["color"] == "red" and obj["estimated"] is True
        assert obj["coverage"]["observed_surface_only"] is True
        assert set(obj["coverage"]["camera_ids"]) == {"1", "2", "3"}
        points = np.asarray(obj["points_world_m"])
        assert points.shape[1] == 3 and len(points) >= 30
        np.testing.assert_allclose(points[:, 2], .7, atol=1e-7)
        np.testing.assert_allclose(obj["visible_centroid_world"], points.mean(axis=0), atol=1e-6)
        assert "T_world_object" not in obj and "size" not in obj
    assert observation["table"]["surface_height_m"] == pytest.approx(.6, abs=.001)
    for view in observation["views"].values():
        assert set(view["instance_masks"]) == identifiers
    public = module.public_observation(observation)
    assert "views" not in public and "SIM_SECRET" not in str(public)
    assert all("points_world_m" not in obj and "points_object_m" not in obj for obj in public["objects"])


def test_scene_update_reuses_table_without_rebuilding_unused_background_points(monkeypatch):
    module = perception_module()
    capture, masks = capture_fixture()
    engine = GeometryPerception(FrozenSegmenter(masks))
    initial = engine.observe(capture, source_hint={'camera_ids': ['3']})
    background = ~np.logical_or.reduce(masks)
    background_calls = []
    deproject = module.deproject_view

    def tracked_deproject(view, mask=None):
        if view['camera_id'] == '3' and np.array_equal(mask, background):
            background_calls.append(view['frame_id'])
        return deproject(view, mask)

    monkeypatch.setattr(module, 'deproject_view', tracked_deproject)
    for view in capture['views'].values():
        for mask in masks:
            view['depth_m'][mask] = .25
    current = engine.observe(capture, source_hint={'table': initial['table']})
    assert background_calls == []
    np.testing.assert_equal(current['table'], initial['table'])
    for obj in current['objects']:
        np.testing.assert_allclose(obj['points_world_m'][:, 2], .75)
        assert set(obj['coverage']['camera_ids']) == {'1', '2', '3'}


def test_default_perception_uses_visual_inventory_and_requires_its_selected_purpose(monkeypatch):
    from geometry_perception_fixture import ScriptedDiscovery, ScriptedVisualInventory

    module = perception_module()
    capture, masks = capture_fixture()
    inventory = ScriptedVisualInventory(('decisions-first', 'decisions-second'))
    monkeypatch.setattr(module.VisualInventory, 'from_environment', lambda: inventory)
    monkeypatch.setattr(module.VisualDiscovery, 'from_environment', lambda: ScriptedDiscovery(('decisions-first', 'decisions-second')))
    engine = module.Perception(FrozenSegmenter(masks))
    with pytest.raises(module.PerceptionError, match='visual_decisions_selection_required'):
        engine.observe(capture)
    result = engine.observe(capture, source_hint={
        'kind': 'cube', 'visual_decision': {'source': 'jev', 'purpose': 'initialize_inventory'}})
    assert {obj['id'] for obj in result['objects']} == {'decisions-first', 'decisions-second'}
    assert all(obj['identity_source'] == 'openai_decisions' for obj in result['objects'])


def test_visual_identity_failure_never_falls_back_to_unambiguous_measured_geometry():
    module = perception_module()
    capture, masks = capture_fixture()

    class UnknownInventory:
        def resolve(self, capture, detections, **kwargs):
            assert len(detections) == 6
            raise module.VisualInventoryError('visual_identity_unknown', evidence={'status': 'unknown'})

    from geometry_perception_fixture import ScriptedDiscovery
    engine = module.Perception(FrozenSegmenter(masks), visual_inventory=UnknownInventory(),
                               visual_discovery=ScriptedDiscovery(('first', 'second')))
    with pytest.raises(module.PerceptionError, match='visual_identity_unknown') as raised:
        engine.observe(capture, source_hint={
            'kind': 'cube', 'visual_decision': {'source': 'jev', 'purpose': 'initialize_inventory'}})
    assert raised.value.details == {'status': 'unknown'}


@pytest.mark.parametrize("kind", ["box", "cylinder"])
def test_airborne_measured_object_does_not_extend_its_height_back_to_table(kind):
    module = perception_module()
    x, y, z = np.meshgrid(np.linspace(-.02, .02, 7), np.linspace(-.02, .02, 7), [.68, .72])
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    group = [{"entity_id": "fixture-entity", "kind": kind, "camera_id": "3", "points": points,
              "pixels": np.full((len(points), 3), [220, 20, 20], np.uint8), "score": .9}]
    observed = module._estimate_object(group, {"surface_height_m": .60, "uncertainty_m": .003})
    assert np.ptp(observed["points_world_m"][:, 2]) == pytest.approx(.04, abs=1e-6)
    assert observed["position"][2] == pytest.approx(.70, abs=1e-6)
    assert "height_inferred_with_table_support" not in observed["coverage"]["limitations"]


def test_actual_gripper_position_and_limits_survive_observe_and_public_projection():
    module = perception_module()
    capture, masks = capture_fixture()
    capture["arms"]["A"].update(gripper_position_rad=.43, gripper_limits_rad=[0., .81],
                                 gripper_closed_fraction=.53)
    capture["arms"]["B"].update(gripper_position_rad=.19, gripper_limits_rad=[-.1, .7],
                                 gripper_closed_fraction=.36)
    observation = GeometryPerception(FrozenSegmenter(masks)).observe(capture)
    public = module.public_observation(observation)
    for arm in ("A", "B"):
        for key in ("gripper_position_rad", "gripper_limits_rad"):
            assert observation["arms"][arm].get(key) == capture["arms"][arm][key]
            assert public["arms"][arm].get(key) == capture["arms"][arm][key]
    capture["arms"]["A"]["gripper_limits_rad"][1] = 9.
    assert observation["arms"]["A"]["gripper_limits_rad"] == [0., .81]


def test_partial_views_with_distant_medians_fuse_only_their_shared_measured_surface():
    capture, _ = capture_fixture()
    masks = {}
    for camera_id, column in {"3": 2, "1": 15, "2": 28}.items():
        view = capture["views"][camera_id]
        mask = np.zeros((80, 80), bool)
        mask[30:50, column:column + 52] = True
        masks[camera_id] = mask
        view["rgb"][:] = 140
        view["rgb"][mask] = [220, 20, 20]
        view["depth_m"][:] = .8
        view["depth_m"][mask] = .5

    class PartialSurfaceDetector:
        def segment(self, rgb, text, timestamp_us):
            camera_id = next(key for key, view in capture["views"].items() if view["rgb"] is rgb)
            if text in {"box", "table"}:
                mask = masks[camera_id] if text == "box" else ~masks[camera_id]
                return [{"mask": mask, "score": .9, "label": text}]
            return []

    # The head/left visible medians differ by 130 mm, but the masks have 50%
    # overlap at identical measured depth. Partial-view centres are not poses.
    observation = GeometryPerception(PartialSurfaceDetector(), identities=('surface',)).observe(capture, source_hint={"kind": "box", "color": "any"})
    assert len(observation["objects"]) == 1
    instance = observation["objects"][0]
    assert set(instance["coverage"]["camera_ids"]) == {"1", "2", "3"}
    np.testing.assert_allclose(instance["points_world_m"][:, 2], .5, atol=1e-7)


def test_capture_mismatch_and_detector_failure_never_return_truth():
    module = perception_module()
    capture, masks = capture_fixture()
    capture["views"]["2"]["capture_time"] = 2.01
    with pytest.raises(module.PerceptionError, match="same captured frame"):
        GeometryPerception(FrozenSegmenter(masks)).observe(capture)

    class Offline:
        def segment(self, *args):
            raise module.PerceptionError("sam3_service_unavailable")

    capture, _ = capture_fixture()
    with pytest.raises(module.PerceptionError, match="sam3_service_unavailable"):
        GeometryPerception(Offline()).observe(capture)


def test_nonfinite_intrinsics_and_detector_confidence_fail_at_sensor_boundary():
    module = perception_module()
    capture, masks = capture_fixture()
    capture["views"]["1"]["intrinsics"]["cx"] = np.nan
    with pytest.raises(module.PerceptionError, match="calibration"):
        GeometryPerception(FrozenSegmenter(masks)).observe(capture)
    capture, masks = capture_fixture()

    class NonfiniteDetector(FrozenSegmenter):
        def segment(self, *args):
            result = super().segment(*args)
            for detection in result:
                detection["score"] = np.nan
            return result

    with pytest.raises(module.PerceptionError, match="invalid_sam3_detection"):
        GeometryPerception(NonfiniteDetector(masks)).observe(capture)


def bin_cloud():
    x, y = np.meshgrid(np.linspace(-.04, .04, 11), np.linspace(-.05, .05, 11))
    floor = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .61)))
    rim = []
    for along in np.linspace(-1, 1, 31):
        rim.extend([[along * .06, -.07, .69], [along * .06, .07, .69],
                    [-.06, along * .07, .69], [.06, along * .07, .69]])
    return np.vstack((floor, rim)).astype(np.float32)


def test_real_rgbd_bin_floor_excludes_nearby_vertical_outer_wall_samples():
    module = perception_module()
    from jev_mujoco.cameras import deproject_view

    root = Path(__file__).parent / "artifacts" / "rgbd" / "perception-windows"
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    view = result["cameras"]["3"]
    with np.load(root / "registered-rgbd.npz") as raw, np.load(root / "sam3-masks.npz") as masks:
        view.update(depth_m=raw["camera_3_depth_m"], valid_depth=raw["camera_3_valid_depth"])
        points = deproject_view(view, masks["request_4_mask_0"].astype(bool))
    reference = module.measured_bin_reference(points, {"surface_height_m": .6008233, "uncertainty_m": .003})
    assert reference["floor_z_m"] == pytest.approx(.611, abs=.002)
    assert any(np.allclose(point, reference["position_world_m"]) for point in points)


def test_real_cube_mask_table_boundary_does_not_expand_the_measured_grasp_cloud():
    module = perception_module()
    from jev_mujoco.cameras import deproject_view

    root = Path(__file__).parent / "artifacts" / "rgbd" / "perception-windows"
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    view = result["cameras"]["3"]
    with np.load(root / "registered-rgbd.npz") as raw, np.load(root / "sam3-masks.npz") as masks:
        view.update(depth_m=raw["camera_3_depth_m"], valid_depth=raw["camera_3_valid_depth"])
        mask = masks["request_0_mask_0"].astype(bool)
        points = deproject_view(view, mask)
        pixels = raw["camera_3_rgb"][mask & view["valid_depth"]]
    group = [{"entity_id": "fixture-entity", "kind": "cube", "camera_id": "3", "points": points,
              "pixels": pixels, "score": .76953125}]
    observed = module._estimate_object(group, {"surface_height_m": .6008233, "uncertainty_m": .003})
    assert "size" not in observed
    np.testing.assert_allclose(observed["position"], observed["points_world_m"].mean(axis=0), atol=1e-6)
    assert np.min(observed["points_world_m"][:, 2]) > .6038
    assert "table_plane_pixels_removed" in observed["coverage"]["limitations"]


def test_bin_floor_reference_remains_available_when_some_walls_are_occluded():
    module = perception_module()
    table = {"surface_height_m": .6, "uncertainty_m": .003}
    for cloud in (bin_cloud(), bin_cloud()[bin_cloud()[:, 0] < .055]):
        reference = module.measured_bin_reference(cloud, table)
        assert reference["floor_z_m"] == pytest.approx(.61, abs=1e-6)
        assert any(np.allclose(point, reference["position_world_m"]) for point in cloud)


def test_bin_floor_reference_can_be_measured_across_complementary_occluded_views():
    capture, _ = capture_fixture()
    points = bin_cloud()
    floor, rim = points[:121], points[121:]
    partial_clouds = {
        "3": np.vstack((floor, rim[(rim[:, 0] < -.055) | (rim[:, 1] < -.065)])),
        "1": np.vstack((floor, rim[rim[:, 0] > .055])),
        "2": np.vstack((floor, rim[rim[:, 1] > .065])),
    }
    masks = {}
    for camera_id, cloud in partial_clouds.items():
        view = capture["views"][camera_id]
        view["T_world_camera"] = np.eye(4).tolist()
        view["rgb"][:] = 140
        view["rgb"][0, 0] = [int(camera_id), 0, 0]
        view["depth_m"][:] = .8
        mask = np.zeros((80, 80), bool)
        columns = np.round(40 + 100 * cloud[:, 0] / cloud[:, 2]).astype(int)
        rows = np.round(40 + 100 * cloud[:, 1] / cloud[:, 2]).astype(int)
        view["depth_m"][rows, columns] = cloud[:, 2]
        mask[rows, columns] = True
        view["rgb"][mask] = [20, 40, 230]
        masks[camera_id] = mask

    class ComplementaryMasks:
        def segment(self, rgb, text, timestamp_us):
            mask = masks[str(int(rgb[0, 0, 0]))]
            if text in {"open box", "blue open box"}:
                return [{"mask": mask, "score": .9, "label": text}]
            if text == "table":
                return [{"mask": ~mask, "score": .9, "label": text}]
            return []

    observation = GeometryPerception(ComplementaryMasks(), identities=('bin',)).observe(capture, checkpoint="fine", source_hint={"kind": "bin", "color": "any"})
    assert len(observation["objects"]) == 1
    container = observation["objects"][0]
    assert container["kind"] == "bin" and container["color"] == "blue"
    assert set(container["coverage"]["camera_ids"]) == {"1", "2", "3"}
    assert container["placement_reference"]["floor_z_m"] == pytest.approx(.61, abs=.001)


def wire_array(array, unit, frame):
    return {"dtype": str(array.dtype), "rank": array.ndim, "shape": list(array.shape),
            "unit": unit, "frame": frame, "data": array.tobytes()}


def test_real_http_sam3_transport_decodes_pinned_masks_and_rejects_wrong_identity():
    module = perception_module()
    identity = dict(module.SAM3_MODEL_IDENTITY)
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                request = msgpack.unpackb(self.rfile.read(int(self.headers["Content-Length"])), raw=False)
                assert set(request) == {"operation", "payload"} and request["operation"] == "segment_text"
                payload = request["payload"]
                assert set(payload) == {"rgb_jpeg", "text", "timestamp_us"}
                assert payload["text"] == "cube" and payload["timestamp_us"] == 2_000_000
                image = np.asarray(Image.open(BytesIO(payload["rgb_jpeg"])))
                assert image.shape == (12, 14, 3) and image[0, 0, 0] > 200
                masks = np.ones((1, 12, 14), np.uint8)
                result = {"masks": wire_array(masks, "mask", "image"),
                          "boxes": wire_array(np.array([[0, 0, 14, 12]], np.float32), "pixel", "image"),
                          "scores": wire_array(np.array([.95], np.float32), "confidence", "model"),
                          "selected_index": 0, "model": dict(identity),
                          "timing_us": {"inference": 50, "total": 80}, "timestamp_us": payload["timestamp_us"]}
                response = msgpack.packb({"ok": True, "result": result}, use_bin_type=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/msgpack")
                self.end_headers()
                self.wfile.write(response)
            except Exception as error:  # noqa: BLE001 - propagate server-thread assertion failures to this test.
                errors.append(error)
                self.send_error(500)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = module.Sam3Client(f"http://127.0.0.1:{server.server_port}")
        rgb = np.full((12, 14, 3), [240, 10, 10], np.uint8)
        detections = client.segment(rgb, " cube ", 2_000_000)
        assert len(detections) == 1 and detections[0]["mask"].all()
        assert detections[0]["score"] == pytest.approx(.95) and detections[0]["label"] == "cube"
        identity["source_sha"] = "wrong"
        with pytest.raises(module.PerceptionError, match="sam3_model_identity_mismatch"):
            client.segment(rgb, "cube", 2_000_000)
        assert not errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_decisions_group_preserves_complementary_wrist_geometry():
    capture, _ = capture_fixture()
    masks = {}
    for camera, view in capture['views'].items():
        mask = np.zeros((80, 80), bool)
        columns = slice(20, 60) if camera == '3' else slice(20, 40) if camera == '2' else slice(40, 60)
        mask[30:55, columns] = True
        view['rgb'][:] = 140
        view['rgb'][mask] = [220, 20, 20]
        view['depth_m'][:] = .4
        view['depth_m'][mask] = .3
        masks[camera] = mask
    class Detector:
        def segment(self, rgb, text, timestamp_us):
            if text != 'red cube':
                return []
            camera = next(c for c, v in capture['views'].items() if v['rgb'] is rgb)
            return [{'score': .99, 'mask': masks[camera]}]
    result = GeometryPerception(detector=Detector(), identities=('source',)).observe(capture, checkpoint='fine', source_hint={'kind':'cube', 'color':'red'})
    assert len(result['objects']) == 1
    assert set(result['objects'][0]['coverage']['camera_ids']) == {'1','2','3'}
