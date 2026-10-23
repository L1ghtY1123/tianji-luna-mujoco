"""Observation inventory bounds every demo cloud and collision layer."""
from copy import deepcopy

import numpy as np
import pytest
from test_demo_perception import demo_discovery, discovery_output
from test_perception import capture_fixture
from test_perception_mapping import wire

from luna_mujoco.demo_perception import DemoPerception
from luna_mujoco.perception import PerceptionError
from luna_mujoco.perception_mapping import decode_mapping_payload


def test_source_queries_have_four_bounded_shape_neutral_variants():
    capture, _ = capture_fixture()
    result = demo_discovery(discovery_output(), []).discover(capture, instruction="move all pink objects")
    for identifier in result["demo_task"]["source_object_ids"]:
        entity = result["entities"][identifier]
        assert entity["segmentation_query"]["prompt"] == "pink object"
        assert entity["segmentation_prompts"] == [
            "pink object", "pink foam wrapped object", "object inside the left blue box", "elliptical pink object"]
    queries = DemoPerception._segmentation_queries(result["entities"])
    assert len(queries) == 5  # Four common source descriptions and the blue box.
    assert all("cup" not in row["prompt"] for row in queries)


def test_demo_has_no_raw_bootstrap_mapping_entry():
    with pytest.raises(PerceptionError, match="demo_initial_observation_required"):
        DemoPerception().bootstrap(capture_fixture()[0])


def test_first_inventory_tables_are_fitted_once_and_reused_without_table_voxels():
    capture, _ = capture_fixture()
    entities = {"table": {"kind": "table", "bbox": [0, 0, 1, 1], "reference_camera_id": "3"},
                "side": {"kind": "table", "bbox": [0, 0, 1, 1], "reference_camera_id": "2"}}
    engine = DemoPerception(robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}})
    reference = {"observation_id": capture["observation_id"], "demo_task": {"destination_table_id": "table"}}
    table = engine._table_for_entities(capture, {}, entities, reference, [])
    assert table["surface_height_m"] == pytest.approx(.6)
    assert table["source"] == "initial_inventory_ransac_table"
    assert set(engine.demo_tables["tables"]) == {"table", "side"}
    frozen = deepcopy(engine.demo_tables)
    changed = deepcopy(capture)
    changed["observation_id"] = "later"
    for view in changed["views"].values():
        view["depth_m"] += .2
    assert engine._table_for_entities(changed, {}, entities, reference, []) == table
    assert engine.demo_tables == frozen


def test_demo_decoder_drops_every_return_outside_named_masks():
    from luna_mujoco.semantic_transport import (
        decode_project_semantic_request,
        encode_project_semantic_request,
    )

    payload = wire()
    decoded = decode_project_semantic_request({key: value for key, value in payload.items()
                                               if key != "gripper_joint_positions_rad"})
    for view in decoded["views"].values():
        view["instance_masks"]["source"][:] = False
        view["instance_masks"]["source"][0, 0] = True
    payload = encode_project_semantic_request(decoded, exclusion_state="selected") | {
        "gripper_joint_positions_rad": payload["gripper_joint_positions_rad"]}
    payload.update(demo_mapping=True, demo_tables={"source_observation_id": "capture-42",
        "primary_table_id": "table", "tables": {"table": {
            "model": {"pose": [.5, 0., .45, 1., 0., 0., 0.], "dims": [.8, 1., .3]},
            "surface_height_m": .6, "normal": [0., 0., 1.], "inlier_count": 100,
            "source": "initial_inventory_ransac_table"}}})
    decoded, identity = decode_mapping_payload(payload)
    for view in decoded["views"].values():
        named = np.logical_or.reduce(list(view["instance_masks"].values())) if view["instance_masks"] else np.zeros_like(
            view["depth_left_m"], bool)
        assert not np.any(view["depth_left_m"][~named])
        assert not np.any(view["valid_left_mask"][~named])
        assert np.count_nonzero(view["depth_left_m"]) == 1
    assert decoded["demo_tables"] == payload["demo_tables"]
    assert identity["collision_world_source"] == "inventory_semantic_esdf_and_cached_table_cuboids"
    assert identity["unlisted_depth_policy"] == "discard_without_free_space_claim"


def test_height_completion_preserves_measured_cloud_and_marks_prior():
    from luna_mujoco.demo_perception import complete_source_height

    points = np.array([[.5+x, .7+y, .7+z] for x in (-.03, .03) for y in (-.02, .02)
                       for z in (-.003, .003)], np.float32)
    obj = {"id": "pink", "points_world_m": points.copy()}
    complete_source_height(obj, support_z_m=.6, support_source="measured_bin_floor",
                           observation_id="capture")
    np.testing.assert_array_equal(obj["points_world_m"], points)
    completion = obj["geometry_completion"]
    assert completion["height_prior_m"] == .11
    assert completion["source"] == "measured_surface_and_user_height_prior"
    assert completion["support_z_m"] == .6
    assert completion["synthetic_points_are_measured"] is False
    completed = np.asarray(obj["completed_geometry_points_world_m"])
    assert completed[:, 2].min() == pytest.approx(.6)
    assert completed[:, 2].max() >= points[:, 2].max()


def test_height_completion_rejects_prior_inconsistent_with_measurements():
    from luna_mujoco.demo_perception import complete_source_height

    obj = {"id": "pink", "points_world_m": np.array([[.5, .7, .85], [.51, .71, .86]])}
    with pytest.raises(PerceptionError, match="demo_height_prior_inconsistent"):
        complete_source_height(obj, support_z_m=.6, support_source="measured_floor", observation_id="capture")


def test_tabletop_pixels_in_named_object_masks_never_enter_geometry():
    capture, _ = capture_fixture()
    engine = DemoPerception(robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}})
    entities = {"table": {"kind": "table", "bbox": [0, 0, 1, 1], "reference_camera_id": "3"}}
    engine._table_for_entities(capture, {}, entities, {
        "observation_id": capture["observation_id"], "demo_task": {"destination_table_id": "table"}}, [])
    mask = np.zeros((80, 80), bool)
    mask[0, :5] = True
    cloud = np.array([[.04, .04, .6], [.04, .04, .7], [.041, .04, .7],
                      [.04, .041, .7], [.041, .041, .7]])
    group = [{"points": cloud, "pixels": np.ones((5, 3)),
              "pixels_rc": np.array([[0, i] for i in range(5)]), "mask": mask}]
    result = engine._filter_geometry_groups([group], {})
    np.testing.assert_allclose(result[0][0]["points"], cloud[1:])
    assert not result[0][0]["mask"][0, 0] and result[0][0]["mask"][0, 1]


def test_sam_attempt_evidence_retains_all_masks_and_unfiltered_clouds():
    capture, masks = capture_fixture()
    engine = DemoPerception()
    evidence = {"prompt": "pink object"}
    engine._record_segmentation_attempt(evidence, capture["views"]["2"],
                                       [{"mask": masks[0], "score": .9}])
    assert len(evidence["rgb_sha256"]) == 64
    proposal = evidence["proposals"][0]
    np.testing.assert_array_equal(proposal["mask"], masks[0])
    assert len(proposal["points_world_m"]) == np.count_nonzero(masks[0])
    assert proposal["identity_status"] == "unassociated_proposal"
    empty = {}
    engine._record_segmentation_attempt(empty, capture["views"]["2"], [])
    assert empty["proposals"] == []


def test_identical_pixel_proposals_deduplicate_without_assigning_permanent_identity():
    _, masks = capture_fixture()
    first = {"camera_id": "2", "silhouette_mask": masks[0], "prompt": "pink object", "score": .8}
    second = {**first, "prompt": "elliptical pink object", "score": .9}
    third = {**first, "silhouette_mask": masks[1]}
    result = DemoPerception()._deduplicate_detections([first, second, third])
    assert len(result) == 2
    assert result[0]["prompt_proposals"] == ["pink object", "elliptical pink object"]
    assert all("entity_id" not in proposal for proposal in result)


def test_decisions_failure_keeps_every_sam_attempt_for_archive():
    from luna_mujoco.visual_api import VisualInventoryError

    capture, masks = capture_fixture()
    class Detector:
        def segment(self, rgb, text, timestamp_us):
            return [{"mask": masks[0], "score": .9}] if text == "pink object" else []
    class Inventory:
        def resolve(self, *args, **kwargs):
            raise VisualInventoryError("visual_identity_unknown", evidence={"confidence": .7})
    engine = DemoPerception(Detector(), visual_inventory=Inventory(),
        visual_discovery=demo_discovery(discovery_output(), []),
        robot_surface_classifier=lambda *args: {"status": "measured", "masks": {}})
    with pytest.raises(PerceptionError, match="visual_identity_unknown") as caught:
        engine.observe(capture, instruction="move all",
            source_hint={"visual_decision": {"source": "responses_program", "purpose": "initialize_inventory"}})
    evidence = caught.value.sam3_queries
    assert len(evidence) == 15  # Three cameras, four pink descriptions and blue box.
    assert sum(len(attempt["proposals"]) for attempt in evidence) == 3
    assert caught.value.initial_inventory["reference"]["observation_id"] == capture["observation_id"]
