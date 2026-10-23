"""Accepted identities need usable geometry; rejected masks leave raw depth intact."""

from copy import deepcopy

import numpy as np
import pytest
from geometry_perception_fixture import GeometryPerception
from test_perception import capture_fixture

from luna_mujoco.perception import PerceptionError


def fixture(*, retained=3, reject_target=False, on_table=False):
    capture, masks = capture_fixture()
    sparse = masks[0].copy() if reject_target else np.zeros((80, 80), bool)
    if not reject_target:
        sparse[55:65, 35:45] = True
    for camera, view in capture["views"].items():
        view["rgb"][0, 0, 0] = int(camera)
        view["rgb"][masks[1]] = [20, 40, 220]
        if not reject_target:
            view["rgb"][sparse] = [40, 40, 40]
            view["depth_m"][sparse] = .3
        if on_table:
            view["depth_m"][sparse] = .4

    class Detector:
        def segment(self, rgb, text, timestamp_us):
            camera = str(rgb[0, 0, 0])
            mask = None
            if text == "cube" and camera == "3":
                mask = masks[0]
            elif text in {"open box", "blue open box"}:
                mask = masks[1]
            elif text == "cylinder" and camera == "2" and not reject_target:
                mask = sparse
            elif text == "table":
                mask = ~(masks[0] | masks[1] | sparse)
            return [] if mask is None else [{"mask": mask, "score": .9}]

    def robot_classifier(capture, union):
        camera = "3" if reject_target else "2"
        removed = sparse.copy()
        rows, columns = np.nonzero(sparse)
        removed[rows[:retained], columns[:retained]] = False
        if on_table:
            removed[:] = False
        return {"masks": {camera: removed}, "matches": {}, "source": "fixture_own_surface_evidence"}

    identities = {'3': ('bin', 'source'), '1': ('bin',),
                  '2': ('bin',) if reject_target else ('bin', 'residual')}
    return capture, masks, sparse, GeometryPerception(
        Detector(), identities=identities, kinds={'bin': 'bin', 'residual': 'cylinder'},
        robot_surface_classifier=robot_classifier)


@pytest.mark.parametrize("retained", [0, 3, 29])
def test_accepted_identity_with_small_robot_residual_rejects_incomplete_geometry(retained):
    capture, masks, sparse, engine = fixture(retained=retained)
    before = deepcopy(capture)
    original_masks = [mask.copy() for mask in (*masks, sparse)]
    result = engine.observe(capture, checkpoint="fine", source_hint={
        "kind": "cube", "color": "any", "destination": {"color": "blue"}})
    assert 'residual' in result['scene_entities']
    assert 'residual' in result['visual_inventory']['missing_geometry_ids']
    assert result['mapping_unresolved'][0]['previous_id'] == 'residual'
    for camera, view in capture["views"].items():
        for field in ("rgb", "depth_m", "valid_depth"):
            np.testing.assert_array_equal(view[field], before["views"][camera][field])
    for current, original in zip((*masks, sparse), original_masks, strict=True):
        np.testing.assert_array_equal(current, original)


@pytest.mark.parametrize("retained", [0, 3, 29])
def test_accepted_target_with_insufficient_geometry_cannot_publish_partial_observation(retained):
    capture, _, _, engine = fixture(retained=retained, reject_target=True)
    result = engine.observe(capture, checkpoint="fine", source_hint={
        "kind": "cube", "color": "any", "destination": {"color": "blue"}})
    assert 'source' in result['scene_entities']
    assert 'source' in result['visual_inventory']['missing_geometry_ids']
    assert any(item['previous_id'] == 'source' for item in result['mapping_unresolved'])


def test_unnamed_depth_remains_unassigned():
    capture, _, sparse, engine = fixture(on_table=True)
    result = engine.observe(capture, checkpoint='fine', source_hint={
        'kind': 'cube', 'destination': {'color': 'blue'}})
    for camera, view in result["views"].items():
        np.testing.assert_array_equal(view["depth_m"], capture["views"][camera]["depth_m"])
        assert not any(np.any(mask & sparse) for mask in view["instance_masks"].values())


def test_other_perception_errors_are_not_suppressed(monkeypatch):
    from luna_mujoco import perception

    capture, _, _, engine = fixture()

    def broken(*args, **kwargs):
        raise PerceptionError("invalid-calibration")

    monkeypatch.setattr(perception, "_estimate_object", broken)
    with pytest.raises(PerceptionError, match="invalid-calibration"):
        engine.observe(capture, checkpoint='fine', source_hint={
            'kind': 'cube', 'destination': {'color': 'blue'}})
