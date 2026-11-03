"""Inventory-only demo maps retain all listed tables without any background."""
from copy import deepcopy

import numpy as np
import pytest
from test_perception_mapping import wire
from test_rgbd_algorithms import observed_scene
from test_semantic_builder import builder, request

from luna_mujoco.algorithms import Algorithms
from luna_mujoco.perception_mapping import decode_mapping_payload, mapping_builder_payload


def table_evidence():
    table = {"model": {"pose": [.55, 0., .45, 1., 0., 0., 0.], "dims": [.8, 1., .3]},
             "surface_height_m": .6, "normal": [0., 0., 1.], "inlier_count": 100,
             "source": "initial_inventory_ransac_table"}
    side = deepcopy(table)
    side["model"]["pose"][1] = .75
    side["model"]["dims"][:2] = [.25, .25]
    return {"source_observation_id": "capture-42", "primary_table_id": "table",
            "tables": {"table": table, "side": side}}


def test_demo_mapping_is_validated_and_bound_to_scene_identity():
    original = wire()
    _, legacy_identity = decode_mapping_payload(original)
    demo, identity = decode_mapping_payload(original | {"demo_mapping": True, "demo_tables": table_evidence()})
    assert demo["scene_policy"] == "demo_inventory"
    assert identity["scene_token"] != legacy_identity["scene_token"]
    with pytest.raises(ValueError, match="demo_mapping"):
        decode_mapping_payload(original | {"demo_mapping": "yes"})
    with pytest.raises(ValueError, match="demo_initial_table_evidence_required"):
        decode_mapping_payload(original | {"demo_mapping": True})


def test_demo_discards_background_and_reuses_all_initial_table_cuboids():
    value, segmenter, mapper, table_inputs = builder(support=True)
    payload = request("selected") | {"demo_mapping": True, "scene_policy": "demo_inventory",
                                   "demo_tables": table_evidence()}
    def reject_crop(*args, **kwargs):
        raise AssertionError("main-table crop deletes listed side-table entities")
    segmenter.crop_semantic_observation = reject_crop
    candidate = value.build(payload)
    assert table_inputs == []
    assert not segmenter.separated
    assert [view.background_point_count for view in mapper.integrated] == [0, 0, 0]
    assert set(candidate.scene_model["cuboid"]) == {"semantic_table::table", "semantic_table::side"}
    assert value.diagnostics["semantic_only_active"] is True
    assert value.diagnostics["background_drop_policy"] == "discard_all_unlisted_and_tabletop_returns"
    frozen = deepcopy(dict(candidate.scene_model["cuboid"]))
    candidate = value.build(payload | {"observation_set_id": "new-observation"})
    assert value.diagnostics["table_cache"]["cache_hit"] is True
    assert dict(candidate.scene_model["cuboid"]) == frozen
    changed = deepcopy(payload)
    changed["demo_tables"]["tables"]["side"]["model"]["pose"][0] += .01
    with pytest.raises(ValueError, match="demo_table_inventory_changed_requires_reset"):
        value.build(changed)


def test_demo_rejects_raw_background_mapping():
    value, _, _, _ = builder()
    payload = request("observation") | {"demo_mapping": True, "mapping_mode": "background",
                                       "confirmed_instance_id": "", "sam3_generation": 0}
    for view in payload["views"].values():
        view["instance_masks"] = {}
    with pytest.raises(ValueError, match="demo_inventory_semantic_mapping_required"):
        value.build(payload)


def test_demo_partial_views_remain_all_three_without_background_fallback():
    decoded, _ = decode_mapping_payload(wire() | {"demo_mapping": True, "demo_tables": table_evidence()})
    decoded["views"]["1"]["depth_left_m"][:] = 0
    decoded["views"]["1"]["valid_left_mask"][:] = 0
    result = mapping_builder_payload(decoded)
    assert list(result["views"]) == ["1", "2", "3"]
    assert not np.any(result["views"]["1"]["depth_left_m"])


def test_asset_identity_never_adds_geometry_priors_to_native_request(monkeypatch):
    import builtins

    original_import = builtins.__import__
    def reject_cad_import(name, *args, **kwargs):
        if name.rsplit(".", 1)[-1] in {"cad_assets", "trimesh"}:
            raise AssertionError("mapping must not import CAD geometry")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_cad_import)
    snapshot = observed_scene()
    snapshot.update(demo_mapping=True, demo_tables=table_evidence())
    snapshot['objects'][0].update(kind='bin', asset_id='blue_box')
    sent = []
    Algorithms(curobo_request=lambda path, body: (sent.append(body) or {'scene_token': 'map'})).build_scene(
        snapshot, 'seen-red')
    decoded, _ = decode_mapping_payload(sent[0])
    assert 'demo_bin_priors' not in sent[0] and 'demo_completions' not in sent[0]
    assert 'demo_bin_priors' not in decoded and 'demo_completions' not in decoded


def test_old_completed_snapshot_is_rejected_before_any_service_call():
    from luna_mujoco.algorithms import AlgorithmError

    snapshot = observed_scene()
    snapshot['objects'][0]['completed_geometry_points_world_m'] = [[0, 0, 0]]
    def service(*args):
        raise AssertionError('obsolete synthetic geometry reached service')
    with pytest.raises(AlgorithmError, match='^synthetic_scene_geometry_unsupported:reobserve_required$'):
        Algorithms(curobo_request=service).build_scene(snapshot, 'seen-red')
