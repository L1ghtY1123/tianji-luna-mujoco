"""Current OBB/CAD geometry replaces the same ID; unseen layers retain provenance."""
from copy import deepcopy

import pytest
from scipy.spatial.transform import Rotation

from luna_mujoco.inventory_map import (
    apply_inventory_update,
    freeze_inventory_scene,
    reconcile_inventory,
    require_native_inventory_geometry,
)
from luna_mujoco.sdk_scene import sdk_scene_model


def obb(capture, identifier="a", *, center=(.4, .2, .7), angle=45):
    quaternion = Rotation.from_euler("z", angle, degrees=True).as_quat(scalar_first=True).tolist()
    return {"pose": [*center, *quaternion], "dims": [.2, .08, .1],
            "geometry_source": "observed_obb", "instance_id": identifier,
            "observation_id": capture, "generation": 1}


def scene(capture, layers):
    identity = {"scene_token": capture, "observation_id": capture, "generation": 1,
                "target_id": "", "exclusion_state": "observation", "disabled_obstacle_names": ()}
    model = {"cuboid": {}, "mesh": {}, "voxel": {}}
    for layer in layers:
        kind = "mesh" if layer["geometry_source"] == "registered_cad" else "cuboid"
        model[kind]["semantic_instance::" + layer["instance_id"]] = layer
    return identity, [], model


def test_current_obb_replaces_previous_geometry_without_accumulating_old_corners():
    older = obb("old")
    newer = obb("new", center=(.48, .19, .71), angle=-30)
    previous = reconcile_inventory(None, *scene("old", [older]))
    incoming = scene("new", [newer])
    untouched = deepcopy((previous, incoming))
    current = reconcile_inventory(previous, *incoming)
    require_native_inventory_geometry(current[2])
    assert set(current[2]["cuboid"]) == {"semantic_instance::a"}
    updated = current[2]["cuboid"]["semantic_instance::a"]
    assert updated["pose"] == newer["pose"]
    assert updated["dims"] == newer["dims"]
    assert updated["geometry_observation_ids"] == ["new"]
    assert updated["observation_id"] == "new"
    assert (previous, incoming) == untouched
    assert "geometry_observation_ids" not in sdk_scene_model(current[2])["cuboid"]["semantic_instance::a"]


def test_repeated_observations_do_not_inflate_current_obb():
    current = None
    for number in range(20):
        capture = str(number)
        current = reconcile_inventory(current, *scene(capture, [obb(capture, angle=number * 5)]))
        assert len(current[2]["cuboid"]) == 1
        assert current[1] == []
        layer = current[2]["cuboid"]["semantic_instance::a"]
        assert layer['dims'] == obb(capture)['dims']
        assert layer['pose'] == obb(capture, angle=number * 5)['pose']
        freeze_inventory_scene(current)
    assert current[2]["cuboid"]["semantic_instance::a"]["geometry_observation_ids"] == ['19']


def test_local_map_excluded_entity_keeps_exact_frozen_geometry_and_capture():
    previous = reconcile_inventory(None, *scene("old", [obb("old"), obb("old", "b")]))
    before = deepcopy(previous[2]["cuboid"]["semantic_instance::b"])
    current = reconcile_inventory(previous, *scene("new", [obb("new"), obb("new", "b", center=(.8, .3, .7))]),
        requested_scope="local", mapping_entity_ids=["a"], strategy_source="decisions", scope_reason="Refresh a")
    assert current[2]["cuboid"]["semantic_instance::b"] == before
    record = current[0]["inventory_map"]["entities"]["b"]
    assert record["last_observation_id"] == "old" and record["status"] == "retained"


def test_action_invalidation_replaces_obb_instead_of_keeping_old_pose():
    previous = reconcile_inventory(None, *scene("old", [obb("old")]))
    apply_inventory_update(previous[0], {"operation": "invalidate", "target_id": "a"}, previous[2])
    unseen = reconcile_inventory(previous, *scene("unseen", []))
    assert unseen[0]["inventory_map"]["entities"]["a"]["status"] == "stale"
    fresh = obb("fresh", center=(.9, -.2, .7), angle=-20)
    current = reconcile_inventory(unseen, *scene("fresh", [fresh]))
    layer = current[2]["cuboid"]["semantic_instance::a"]
    assert layer["pose"] == fresh["pose"] and layer["dims"] == fresh["dims"]
    assert layer["geometry_observation_ids"] == ["fresh"]
    assert current[0]["inventory_map"]["entities"]["a"]["pose_replaced_after_invalidation"] is True


def test_current_cad_replaces_old_obb_and_subsequent_pose_without_ghost_layers():
    previous = reconcile_inventory(None, *scene("old", [obb("old")]))
    mesh = {"pose": [.5, 0, .7, 1, 0, 0, 0], "file_path": "/verified/mesh.stl",
            "geometry_source": "registered_cad", "instance_id": "a", "observation_id": "cad-1",
            "generation": 1, "asset_id": "part", "source_sha256": "a" * 64, "mesh_sha256": "b" * 64}
    current = reconcile_inventory(previous, *scene("cad-1", [mesh]))
    assert not current[2]["cuboid"]
    assert set(current[2]["mesh"]) == {"semantic_instance::a"}
    later = deepcopy(mesh)
    later.update(observation_id="cad-2", pose=[.6, 0, .7, 1, 0, 0, 0])
    current = reconcile_inventory(current, *scene("cad-2", [later]))
    assert current[2]["mesh"] == {"semantic_instance::a": later}


@pytest.mark.parametrize("field,value", [("dims", [-1, 1, 1]), ("pose", [0, 0, 0, 0, 0, 0, 0]),
                                         ("geometry_observation_ids", ["old", None])])
def test_invalid_hybrid_geometry_cannot_be_archived(field, value):
    layer = obb("old")
    layer[field] = value
    with pytest.raises(ValueError, match="scene_geometry_policy_requires_rebuild"):
        reconcile_inventory(None, *scene("old", [layer]))
