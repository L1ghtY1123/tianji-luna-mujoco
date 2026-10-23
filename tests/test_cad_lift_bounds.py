"""The first real CAD run must use its validated mesh shell for source exit."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from luna_mujoco.demo_lift import source_clearance_lift


def frozen():
    directory = Path(__file__).parent / "fixtures"
    value = json.loads((directory / "cad_lift_dafba328.json").read_text(encoding="utf-8"))
    with np.load(directory / "cad_lift_dafba328.npz", allow_pickle=False) as arrays:
        for obj in value["snapshot"]["objects"]:
            for key in ("points_world_m", "completed_geometry_points_world_m"):
                if key in obj:
                    obj[key] = arrays[obj[key]["fixture_array"]]
    return value


def derive(value):
    return source_clearance_lift(value["snapshot"], value["target_id"], value["arm"],
                                 value["candidates"], value["scene"], value["settings"])


def test_actual_cad_source_lift_uses_all_49_mesh_bounds_without_changing_scene():
    value = frozen()
    before = copy.deepcopy(value["scene"])
    settings, proof = derive(value)
    prefix = "semantic_instance::" + proof["source_container_id"] + "::bin_shell::"
    parts = [layer for layer in proof["retained_source_layers"] if layer["name"].startswith(prefix)]
    assert len(parts) == 49
    assert all(layer["source"] == "published_mesh_bounds" for layer in parts)
    assert all(layer["geometry_source"] == "known_cad_measured_registration" for layer in parts)
    assert all(layer["synthetic_geometry_is_measured"] is False for layer in parts)
    assert proof["source_neighbor_ids"] == value["expected_neighbor_ids"]
    assert proof["source_occupied_top_world_z_m"] == pytest.approx(max(
        layer["maximum_world_m"][2] for layer in proof["retained_source_layers"]))
    assert settings["lift_offset_m"] >= value["settings"]["lift_offset_m"]
    for candidate in proof["candidates"]:
        assert candidate["lowest_sphere_world_z_m"] + settings["lift_offset_m"] >= (
            proof["source_occupied_top_world_z_m"] + .01 - 1e-12)
    assert [c["candidate_index"] for c in proof["candidates"]] == [
        c["candidate_index"] for c in value["candidates"]]
    assert proof["required_clearance_m"] == .01
    assert value["scene"] == before


@pytest.mark.parametrize("damage", ["missing_part", "stale_part", "hash", "disabled", "wrong_id"])
def test_cad_lift_does_not_bypass_complete_current_catalog_validation(damage):
    value = frozen()
    layers = value["scene"]["collision_layer_bounds"]["layers"]
    source = value["snapshot"]["scene_entities"][value["target_id"]]["initial_parent_id"]
    name = f"semantic_instance::{source}::bin_shell::cad_part_00"
    if damage == "missing_part":
        layers.pop(name)
    elif damage == "stale_part":
        layers[name]["observation_id"] = "old"
    elif damage == "hash":
        layers[name]["part_sha256"] = "unverified"
    elif damage == "disabled":
        value["scene"]["disabled_obstacle_names"].append(name)
    else:
        layers[name]["instance_id"] = "another-bin"
    with pytest.raises(ValueError, match="demo_bin_shell"):
        derive(value)


@pytest.mark.parametrize("role", ["source", "neighbor", "height_completion"])
def test_mesh_type_is_not_authorized_for_unverified_source_or_neighbor_layers(role):
    value = frozen()
    source = value["snapshot"]["scene_entities"][value["target_id"]]["initial_parent_id"]
    identifier = source if role == "source" else value["expected_neighbor_ids"][0]
    name = "semantic_instance::" + identifier
    layers = value["scene"]["collision_layer_bounds"]["layers"]
    if role == "height_completion":
        layer = copy.deepcopy(layers[name])
        name += "::height_completion"
        layers[name] = layer | {"kind": "mesh"}
    layers[name]["source"] = "published_mesh_bounds"
    with pytest.raises(ValueError, match="grasp_lift_source_bounds_unavailable:" + name):
        derive(value)
