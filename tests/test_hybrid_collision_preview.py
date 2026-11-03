"""Published CAD diagnostics use the exact verified mesh, never a replacement box."""
import struct
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from test_hybrid_map_lifecycle import obb, scene

from luna_mujoco.cad_perception import CadCatalog
from luna_mujoco.collision_mapping import collision_layer_bounds, collision_scene_preview
from luna_mujoco.demo_lift import _reference_layers
from luna_mujoco.hybrid_mapping import build_geometry_model


@pytest.fixture
def published():
    catalog = CadCatalog.from_file()
    record = catalog.record("THREADED_VISUAL_SAMPLE_MM")
    geometry = catalog.geometry(record["mesh_key"])
    world = np.array([[0, -1, 0, .4], [1, 0, 0, -.2], [0, 0, 1, .8], [0, 0, 0, 1.]])
    row = {"id": "part", "kind": "object", "observation_id": "capture", "generation": 1,
           "cad": {"instance_id": "part", "asset_id": record["asset_id"], "mesh_key": record["mesh_key"],
                   "mesh_sha256": geometry["mesh_sha256"],
                   "source_sha256": record["source_sha256"], "observation_id": "capture", "generation": 1,
                   "T_world_object": world.tolist(),
                   "T_world_source": (world @ np.linalg.inv(geometry["T_source_canonical_m"])).tolist(),
                   "pose_validation": {"accepted": True}}}
    model = build_geometry_model([row], "capture", 1, catalog=catalog)
    model["cuboid"]["semantic_table::table"] = {"pose": [.5, 0, .45, 1, 0, 0, 0], "dims": [1, 1, .3]}
    identity = scene("capture", [])[0]
    identity["disabled_obstacle_names"] = ["semantic_instance::part"]
    table = SimpleNamespace(model=model["cuboid"]["semantic_table::table"], height_m=.6,
                            normal_xyz=(0, 0, 1), inlier_count=100)
    candidate = SimpleNamespace(table=table, camera_metrics={},
                               integration=SimpleNamespace(pointcloud_data=struct.pack("<I", 0)))
    return identity, model, geometry, world, candidate


def test_cad_bounds_cover_full_rotated_mesh_and_retain_fingerprint(published):
    identity, model, geometry, world, _ = published
    original = deepcopy(model)
    result = collision_layer_bounds([], model, identity)
    layer = result["layers"]["semantic_instance::part"]
    points = geometry["vertices_m"] @ world[:3, :3].T + world[:3, 3]
    np.testing.assert_allclose(layer["minimum_world_m"], points.min(0))
    np.testing.assert_allclose(layer["maximum_world_m"], points.max(0))
    assert layer["source"] == "published_cad_mesh_bounds" and layer["kind"] == "mesh"
    assert layer["enabled"] is False
    assert layer["mesh_sha256"] == geometry["mesh_sha256"]
    assert model == original


def test_preview_preserves_every_mesh_triangle_with_deduplicated_vertices(published):
    identity, model, geometry, _, candidate = published
    result = collision_scene_preview(candidate, [], model, identity)
    layer = result["mesh_layers"]["semantic_instance::part"]
    vertices, faces = np.asarray(layer["vertices_local_m"]), np.asarray(layer["faces"])
    np.testing.assert_array_equal(vertices[faces], geometry["vertices_m"][geometry["faces"]])
    assert len(vertices) <= len(geometry["vertices_m"])
    assert layer["sampling"] == "none" and layer["enabled"] is False
    assert layer["instance_id"] == "part" and layer["geometry_source"] == "registered_cad"
    assert layer["pose_wxyz"] == model["mesh"]["semantic_instance::part"]["pose"]


@pytest.mark.parametrize("field,value", [("mesh_sha256", "f" * 64), ("source_sha256", "f" * 64),
                                         ("file_path", "C:/untrusted.stl"), ("scale", [2, 2, 2])])
def test_unverified_or_altered_mesh_cannot_publish_bounds(published, field, value):
    identity, model, _, _, _ = published
    model["mesh"]["semantic_instance::part"][field] = value
    with pytest.raises(ValueError):
        collision_layer_bounds([], model, identity)


def test_obb_preview_is_an_instance_and_bounds_can_supply_lift_height():
    identity, _, model = scene("capture", [obb("capture")])
    table = SimpleNamespace(model={"pose": [0, 0, 0, 1, 0, 0, 0], "dims": [1, 1, .1]},
                            height_m=.05, normal_xyz=(0, 0, 1), inlier_count=100)
    candidate = SimpleNamespace(table=table, camera_metrics={},
                               integration=SimpleNamespace(pointcloud_data=struct.pack("<I", 0)))
    preview = collision_scene_preview(candidate, [], model, identity)
    assert preview["cuboid_layers"][0]["role"] == "semantic_instance"
    bounds = collision_layer_bounds([], model, identity)
    layers = _reference_layers(bounds, identity, "a")
    assert layers[0]["geometry_source"] == "observed_obb"


def test_real_mapper_readonly_camera_metrics_can_be_exported(published):
    identity, model, _, _, candidate = published
    original = {'3': {'point_count': 100}}
    candidate.camera_metrics = MappingProxyType(original)
    preview = collision_scene_preview(candidate, [], model, identity)
    assert preview['status'] == 'available'
    preview['camera_metrics']['3']['point_count'] = 0
    assert original['3']['point_count'] == 100


def test_cad_lift_bounds_require_enabled_exact_registered_instance(published):
    identity, model, _, _, _ = published
    identity["disabled_obstacle_names"] = []
    bounds = collision_layer_bounds([], model, identity)
    layer = _reference_layers(bounds, identity, "part")[0]
    assert layer["source"] == "published_cad_mesh_bounds"
    bounds["layers"]["semantic_instance::part"]["geometry_source"] = "unverified_mesh"
    with pytest.raises(ValueError, match="grasp_lift_source_bounds_unavailable"):
        _reference_layers(bounds, identity, "part")
