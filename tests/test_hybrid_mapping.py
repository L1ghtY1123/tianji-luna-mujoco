"""CAD/OBB scene composition exercises registered depth without GPU grids."""
import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_demo_mapping import table_evidence
from test_semantic_builder import Segmenter, request
from test_semantic_depth import CameraObservation


def obb_row(identifier="source"):
    return {"id": identifier, "kind": "object", "observation_id": "capture-42", "generation": 4,
            "obb": {"frame": "world", "instance_id": identifier,
                    "source": "observed_points_obb_approximation",
                    "T_frame_obb": [[0, -1, 0, .5], [1, 0, 0, .1], [0, 0, 1, .8], [0, 0, 0, 1]],
                    "half_extents_m": [.1, .2, .3], "points_sha256": "a" * 64, "measured_point_count": 20,
                    "observation_provenance": {"observation_id": "capture-42", "generation": 4}}}


def cad_row():
    canonical = np.eye(4)
    canonical[:3, 3] = [.4, -.2, .7]
    source = canonical.copy()
    source[2, 3] += .1
    return {"id": "source", "kind": "object", "observation_id": "capture-42", "generation": 4,
            "cad": {"instance_id": "source", "asset_id": "part", "mesh_key": "black_core",
                    "source_sha256": "b" * 64, "mesh_sha256": "c" * 64,
                    "T_world_object": canonical.tolist(), "T_world_source": source.tolist(),
                    "observation_id": "capture-42", "generation": 4,
                    "pose_validation": {"accepted": True}}}


class Catalog:
    def record(self, asset_id):
        if asset_id != "part":
            raise ValueError("unknown_cad_asset")
        return {"asset_id": "part", "mesh_key": "black_core", "source_sha256": "b" * 64}

    def geometry(self, mesh_key):
        if mesh_key != "black_core":
            raise ValueError("unknown_cad_mesh")
        source_from_canonical = np.eye(4)
        source_from_canonical[2, 3] = -.1
        return {"asset_id": "part", "source_sha256": "b" * 64, "mesh_sha256": "c" * 64,
                "T_source_canonical_m": source_from_canonical, "path": "/verified/black_core/visual.stl"}


def build_model(rows, **kwargs):
    from luna_mujoco.hybrid_mapping import build_geometry_model

    return build_geometry_model(rows, "capture-42", 4, **kwargs)


def test_obb_becomes_rotated_solid_cuboid_with_current_identity():
    model = build_model([obb_row()])
    box = model["cuboid"]["semantic_instance::source"]
    assert box["dims"] == [.2, .4, .6]
    np.testing.assert_allclose(box["pose"], [.5, .1, .8, 2 ** -.5, 0, 0, 2 ** -.5])
    assert box["geometry_source"] == "observed_obb"
    assert box["instance_id"] == "source" and box["observation_id"] == "capture-42"
    assert model["mesh"] == {} and model["voxel"] == {}


def test_cad_uses_verified_metric_canonical_mesh_and_canonical_world_pose():
    row = cad_row()
    row["cad"]["file_path"] = "../../untrusted.stl"
    model = build_model([row], catalog=Catalog())
    mesh = model["mesh"]["semantic_instance::source"]
    assert mesh["file_path"] == "/verified/black_core/visual.stl"
    assert mesh["pose"] == [.4, -.2, .7, 1., 0., 0., 0.]
    assert mesh["geometry_source"] == "registered_cad"
    assert mesh["source_sha256"] == "b" * 64 and mesh["mesh_sha256"] == "c" * 64
    assert not model["cuboid"] and not model["voxel"]


@pytest.mark.parametrize("change", ["observation", "generation", "instance", "source", "negative_size",
                                   "nonrigid", "provenance", "missing_digest", "both", "missing", "duplicate",
                                   "missing_kind"])
def test_rejects_stale_or_invalid_obb_contract(change):
    row = obb_row()
    if change == "observation":
        row["observation_id"] = "old"
    elif change == "generation":
        row["generation"] = 3
    elif change == "instance":
        row["obb"]["instance_id"] = "other"
    elif change == "source":
        row["obb"]["source"] = "cad"
    elif change == "negative_size":
        row["obb"]["half_extents_m"][0] = -1
    elif change == "nonrigid":
        row["obb"]["T_frame_obb"][2][2] = 2
    elif change == "provenance":
        row["obb"]["observation_provenance"]["observation_id"] = "old"
    elif change == "missing_digest":
        row["obb"].pop("points_sha256")
    elif change == "both":
        row["cad"] = cad_row()["cad"]
    elif change == "missing":
        row.pop("obb")
    elif change == "missing_kind":
        row.pop("kind")
    with pytest.raises(ValueError):
        build_model([row, deepcopy(row)] if change == "duplicate" else [row])


@pytest.mark.parametrize("change", ["unaccepted", "stale", "hash", "asset", "mesh", "frame_relation"])
def test_rejects_unvalidated_or_misbound_cad(change):
    row = cad_row()
    cad = row["cad"]
    if change == "unaccepted":
        cad["pose_validation"]["accepted"] = False
    elif change == "stale":
        cad["observation_id"] = "old"
    elif change == "hash":
        cad["source_sha256"] = "0" * 64
    elif change == "asset":
        cad["asset_id"] = "other"
    elif change == "mesh":
        cad["mesh_key"] = "blue_box"
    else:
        cad["T_world_source"][2][3] += .01
    with pytest.raises(ValueError):
        build_model([row], catalog=Catalog())


def test_same_capture_cad_pose_cannot_be_reassigned_to_another_permanent_id():
    first, second = cad_row(), cad_row()
    second["id"] = second["cad"]["instance_id"] = "other"
    second["cad"]["T_world_object"][0][3] += .3
    second["cad"]["T_world_source"][0][3] += .3
    model = build_model([first, second], catalog=Catalog())
    assert model["mesh"]["semantic_instance::source"]["pose"][0] == .4
    assert model["mesh"]["semantic_instance::other"]["pose"][0] == .7
    first["cad"], second["cad"] = second["cad"], first["cad"]
    with pytest.raises(ValueError, match="cad_instance_binding_changed"):
        build_model([first, second], catalog=Catalog())


@pytest.mark.parametrize("value", [None, "", "other"])
def test_cad_requires_its_bound_permanent_id(value):
    row = cad_row()
    if value is None:
        row["cad"].pop("instance_id")
    else:
        row["cad"]["instance_id"] = value
    with pytest.raises(ValueError, match="cad_instance_binding_changed"):
        build_model([row], catalog=Catalog())


@pytest.mark.parametrize("value", [None, "", "d" * 64])
def test_cad_requires_the_current_catalog_mesh_hash(value):
    row = cad_row()
    if value is None:
        row["cad"].pop("mesh_sha256")
    else:
        row["cad"]["mesh_sha256"] = value
    with pytest.raises(ValueError, match="cad_model_binding_changed"):
        build_model([row], catalog=Catalog())


@pytest.mark.parametrize("validation", [None, {}, {"accepted": False},
                                      {"accepted": 1}, {"accepted": "true"}])
def test_cad_requires_explicit_boolean_pose_acceptance(validation):
    row = cad_row()
    row["cad"]["pose_validation"] = validation
    with pytest.raises(ValueError, match="cad_pose_validation_required"):
        build_model([row], catalog=Catalog())


class Scene:
    def __init__(self, *, cuboid, mesh, voxel):
        self.cuboid, self.mesh, self.voxel = cuboid, mesh, voxel

    @staticmethod
    def create(model):
        for kind, rows in model.items():
            allowed = {"pose", "dims"} if kind == "cuboid" else {"pose", "file_path"}
            assert all(set(row) <= allowed for row in rows.values())
        return SimpleNamespace(cuboid=list(model["cuboid"]), mesh=list(model["mesh"]))


def mapper(monkeypatch):
    from luna_mujoco.hybrid_mapping import HybridGeometryMapper

    monkeypatch.setitem(sys.modules, "curobo.types", SimpleNamespace(CameraObservation=CameraObservation))
    boundary = Segmenter()
    boundary.robot_spheres = lambda joints: np.array([[0., 0., 0., .04]])
    return HybridGeometryMapper(boundary, scene_type=Scene)


def payload():
    result = request("selected")
    result.update(demo_mapping=True, scene_policy="demo_inventory", demo_tables=table_evidence(),
                  geometry_objects=[obb_row("source"), obb_row("other")])
    return result


def test_registered_instances_and_measured_tables_build_without_voxel_grids(monkeypatch):
    import tianji_curobo.volumetric_mapper as volumetric

    def forbidden(*args, **kwargs):
        raise AssertionError("hybrid scenes must never allocate a volumetric mapper")

    monkeypatch.setattr(volumetric, "CuroboVolumetricMapperBoundary", forbidden)
    value = mapper(monkeypatch)
    candidate, grids, model = value.build(payload())
    assert grids == candidate.scene_data.voxel == []
    assert set(model["cuboid"]) == {"semantic_instance::source", "semantic_instance::other",
                                     "semantic_table::table", "semantic_table::side"}
    assert not model["voxel"]
    assert candidate.instance_metrics["source"]["input_point_count"] == 2
    assert candidate.instance_metrics["source"]["geometry"] == "cuboid"
    assert candidate.camera_metrics["3"]["background_point_count"] == 8
    assert candidate.integration.point_count == 0
    assert candidate.integration.pointcloud_data == b"\0\0\0\0"
    assert value.diagnostics["grids"] == []
    assert value.diagnostics["geometry_policy"] == "registered_cad_or_observed_obb"
    assert list(value.capture_segmenter.filtered_depths) == ["1", "2", "3"]


@pytest.mark.parametrize("change", ["extra_geometry", "missing_geometry", "empty_depth"])
def test_mapping_rejects_geometry_without_current_registered_depth(monkeypatch, change):
    value, data = mapper(monkeypatch), payload()
    if change == "extra_geometry":
        data["geometry_objects"].append(obb_row("unseen"))
    elif change == "missing_geometry":
        data["geometry_objects"].pop()
    else:
        for view in data["views"].values():
            view["valid_left_mask"][:] = 0
    with pytest.raises(ValueError):
        value.build(data)
    assert value.capture_segmenter.capture is None


def test_segmenter_update_preserves_table_and_reset_invalidates_it(monkeypatch):
    value = mapper(monkeypatch)
    value.build(payload())
    table = value.table_cache.table
    replacement = Segmenter()
    replacement.robot_spheres = lambda joints: np.array([[0., 0., 0., .04]])
    value.update_segmenter(replacement)
    assert value.table_cache.table is table
    assert value.capture_segmenter.boundary is replacement
    value.build(payload())
    assert value.diagnostics["table_cache"]["cache_hit"] is True
    value.reset()
    assert value.table_cache.table is None and value.capture_segmenter.capture is None
    assert value.diagnostics == {}


def test_normal_mapping_initializes_table_from_current_head_measurement(monkeypatch):
    from tianji_curobo.mixed_esdf_scene import TableCompletion

    value, data = mapper(monkeypatch), payload()
    data.pop("demo_mapping")
    data.pop("demo_tables")
    data["scene_policy"] = "all_instances"
    captured = []

    def table_builder(groups, spheres, **kwargs):
        captured.extend(groups)
        return TableCompletion({"pose": [2, 0, .4, 1, 0, 0, 0], "dims": [10, 10, .1]},
                               .45, (0, 0, 1), 100, .05)

    value.table_cache.table_builder = table_builder
    _, grids, model = value.build(data)
    assert len(captured) == 1 and len(captured[0]) == 8
    assert grids == []
    assert "semantic_table_forbidden" in model["cuboid"]


@pytest.mark.parametrize("asset_id", ["black_workpiece", "blue_box", "THREADED_VISUAL_SAMPLE_MM", "blue_box_410x310x145"])
def test_real_project_catalog_mesh_hashes_and_canonical_poses(asset_id):
    from luna_mujoco.cad_perception import CadCatalog

    catalog = CadCatalog.from_file()
    record = catalog.record(asset_id)
    geometry = catalog.geometry(record["mesh_key"])
    canonical = np.array([[0, -1, 0, .5], [1, 0, 0, .2], [0, 0, 1, .8], [0, 0, 0, 1.]])
    row = cad_row()
    row["cad"].update(asset_id=asset_id, mesh_key=record["mesh_key"], source_sha256=record["source_sha256"],
                      mesh_sha256=geometry["mesh_sha256"],
                      T_world_object=canonical.tolist(),
                      T_world_source=(canonical @ np.linalg.inv(geometry["T_source_canonical_m"])).tolist())
    model = build_model([row], catalog=catalog)
    mesh = model["mesh"]["semantic_instance::source"]
    path = Path(mesh["file_path"])
    assert path.is_relative_to(Path(__file__).resolve().parents[1] / "resources/cad")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == mesh["mesh_sha256"]
    np.testing.assert_allclose(mesh["pose"], [.5, .2, .8, 2 ** -.5, 0, 0, 2 ** -.5])
