"""CAD units, cavity contacts and stable mesh frames are physical boundaries."""
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from luna_mujoco.model_builder import ModelBuilder, normalize_scene


def cad_item(asset_id, **changes):
    return {"id": "sample", "kind": "cad", "asset_id": asset_id,
            "position": [.7, 0., .8], "color": "#4080c0", "yaw_deg": 31,
            "mass_kg": .5, **changes}


def test_cad_normalization_uses_native_dimensions_and_has_no_object_count_limit():
    objects = [cad_item("pink_wrapped", id=f"sample_{index}") for index in range(35)]
    scene = normalize_scene({"objects": objects})
    assert len(scene["objects"]) == 35
    item = scene["objects"][0]
    assert item["asset_id"] == "pink_wrapped"
    np.testing.assert_allclose(item["size"], [.1110853500366211, .06278736877441406, .11])
    assert item["color"] == "#4080c0" and item["mass_kg"] == .5
    with pytest.raises(ValueError, match="asset"):
        normalize_scene({"objects": [cad_item("missing")]})
    with pytest.raises(ValueError, match="size"):
        normalize_scene({"objects": [cad_item("pink_wrapped", size=[.06, .04, .1])]})


def test_catalog_exposes_real_units_without_mutable_shared_records():
    from luna_mujoco.cad_assets import cad_asset, cad_catalog

    catalog = cad_catalog()
    assets = {item["id"]: item for item in catalog["assets"]}
    assert set(assets) == {"blue_box", "pink_wrapped", "black_core"}
    assert assets["blue_box"]["size_m"] == pytest.approx([.41, .31, .145])
    assert assets["black_core"]["size_m"] == pytest.approx([.044, .056, .0745])
    assert assets["pink_wrapped"]["default_color"] == "pink"
    record = cad_asset("blue_box")
    assert record["cavity_floor_z_m"] == pytest.approx(-.0675)
    section = record["cavity_bottom_section"]
    polygon = np.asarray(section["polygon_xy_m"])
    assert section["z_m"] > record["cavity_floor_z_m"]
    assert len(polygon) > 20
    assert section["area_m2"] < float(np.prod(np.ptp(polygon, axis=0)))
    assert section["area_m2"] > .09
    # The corner of the bounding rectangle is not inside the rounded cavity.
    corner = polygon.max(axis=0)
    assert np.min(np.linalg.norm(polygon - corner, axis=1)) > .001
    record["size_m"][0] = 99
    assert cad_asset("blue_box")["size_m"][0] == pytest.approx(.41)


def test_blue_cad_collision_preserves_cavity_and_bottom():
    builder = ModelBuilder()
    try:
        scene = normalize_scene({"objects": [cad_item("blue_box", yaw_deg=0)],
                                 "table": {"position": [2, 0, .2], "size": [.5, .5, .1]}})
        model = builder.build(scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body = model.body("object_sample")
        ids = np.flatnonzero((model.geom_bodyid == body.id) & (model.geom_contype != 0))
        assert len(ids) > 4
        assert all(model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH for i in ids)
        assert float(body.mass[0]) == pytest.approx(.5)
        # A sphere at the central cavity is separated from every wall/floor.
        # Use the real compiled CAD hulls and public geom-distance API.
        probe_scene = deepcopy(scene)
        probe_scene["objects"].append({"id": "probe", "kind": "sphere", "color": "red",
                                        "size": [.016]*3, "position": [.7, 0., .8], "yaw_deg": 0})
        model = builder.build(normalize_scene(probe_scene)); data = mujoco.MjData(model)
        probe_id = model.geom("object_geom_probe").id
        bin_id = model.body("object_sample").id
        ids = np.flatnonzero((model.geom_bodyid == bin_id) & (model.geom_contype != 0))
        q = int(model.joint("free_probe").qposadr[0])
        distances = []
        for position in ([.7, 0., .8], [.7, 0., .8 - .0675], [.7 + .19, 0., .8]):
            data.qpos[q:q+3] = position; mujoco.mj_forward(model, data)
            distances.append(min(mujoco.mj_geomDistance(model, data, int(i), probe_id, 1., None) for i in ids))
        assert distances[0] > .04
        assert distances[1] < 0
        assert distances[2] < 0
    finally:
        builder.close()


def test_cad_sources_and_generated_meshes_are_fingerprinted_without_visual_repairs():
    import hashlib
    import json

    import trimesh

    from luna_mujoco.cad_assets import CAD_ROOT, cad_asset

    manifest = json.loads((CAD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    script = Path(__file__).resolve().parents[1] / manifest["generator"]["script"]
    assert hashlib.sha256(script.read_bytes()).hexdigest() == manifest["generator"]["script_sha256"]
    assert set(manifest["generator"]["converter"]["files_sha256"]) == {
        "occt-import-js.js", "occt-import-js.wasm"}
    for asset_id in ("blue_box", "pink_wrapped", "black_core"):
        record = cad_asset(asset_id)
        for file in [record["source"], record["visual"], *record["collision"]]:
            assert hashlib.sha256((CAD_ROOT / file["path"]).read_bytes()).hexdigest() == file["sha256"]
        if asset_id == "blue_box":
            continue
        raw = trimesh.load(CAD_ROOT / record["source"]["path"], force="mesh", process=False)
        visual = trimesh.load(CAD_ROOT / record["visual"]["path"], force="mesh", process=False)
        expected = (raw.triangles - np.array(record["source"]["aabb_center_mm"]))*.001
        assert len(visual.faces) == len(raw.faces)
        np.testing.assert_allclose(visual.triangles, expected, atol=1e-8, rtol=0)


def test_cad_render_local_transform_recovers_body_centered_source_bounds():
    from luna_mujoco.cad_assets import cad_asset

    builder = ModelBuilder()
    try:
        scene = normalize_scene({"objects": [cad_item("black_core")]})
        model = builder.build(scene); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        rows = [row for row in builder.render_geometries(model, data) if row.get("object_id") == "sample"]
        assert len(rows) == 1
        row = rows[0]
        assert row["asset_id"] == "black_core" and row["object_kind"] == "cad"
        assert row["object_size_m"] == cad_asset("black_core")["size_m"]
        geom = model.geom(row["name"]); mesh = model.mesh(int(geom.dataid[0]))
        start = int(mesh.vertadr[0]); count = int(mesh.vertnum[0])
        points = model.mesh_vert[start:start+count]
        rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, np.array(row["local_quaternion"]))
        centered = points @ rotation.reshape(3,3).T + row["local_position_m"]
        np.testing.assert_allclose(centered.min(axis=0), -np.array(row["object_size_m"])/2, atol=1e-7)
        np.testing.assert_allclose(centered.max(axis=0), np.array(row["object_size_m"])/2, atol=1e-7)
        assert Path("assets/render_meshes", Path(row["mesh_url"]).name).is_file()
    finally:
        builder.close()


def test_replacement_pink_keeps_authored_flat_bottom_and_exact_source():
    import hashlib

    import trimesh

    from luna_mujoco.cad_assets import CAD_ROOT, cad_asset

    record = cad_asset("pink_wrapped")
    raw_path = CAD_ROOT / record["source"]["path"]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == "200868327093f29d38856dd769ed11855a0cab7f26e1d8aa72207e6cf0298f3f"
    assert b"Units:mm" in raw_path.read_bytes()[:80]
    assert record["source"]["aabb_center_mm"] == [0., 0., 55.]
    raw = trimesh.load_mesh(raw_path, process=False)
    bottom = np.all(abs(raw.triangles[:, :, 2]) < 1e-5, axis=1)
    assert np.count_nonzero(bottom) == 368
    assert raw.area_faces[bottom].sum() > 3200  # Authored mm², not inferred contact area.
    collision = trimesh.load_mesh(CAD_ROOT / record["collision"][0]["path"])
    bottom = np.all(abs(collision.triangles[:, :, 2] + .055) < 1e-8, axis=1)
    assert collision.area_faces[bottom].sum() > .0032
    assert collision.is_watertight and collision.is_winding_consistent
