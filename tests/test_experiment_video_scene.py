"""Offline recording rebuilds must preserve the full authored scene topology."""
import gzip
import json
import time
from copy import deepcopy

import mujoco
import numpy as np
import pytest

from luna_mujoco import experiment_video as video
from luna_mujoco.cad_assets import cad_asset
from luna_mujoco.demo_scene import demo_scene
from luna_mujoco.model_builder import ModelBuilder


@pytest.fixture
def demo_recording():
    scene = demo_scene(3)
    builder = ModelBuilder()
    try:
        model = builder.build(scene)
        geometries = [
            {"id": str(index), "name": model.geom(index).name,
             "type": mujoco.mjtGeom(int(model.geom_type[index])).name.removeprefix("mjGEOM_").lower(),
             "size": model.geom_size[index].tolist()}
            for index in range(model.ngeom) if model.geom_group[index] != 3
        ]
    finally:
        builder.close()
    return {"schema_version": 1, "generation": 7, "scene": scene,
            "geometry_manifest": {"geometries": geometries}, "duration_s": .2}


def test_recorder_preserves_support_tables_and_cad_objects(tmp_path, monkeypatch, demo_recording):
    class FrozenSimulation:
        def snapshot(self):
            return {**deepcopy(demo_recording["scene"]), "generation": 7}

        def manifest(self):
            return demo_recording["geometry_manifest"]

        def recording_snapshot(self, known_revision=None):
            snapshot = {**self.snapshot(), "manifest_revision": 1}
            snapshot["geometry_update"] = {
                "scene": deepcopy(demo_recording["scene"]),
                "geometry_manifest": {**self.manifest(), "revision": 1}}
            return snapshot

    recorder = video.VideoRecorder(tmp_path, FrozenSimulation(), time.monotonic(), 7)
    monkeypatch.setattr(recorder, "_sample_geometry", lambda: None)
    monkeypatch.setattr(recorder, "_sample_preview", lambda: None)
    recorder.start()
    recorder.stop()
    for thread in recorder._threads:
        thread.join()
    archived = json.loads((tmp_path / "videos/recording.json").read_text(encoding="utf-8"))
    assert archived["scene"]["support_tables"] == demo_recording["scene"]["support_tables"]
    item = archived["scene"]["objects"][-1]
    assert item == demo_recording["scene"]["objects"][-1]
    assert item["kind"] == "cad"
    assert item["asset_id"] == "pink_wrapped"
    assert item["size"] == cad_asset("pink_wrapped")["size_m"]


class ReachedRenderer(Exception):
    pass


def stop_before_gl(*args, **kwargs):
    raise ReachedRenderer


def legacy_recording(tmp_path, recording, *, generation=7):
    scene = deepcopy(recording["scene"])
    (tmp_path / "config.json").write_text(
        json.dumps({"simulation_initial": {**scene, "generation": generation}}), encoding="utf-8")
    legacy = deepcopy(recording)
    legacy["scene"].pop("support_tables")
    return legacy


def test_old_recording_recovers_supports_from_its_frozen_initial_scene(tmp_path, monkeypatch, demo_recording):
    legacy = legacy_recording(tmp_path, demo_recording)
    monkeypatch.setattr(mujoco, "Renderer", stop_before_gl)
    # Model compilation and every recorded ID/name check run before the GL boundary.
    with pytest.raises(ReachedRenderer):
        next(video._scene_images(tmp_path, legacy))
    assert "support_tables" not in legacy["scene"]


def test_complete_recording_rebuilds_without_a_config_fallback(tmp_path, monkeypatch, demo_recording):
    monkeypatch.setattr(mujoco, "Renderer", stop_before_gl)
    with pytest.raises(ReachedRenderer):
        next(video._scene_images(tmp_path, demo_recording))


def test_missing_frozen_support_evidence_cannot_be_replaced_by_current_defaults(tmp_path, demo_recording):
    legacy = deepcopy(demo_recording)
    legacy["scene"].pop("support_tables")
    with pytest.raises(RuntimeError, match="frozen config.json"):
        next(video._scene_images(tmp_path, legacy))


def test_support_recovery_rejects_a_different_simulation_generation(tmp_path, monkeypatch, demo_recording):
    legacy = legacy_recording(tmp_path, demo_recording, generation=8)
    monkeypatch.setattr(mujoco, "Renderer", stop_before_gl)
    with pytest.raises(RuntimeError, match="frozen.*generation"):
        next(video._scene_images(tmp_path, legacy))


def test_recovered_scene_still_checks_each_geometry_id(tmp_path, monkeypatch, demo_recording):
    legacy = legacy_recording(tmp_path, demo_recording)
    legacy["geometry_manifest"]["geometries"][-1]["name"] = "not_the_recorded_object"
    monkeypatch.setattr(mujoco, "Renderer", stop_before_gl)
    with pytest.raises(RuntimeError, match="geometry IDs"):
        next(video._scene_images(tmp_path, legacy))


def test_recorded_support_sizes_are_checked_not_only_their_ids(tmp_path, monkeypatch, demo_recording):
    legacy = legacy_recording(tmp_path, demo_recording)
    config = json.loads((tmp_path / "config.json").read_text())
    config["simulation_initial"]["support_tables"][0]["size"][0] += .02
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(mujoco, "Renderer", stop_before_gl)
    with pytest.raises(RuntimeError, match="geometry.*size"):
        next(video._scene_images(tmp_path, legacy))


def test_scene_encoder_switches_actual_model_geometry_at_recorded_revision(tmp_path, monkeypatch, demo_recording):
    from types import SimpleNamespace

    changed = deepcopy(demo_recording)
    item = changed["scene"]["objects"][-1]
    black = cad_asset("black_core")
    item.update(kind="cad", asset_id="black_core", color="black", size=black["size_m"], mass_kg=.06)
    builder = ModelBuilder()
    try:
        model = builder.build(changed["scene"])
        changed["geometry_manifest"] = {"revision": 2, "geometries": builder.render_geometries(model, mujoco.MjData(model))}
    finally:
        builder.close()
    recording = {**demo_recording, "duration_s": .2,
                 "geometry_manifest": {**demo_recording["geometry_manifest"], "revision": 1},
                 "geometry_revisions": {"2": changed}}
    with gzip.open(tmp_path / "frames.jsonl.gz", "wt", encoding="utf-8") as output:
        for revision, timestamp in ((1, 0), (2, .1)):
            output.write(json.dumps({"t_s": timestamp, "sim_time": timestamp, "manifest_revision": revision,
                                     "geometry_poses": []}) + "\n")
    rendered = []
    class Renderer:
        def __init__(self, model, **kwargs):
            geom = model.geom("object_visual_" + item["id"])
            mesh_id = int(model.geom_dataid[geom.id])
            rendered.append((int(geom.type[0]), model.mesh(mesh_id).name,
                             int(model.mesh_facenum[mesh_id]), model.geom_rgba[geom.id].copy()))
            self.scene = SimpleNamespace(geoms=[], ngeom=0)
        def update_scene(self, *args, **kwargs):
            pass
        def render(self):
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    assert len(list(video._scene_images(tmp_path, recording))) == 3
    assert [row[0] for row in rendered] == [mujoco.mjtGeom.mjGEOM_MESH] * 2
    assert [row[1] for row in rendered] == ["cad_pink_wrapped_visual", "cad_black_core_visual"]
    assert [row[2] for row in rendered] == [10924, 90916]
    assert max(rendered[-1][3][:3]) < .1
