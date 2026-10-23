"""Read-only array previews preserve archive identity and bound browser payloads."""
import json

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_experiments import begin

from luna_mujoco.experiment_api import experiment_router
from luna_mujoco.experiments import ExperimentStore


def test_array_preview_reads_exact_mask_and_reports_sampled_cloud_indices(tmp_path):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    session.artifact("artifacts/evidence", {
        "cloud": np.arange(300, dtype=np.float32).reshape(100, 3),
        "mask": np.array([[True, False], [False, True]]),
    })
    session.finish({"status": "failed"})
    data = json.loads(store.file(session.id, "artifacts/evidence.json").read_text())
    app = FastAPI()
    app.include_router(experiment_router(store))
    with TestClient(app) as client:
        base = f"/api/experiments/{session.id}/arrays"
        cloud = client.get(base, params={**{k: data["cloud"][k] for k in ("path", "key")}, "max_points": 3})
        assert cloud.status_code == 200
        assert cloud.json()["shape"] == [100, 3]
        assert cloud.json()["indices"] == [0, 49, 99]
        assert cloud.json()["data"] == [[0, 1, 2], [147, 148, 149], [297, 298, 299]]
        mask = client.get(base, params={k: data["mask"][k] for k in ("path", "key")})
        assert mask.status_code == 200
        assert mask.json()["data"] == [[True, False], [False, True]]
        assert mask.json()["sampled"] is False
        assert client.get(base, params={"path": "../private.npz", "key": "x"}).status_code in {404, 409}
        assert client.get(base, params={"path": "artifacts/evidence.npz", "key": "missing"}).status_code == 404
        assert client.get(base, params={"path": "artifacts/evidence.json", "key": "x"}).status_code == 409
    assert not store.get(session.id)["busy"]
