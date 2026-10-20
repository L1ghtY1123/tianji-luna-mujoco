"""Offline fitting must preserve final results and reject unsafe output/input states."""

import importlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fitter(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("fit_official_motion")


def test_final_results_do_not_read_history(fitter):
    class Result:
        centers = np.array([[0.1, 0.2, 0.3]])
        radii = np.array([0.01])
        num_spheres = 1

        @property
        def history(self):
            raise AssertionError("History must not be merged into final spheres")

    assert fitter.final_spheres(Result(), 2) == [{"center": [0.1, 0.2, 0.3], "radius": 0.01}]


def test_final_result_over_budget_is_rejected(fitter):
    result = SimpleNamespace(centers=np.zeros((2, 3)), radii=np.ones(2), num_spheres=2)
    with pytest.raises(ValueError, match="budget"):
        fitter.final_spheres(result, 1)


def test_unrecognised_live_state_blocks_offline_gpu(fitter, monkeypatch):
    monkeypatch.setattr(fitter.urllib.request, "urlopen", lambda *a, **kw: io.StringIO("{}"))
    with pytest.raises(RuntimeError, match="live state"):
        fitter.check_live_state("unused")


def test_sdk_output_cannot_be_created(fitter, tmp_path):
    sdk = tmp_path / "site-packages" / "curobo"
    output = sdk / "new-candidate"
    with pytest.raises(ValueError, match="SDK"):
        fitter.validate_output_path(ROOT, ROOT / "source.yml", output, sdk_root=sdk)
    assert not output.exists()


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"num_spheres": 2.5, "parts": [{"num_spheres": 2}]}, "positive integer link budget"),
        ({"num_spheres": 2, "parts": []}, "at least one fitted part"),
    ],
)
def test_invalid_link_plan_is_rejected_before_sdk_import(tmp_path, spec, message):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"Link4_L": spec}))
    output = tmp_path / "candidate"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fit_official_motion.py"),
            "--project",
            str(ROOT),
            "--source-config",
            str(ROOT / "data/collision-sphere-source-2026-09-30.yml"),
            "--plan",
            str(plan),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()
