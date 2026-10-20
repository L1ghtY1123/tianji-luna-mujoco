"""The offline fitter must never overwrite mounted robot resources."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "fit_collision_model.py"


def test_fitter_rejects_robot_asset_output_before_sdk_import():
    output = PROJECT_ROOT / "resources" / "forbidden_fit_test_output"
    assert not output.exists()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--project", str(PROJECT_ROOT), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Output must not overwrite project assets or resources" in result.stderr
    assert not output.exists()


def test_fitter_requires_new_output_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--project", str(PROJECT_ROOT), "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "existing artifacts are not overwritten" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("budget_args", [[], ["--num-spheres", "0"]])
def test_fitter_requires_explicit_positive_link_budget_before_sdk_import(tmp_path, budget_args):
    output = tmp_path / "candidate"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project",
            str(PROJECT_ROOT),
            "--output",
            str(output),
            "--links",
            "Link4_L",
            *budget_args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--num-spheres must specify a positive per-link sphere budget" in result.stderr
    assert not output.exists()
