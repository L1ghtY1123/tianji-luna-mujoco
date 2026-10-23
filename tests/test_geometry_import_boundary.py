"""The planning service image deliberately has no MuJoCo renderer installed."""
import os
import subprocess
import sys
from pathlib import Path


def test_planning_settings_import_does_not_require_mujoco():
    source = Path(__file__).resolve().parents[1] / 'src'
    code = '''
import sys
from importlib.abc import MetaPathFinder
class NoRenderer(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mujoco" or fullname.startswith("mujoco."):
            raise ModuleNotFoundError("GPU service has no MuJoCo renderer")
sys.meta_path.insert(0, NoRenderer())
from luna_mujoco.planning_settings import PlanningSettings
assert PlanningSettings().attachment.sphere_budget == 24
assert "mujoco" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(source)},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
