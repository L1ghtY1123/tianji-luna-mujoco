"""The single web container must stop both services together."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Container entrypoint runs on Linux")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def web_processes(tmp_path):
    project = tmp_path / "project with spaces"
    (project / "deploy").mkdir(parents=True)
    (project / "web" / "node_modules").mkdir(parents=True)
    (project / ".web-node_modules").mkdir()
    (project / ".web-node_modules" / "dependency-marker").write_text("from image")
    entrypoint = project / "deploy" / "run-web.sh"
    shutil.copy2(ROOT / "deploy" / "run-web.sh", entrypoint)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    child = (
        f"#!{sys.executable}\n"
        "import json, os, signal, sys, time\n"
        "from pathlib import Path\n"
        "name = Path(sys.argv[0]).name\n"
        "record = Path(os.environ['CHILD_RECORD'])\n"
        "def stop(signum, frame):\n"
        "    (record / (name + '.stopped')).touch()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "(record / (name + '.started')).write_text(json.dumps(sys.argv[1:]))\n"
        "while True:\n"
        "    trigger = record / (name + '.exit')\n"
        "    if trigger.exists():\n"
        "        sys.exit(int(trigger.read_text()))\n"
        "    time.sleep(.01)\n"
    )
    for name in ("python", "node"):
        executable = binaries / name
        executable.write_text(child)
        executable.chmod(0o755)
    environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       CHILD_RECORD=str(tmp_path))
    process = subprocess.Popen(["bash", str(entrypoint)], env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not all((tmp_path / (name + ".started")).exists() for name in ("python", "node")):
            if process.poll() is not None:
                pytest.fail(process.communicate()[1].decode())
            assert time.monotonic() < deadline, "Both services must start"
            time.sleep(.01)
        yield process, tmp_path, project
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)


def test_container_stop_terminates_backend_and_frontend(web_processes):
    process, record, project = web_processes
    assert (project / "web/node_modules/dependency-marker").read_text() == "from image"
    backend_args = json.loads((record / "python.started").read_text())
    assert "--reload" in backend_args
    assert "8341" in backend_args
    frontend_args = json.loads((record / "node.started").read_text())
    assert "8340" in frontend_args and "--strictPort" in frontend_args
    process.terminate()
    process.communicate(timeout=5)
    assert process.returncode == 143
    assert (record / "python.stopped").exists()
    assert (record / "node.stopped").exists()


@pytest.mark.parametrize("service,code,expected", [("python", 17, 17), ("node", 0, 1)])
def test_either_service_exit_stops_its_partner(web_processes, service, code, expected):
    process, record, _ = web_processes
    (record / (service + ".exit")).write_text(str(code))
    process.communicate(timeout=5)
    assert process.returncode == expected
    partner = "node" if service == "python" else "python"
    assert (record / (partner + ".stopped")).exists()
