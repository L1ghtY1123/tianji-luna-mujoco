"""Exercise the Linux launchers without creating or stopping real containers."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Bash deployment runs on Linux")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    project = tmp_path / "project with spaces"
    (project / "scripts").mkdir(parents=True)
    (project / "deploy").mkdir()
    (project / "web").mkdir()
    (project / ".env").touch()
    for name in ("web-service.sh", "algorithms-service.sh"):
        source = ROOT / "scripts" / name
        if source.exists():
            shutil.copy2(source, project / "scripts" / name)
    for name in ("start.sh", "stop.sh"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, project / name)
    for name in ("web.compose.yaml", "algorithms.compose.yaml"):
        (project / "deploy" / name).touch()

    binaries = tmp_path / "bin"
    binaries.mkdir()
    record = tmp_path / "docker.jsonl"
    docker = binaries / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "keys = ('JEV_WEB_IMAGE_TAG', 'JEV_ALGORITHM_IMAGE_TAG', 'JEV_UID', 'JEV_GID')\n"
        "with open(os.environ['FAKE_DOCKER_RECORD'], 'a') as output:\n"
        "    output.write(json.dumps({'args': args, 'frontend_mountpoint_exists': "
        "os.path.isdir('web/node_modules'), 'env': {key: os.environ.get(key) "
        "for key in keys}}) + '\\n')\n"
        "if args[:2] == ['image', 'inspect']:\n"
        "    if args[-1] in json.loads(os.environ.get('FAKE_MISSING_IMAGES', '[]')):\n"
        "        sys.exit(1)\n"
        "    sys.exit(0 if os.environ.get('FAKE_IMAGE_EXISTS', '1') == '1' else 1)\n"
        "if args[:2] == ['container', 'inspect']:\n"
        "    if '--format' in args and os.environ.get('FAKE_FRONTEND_EXISTS') == '1':\n"
        "        print(os.environ.get('FAKE_FRONTEND_PROJECT', 'tianji-jev-mujoco-web-guangyu'))\n"
        "    sys.exit(0 if os.environ.get('FAKE_FRONTEND_EXISTS') == '1' else 1)\n"
        "if args[:2] == ['rm', '-f'] and os.environ.get('FAKE_REMOVE_FAIL') == '1':\n"
        "    sys.exit(23)\n"
        "if args and args[0] == 'compose':\n"
        "    fail = os.environ.get('FAKE_COMPOSE_FAIL')\n"
        "    if fail and fail in args:\n"
        "        sys.exit(23)\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    outside = tmp_path / "outside"
    outside.mkdir()
    environment = dict(os.environ)
    for key in ("JEV_WEB_IMAGE_TAG", "JEV_ALGORITHM_IMAGE_TAG", "JEV_UID", "JEV_GID"):
        environment.pop(key, None)
    environment["PATH"] = str(binaries) + os.pathsep + environment["PATH"]
    environment["FAKE_DOCKER_RECORD"] = str(record)

    def run(script, *arguments, env=None):
        record.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(project / script), *arguments],
            cwd=outside,
            env=environment | (env or {}),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        calls = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        return result, calls

    return project, run


def compose_actions(calls, project, filename):
    actions = []
    for call in calls:
        arguments = call["args"]
        if not arguments or arguments[0] != "compose":
            continue
        # Version probes are permitted; they do not operate on a Compose project.
        if "-f" not in arguments:
            assert arguments[1:] in (["version"], ["version", "--short"])
            continue
        filenames = [arguments[index + 1] for index, value in enumerate(arguments) if value == "-f"]
        expected_files = [str(project / "deploy" / filename)]
        assert filenames == expected_files
        for action in ("build", "up", "stop", "ps", "logs", "down"):
            if action in arguments:
                actions.append(arguments[arguments.index(action):])
                break
        else:
            pytest.fail(f"Unexpected Compose operation: {arguments}")
    return actions


@pytest.mark.parametrize("image_exists", [True, False])
def test_web_start_never_builds_and_waits_for_health_when_image_exists(launcher, image_exists):
    project, run = launcher
    result, calls = run(
        "scripts/web-service.sh", "start", env={"FAKE_IMAGE_EXISTS": "1" if image_exists else "0"}
    )
    assert result.returncode == (0 if image_exists else 1), result.stderr
    expected = [["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "web"]]
    if not image_exists:
        expected = []
    assert compose_actions(calls, project, "web.compose.yaml") == expected
    inspections = [call for call in calls if call["args"][:2] == ["image", "inspect"]]
    assert len(inspections) == 1
    assert inspections[0]["args"][-1] == "tianji-jev-mujoco:0.4.0-experiments-guangyu"
    assert calls[-1]["env"]["JEV_WEB_IMAGE_TAG"] == "0.4.0-experiments-guangyu"
    assert calls[-1]["env"]["JEV_UID"] == str(os.getuid())
    assert calls[-1]["env"]["JEV_GID"] == str(os.getgid())


def test_web_explicit_build_uses_requested_private_tag_and_does_not_start(launcher):
    project, run = launcher
    result, calls = run("scripts/web-service.sh", "build", "0.2.0-guangyu-test")
    assert result.returncode == 0, result.stderr
    assert compose_actions(calls, project, "web.compose.yaml") == [["build", "web"]]
    assert calls[-1]["env"]["JEV_WEB_IMAGE_TAG"] == "0.2.0-guangyu-test"


@pytest.mark.parametrize("arguments", [
    ("--dev",), ("--dev", "0.2.0-guangyu-test"), ("0.2.0-guangyu-test", "--dev"),
])
def test_legacy_dev_flag_is_only_an_alias_for_the_single_web_service(launcher, arguments):
    project, run = launcher
    for _ in range(2):
        result, calls = run("scripts/web-service.sh", "start", *arguments)
        assert result.returncode == 0, result.stderr
        assert compose_actions(calls, project, "web.compose.yaml") == [
            ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "web"]
        ]
        tag = "0.2.0-guangyu-test" if len(arguments) == 2 else "0.4.0-experiments-guangyu"
        inspected = [call["args"][-1] for call in calls if call["args"][:2] == ["image", "inspect"]]
        assert inspected == [f"tianji-jev-mujoco:{tag}"]
        assert "8340" in result.stdout
        assert "重载" in result.stdout and "仿真" in result.stdout


@pytest.mark.parametrize(("arguments", "created"), [
    (("start", "--dev"), True), (("start",), True), (("build", "--dev"), False),
])
def test_start_prepares_empty_dependency_mountpoint_before_up(launcher, arguments, created):
    project, run = launcher
    mountpoint = project / "web" / "node_modules"
    assert not mountpoint.exists()
    result, calls = run("scripts/web-service.sh", *arguments)
    assert result.returncode == 0, result.stderr
    assert mountpoint.is_dir() is created
    if created:
        assert list(mountpoint.iterdir()) == []
    starts = [call for call in calls if call["args"][0] == "compose" and "up" in call["args"]]
    for call in starts:
        assert call["frontend_mountpoint_exists"] is created


def test_legacy_dev_start_rejects_missing_image_without_building(launcher):
    project, run = launcher
    result, calls = run("scripts/web-service.sh", "start", "--dev", env={
        "FAKE_MISSING_IMAGES": json.dumps(["tianji-jev-mujoco:0.4.0-experiments-guangyu"]),
    })
    assert result.returncode == 1, result.stderr
    assert compose_actions(calls, project, "web.compose.yaml") == []


@pytest.mark.parametrize("arguments", [
    ("--dev",), ("--dev", "0.2.0-guangyu-test"), ("0.2.0-guangyu-test", "--dev"),
])
def test_legacy_dev_build_builds_only_web_without_env_or_starting(launcher, arguments):
    project, run = launcher
    (project / ".env").unlink()
    result, calls = run("scripts/web-service.sh", "build", *arguments)
    assert result.returncode == 0, result.stderr
    assert compose_actions(calls, project, "web.compose.yaml") == [["build", "web"]]


@pytest.mark.parametrize("image_exists", [True, False])
def test_start_removes_only_owned_legacy_frontend_after_image_is_available(launcher, image_exists):
    project, run = launcher
    result, calls = run("scripts/web-service.sh", "start", env={
        "FAKE_FRONTEND_EXISTS": "1", "FAKE_IMAGE_EXISTS": "1" if image_exists else "0",
    })
    removals = [index for index, call in enumerate(calls) if call["args"][:2] == ["rm", "-f"]]
    if not image_exists:
        assert result.returncode == 1
        assert removals == []
        assert compose_actions(calls, project, 'web.compose.yaml') == []
        return
    assert result.returncode == 0, result.stderr
    assert len(removals) == 1
    removed = removals[0]
    assert calls[removed]["args"] == ["rm", "-f", "jev-mujoco-frontend-dev-guangyu"]
    inspected = [call["args"] for call in calls[:removed] if call["args"][:2] == ["container", "inspect"]]
    assert any("--format" in arguments and "com.docker.compose.project" in " ".join(arguments)
               for arguments in inspected)
    assert any("up" in call["args"] for call in calls[removed + 1:])
    expected = [["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "web"]]
    assert compose_actions(calls, project, "web.compose.yaml") == expected


@pytest.mark.parametrize("owner", ["another-project", ""])
def test_unowned_legacy_frontend_is_not_removed_or_replaced(launcher, owner):
    project, run = launcher
    result, calls = run("scripts/web-service.sh", "start", env={
        "FAKE_FRONTEND_EXISTS": "1", "FAKE_FRONTEND_PROJECT": owner,
    })
    assert result.returncode != 0
    assert "归属" in result.stderr
    assert not compose_actions(calls, project, "web.compose.yaml")
    assert not any(call["args"][0] == "rm" for call in calls)


def test_failed_legacy_frontend_removal_does_not_start_web(launcher):
    project, run = launcher
    result, calls = run("scripts/web-service.sh", "start", env={
        "FAKE_FRONTEND_EXISTS": "1", "FAKE_REMOVE_FAIL": "1",
    })
    assert result.returncode != 0
    assert not compose_actions(calls, project, "web.compose.yaml")


def test_missing_image_never_removes_legacy_frontend_or_attempts_start(launcher):
    project, run = launcher
    result, calls = run("scripts/web-service.sh", "start", "--dev", env={
        "FAKE_IMAGE_EXISTS": "0", "FAKE_COMPOSE_FAIL": "build", "FAKE_FRONTEND_EXISTS": "1",
    })
    assert result.returncode != 0
    assert compose_actions(calls, project, "web.compose.yaml") == []
    assert not any(call["args"][0] == "rm" for call in calls)


def test_failed_start_shows_only_web_service_logs(launcher):
    project, run = launcher
    result, calls = run("scripts/web-service.sh", "start", "--dev", env={"FAKE_COMPOSE_FAIL": "up"})
    assert result.returncode != 0
    assert compose_actions(calls, project, "web.compose.yaml") == [
        ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "web"],
        ["logs", "--tail", "80", "web"],
    ]


def test_algorithm_start_includes_rgbd_segmentation_and_waits_for_all_health(launcher):
    project, run = launcher
    result, calls = run("scripts/algorithms-service.sh", "start", "0.2.0-guangyu-test")
    assert result.returncode == 0, result.stderr
    assert compose_actions(calls, project, "algorithms.compose.yaml") == [
        ["up", "-d", "--no-build", "--wait", "--wait-timeout", "300", "sam3", "curobo", "graspgen"]
    ]
    assert calls[-1]["env"]["JEV_ALGORITHM_IMAGE_TAG"] == "0.2.0-guangyu-test"


def test_algorithm_start_never_builds_a_missing_image(launcher):
    project, run = launcher
    result, calls = run('scripts/algorithms-service.sh', 'start', env={'FAKE_IMAGE_EXISTS': '0'})
    assert result.returncode == 1
    assert compose_actions(calls, project, 'algorithms.compose.yaml') == []


@pytest.mark.parametrize(
    ("script", "filename", "services"),
    [
        ("scripts/web-service.sh", "web.compose.yaml", ["web"]),
        ("scripts/algorithms-service.sh", "algorithms.compose.yaml", ["sam3", "curobo", "graspgen"]),
    ],
)
@pytest.mark.parametrize(
    ("action", "expected"),
    [("stop", ["stop"]), ("status", ["ps"]), ("logs", ["logs", "--follow", "--tail", "100"])],
)
def test_management_commands_only_target_their_own_services(launcher, script, filename, services, action, expected):
    project, run = launcher
    result, calls = run(script, action)
    assert result.returncode == 0, result.stderr
    assert compose_actions(calls, project, filename) == [
        [*expected, *services]
    ]


@pytest.mark.parametrize("script", ["scripts/web-service.sh", "scripts/algorithms-service.sh"])
@pytest.mark.parametrize(
    "arguments",
    [(), ("unknown",), ("start", "latest"), ("stop", "0.2.0-guangyu"), ("start", "0.2.0-guangyu", "extra")],
)
def test_invalid_arguments_exit_two_before_calling_docker(launcher, script, arguments):
    _, run = launcher
    result, calls = run(script, *arguments)
    assert result.returncode == 2, result.stderr
    assert calls == []


@pytest.mark.parametrize("arguments", [
    ("start", "--unknown"), ("start", "--dev", "--dev"), ("start", "--dev", "latest"),
    ("start", "--dev", "0.2.0-guangyu", "another-guangyu"), ("stop", "--dev"),
])
def test_invalid_dev_arguments_exit_two_before_calling_docker(launcher, arguments):
    _, run = launcher
    result, calls = run("scripts/web-service.sh", *arguments)
    assert result.returncode == 2, result.stderr
    assert calls == []


def test_web_requires_env_file_before_start(launcher):
    project, run = launcher
    (project / ".env").unlink()
    result, calls = run("scripts/web-service.sh", "start")
    assert result.returncode != 0
    assert not compose_actions(calls, project, "web.compose.yaml")


def test_web_build_does_not_require_runtime_env_file(launcher):
    project, run = launcher
    (project / ".env").unlink()
    result, calls = run("scripts/web-service.sh", "build")
    assert result.returncode == 0, result.stderr
    assert compose_actions(calls, project, "web.compose.yaml") == [["build", "web"]]


def test_failed_web_build_does_not_attempt_start(launcher):
    project, run = launcher
    result, calls = run(
        "scripts/web-service.sh", "build", env={"FAKE_IMAGE_EXISTS": "0", "FAKE_COMPOSE_FAIL": "build"}
    )
    assert result.returncode != 0
    assert compose_actions(calls, project, "web.compose.yaml") == [["build", "web"]]


@pytest.mark.parametrize(
    ("script", "expected"),
    [("start.sh", ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "web"]),
     ("stop.sh", ["stop", "web"])],
)
def test_root_shortcuts_delegate_to_docker_web_service(launcher, script, expected):
    project, run = launcher
    result, calls = run(script)
    assert result.returncode == 0, result.stderr
    assert compose_actions(calls, project, "web.compose.yaml") == [expected]


def test_root_start_forwards_dev_and_version_arguments(launcher):
    project, run = launcher
    result, calls = run("start.sh", "--dev", "0.2.0-guangyu-test")
    assert result.returncode == 0, result.stderr
    assert compose_actions(calls, project, "web.compose.yaml") == [
        ["up", "-d", "--no-build", "--wait", "--wait-timeout", "120", "web"]
    ]
    assert calls[-1]["env"]["JEV_WEB_IMAGE_TAG"] == "0.2.0-guangyu-test"
