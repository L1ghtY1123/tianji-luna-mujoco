import pytest
from fastapi.testclient import TestClient
from test_api import Algorithms, Simulation

from luna_mujoco.app import create_app
from luna_mujoco.demo_runner import DemoRunner
from luna_mujoco.planning_settings import PlanningSettings, save_settings


def test_demo_composition_does_not_require_typesafe_key(tmp_path, monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    monkeypatch.setattr('luna_mujoco.demo_scene.demo_observation_poses', lambda: {'A': [0.] * 7, 'B': [0.] * 7})
    app = create_app(Simulation(), Algorithms(), data_dir=tmp_path, demo_mode=True)
    assert isinstance(app.state.runner, DemoRunner)
    with TestClient(app) as client:
        health = client.get('/api/health').json()
        assert 'jev' not in health
        assert health['controller']['ready'] is True
        assert health['controller']['name'] == 'responses_program'
        result = client.post('/api/demo/preset', json={'object_count': 2})
        assert result.status_code == 200
        assert len(app.state.runner.simulation.last_scene['objects']) == 4
        assert client.post('/api/demo/preset', json={'object_count': 5}).status_code == 422


def test_standalone_demo_uses_its_own_experiment_store(tmp_path, monkeypatch):
    root = tmp_path / 'demo'
    root.mkdir()
    monkeypatch.setattr('luna_mujoco.app.ROOT', root)
    monkeypatch.delenv('LUNA_EXPERIMENT_DIR', raising=False)
    monkeypatch.setattr('luna_mujoco.demo_scene.demo_observation_poses', lambda: {'A': [0.] * 7, 'B': [0.] * 7})
    app = create_app(Simulation(), Algorithms(), demo_mode=True)
    assert app.state.runner.experiments.root == root / 'data' / 'experiments'
    assert not (tmp_path / 'tianji-experiments').exists()


@pytest.mark.parametrize("saved_budget", [1, 16, 24, 31])
def test_demo_defaults_to_24_spheres_and_honors_legal_saved_settings(tmp_path, monkeypatch, saved_budget):
    monkeypatch.setattr('luna_mujoco.demo_scene.demo_observation_poses', lambda: {'A': [0.] * 7, 'B': [0.] * 7})
    app = create_app(Simulation(), Algorithms(), data_dir=tmp_path / 'runs', demo_mode=True)
    assert app.state.runner.planning_settings.attachment.sphere_budget == 24
    saved = PlanningSettings(attachment={"sphere_budget": saved_budget})
    save_settings(tmp_path / 'planning-settings.json', saved)
    app = create_app(Simulation(), Algorithms(), data_dir=tmp_path / 'runs', demo_mode=True)
    assert app.state.runner.planning_settings.attachment.sphere_budget == saved_budget
    assert list(app.state.runner.planning_settings.observation.A) == [0.] * 7


@pytest.mark.parametrize('saved_budget', [32, 128, 256])
def test_demo_explicitly_migrates_saved_excessive_budget(tmp_path, monkeypatch, saved_budget):
    import json
    monkeypatch.setattr('luna_mujoco.demo_scene.demo_observation_poses', lambda: {'A': [0.] * 7, 'B': [0.] * 7})
    (tmp_path / 'planning-settings.json').write_text(json.dumps({'attachment': {'sphere_budget': saved_budget}}))
    with pytest.warns(UserWarning, match='migrated to 24'):
        app = create_app(Simulation(), Algorithms(), data_dir=tmp_path / 'runs')
    assert app.state.runner.planning_settings.attachment.sphere_budget == 24
