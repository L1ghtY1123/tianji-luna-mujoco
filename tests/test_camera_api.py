from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from luna_mujoco.app import create_app
from luna_mujoco.simulation import Simulation


def test_camera_preview_is_three_same_frame_local_pngs_and_reset_rejects_old_frame(tmp_path):
    simulation = Simulation()
    with TestClient(create_app(simulation=simulation, data_dir=tmp_path)) as client:
        assert client.get('/api/cameras').status_code == 503
        captured = simulation.capture_rgbd()  # explicit observation checkpoint
        response = client.get('/api/cameras')
        assert response.status_code == 200
        info = response.json()
        assert info['observation_id'] == captured['observation_id']
        assert client.get('/api/cameras').json()['observation_id'] == captured['observation_id']
        assert {item['id'] for item in info['cameras']} == {'1', '2', '3'}
        assert all(item['frame_id'] == info['observation_id'] for item in info['cameras'])
        for item in info['cameras']:
            preview = client.get(item['rgb_url'])
            assert preview.status_code == 200
            assert preview.headers['content-type'] == 'image/png'
            assert Image.open(BytesIO(preview.content)).size[0] == 640
        old_url = info['cameras'][0]['rgb_url']
        assert client.post('/api/scene', json={'objects': []}).status_code == 200
        assert client.get(old_url).status_code == 409
        assert client.get('/api/cameras').status_code == 503


def test_live_rgb_preview_does_not_create_or_overwrite_task_capture(tmp_path, monkeypatch):
    simulation = Simulation()
    with TestClient(create_app(simulation=simulation, data_dir=tmp_path)) as client:
        task = simulation.capture_rgbd()
        def forbidden():
            raise AssertionError('preview called task capture')
        monkeypatch.setattr(simulation, 'capture_rgbd', forbidden)
        result = client.get('/api/camera-preview')
        assert result.status_code == 200, result.text
        info = result.json()
        assert [c['id'] for c in info['cameras']] == ['2', '3', '1']
        assert info['preview_frame_id'].startswith('preview-')
        assert info['preview_frame_id'] != task['observation_id']
        assert all('depth_url' not in c for c in info['cameras'])
        assert simulation.latest_rgbd()['observation_id'] == task['observation_id']
        assert client.get('/api/camera-preview').json()['preview_frame_id'] == info['preview_frame_id']
        assert client.get(info['cameras'][0]['rgb_url']).status_code == 200
        assert client.post('/api/scene', json={'objects': []}).status_code == 200
        assert client.get(info['cameras'][0]['rgb_url']).status_code == 409
