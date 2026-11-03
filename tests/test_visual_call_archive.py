import asyncio
import base64
import hashlib
import json
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from test_experiments import begin

from luna_mujoco.experiments import ExperimentStore


def encoded_image(format):
    stream = BytesIO()
    Image.new('RGB', (8, 6), color=(17, 129, 240)).save(stream, format=format)
    return stream.getvalue()


def test_question_images_archive_and_restore_the_exact_request(tmp_path):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    image = encoded_image('PNG')
    request = {'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'reference'}]}],
        'questions': [{'name': 'on_table', 'type': 'predicate', 'instructions': [
            {'type': 'input_text', 'text': 'CURRENT camera 3'},
            {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(image).decode()}]}]}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    relative = session.visual_decision({'request': request, 'request_hash': key})
    session.finish({'status': 'succeeded'})
    saved = json.loads(store.file(session.id, relative).read_text('utf-8'))
    assert 'base64' not in json.dumps(saved)
    part = saved['request']['questions'][0]['instructions'][1]
    descriptor = part['image_url']
    assert store.file(session.id, descriptor['path']).read_bytes() == image
    part['image_url'] = 'data:image/png;base64,' + base64.b64encode(image).decode()
    assert saved['request'] == request
    assert hashlib.sha256(json.dumps(saved['request'], sort_keys=True).encode()).hexdigest() == key


@pytest.mark.parametrize('format,media_type,extension', [('JPEG', 'image/jpeg', '.jpg'), ('PNG', 'image/png', '.png')])
def test_actual_visual_call_archive_preserves_inputs_images_and_failure(tmp_path, format, media_type, extension):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    image = encoded_image(format)
    request = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'candidate-0'},
        {'type': 'input_image', 'image_url': 'data:' + media_type + ';base64,' + base64.b64encode(image).decode(),
         'detail': 'auto'}]}], 'questions': [{'name': 'identity_0', 'type': 'choice'}]}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = session.visual_decision({'observation_id': 'frame-1', 'request_hash': key,
        'purpose': 'initialize_inventory', 'request': request,
        'response': {'answers': [], 'usage': {}}, 'elapsed_s': 1.2,
        'http_status': 429, 'error': 'visual_decisions_http_429'})
    request['questions'].clear()
    session.finish({'status': 'needs_attention'})
    saved = json.loads(store.file(session.id, path).read_text(encoding='utf-8'))
    assert saved['request']['questions'][0]['name'] == 'identity_0'
    image_ref = saved['request']['input'][0]['content'][1]['image_url']
    assert store.file(session.id, image_ref['path']).read_bytes() == image
    assert image_ref['sha256'] == hashlib.sha256(image).hexdigest()
    assert image_ref['media_type'] == media_type and image_ref['path'].endswith(extension)
    assert store.get(session.id)['recording_complete'] is True
    assert saved['http_status'] == 429 and saved['elapsed_s'] == 1.2
    assert 'base64' not in json.dumps(saved)


def test_mixed_jpeg_png_archive_can_reconstruct_request_hash_and_serve_exact_mime(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from luna_mujoco.experiment_api import experiment_router

    store = ExperimentStore(tmp_path)
    session = begin(store)
    originals = [encoded_image('JPEG'), encoded_image('PNG')]
    request = {'input': [{'role': 'user', 'content': [
        {'type': 'input_image', 'image_url': 'data:' + media_type + ';base64,' + base64.b64encode(image).decode()}
        for media_type, image in zip(('image/jpeg', 'image/png'), originals, strict=True)]}], 'questions': []}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    relative = session.visual_decision({'request': request, 'response': {'answers': []}, 'request_hash': key,
        'call_id': 'b' * 32, 'observation_id': 'current', 'http_status': 200, 'error': None})
    session.finish({'status': 'succeeded'})
    assert store.get(session.id)['errors'] == []
    saved = json.loads(store.file(session.id, relative).read_text('utf-8'))
    app = FastAPI()
    app.include_router(experiment_router(store))
    with TestClient(app) as client:
        for part, original in zip(saved['request']['input'][0]['content'], originals, strict=True):
            descriptor = part['image_url']
            response = client.get(f'/api/experiments/{session.id}/files/' + descriptor['path'])
            assert response.status_code == 200
            assert response.headers['content-type'] == descriptor['media_type']
            assert response.content == original
            assert hashlib.sha256(response.content).hexdigest() == descriptor['sha256']
            part['image_url'] = 'data:' + descriptor['media_type'] + ';base64,' + base64.b64encode(response.content).decode()
    assert saved['request'] == request
    assert hashlib.sha256(json.dumps(saved['request'], sort_keys=True).encode()).hexdigest() == key
    assert json.loads(store.file(session.id, 'decisions/calls.jsonl').read_text('utf-8'))['artifact'] == relative


def test_production_sam_subject_builder_archives_png_without_reencoding(tmp_path):
    from test_identity_subject_views import subject_case

    from luna_mujoco.identity_subject_views import append_identity_subject_views

    capture, payload, subjects = subject_case()
    payload['input'][0]['content'][1]['image_url'] = (
        'data:image/jpeg;base64,' + base64.b64encode(encoded_image('JPEG')).decode())
    request = append_identity_subject_views(payload, capture, subjects)
    original_png = base64.b64decode(request['input'][0]['content'][-1]['image_url'].split(',')[1])
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    store = ExperimentStore(tmp_path)
    session = begin(store)
    relative = session.visual_decision({'request': request, 'request_hash': key,
        'purpose': 'synthetic_subject_archive_test', 'observation_id': capture['observation_id']})
    session.finish({'status': 'succeeded'})
    assert store.get(session.id)['recording_complete'] is True
    archived = json.loads(store.file(session.id, relative).read_text('utf-8'))
    descriptor = archived['request']['input'][0]['content'][-1]['image_url']
    assert descriptor['media_type'] == 'image/png' and descriptor['path'].endswith('.png')
    assert store.file(session.id, descriptor['path']).read_bytes() == original_png


@pytest.mark.parametrize('case', ['unsupported', 'invalid_base64', 'wrong_mime', 'not_an_image', 'truncated'])
def test_visual_archive_rejects_unsupported_corrupt_or_mislabeled_images(tmp_path, case):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    media_type, raw = 'image/jpeg', encoded_image('JPEG')
    if case == 'unsupported':
        media_type, raw = 'image/gif', encoded_image('GIF')
    elif case == 'wrong_mime':
        raw = encoded_image('PNG')
    elif case == 'not_an_image':
        raw = b'jpeg-image-bytes'
    elif case == 'truncated':
        raw = raw[:len(raw) // 2]
    encoded = '@@not-base64@@' if case == 'invalid_base64' else base64.b64encode(raw).decode()
    request = {'input': [{'content': [{'type': 'input_image', 'image_url': 'data:' + media_type + ';base64,' + encoded}]}]}
    relative = session.visual_decision({'request': request, 'request_hash': 'c' * 64})
    session.finish({'status': 'failed'})
    manifest = store.get(session.id)
    assert manifest['recording_complete'] is False
    assert manifest['errors'] == ['archive_write:ValueError']
    with pytest.raises(FileNotFoundError):
        store.file(session.id, relative)


def test_cancelled_run_keeps_inflight_call_archive_without_late_live_event(tmp_path):
    from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms

    from luna_mujoco.runner import Runner

    store = ExperimentStore(tmp_path / 'experiments')
    session = begin(store)
    runner = Runner(SensorRobot(), VisionAlgorithms(), None, tmp_path)
    runner.run_id = session.id
    runner.experiment = session
    runner.state['status'] = 'running'
    audit, done = runner.visual_audit_sink()
    runner.clear()
    session.stop({'status': 'cancelled'})
    assert session.busy
    request = {'input': [], 'questions': []}
    key = hashlib.sha256(json.dumps(request).encode()).hexdigest()
    audit({'request': request, 'response': {'answers': []}, 'observation_id': 'cancelled-frame',
           'request_hash': key, 'elapsed_s': .2, 'http_status': 200, 'error': None})
    done.set()
    store.close()
    assert not runner.events
    assert runner.state['status'] == 'idle'
    saved = json.loads(store.file(session.id, f'decisions/{key}.json').read_text(encoding='utf-8'))
    assert saved['observation_id'] == 'cancelled-frame'
    index = store.file(session.id, 'decisions/calls.jsonl').read_text(encoding='utf-8')
    assert key in index


async def test_running_observe_owns_archive_across_cancel_and_reset(tmp_path):
    from rgbd_workflow_fixture import SensorRobot, VisionAlgorithms

    from luna_mujoco.runner import Runner

    started, release = threading.Event(), threading.Event()
    event = {'request': {'input': [], 'questions': []}, 'response': {'answers': []},
             'request_hash': 'a' * 64, 'observation_id': 'in-flight', 'http_status': 200, 'elapsed_s': .1}
    class BlockingVision(VisionAlgorithms):
        def __init__(self):
            super().__init__()
            self.perception = SimpleNamespace(visual_inventory=SimpleNamespace(audit=None),
                cad_resolver=SimpleNamespace(matcher=SimpleNamespace(audit=None)))

        def observe(self):
            started.set()
            assert release.wait(3)
            assert self.perception.cad_resolver.matcher.audit is self.perception.visual_inventory.audit
            self.perception.cad_resolver.matcher.audit(event)
            return {}

    store = ExperimentStore(tmp_path / 'experiments')
    session = begin(store)
    algorithms = BlockingVision()
    runner = Runner(SensorRobot(), algorithms, None, tmp_path)
    runner.run_id, runner.experiment = session.id, session
    runner.state['status'] = 'running'
    task = asyncio.create_task(runner.call_algorithm(algorithms.observe))
    assert await asyncio.to_thread(started.wait, 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    session.stop({'status': 'cancelled'})
    runner.clear()
    release.set()
    # Synchronous close must not depend on a pending event-loop done callback.
    store.close()
    await runner.planning_worker
    assert not runner.events
    assert store.file(session.id, 'decisions/' + 'a' * 64 + '.json').is_file()
