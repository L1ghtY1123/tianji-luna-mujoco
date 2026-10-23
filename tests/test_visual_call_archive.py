import asyncio
import base64
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from test_experiments import begin

from luna_mujoco.experiments import ExperimentStore


def test_actual_visual_call_archive_preserves_inputs_images_and_failure(tmp_path):
    store = ExperimentStore(tmp_path)
    session = begin(store)
    image = b'jpeg-image-bytes'
    request = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'candidate-0'},
        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(image).decode(),
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
    assert saved['http_status'] == 429 and saved['elapsed_s'] == 1.2
    assert 'base64' not in json.dumps(saved)


def test_cancelled_run_keeps_inflight_call_archive_without_late_live_event(tmp_path):
    from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

    from luna_mujoco.runner import Runner

    store = ExperimentStore(tmp_path / 'experiments')
    session = begin(store)
    runner = Runner(SensorRobot(), VisionAlgorithms(), lambda record: Choices(), tmp_path)
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
    from test_rgbd_workflow import Choices, SensorRobot, VisionAlgorithms

    from luna_mujoco.runner import Runner

    started, release = threading.Event(), threading.Event()
    event = {'request': {'input': [], 'questions': []}, 'response': {'answers': []},
             'request_hash': 'a' * 64, 'observation_id': 'in-flight', 'http_status': 200, 'elapsed_s': .1}
    class BlockingVision(VisionAlgorithms):
        def __init__(self):
            super().__init__()
            self.perception = SimpleNamespace(visual_inventory=SimpleNamespace(audit=None))

        def observe(self):
            started.set()
            assert release.wait(3)
            self.perception.visual_inventory.audit(event)
            return {}

    store = ExperimentStore(tmp_path / 'experiments')
    session = begin(store)
    algorithms = BlockingVision()
    runner = Runner(SensorRobot(), algorithms, lambda record: Choices(), tmp_path)
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
