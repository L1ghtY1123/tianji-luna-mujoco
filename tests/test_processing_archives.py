"""Independent processing calls retain immutable audit archives through cancellation."""
import asyncio
import hashlib
import json
import threading

import pytest
from policy_processing_fixture import runner_and_context
from processing_fixtures import before_kwargs, capture_and_action, release_evidence, work_area
from test_experiments import begin

from luna_mujoco.experiments import ExperimentStore


async def test_processing_visual_calls_retain_archive_until_actual_call_finishes(tmp_path, monkeypatch):
    runner, _, _ = runner_and_context(tmp_path)
    leases = []

    def retain():
        done = threading.Event()
        leases.append(done)
        return lambda event: None, done

    monkeypatch.setattr(runner, 'visual_audit_sink', retain)
    capture, action = capture_and_action()
    function, extra = runner.verify_unpacking, {'release_evidence': release_evidence(), 'work_area': work_area(), **before_kwargs()}
    result = await runner.call_algorithm(function,
        capture, target_id='pink', action_evidence=action, owner_id='run', **extra)
    assert result['success']
    assert len(leases) == 1, 'new processing request must own an in-flight archive lease'
    assert leases[0].is_set()



async def test_bound_processing_call_keeps_real_archive_after_cancellation(tmp_path, monkeypatch):
    runner, _, _ = runner_and_context(tmp_path / 'logs')
    store = ExperimentStore(tmp_path / 'experiments')
    session = begin(store)
    runner.run_id, runner.experiment = session.id, session
    runner.visual_inventory.api_key = 'archive-regression-secret'
    started, release = threading.Event(), threading.Event()
    post = runner.visual_inventory.client.post

    def blocked_post(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return post(*args, **kwargs)

    monkeypatch.setattr(runner.visual_inventory.client, 'post', blocked_post)
    capture, action = capture_and_action()
    function, extra = runner.verify_unpacking, {'release_evidence': release_evidence(), 'work_area': work_area(), **before_kwargs()}
    action['owner_id'] = session.id
    extra['table_release_reference']['owner_id'] = session.id
    task = asyncio.create_task(runner.call_algorithm(function, capture, target_id='pink',
        action_evidence=action, owner_id=session.id, **extra))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        session.stop({'status': 'cancelled'})
        runner.clear()
        assert session.busy
    finally:
        release.set()
        store.close()
        await runner.planning_worker
    assert not runner.events, 'late response must not enter a reset run'
    manifest = store.get(session.id)
    assert manifest['status'] == 'cancelled' and manifest['recording_complete']
    assert manifest['errors'] == []
    archived_calls = list((session.directory / 'decisions').glob('*.json'))
    assert len(archived_calls) == 3
    assert {json.loads(json.loads(path.read_text(encoding='utf-8'))['request']['input'][0]['content'][0]['text'])['camera_id']
                for path in archived_calls} == {'1', '2', '3'}
    for path in session.directory.rglob('*.json'):
        serialized = path.read_text(encoding='utf-8')
        assert 'archive-regression-secret' not in serialized and 'api_key' not in serialized
    for path in archived_calls:
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['response']['answers']
        images = [part['image_url'] for part in saved['request']['input'][0]['content'] if part['type'] == 'input_image']
        assert len(images) == 1  # The fixture's identical before/current RGB is deduplicated.
        for image in images:
            assert hashlib.sha256(store.file(session.id, image['path']).read_bytes()).hexdigest() == image['sha256']


