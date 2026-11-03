"""Graph goals and verified effects replace fixed route/appearance phases."""
from copy import deepcopy

import pytest
from test_agent_model_layer import task_data

from luna_mujoco.agent.schema import TaskSpec
from luna_mujoco.runtime.inventory import PolicyTaskProgress


def ledger(processing=()):
    task = TaskSpec.model_validate(task_data() | {'processing': processing, 'work_area_ref': 'bench'})
    result = PolicyTaskProgress(task.instruction, 'collection')
    result.records = {'item': {'id': 'item', 'status': 'observed', 'display_id': 1}}
    result.scene_entities = {'item': {'id': 'item', 'kind': 'object'}}
    result.membership = {'item': 'required'}
    result.configure_task(task, ['item'])
    return result


def release(progress, identifier, destination, number):
    progress.select({'id': identifier}, destination)
    progress.record_release(identifier, feedback_id='feedback-' + str(number))
    return progress.confirm_visual_transfer(identifier, {
        **{key: progress.records[identifier]['visual_release'][key] for key in ('region_id', 'slot_id')
           if key in progress.records[identifier]['visual_release']},
        'source': 'openai_decisions', 'status': 'at_destination',
        'target_id': identifier, 'destination_id': destination, 'observation_id': 'obs-' + str(number)})


def product(identifier='output', source='item', skill='vla_unpack'):
    return {'id': identifier, 'input_entity_id': source, 'process_id': 'process-' + identifier,
        'processing_skill': skill, 'kind': 'object', 'color': 'black',
        'creation_evidence': {'source': 'confirmed_process_output', 'observation_id': 'new-obs',
            'request_hash': 'hash', 'action_id': 'action-' + identifier, 'generation': 1}}


def test_arbitrary_intermediate_transfers_do_not_finish_task_or_impose_two_legs():
    progress = ledger()
    for number, destination in enumerate(['bench', 'buffer', 'inspection'], 1):
        assert release(progress, 'item', destination, number)
        assert progress.completed_input_ids() == set()
    assert release(progress, 'item', 'tray', 4)
    assert progress.completed_input_ids() == {'item'}
    assert len(progress.records['item']['transfers']) == 4


def test_processing_requirement_needs_verified_lineage_and_final_product_delivery():
    progress = ledger(('vla_unpack',))
    assert release(progress, 'item', 'tray', 1)
    assert progress.completed_input_ids() == set()
    assert release(progress, 'item', 'bench', 2)
    progress.register_product(product())
    assert progress.records['item']['status'] == 'consumed'
    assert progress.completed_input_ids() == set()
    assert release(progress, 'output', 'tray', 3)
    assert progress.completed_input_ids() == {'item'}
    assert progress.records['output']['id'] != progress.records['item']['id']
    assert 'item_lifecycle' not in progress.snapshot()


def test_multiple_processing_capabilities_compose_without_new_task_phases():
    progress = ledger(('inspect_part', 'polish_part'))
    progress.register_product(product('inspected', skill='inspect_part'))
    progress.register_product(product('polished', source='inspected', skill='polish_part'))
    assert release(progress, 'polished', 'tray', 1)
    assert progress.completed_input_ids() == {'item'}


def test_transform_replay_is_idempotent_but_conflicting_output_is_rejected():
    progress = ledger(('vla_unpack',))
    output = product()
    progress.register_product(output)
    snapshot = deepcopy(progress.snapshot())
    progress.register_product(output)
    assert progress.snapshot() == snapshot
    with pytest.raises(ValueError):
        progress.register_product(product('different'))


def test_later_observation_cannot_resurrect_consumed_input_geometry():
    progress = ledger(('vla_unpack',))
    progress.register_product(product())
    progress.observe([{'id': 'output', 'kind': 'object', 'color': 'black'}], 'later-capture')
    assert progress.scene_entities['item']['geometry_status'] == 'historical_consumed'
    assert progress.inventory_state()['retired_entity_ids'] == ['item']
    assert progress.membership['item'] == 'excluded'


def test_next_transfer_drops_previous_correspondence_but_keeps_verified_history():
    progress = ledger()
    assert release(progress, 'item', 'bench', 1)
    history = deepcopy(progress.records['item']['transfers'])
    progress.records['item']['completed_identity_track'] = {'binding': 'old-transfer'}
    progress.select({'id': 'item'}, 'tray')
    assert 'completed_identity_track' not in progress.visual_progress()['item']
    assert progress.records['item']['transfers'] == history
    assert progress.completed_input_ids() == set()
