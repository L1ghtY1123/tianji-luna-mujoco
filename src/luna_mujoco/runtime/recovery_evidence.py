"""Expose the existing no-progress guard before Decisions chooses an operation."""
import hashlib
import json

import numpy as np

from ..inventory_reference import visual_inventory_state
from .execution_state import result_status

# Only for detecting no-progress sensing retries, never for geometry, collision
# or pose acceptance. Keep a fixed depth anchor so tiny changes cannot accumulate
# indefinitely by sliding the comparison window.
OBSERVATION_DEPTH_NOISE_M = 1e-5


def _observation_depth_digest(context, camera, depth):
    references = context.setdefault('_recovery_depth_reference', {})
    previous = references.get(camera)
    depth = np.ascontiguousarray(depth)
    if (previous is None or previous['depth'].shape != depth.shape
            or not np.allclose(previous['depth'], depth, atol=OBSERVATION_DEPTH_NOISE_M,
                               rtol=0, equal_nan=True)):
        previous = {'depth': depth.copy(), 'digest': hashlib.sha256(depth.tobytes()).digest()}
        references[camera] = previous
    return previous['digest']


def reusable_observation_result(context, history, observation_epoch, contracts):
    """Current successful sensing result only; this never chooses a continuation."""
    latest = history[-1] if history else {}
    node = latest.get('node', {})
    result = latest.get('result', {})
    contract = contracts.get(node.get('skill'))
    if (contract is None or not contract.recovery_result_reusable or not contract.recovery_retry_safe
            or contract.invalidates_observation or latest.get('status') != 'success'
            or result_status(result) != 'success'):
        return None
    identifier = result.get('observation_id')
    capture = context.get('latest_capture', {})
    observation = context.get('observation', {})
    if (not identifier or identifier != capture.get('observation_id')
            or identifier != observation.get('observation_id')
            or result.get('generation') != capture.get('generation')
            or result.get('generation') != observation.get('generation')
            or context.get('capture_epochs', {}).get(identifier) != observation_epoch):
        return None
    return {'node_id': node['id'], 'observation_id': identifier, 'generation': result['generation'],
            'action_epoch': observation_epoch}


def observation_inputs(context, facts):
    """Inputs to observation/identity interpretation, independent of planning state."""
    capture = context.get('latest_capture', {})
    reference = context.get('visual_inventory_reference', {})
    entity_fields = ('id', 'kind', 'color', 'description', 'appearance_keywords', 'initial_parent_id',
        'roles', 'segmentation_query', 'reference_camera_id', 'bbox')
    entities = reference.get('inventory_state', {}).get('entities', reference.get('entities', {}))
    return {
        'capture_version': {key: capture.get(key) for key in ('generation', 'manifest_revision', 'command_revision')},
        'action_epoch': context.get('capture_epochs', {}).get(capture.get('observation_id')),
        'cameras': {camera: {key: view.get(key) for key in
            ('camera_id', 'calibration_id', 'intrinsics', 'T_world_camera', 'registered_to')}
            for camera, view in capture.get('views', {}).items()},
        'reference_id': reference.get('observation_id'),
        'task_spec': reference.get('task_spec'), 'task_inventory': reference.get('task_inventory'),
        'entities': {identifier: {key: entity[key] for key in entity_fields if key in entity}
                     for identifier, entity in entities.items()},
        'lifecycle': visual_inventory_state(reference),
        'physical_facts': {key: facts.get(key) for key in ('grasp_held', 'released', 'place_execution_completed',
            'attached', 'unpack_pending', 'unpack_verified', 'recovery_needed')},
        'settings': {key: context.get('planning_settings', {}).get(key) for key in ('observation', 'pointcloud_filter')},
    }


def operation_evidence(context, node_id, reason, facts, completed_inputs, *, failed_skill=None, target_id=None):
    pixels = hashlib.sha256()
    observation_pixels = hashlib.sha256()
    rgb_pixels = hashlib.sha256()
    for key, view in sorted(context.get('latest_capture', {}).get('views', {}).items()):
        pixels.update(key.encode())
        pixels.update((view['rgb'][::8, ::8] // 16).tobytes())
        observation_pixels.update(key.encode())
        observation_pixels.update((view['rgb'][::8, ::8] // 16).tobytes())
        rgb_pixels.update(key.encode())
        rgb_pixels.update((view['rgb'][::8, ::8] // 16).tobytes())
        for field in ('depth_m', 'valid_depth'):
            if field in view:
                pixels.update(field.encode())
                pixels.update(np.ascontiguousarray(view[field]).tobytes())
                observation_pixels.update(field.encode())
                observation_pixels.update(_observation_depth_digest(context, key, view[field])
                    if field == 'depth_m' else np.ascontiguousarray(view[field]).tobytes())
    return {'node': node_id, 'pixels': pixels.hexdigest(), 'facts': facts,
        'rgb_pixels': rgb_pixels.hexdigest(),
        'observation_pixels': observation_pixels.hexdigest(),
        'measured_ids': sorted(item['id'] for item in context.get('observation', {}).get('objects', [])),
        'map_coverage': context.get('scene', {}).get('coverage', {}),
        'completed_inputs': sorted(completed_inputs), 'reason': reason,
        'failed_skill': failed_skill, 'observation_inputs': observation_inputs(context, facts),
        'release_execution_key': context.get('release_execution_evidence', {}).get('sha256'),
        'map_inputs': {'observation_id': context.get('observation', {}).get('observation_id'),
            'target_id': target_id,
            'geometry': [{'id': item['id'], 'position': item.get('position'),
                'cad': {key: item.get('cad', {}).get(key) for key in ('mesh_sha256', 'T_world_object')},
                'obb': {key: item.get('obb', {}).get(key) for key in ('T_frame_obb', 'half_extents_m')}}
                         for item in context.get('observation', {}).get('objects', [])]}}


def operation_signature(evidence, proposal):
    observe = proposal.get('skill') == 'observe_scene' or (
        proposal.get('action') == 'retry' and evidence.get('failed_skill') == 'observe_scene')
    release = proposal.get('skill') == 'verify_place' or (
        proposal.get('action') == 'retry' and evidence.get('failed_skill') == 'verify_place')
    if proposal.get('skill') == 'build_scene':
        # A new map token and the facts produced by publishing it are outputs,
        # not new sensing inputs. They cannot authorize rebuilding it forever.
        bound = {key: evidence[key] for key in ('node', 'pixels', 'completed_inputs', 'reason', 'map_inputs')}
    elif observe or release:
        bound = {key: evidence[key] for key in
                 ('node', 'pixels', 'completed_inputs', 'reason', 'observation_inputs')}
        bound['pixels'] = evidence.get('observation_pixels', evidence['pixels'])
        if release:
            # This verifier consumes RGB and the held/released identity chain,
            # not new depth, SAM masks, measured CAD/OBB or measured entity IDs.
            bound['pixels'] = evidence.get('rgb_pixels', bound['pixels'])
            bound['observation_inputs'] = {key: value for key, value in bound['observation_inputs'].items()
                                            if key not in {'lifecycle', 'settings', 'physical_facts'}}
            bound['release_execution_key'] = evidence.get('release_execution_key')
    else:
        # Preserve the original evidence boundary for every other operation.
        bound = {key: value for key, value in evidence.items()
                 if key not in ('failed_skill', 'observation_inputs', 'observation_pixels', 'rgb_pixels',
                                'release_execution_key', 'map_inputs')}
    return json.dumps({**bound, **{key: proposal.get(key) for key in ('action', 'skill', 'arguments')}},
        sort_keys=True, default=str)


def record_failed_operation(seen, evidence, skill, argument_variants):
    """Include the initial failed evaluation and its resolved argument aliases."""
    for arguments in argument_variants:
        seen.update(operation_signatures(evidence, {'action': 'skill', 'skill': skill, 'arguments': arguments}))
    seen.add(operation_signature(evidence, {'action': 'retry'}))


def operation_signatures(evidence, proposal):
    """Reanalyzing a fresh attempt's unchanged input is not a new recovery."""
    signatures = {operation_signature(evidence, proposal)}
    if (proposal.get('action') == 'skill' and proposal.get('skill') == 'observe_scene'
            and proposal.get('arguments', {}).get('capture_mode') == 'fresh'):
        for mode in ('reuse_current', None):
            arguments = dict(proposal['arguments'])
            if mode is None:
                arguments.pop('capture_mode')
            else:
                arguments['capture_mode'] = mode
            signatures.add(operation_signature(evidence, {**proposal, 'arguments': arguments}))
    return signatures


def blocked_operations(seen, evidence, *, capture_current=True):
    """Only suppress choices the unchanged dispatch guard would already reject."""
    if not capture_current:
        # _judge will acquire current raw input; old pixels cannot remove choices
        # before that capture exists. The dispatch guard still checks afterwards.
        return []
    result = []
    for signature in sorted(seen):
        row = json.loads(signature)
        operation = {key: row.pop(key) for key in ('action', 'skill', 'arguments')}
        if (operation['action'] in ('skill', 'retry')
                and operation_signature(evidence, operation) == signature):
            result.append({**operation, 'reason': 'identical_evidence_and_operation_no_progress'})
    return result
