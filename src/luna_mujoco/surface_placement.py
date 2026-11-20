"""Read current execution receipts and previously archived visual placement evidence."""
import hashlib
import json

SOURCE = 'measured_surface_placement'


def evidence_digest(receipt):
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, allow_nan=False).encode()).hexdigest()


def placement_evidence_id(visual):
    if visual.get('source') == 'bound_release_execution':
        from .demo_transfer_verification import release_execution_evidence_id
        return release_execution_evidence_id(visual)
    if visual.get('source') == 'openai_decisions':
        return visual.get('request_hash')
    receipt = visual.get('surface_measurement')
    if (visual.get('source') != SOURCE or not isinstance(receipt, dict)
            or visual.get('placement_outcome') != 'direct_success'
            or any(not receipt.get(k) or receipt[k] != visual.get(k) for k in
                   ('observation_id', 'binding_id', 'target_id', 'destination_id', 'support_entity_id'))
            or receipt.get('relation') != 'on' or '3' not in receipt.get('camera_ids', [])
            or not {'1', '2'}.intersection(receipt['camera_ids'])):
        return None
    digest = evidence_digest(receipt)
    return digest if visual.get('measurement_hash') == digest else None
