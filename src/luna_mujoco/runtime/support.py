"""Separate current measurements from dated, verified inventory memory."""
from copy import deepcopy

import numpy as np

from ..task_binding_images import current_instance_locations


def cad_support_geometry(cad, subject_id, capture, tables):
    """Expose current model geometry separately from the visible surface samples."""
    shape = cad.get('shape_bounds')
    if shape is None:
        return {}  # Older observations did not publish complete shape bounds.
    validation = cad.get('pose_validation', {})
    direct = (validation.get('status') == 'accepted_by_policy'
              and validation.get('method') == 'foundationpose_direct')
    lower = np.asarray(shape.get('minimum_world_m', []), dtype=float)
    upper = np.asarray(shape.get('maximum_world_m', []), dtype=float)
    tolerance = None if direct else validation.get('surface', {}).get('tolerance_m')
    if not direct and (type(tolerance) not in (int, float)
                       or not np.isfinite(tolerance) or not 0 < tolerance <= .05):
        raise ValueError('observed_support_cad_geometry_invalid')
    if (cad.get('instance_id') != subject_id
            or any(cad.get(key) != capture.get(key) for key in ('observation_id', 'generation'))
            or (not direct and validation.get('status') != 'validated') or validation.get('accepted') is not True
            or not cad.get('mesh_sha256') or shape.get('mesh_sha256') != cad['mesh_sha256']
            or not np.array_equal(shape.get('T_world_object'), cad.get('T_world_object'))
            or lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(upper <= lower)):
        raise ValueError('observed_support_cad_geometry_invalid')
    model = {key: cad[key] for key in ('instance_id', 'asset_id', 'observation_id', 'generation', 'mesh_sha256')}
    model.update(bounds_world_m=[lower.tolist(), upper.tolist()], pose_validation_status=validation['status'],
        residual_p95_m=validation.get('surface', {}).get('residual_p95_m'),
        geometry_role='Registered CAD shape prior at a current accepted pose; hidden surfaces are '
            'model prior, not visible depth. Bounds do not by themselves prove contact, support identity or clearance.')
    if direct:
        model.pop('residual_p95_m', None)
        model['geometry_role'] = ('Current FoundationPose CAD geometry, accepted as accurate under the '
            'user-authorized project policy. Use its supplied bounds for support reasoning without '
            'requesting an additional geometry residual. Motion clearance is checked separately.')
    relationships = {}
    for identifier, table in tables.items():
        pose = np.asarray(table['model']['pose'], dtype=float)
        dims = np.asarray(table['model']['dims'], dtype=float)
        # Current measured table cuboids are horizontal/world-aligned. Do not
        # interpret a rotated cuboid's local dimensions as world bounds.
        if (pose.shape != (7,) or dims.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(dims).all() or np.any(dims <= 0)
                or not np.allclose(pose[3:], [1, 0, 0, 0], atol=1e-12, rtol=0)
                or not np.isfinite(table['surface_height_m'])
                or not np.isclose(pose[2] + dims[2] / 2, table['surface_height_m'], atol=1e-8, rtol=0)):
            continue
        gap = float(lower[2] - table['surface_height_m'])
        relationships[identifier] = {
            'source': 'current_registered_CAD_bounds_and_retained_measured_table',
            'table_source_observation_id': table['source_observation_id'],
            'cad_bottom_minus_table_surface_m': gap,
            'cad_xy_bounds_inside_table_xy_bounds': bool(np.all(lower[:2] >= pose[:2] - dims[:2] / 2)
                and np.all(upper[:2] <= pose[:2] + dims[:2] / 2)),
            'cad_pose_surface_tolerance_m': tolerance,
            'bottom_gap_within_pose_residual_tolerance': None if direct else abs(gap) <= tolerance,
            'meaning': 'Numerically consistent with this table support only when the reported checks pass; '
                'not proof of contact, object identity or unseen free space.'}
        if direct:
            relationships[identifier].pop('cad_pose_surface_tolerance_m')
            relationships[identifier].pop('bottom_gap_within_pose_residual_tolerance')
            relationships[identifier]['meaning'] = ('The accepted CAD pose supplies the object bottom '
                'height and XY containment; no residual check is required. Combine this numeric '
                'relationship with RGB and verified support history to select the support. '
                'This judgment does not replace robot collision checking.')
    return {'registered_cad': model, 'candidate_table_relationships': relationships}


def support_memory(entity, record, failed_attempt_count):
    initial = record.get('initial_source', {}).get('task_source_evidence', {})
    verified_initial = None
    if (initial.get('source') == 'openai_decisions' and initial.get('status') == 'yes'
            and initial.get('target_id') == entity['id'] and initial.get('source_id')
            and initial.get('request_hash') and initial.get('observation_id')):
        verified_initial = {'support_entity_id': initial['source_id'],
            'observation_id': initial['observation_id'], 'request_hash': initial['request_hash'],
            'source': 'openai_decisions', 'relation': 'verified_task_source_membership'}
    previous = entity.get('observed_support')
    if not (isinstance(previous, dict) and previous.get('source') == 'openai_decisions'
            and previous.get('request_hash') and previous.get('observation_id')):
        previous = None
    else:
        previous = {key: previous[key] for key in ('support_entity_id', 'observation_id', 'generation',
            'source', 'request_hash', 'confidence', 'assessment_basis') if key in previous}
    return deepcopy({'verified_initial_support': verified_initial,
        'previous_support_assessment': previous,
        'last_seen_observation_id': record.get('last_seen_observation_id'),
        'last_observed_position': record.get('last_observed_position'),
        'visibility': entity.get('visibility', record.get('visibility', 'unknown')),
        'status': record.get('status'), 'transfers': record.get('transfers', []),
        'failed_attempt_count': failed_attempt_count,
        'interpretation': 'These are dated observations and recorded actions, not new pixels. '
            'Occlusion alone does not erase an established relation. Assess continuity using history '
            'and current evidence; changed position, manipulation, contacts or conflicting evidence '
            'can invalidate it. Absence of recorded actions is not proof of no external change. '
            'Retained location does not authorize stale grasp geometry or certify motion clearance.'})


def support_context(capture, observation, subject, candidates):
    proof = observation.get('visual_inventory', {}).get('evidence', {})
    if (observation.get('observation_id') != capture['observation_id']
            or observation.get('generation') != capture.get('generation')
            or proof.get('source') != 'openai_decisions' or not proof.get('request_hash')):
        raise ValueError('observed_support_current_geometry_required')
    obj = next((obj for obj in observation.get('objects', []) if obj['id'] == subject['id']), {})
    points = np.asarray(obj.get('points_world_m', []), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
        raise ValueError('observed_support_current_geometry_required')
    tables = observation.get('demo_tables', {})
    measured = {'observation_id': capture['observation_id'], 'generation': capture.get('generation'),
        'source': 'current_identity_bound_sam3_registered_rgbd', 'identity_request_hash': proof['request_hash'],
        'frame': 'world', 'unit': 'm', 'subject_id': subject['id'], 'point_count': len(points),
        'visible_bounds_world_m': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        'measured_tables': {identifier: {'model': table['model'], 'surface_height_m': table['surface_height_m'],
            'source_observation_id': tables.get('source_observation_id')}
            for identifier, table in tables.get('tables', {}).items() if identifier in candidates},
        'limits': 'Visible surfaces only; the lowest measured point need not be the physical bottom. '
            'Table models are retained measured references, not newly seen surfaces. '
            'No simulator object pose or hidden shape is supplied; this is not a clearance certificate.'}
    if obj.get('cad'):
        model_geometry = cad_support_geometry(obj['cad'], subject['id'], capture, measured['measured_tables'])
        if model_geometry:
            measured.update(model_geometry)
            measured['limits'] = ('Visible bounds describe measured surfaces; their minimum Z need not be '
                'the physical bottom. registered_cad separately describes the complete known shape at the '
                'current accepted pose. Table models remain dated measured references. No simulator pose '
                'is used; support remains a visual and semantic judgment, not a clearance certificate.')
    identity_fields = ('id', 'kind', 'color', 'appearance_keywords')
    return {'subject': {key: subject[key] for key in identity_fields if key in subject},
        'support_candidates': {identifier: {key: value[key] for key in identity_fields if key in value}
                               for identifier, value in candidates.items()}, 'current_measured_geometry': measured,
        'verification_scope': 'current_support',
        'current_instance_locations': current_instance_locations(observation),
        'interpretation': 'Determine current location/support, not task completion or planned destination.'}
