"""Necessary geometric exclusions for simultaneous masks of a known rigid instance."""
import hashlib

import numpy as np


def _bounds(points):
    points = np.asarray(points, dtype='<f8')
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 32 or not np.isfinite(points).all():
        return None
    return points.min(0), points.max(0), hashlib.sha256(points.tobytes()).hexdigest()


def cad_identity_exclusions(capture, detections, anchors, models):
    """Separated observed surfaces cannot belong to one smaller rigid CAD.

    Only a current measured correspondence supplies an anchor. Historical CAD
    contributes shape size, never a historical world position or a new identity.
    """
    bounds = [_bounds(item.get('points', [])) for item in detections]
    rows = []
    for anchor in anchors:
        identifier = anchor['entity_id']
        model = models.get(identifier, {})
        if (anchor.get('observation_id') != capture['observation_id']
                or anchor.get('generation') != capture['generation']
                or anchor.get('source') != 'measured_container_correspondence'
                or not anchor.get('binding_hash')
                or model.get('generation') != capture['generation']
                or model.get('instance_id') != identifier
                or model.get('source') != 'validated_cad_model_binding'
                or not model.get('validated_observation_id')
                or not model.get('source_sha256') or not model.get('mesh_sha256')):
            continue
        maximum = model['maximum_diameter_m'] + 2 * model['surface_tolerance_m']
        if not np.isfinite(maximum) or maximum <= 0:
            continue
        index = int(anchor['candidate'].removeprefix('candidate-'))
        if not 0 <= index < len(detections) or detections[index]['camera_id'] != anchor['camera']:
            continue
        fixed = bounds[index]
        if fixed is None:
            continue
        for number, candidate in enumerate(bounds):
            if candidate is None or number == index:
                continue
            gap = np.maximum(np.maximum(candidate[0] - fixed[1], fixed[0] - candidate[1]), 0.)
            separation = float(np.linalg.norm(gap))
            if separation <= maximum:
                continue
            rows.append({'candidate': 'candidate-' + str(number),
                'camera': detections[number]['camera_id'], 'excluded_instance_id': identifier,
                'assignment': 'excluded_not_reassigned', 'source': 'simultaneous_surface_distance_bound',
                'observation_id': capture['observation_id'], 'generation': capture['generation'],
                'minimum_surface_separation_m': separation,
                'maximum_same_instance_separation_m': float(maximum),
                'candidate_points_sha256': candidate[2], 'anchor_points_sha256': fixed[2],
                'anchor_candidate': anchor['candidate'], 'anchor_binding_hash': anchor['binding_hash'],
                'asset_id': model['asset_id'], 'source_sha256': model['source_sha256'],
                'mesh_sha256': model['mesh_sha256'],
                'model_validated_observation_id': model['validated_observation_id']})
    return rows
