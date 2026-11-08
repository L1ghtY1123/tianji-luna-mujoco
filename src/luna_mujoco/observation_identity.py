"""Defer individual identity only for currently bounded, already-delivered contents."""
import numpy as np

from .demo_container_tracking import stationary_container_matches


def covered_container_candidates(capture, detections, scope, containers):
    """Keep covered surfaces in the collision envelope without assigning an object ID."""
    if (not scope or scope.get('generation') != capture.get('generation') or not scope.get('scene_token')
            or not containers or containers.get('status') != 'matched'
            or containers.get('observation_id') != capture.get('observation_id')
            or containers.get('generation') != capture.get('generation')):
        return []
    confirmed = {identifier for identifier in containers['candidate_mapping'].values()
                 if stationary_container_matches(containers, identifier)}
    covered = []
    for index, detection in enumerate(detections):
        if str(index) in containers['candidate_mapping'] or detection.get('kind') in {'bin', 'table', 'marker'}:
            continue
        points = np.asarray(detection.get('points', []), dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 30
                or not np.isfinite(points).all()):
            continue
        for occupancy in scope.get('container_occupancies', ()):
            if occupancy['container_id'] not in confirmed:
                continue
            lower = np.asarray(occupancy['minimum_world_m'], dtype=float)
            upper = np.asarray(occupancy['maximum_world_m'], dtype=float)
            if (lower.shape != (3,) or upper.shape != (3,)
                    or not np.isfinite([lower, upper]).all() or np.any(upper <= lower)):
                continue
            if np.all((points >= lower) & (points <= upper)):
                covered.append({
                    'candidate_index': index, 'candidate_id': 'candidate-' + str(index),
                    'camera_id': detection['camera_id'], 'container_id': occupancy['container_id'],
                    'collision_layer_name': occupancy['layer_name'],
                    'entry_receipt_id': occupancy['entry_receipt_id'],
                    'scene_token': scope['scene_token'], 'observation_id': capture['observation_id'],
                    'generation': capture['generation'],
                    'container_correspondence_binding_hash': containers['binding_hash'],
                    'point_count': len(points), 'measured_bounds_world_m': [
                        points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                    'occupancy_bounds_world_m': [lower.tolist(), upper.tolist()],
                    'reason': 'delivered_container_occupancy_covers_current_surface',
                    'identity_assigned': False})
                break
    return covered
