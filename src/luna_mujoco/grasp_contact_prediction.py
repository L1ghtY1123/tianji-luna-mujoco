"""Reject bounded commands that cannot reach observed opposing contact patches."""
import numpy as np

from .demo_gripper_quality import gripper_width_model
from .geometry import mount_tcp_pose


def is_proven_contact_shortfall(proof):
    """Only numerical rejection evidence permits another frozen candidate pool."""
    if (not isinstance(proof, dict) or proof.get('source') != 'measured_points_and_own_finite_pad_planes'
            or proof.get('opposing_contact_predicted') is not False
            or proof.get('is_holding_evidence') is not False or not proof.get('model_fingerprint')
            or type(proof.get('observed_point_count')) is not int or proof['observed_point_count'] <= 0):
        return False
    limit, probe = proof.get('driver_limit_rad'), proof.get('contact_probe_rad')
    if (not all(type(value) in (int, float) and np.isfinite(value) for value in (limit, probe))
            or not 0 < probe < limit):
        return False
    pads = proof.get('pads')
    if not isinstance(pads, list) or len(pads) != 2 or not all(isinstance(row, dict) for row in pads):
        return False
    gaps = [row.get('minimum_signed_gap_m') for row in pads]
    return (all(type(value) in (int, float) and np.isfinite(value) for value in gaps)
            and all(type(row.get('observed_patch_point_count')) is int
                    and row['observed_patch_point_count'] > 0 for row in pads)
            and any(value > 0 for value in gaps))


def bounded_contact_prediction(points_world_m, T_world_tcp, arm, driver_limit_rad, tolerance_fraction):
    points = np.asarray(points_world_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('bounded contact prediction requires finite measured points')
    model = gripper_width_model(arm)
    probe = driver_limit_rad - tolerance_fraction * (model.upper - model.lower)
    if not model.lower < probe <= driver_limit_rad <= model.upper:
        raise ValueError('bounded contact prediction requires a usable closing interval')
    pose = mount_tcp_pose(T_world_tcp, arm)
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    patches = model.contact_patch_gaps(local, probe)
    return {'source': 'measured_points_and_own_finite_pad_planes', 'is_holding_evidence': False,
        'driver_limit_rad': driver_limit_rad, 'contact_probe_rad': probe,
        'model_fingerprint': model.fingerprint, 'observed_point_count': len(points),
        'opposing_contact_predicted': all(row['minimum_signed_gap_m'] is not None
            and row['minimum_signed_gap_m'] <= 0. for row in patches), 'pads': patches,
        'limitations': ['unobserved_surfaces_unknown', 'rigid_fixed_cloud_prediction',
                       'actual_stable_gripper_feedback_required_before_lift']}
