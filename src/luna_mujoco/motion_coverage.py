"""Project-side unknown-space veto, independent of semantic identities and SDK internals.

Depth is optical-axis Z in metres. Pixel cells carry the measured return with a
10 mm uncertainty allowance; invalid cells and everything behind a return stay
unknown. This is a sensor/model certificate, not a claim about hidden geometry.
"""
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

from .planning_timing import measured


def task_interaction_links(urdf_path, robot_config, tool_roots):
    """Select complete configured TCP subtrees plus every attached model link.

    This is an observation scope, never a collision-model filter. Contact
    exemptions do not define tools: camera mounts and other descendants count.
    """
    root = ET.parse(urdf_path).getroot()
    known = {link.get('name') for link in root.findall('link')}
    parents = {joint.find('child').get('link'): joint.find('parent').get('link')
               for joint in root.findall('joint')}
    kin = robot_config['robot_cfg']['kinematics']
    extras = kin.get('extra_links', {})
    known.update(extras)
    parents.update({link: value['parent_link_name'] for link, value in extras.items()})
    roots = set(tool_roots)
    collision_links = set(kin['collision_link_names'])
    if not roots or not roots <= known or not roots <= collision_links:
        raise ValueError('task coverage TCP roots missing from configured robot geometry')
    selected = set()
    for link in collision_links:
        current, ancestors = link, set()
        while True:
            if current not in known or current in ancestors:
                raise ValueError('task coverage collision link has an invalid robot chain')
            ancestors.add(current)
            if current not in parents:
                if current != kin['base_link']:
                    raise ValueError('task coverage collision link is disconnected from model base')
                break
            current = parents[current]
        if ancestors & roots or link in extras:
            selected.add(link)
    for link in roots:
        balls = kin['collision_spheres'].get(link, [])
        if not any(np.isfinite(ball['radius']) and ball['radius'] > 0
                   and np.asarray(ball['center']).shape == (3,)
                   and np.isfinite(ball['center']).all() for ball in balls):
            raise ValueError('task coverage TCP geometry has no finite enabled sphere')
    return selected


def joint_motion_bounds(urdf_path, robot_config, joint_names, *, per_link=False, per_sphere=False):
    """Bound centre travel using joint axes and enclosing downstream orbit balls.

    A point rotating about an axis traces a circle: its axial component does
    not travel. Keep that circle's centre and radius while walking towards the
    root. Rigid transforms are exact; downstream rotations add orbit radii by
    triangle inequality. These bounds hold throughout every joint-linear arc,
    not just at its endpoints. Per-sphere rows retain configured link order.
    """
    from scipy.spatial.transform import Rotation

    root = ET.parse(urdf_path).getroot()
    parents = {}
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        offset = np.fromstring('0 0 0' if origin is None else origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring('0 0 0' if origin is None else origin.get('rpy', '0 0 0'), sep=' ')
        axis = joint.find('axis')
        axis = np.fromstring('1 0 0' if axis is None else axis.get('xyz', '1 0 0'), sep=' ')
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
            raise ValueError('invalid motion coverage joint axis')
        axis = axis / np.linalg.norm(axis)
        limit = joint.find('limit')
        travel = None if limit is None else (float(limit.get('lower', 'nan')), float(limit.get('upper', 'nan')))
        mimic = joint.find('mimic')
        parents[joint.find('child').get('link')] = (
            joint.find('parent').get('link'), joint.get('name'), joint.get('type'), offset,
            Rotation.from_euler('xyz', rpy).as_matrix(), axis, travel,
            None if mimic is None else (mimic.get('joint'), float(mimic.get('multiplier', '1'))))
    kin = robot_config['robot_cfg']['kinematics']
    for link, value in kin.get('extra_links', {}).items():
        pose = value['fixed_transform']
        parents[link] = (value['parent_link_name'], value.get('joint_name'), 'fixed',
                         np.asarray(pose[:3]), Rotation.from_quat([*pose[4:7], pose[3]]).as_matrix(),
                         np.array([1., 0., 0.]), None, None)
    by_link = {}
    known_links = set(parents) | {item[0] for item in parents.values()}
    for link, spheres in kin['collision_spheres'].items():
        sphere_bounds = []
        if link not in known_links:
            raise ValueError('motion coverage collision link absent from robot model')
        for sphere in spheres:
            bounds = dict.fromkeys(joint_names, 0.)
            if sphere['radius'] <= 0:
                sphere_bounds.append(np.zeros(len(joint_names)))
                continue
            point, uncertainty = np.asarray(sphere['center'], dtype=float), 0.
            current, visited = link, set()
            while current in parents:
                if current in visited:
                    raise ValueError('motion coverage URDF cycle')
                visited.add(current)
                parent, joint, kind, offset, rotation, axis, travel, mimic = parents[current]
                source, multiplier = (joint, 1.) if mimic is None else mimic
                if kind in ('revolute', 'continuous'):
                    axial = axis * np.dot(axis, point)
                    lever = float(np.linalg.norm(point - axial)) + uncertainty
                    if source in bounds:
                        bounds[source] += abs(multiplier) * lever
                    point, uncertainty = axial, lever
                elif kind == 'prismatic':
                    if source in bounds:
                        bounds[source] += abs(multiplier)
                    if travel is None or not np.isfinite(travel).all() or travel[1] < travel[0]:
                        raise ValueError('bounded prismatic travel required for motion coverage')
                    point = point + axis * (travel[0] + travel[1]) / 2
                    uncertainty += (travel[1] - travel[0]) / 2
                elif kind != 'fixed':
                    raise ValueError('unsupported motion coverage joint type')
                point = rotation @ point + offset
                current = parent
            sphere_bounds.append(np.array([bounds[name] for name in joint_names]))
        by_link[link] = np.asarray(sphere_bounds).reshape(-1, len(joint_names))
    known = {item[1] for item in parents.values()}
    if set(joint_names) - known:
        raise ValueError('motion coverage joint absent from robot model')
    if per_sphere:
        return by_link
    maximum = {link: np.max(values, axis=0) for link, values in by_link.items() if len(values)}
    return maximum if per_link else np.max(list(maximum.values()), axis=0)


def segment_motion_bounds(positions, coefficients, storage_dtype):
    """Bound joint-linear segments in the public FK input representation.

    Both the requested and represented deltas bound every changing dependency.
    A sphere whose entire influencing joint vector has identical represented
    endpoints is stationary in this model, including all intermediate inputs.
    No spatial epsilon or angular threshold is used to admit new volume.
    """
    positions = np.asarray(positions, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    storage_dtype = np.dtype(storage_dtype)
    if (positions.ndim != 2 or coefficients.ndim not in (1, 2)
            or coefficients.shape[-1] != positions.shape[1]
            or not np.issubdtype(storage_dtype, np.floating)
            or not np.isfinite(positions).all() or not np.isfinite(coefficients).all()
            or np.any(coefficients < 0)):
        raise ValueError('invalid represented joint motion input')
    represented = positions.astype(storage_dtype).astype(float)
    if not np.isfinite(represented).all():
        raise ValueError('joint positions overflow public FK representation')
    original_steps = np.abs(np.diff(positions, axis=0, append=positions[-1:]))
    model_steps = np.abs(np.diff(represented, axis=0, append=represented[-1:]))
    displacement = np.maximum(original_steps, model_steps) @ coefficients.T
    stationary = (model_steps != 0).astype(int) @ (coefficients != 0).astype(int).T == 0
    displacement[stationary] = 0.
    return displacement, stationary


class MotionCoverage:
    """Require whole swept spheres to lie before measured first returns."""

    def __init__(self, views, scene_token, *, uncertainty_m=.01):
        if not np.isfinite(uncertainty_m) or uncertainty_m < 0:
            raise ValueError('depth uncertainty must be finite and nonnegative')
        self.scene_token = scene_token
        self.capture_scene_token = scene_token
        self.uncertainty_m = uncertainty_m
        self.invalid_reason = None
        self.capture_robot_spheres = np.empty((0, 4))
        self.bounded_updates = []
        self.views = []
        self._certificate_cache = OrderedDict()
        self._cache_capture_spheres = np.empty((0, 4))
        for camera, view in views.items():
            depth = np.asarray(view['depth_left_m'], dtype=float)
            valid = np.asarray(view['valid_left_mask']) != 0
            intrinsics = np.array(view['left_intrinsics'], dtype=float, copy=True)
            transform = np.asarray(view['T_base_left_camera'], dtype=float)
            if (depth.ndim != 2 or valid.shape != depth.shape or intrinsics.shape != (3, 3)
                    or transform.shape != (4, 4) or not np.isfinite(intrinsics).all()
                    or not np.isfinite(transform).all() or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0):
                raise ValueError('invalid registered depth coverage input')
            if (not np.allclose(intrinsics[2], [0, 0, 1]) or intrinsics[0, 1] != 0
                    or intrinsics[1, 0] != 0 or not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-5)):
                raise ValueError('coverage requires calibrated pinhole intrinsics and rigid camera transform')
            measured = np.where(valid & np.isfinite(depth) & (depth > 0), depth, 0.)
            measured.setflags(write=False)
            intrinsics.setflags(write=False)
            self.views.append((camera, measured, intrinsics, np.linalg.inv(transform)))

    def invalidate(self, reason):
        self.invalid_reason = reason

    def sphere_observed(self, center, radius):
        """A bounding sphere certificate also certifies any contained cell."""
        radius += self.uncertainty_m
        for _camera, depth, intrinsics, inverse in self.views:
            x, y, z = inverse[:3, :3] @ center + inverse[:3, 3]
            if z <= radius:
                continue
            denominator = z*z - radius*radius
            horizontal = radius * np.sqrt(z*z + x*x - radius*radius)
            vertical = radius * np.sqrt(z*z + y*y - radius*radius)
            left = int(np.floor(intrinsics[0, 0]*(x*z-horizontal)/denominator + intrinsics[0, 2]))
            right = int(np.ceil(intrinsics[0, 0]*(x*z+horizontal)/denominator + intrinsics[0, 2]))
            top = int(np.floor(intrinsics[1, 1]*(y*z-vertical)/denominator + intrinsics[1, 2]))
            bottom = int(np.ceil(intrinsics[1, 1]*(y*z+vertical)/denominator + intrinsics[1, 2]))
            if left < 0 or top < 0 or right >= depth.shape[1] or bottom >= depth.shape[0]:
                continue
            if np.all(depth[top:bottom+1, left:right+1] >= z + radius):
                return True
        return False

    def covered_by_union(self, center, radius, robot_allowed, contact_bounds):
        """Certify intersecting cells by rays, admitted start self volume, or contact.

        Unknown boundary cells fail closed after bounded 5 mm subdivision.
        Self volume is the measured capture configuration, never its sweep.
        """
        captured = np.asarray(self.capture_robot_spheres, dtype=float).reshape(-1, 4)
        if (robot_allowed and len(captured)
                and np.any(np.linalg.norm(captured[:, :3] - center, axis=1) + radius <= captured[:, 3])):
            return True
        if (contact_bounds is not None and np.all(center - radius >= contact_bounds[0])
                and np.all(center + radius <= contact_bounds[1])):
            return True
        if not robot_allowed and contact_bounds is None:
            return False
        if len(captured):
            captured = captured[np.linalg.norm(captured[:, :3] - center, axis=1) <= captured[:, 3] + radius]
        pending = [(np.asarray(center), radius)]
        checked = 0
        offsets = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        while pending:
            midpoint, half = pending.pop()
            checked += 1
            if checked > 8192:
                return False
            if np.linalg.norm(np.maximum(np.abs(midpoint - center) - half, 0.)) > radius:
                continue
            if robot_allowed and len(captured):
                farthest = np.linalg.norm(np.abs(captured[:, :3] - midpoint) + half, axis=1)
                # Only Q (the query sphere) intersected with this cell needs a
                # certificate. For x in Q, ||x-c||² <= r² + ||q-c||²
                # + 2 (q-c)·(x-q); maximize the linear term over the cell.
                # This removes exterior cube corners without sampling away an
                # unknown sliver of the actual sphere.
                difference = center - captured[:, :3]
                clipped_squared = (radius * radius + np.sum(difference * difference, axis=1)
                    + 2 * (difference @ (midpoint - center) + half * np.sum(np.abs(difference), axis=1)))
                if np.any(np.minimum(farthest * farthest, clipped_squared) <= captured[:, 3] ** 2):
                    continue
            if contact_bounds is not None and np.all(midpoint-half >= contact_bounds[0]) and np.all(midpoint+half <= contact_bounds[1]):
                continue
            if self.sphere_observed(midpoint, np.sqrt(3)*half):
                continue
            if half * 2 <= .005:
                return False
            pending.extend((midpoint + offset * half/2, half/2) for offset in offsets)
        return True

    def camera_diagnostics(self, center, radius):
        """Explain the whole-sphere certificate; this does not classify hidden space as occupied."""
        radius += self.uncertainty_m
        results = []
        for camera, depth, intrinsics, inverse in self.views:
            x, y, z = inverse[:3, :3] @ center + inverse[:3, 3]
            item = {"camera": camera, "sphere_camera_center_m": [float(x), float(y), float(z)],
                    "certificate_radius_m": float(radius), "image_size_px": list(depth.shape[::-1])}
            results.append(item)
            if z <= radius:
                item["reason"] = "sphere_intersects_or_behind_camera_plane"
                continue
            denominator = z*z - radius*radius
            horizontal = radius * np.sqrt(z*z + x*x - radius*radius)
            vertical = radius * np.sqrt(z*z + y*y - radius*radius)
            left = int(np.floor(intrinsics[0, 0]*(x*z-horizontal)/denominator + intrinsics[0, 2]))
            right = int(np.ceil(intrinsics[0, 0]*(x*z+horizontal)/denominator + intrinsics[0, 2]))
            top = int(np.floor(intrinsics[1, 1]*(y*z-vertical)/denominator + intrinsics[1, 2]))
            bottom = int(np.ceil(intrinsics[1, 1]*(y*z+vertical)/denominator + intrinsics[1, 2]))
            item["projected_bounds_px"] = [left, top, right, bottom]
            if left < 0 or top < 0 or right >= depth.shape[1] or bottom >= depth.shape[0]:
                item["reason"] = "sphere_projection_outside_image"
                continue
            patch = depth[top:bottom+1, left:right+1]
            item.update(required_free_depth_m=float(z + radius),
                        minimum_depth_m=float(patch.min()), invalid_depth_pixels=int(np.count_nonzero(patch <= 0)),
                        insufficient_depth_pixels=int(np.count_nonzero((patch > 0) & (patch < z + radius))))
            item["reason"] = ("invalid_depth_in_projection" if item["invalid_depth_pixels"] else
                              "first_return_before_required_depth" if item["insufficient_depth_pixels"] else
                              "whole_sphere_certified")
        return results

    @measured('coverage_certificate')
    def check(self, spheres, displacement_bounds_m, *, scene_token, robot_sphere_indices=(),
              contact_bounds=None, contact_sphere_indices=(), stop_on_first_unknown=False,
              required_sphere_indices=None):
        """Each row encloses the entire following joint-linear segment.

        Caller supplies a proven upper bound on centre travel; the final row
        still checks its endpoint. The optional explicit task scope only limits
        the sensor certificate; excluded geometry is not classified as covered.
        """
        spheres = np.asarray(spheres, dtype=float)
        bounds = np.asarray(displacement_bounds_m, dtype=float)
        if (spheres.ndim != 3 or spheres.shape[2] != 4 or not spheres.shape[0]
                or bounds.shape not in ((len(spheres),), spheres.shape[:2]) or not np.isfinite(spheres).all()
                or not np.isfinite(bounds).all() or np.any(bounds < 0)):
            raise ValueError('invalid swept sphere coverage input')
        if bounds.ndim == 1:
            bounds = np.broadcast_to(bounds[:, None], spheres.shape[:2])
        required = np.ones(spheres.shape[:2], dtype=bool)
        scoped = required_sphere_indices is not None
        if scoped:
            indices = np.asarray(required_sphere_indices)
            if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= spheres.shape[1])):
                raise ValueError('invalid task coverage sphere scope')
            required[:] = False
            required[:, indices] = True
            if not np.any(required & (spheres[..., 3] > 0)):
                raise ValueError('task coverage scope has no enabled geometry')
        proof = {'covered': False, 'failure_reason': 'observation_coverage_insufficient',
                 'scene_token': scene_token, 'capture_scene_token': self.capture_scene_token,
                 'policy': ('task_interaction_swept_spheres_in_measured_free_space' if scoped else
                            'all_robot_and_attachment_swept_spheres_in_measured_free_space'),
                 'coverage_scope': 'task_interaction_geometry' if scoped else 'all_model_geometry',
                 'whole_robot_unknown_space_certified': False,
                 'excluded_sphere_count': int((~required).sum()),
                 'excluded_geometry_coverage': 'not_evaluated',
                 'continuous_segment_envelope_checked': False,
                 'interpolation_assumption': 'joint_linear_between_checked_command_samples',
                 'maximum_motion_envelope_m': float(bounds[required].max()),
                 'depth_uncertainty_m': self.uncertainty_m,
                 'unknown_space_policy': 'reject_and_require_observation'}
        proof.update(initial_robot_volume_is_observed_free=False,
                     initial_robot_volume_policy='unchanged_measured_capture_model_admissible_volume_only',
                     contact_volume_is_observed_free=False,
                     bounded_occupied_updates=list(self.bounded_updates),
                     requires_separate_world_collision_check=True)
        if contact_bounds is not None:
            contact_bounds = np.asarray(contact_bounds, dtype=float)
            if (contact_bounds.shape != (2, 3) or not np.isfinite(contact_bounds).all()
                    or np.any(contact_bounds[1] <= contact_bounds[0])):
                raise ValueError('invalid target contact volume')
        if scene_token != self.scene_token or self.invalid_reason is not None:
            return proof | {'reason': self.invalid_reason or 'stale_capture_scene_token'}
        proof['continuous_segment_envelope_checked'] = True
        covered = required & (spheres[..., 3] <= 0)  # SDK disabled slots are not geometry.
        evaluated = covered.copy()
        captured = np.asarray(self.capture_robot_spheres, dtype=float).reshape(-1, 4)
        if not np.array_equal(captured, self._cache_capture_spheres):
            self._certificate_cache.clear()
            self._cache_capture_spheres = captured.copy()
        robot_indices, contact_indices = set(robot_sphere_indices), set(contact_sphere_indices)
        contact_key = None if contact_bounds is None else tuple(contact_bounds.flat)
        cache = self._certificate_cache
        hits, misses = 0, 0
        for sample, index in zip(*np.where(required & ~covered), strict=True):
            sphere = spheres[sample, index]
            radius = float(sphere[3] + bounds[sample, index])
            robot_allowed = index in robot_indices
            contact = contact_bounds if index in contact_indices else None
            key = (*sphere[:3], radius, robot_allowed,
                   contact_key if contact is not None else None, self.uncertainty_m)
            if key in cache:
                hits += 1
                accepted = cache.pop(key)
            else:
                misses += 1
                # The exact admitted start volume can prove stationary geometry
                # before any camera projection; this is the same predicate used
                # by covered_by_union, without enlarging the admitted volume.
                accepted = bool(robot_allowed and len(captured) and np.any(
                    np.linalg.norm(captured[:, :3] - sphere[:3], axis=1) + radius <= captured[:, 3]))
                accepted = accepted or self.sphere_observed(sphere[:3], radius)
                accepted = accepted or self.covered_by_union(sphere[:3], radius, robot_allowed, contact)
            cache[key] = accepted
            if len(cache) > 8192:
                cache.popitem(last=False)
            covered[sample, index] = accepted
            evaluated[sample, index] = True
            if stop_on_first_unknown and not accepted:
                break
        proof.update(certificate_cache_hits=hits, certificate_cache_misses=misses)
        unknown = np.argwhere(evaluated & ~covered)
        complete = bool(evaluated[required].all())
        proof.update(covered=complete and not len(unknown), checked_sphere_count=int(evaluated.sum()),
                     requested_sphere_count=int(required.sum()), coverage_evaluation_complete=complete,
                     unobserved_sphere_count=len(unknown), unobserved_sphere_count_is_lower_bound=not complete)
        if len(unknown):
            sample, index = map(int, unknown[0])
            proof.update(first_unobserved_sample_index=sample, first_unobserved_sphere_index=index,
                         first_unobserved_sphere_world_m=spheres[sample, index].tolist(),
                         first_unobserved_motion_envelope_m=float(bounds[sample, index]),
                         camera_certificates=self.camera_diagnostics(spheres[sample, index, :3],
                             float(spheres[sample, index, 3] + bounds[sample, index])),
                         diagnosis_scope="whole_sphere_certificate_failure_not_proof_of_physical_collision",
                         union_subdivision_resolution_m=.005)
            if len(captured):
                gaps = np.linalg.norm(captured[:, :3] - spheres[sample, index, :3], axis=1) + (
                    spheres[sample, index, 3] + bounds[sample, index]) - captured[:, 3]
                closest = int(np.argmin(gaps))
                proof.update(closest_capture_robot_sphere_world_m=captured[closest].tolist(),
                             minimum_capture_containment_gap_m=float(gaps[closest]))
        else:
            proof['failure_reason'] = None
        return proof
