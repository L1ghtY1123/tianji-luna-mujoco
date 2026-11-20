"""User-declared CAD bindings and measured validation of pose hypotheses."""
import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tianji_foundationpose.contracts import validate_pose

from .cad_paths import cad_resource_root
from .pose_estimation import CadPoseEstimator

RESOURCE_ROOT = Path(__file__).resolve().parents[2] / 'resources'


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scoped_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('cad_asset_path_invalid')
    return path


class CadCatalog:
    """Static asset geometry only; simulator poses and physics are never read."""

    def __init__(self, records, resource_root=RESOURCE_ROOT):
        self.resource_root = Path(resource_root).resolve()
        self.records = {}
        mesh_keys = set()
        for record in records:
            if (record.get('geometry') != 'rigid_reference' or not record.get('asset_id')
                    or record['asset_id'] in self.records or record.get('mesh_key') in mesh_keys):
                raise ValueError('cad_catalog_invalid')
            self.records[record['asset_id']] = deepcopy(record)
            mesh_keys.add(record['mesh_key'])

    @classmethod
    def from_file(cls, path=None):
        path = Path(path) if path else cad_resource_root() / 'cad_catalog.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') != 1:
            raise ValueError('cad_catalog_version')
        return cls(data['assets'], resource_root=path.parent)

    def record(self, asset_id):
        if asset_id not in self.records:
            aliases = [key for key, row in self.records.items() if row.get('source_asset_id') == asset_id]
            if len(aliases) != 1:
                raise ValueError('cad_asset_unknown')
            asset_id = aliases[0]
        return deepcopy(self.records[asset_id])

    def geometry(self, mesh_key):
        records = [record for record in self.records.values() if record['mesh_key'] == mesh_key]
        if len(records) != 1:
            raise ValueError('cad_mesh_key_unknown')
        record = records[0]
        root = self.resource_root / 'cad'
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        asset = manifest['assets'][mesh_key]
        source = asset['source']
        if (manifest['units'] != 'm' or source['units'] != 'mm'
                or source['sha256'] != record['source_sha256']
                or _hash(_scoped_path(self.resource_root.parent, source['library_path'])
                         if source.get('library_path') else _scoped_path(root, source['path'])) != record['source_sha256']):
            raise ValueError('cad_source_hash_or_units_mismatch')
        path = _scoped_path(root, asset['visual']['path'])
        if _hash(path) != asset['visual']['sha256']:
            raise ValueError('cad_mesh_hash_mismatch')
        mesh = trimesh.load(path, force='mesh', process=False)
        center = np.asarray(source['aabb_center_mm'], dtype=float) * .001
        source_bounds = np.asarray(source['bounds_mm'], dtype=float) * .001
        if (not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all()
                or not len(mesh.faces) or not np.allclose(mesh.bounds, source_bounds - center, atol=1e-6)):
            raise ValueError('cad_mesh_frame_mismatch')
        transform = np.eye(4)
        transform[:3, 3] = center
        return {'asset_id': record['asset_id'], 'vertices_m': np.asarray(mesh.vertices).copy(),
                'faces': np.asarray(mesh.faces).copy(), 'T_source_canonical_m': transform,
                'mesh_sha256': asset['visual']['sha256'], 'source_sha256': record['source_sha256'],
                'path': str(path)}

    def preview(self, record):
        # A configured but not yet rendered preview is explicitly absent.
        if not record.get('preview_sha256'):
            return None
        path = _scoped_path(self.resource_root, record['preview_path'])
        if _hash(path) != record['preview_sha256']:
            raise ValueError('cad_preview_hash_mismatch')
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB')).copy()










def _identity_silhouette(entry):
    """Keep the original silhouette for numerical FoundationPose input."""
    geometry = np.asarray(entry['mask'], dtype=bool)
    silhouette = np.asarray(entry.get('silhouette_mask', geometry), dtype=bool)
    if silhouette.shape != geometry.shape or np.any(geometry & ~silhouette):
        raise ValueError('cad_mask_provenance_invalid')
    return silhouette


def _mask_evidence(entry):
    silhouette, geometry = _identity_silhouette(entry), np.asarray(entry['mask'], dtype=bool)
    return {'camera_id': entry['camera_id'], 'source': 'identity_bound_silhouette',
            'silhouette_pixels': int(silhouette.sum()), 'geometry_pixels': int(geometry.sum()),
            'geometry_filtered_pixels': int((silhouette & ~geometry).sum()),
            'silhouette_sha256': hashlib.sha256(np.ascontiguousarray(silhouette).tobytes()).hexdigest(),
            'geometry_mask_sha256': hashlib.sha256(np.ascontiguousarray(geometry).tobytes()).hexdigest()}


def cad_tolerance_policy(geometry, configured_cap_m):
    """Tighten the existing cap for small assets; never relax configured tolerance."""
    shortest = float(np.min(np.ptp(geometry['vertices_m'], axis=0)))
    if (not math.isfinite(configured_cap_m) or not 0 < configured_cap_m <= .05
            or not math.isfinite(shortest) or shortest <= 0):
        raise ValueError('cad_validation_configuration_invalid')
    return {'role': 'upper_bound_tightening', 'configured_cap_m': configured_cap_m,
            'shortest_extent_m': shortest, 'shortest_extent_fraction': .05,
            'tolerance_m': min(configured_cap_m, .05 * shortest)}


def point_cloud_fingerprint(points):
    """Bind ordered XYZ measurements using explicit little-endian float64 bytes."""
    array = np.ascontiguousarray(points, dtype='<f8')
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError('cad_point_cloud_invalid')
    return {'point_count': len(array), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


class CadGeometryResolver:
    def __init__(self, catalog, pose_estimator=None, *, tolerance_m=.015):
        self.catalog, self.pose_estimator = catalog, pose_estimator
        self.tolerance_m = tolerance_m
        self.confirmed_models = {}
        self.subject_policy = json.loads((RESOURCE_ROOT / 'cad_subject_policy.json').read_text(encoding='utf-8'))

    @classmethod
    def from_environment(cls):
        return cls(CadCatalog.from_file(),
                   tolerance_m=float(os.getenv('LUNA_CAD_RESIDUAL_TOLERANCE_M', '.015')))

    def identity_constraints(self, generation):
        """Validated shape bindings only; no cached world pose assigns an identity."""
        result = {}
        for (saved_generation, identifier), saved in self.confirmed_models.items():
            if saved_generation != generation or saved['asset_id'] not in self.catalog.records:
                continue
            record = self.catalog.record(saved['asset_id'])
            if record['source_sha256'] != saved['source_sha256']:
                continue
            result[identifier] = {**deepcopy(saved['identity_constraint']),
                'instance_id': identifier, 'generation': generation}
        return result

    def candidate_pose(self, capture, candidate):
        """Register a model hypothesis before any instance ID exists."""
        hints = candidate.get('retrieval_hypotheses') or [candidate]
        records = [row for asset_id, row in self.catalog.records.items()
            if any(hint.get('color') in self.subject_policy['allowed_subject_colors'].get(asset_id, [])
                and hint.get('kind') in self.subject_policy['allowed_subject_kinds'].get(asset_id, []) for hint in hints)]
        declared = {h['asset_id'] for h in hints if h.get('asset_id')
                    and h.get('scene_quote') == candidate.get('scene_quote')}
        if declared:
            records = [self.catalog.record(code) for code in sorted(declared)]
        provenance = {'candidate_id': candidate['candidate_id'], 'observation_id': capture['observation_id'],
            'generation': capture['generation'], 'camera_id': candidate['camera_id'],
            'calibration_id': candidate['calibration_id'], 'mask_sha256': candidate['mask_sha256'],
            'usage': 'model_and_pose_hypothesis_not_instance_identity', 'official_pose_score': None}
        if len(records) != 1 or candidate['depth_evidence']['status'] != 'measured':
            return {**provenance, 'status': 'unavailable', 'reason': 'eligible_model_and_depth_required'}
        record = records[0]
        try:
            geometry = self.catalog.geometry(record['mesh_key'])
            estimated, pose, _, validation = self._estimate_view(capture,
                {'points_world_m': candidate['points']}, candidate, record, geometry)
            world = geometry['vertices_m'] @ pose[:3, :3].T + pose[:3, 3]
            candidate['_pose_world_points'] = world
            from .association_geometry import head_wrist_pair, reproject_surface
            return {**provenance, 'status': 'estimated', 'asset_id': record['asset_id'],
                'mesh_sha256': geometry['mesh_sha256'], 'source_sha256': record['source_sha256'],
                'estimated': estimated, 'validation': validation,
                'bounds_world_m': [world.min(0).tolist(), world.max(0).tolist()],
                'projection_by_camera': {camera: reproject_surface(world, view,
                    candidate['silhouette_mask'] if camera == candidate['camera_id'] else np.zeros_like(view['valid_depth']))
                    for camera, view in capture['views'].items()
                    if camera == candidate['camera_id'] or head_wrist_pair(camera, candidate['camera_id'])}}
        except (ValueError, RuntimeError, OSError) as error:
            return {**provenance, 'status': 'failed', 'asset_id': record['asset_id'], 'reason': str(error)}

    def resolve(self, capture, group, target, *, instruction):
        if not group or any(entry.get('entity_id') != target['id'] for entry in group):
            raise ValueError('cad_identity_binding_required')
        for entry in group:
            view = capture['views'][entry['camera_id']]
            if (view['frame_id'] != capture['observation_id'] or view['camera_id'] != entry['camera_id']
                    or view['robot_generation'] != capture['generation']):
                raise ValueError('cad_capture_mismatch')
        generation = capture['generation']
        color = target.get('color', 'unknown').strip().lower()
        allowed_colors = self.subject_policy['allowed_subject_colors']
        allowed_kinds = self.subject_policy['allowed_subject_kinds']
        records = [record for asset_id, record in self.catalog.records.items()
                   if color in allowed_colors.get(asset_id, [])
                   and target.get('kind') in allowed_kinds.get(asset_id, [])]
        declared = (target.get('segmentation_query') or {}).get('asset_id')
        if declared:
            records = [self.catalog.record(declared)]
        cache_key = (generation, target['id'])
        if not records:
            self.confirmed_models.pop(cache_key, None)
            result = deepcopy(target)
            result.pop('cad', None)
            result['cad_match'] = {'status': 'not_requested', 'geometry_policy': 'user_declared_tsdf',
                'evidence': {**deepcopy(self.subject_policy), 'instance_id': target['id'],
                            'confirmed_identity_color': color},
                'observation_id': capture['observation_id'], 'generation': generation}
            return result
        policy = target.get('geometry_policy', {})
        # Read the former stored policy name while emitting only TSDF provenance.
        if policy.get('mode') in {'unknown_tsdf', 'unknown_obb'}:
            quote = policy.get('instruction_quote')
            if (policy.get('source') != 'user_instruction_interpreted_by_responses'
                    or policy.get('instance_id') != target['id'] or not isinstance(quote, str)
                    or not quote.strip() or quote not in instruction):
                raise ValueError('geometry_policy_binding_invalid')
            if (generation, target['id']) in self.confirmed_models:
                raise ValueError('geometry_policy_conflicts_with_validated_cad')
            result = deepcopy(target)
            result['cad_match'] = {'status': 'not_requested', 'geometry_policy': 'explicit_unknown_tsdf',
                'evidence': deepcopy(policy), 'observation_id': capture['observation_id'],
                'generation': generation, 'identity_source': target.get('identity_source', 'preconfirmed_group')}
            return result
        self.confirmed_models = {key: value for key, value in self.confirmed_models.items()
                                 if key[0] == generation}
        if len(records) != 1:
            raise ValueError('user_declared_cad_binding_ambiguous')
        record = records[0]
        saved = self.confirmed_models.get(cache_key)
        if saved and (saved['asset_id'] != record['asset_id']
                      or saved['source_sha256'] != record['source_sha256']):
            self.confirmed_models.pop(cache_key)
            saved = None
        match = {'evidence': {**deepcopy(self.subject_policy), 'asset_id': record['asset_id'],
            'instance_id': target['id'], 'confirmed_identity_color': color,
            'observation_id': capture['observation_id'], 'generation': generation,
            'role': 'user_declared_model_binding', 'current_foundationpose_required': True}}
        result = deepcopy(target)
        geometry = self.catalog.geometry(record['mesh_key'])
        tolerance_policy = cad_tolerance_policy(geometry, self.tolerance_m)
        tolerance_m = tolerance_policy['tolerance_m']
        candidates = {}
        for source in group:
            camera = source['camera_id']
            if camera not in candidates:
                candidates[camera] = {**source, 'silhouette_mask': _identity_silhouette(source).copy()}
            else:
                candidates[camera]['silhouette_mask'] |= _identity_silhouette(source)
        for entry in candidates.values():
            view = capture['views'][entry['camera_id']]
            entry['valid_depth_count'] = int(np.count_nonzero(_identity_silhouette(entry)
                & view['valid_depth'] & np.isfinite(view['depth_m']) & (view['depth_m'] > 0)))
        entry = candidates.get('3')
        if entry is None or entry['valid_depth_count'] <= 32:
            raise ValueError('cad_pose_head_depth_insufficient')
        view_estimation = {'policy': 'head_camera_foundationpose',
            'eligible_camera_ids': [camera for camera, row in candidates.items() if row['valid_depth_count'] > 32],
            'selected_camera_id': '3', 'attempts': []}
        seed = saved.get('pose_seed') if saved else None
        motion_history = capture.get('cad_motion_context')
        motion_context = deepcopy(motion_history.get(target['id'])) if isinstance(motion_history, dict) else None
        tracking_seed = None
        if (seed and motion_context is not None
                and seed.get('motion_context') == motion_context
                and seed['mesh_sha256'] == geometry['mesh_sha256']
                and seed['observation_id'] != capture['observation_id']
                and seed['capture_time'] < capture['time']):
            tracking_seed = {**seed, 'instance_id': target['id'], 'source_sha256': record['source_sha256']}
        estimated, pose, source_pose, validation = self._estimate_view(
            capture, target, entry, record, geometry, tracking_seed=tracking_seed)
        view_estimation['attempts'].append({'camera_id': entry['camera_id'],
            'valid_depth_count': entry['valid_depth_count'], 'status': validation['status'],
            'pose_evidence': deepcopy(estimated), 'pose_validation': deepcopy(validation)})
        mask_evidence = [_mask_evidence(item) for item in group]
        view = capture['views'][view_estimation['selected_camera_id']]
        match_evidence = deepcopy(match['evidence'])
        match_evidence.update(
            historical_match_reused=saved is not None,
            match_source_observation_id=match['evidence'].get('observation_id'),
            current_identity_binding={'entity_id': target['id'], 'observation_id': capture['observation_id'],
                'generation': generation, 'source': target.get('identity_source', 'preconfirmed_group'),
                'evidence': deepcopy(target.get('identity_evidence', []))})
        self.confirmed_models[cache_key] = {'asset_id': record['asset_id'],
            'source_sha256': record['source_sha256'], 'evidence': deepcopy(match['evidence']),
            'pose_seed': {'T_world_object': pose.tolist(), 'mesh_sha256': geometry['mesh_sha256'],
                'observation_id': capture['observation_id'], 'capture_time': capture['time'],
                'motion_context': motion_context},
            'identity_constraint': {'source': 'validated_cad_model_binding',
                'asset_id': record['asset_id'], 'source_sha256': record['source_sha256'],
                'mesh_sha256': geometry['mesh_sha256'],
                'maximum_diameter_m': float(np.linalg.norm(np.ptp(geometry['vertices_m'], axis=0))),
                'surface_tolerance_m': tolerance_m,
                'validated_observation_id': capture['observation_id']}}
        world_vertices = geometry['vertices_m'] @ pose[:3, :3].T + pose[:3, 3]
        result['cad'] = {'asset_id': record['asset_id'], 'model_id': record['model_id'],
            'display_name': record['display_name'], 'source_asset_id': record.get('source_asset_id', record['asset_id']),
            'instance_id': target['id'],
            'mesh_key': record['mesh_key'], 'source_sha256': record['source_sha256'],
            'mesh_sha256': geometry['mesh_sha256'], 'T_world_source': source_pose.tolist(),
            'T_world_object': pose.tolist(), 'observation_id': capture['observation_id'],
            'generation': capture['generation'], 'calibration_id': view['calibration_id'],
            'match_evidence': match_evidence, 'pose_validation': validation, 'pose_evidence': estimated,
            'mask_evidence': mask_evidence, 'view_estimation': view_estimation,
            'shape_bounds': {'minimum_world_m': world_vertices.min(axis=0).tolist(),
                'maximum_world_m': world_vertices.max(axis=0).tolist(),
                'mesh_sha256': geometry['mesh_sha256'], 'T_world_object': pose.tolist()}}
        result.pop('obb', None)
        result.pop('cad_match', None)
        from .cad_placement_region import complete_placement_reference

        result = complete_placement_reference(result, geometry, tolerance_m,
                                              support_spec=record.get('placement_support'))
        return result

    def _estimate_view(self, capture, target, entry, record, geometry, *, tracking_seed=None):
        view = capture['views'][entry['camera_id']]
        estimator = self.pose_estimator or CadPoseEstimator()
        options = {'tracking_seed': tracking_seed} if tracking_seed is not None else {}
        hypothesis = entry.get('pose_hypothesis') or {}
        reuse = (hypothesis.get('status') == 'estimated'
            and hypothesis.get('observation_id') == capture['observation_id']
            and hypothesis.get('generation') == capture['generation']
            and hypothesis.get('camera_id') == entry['camera_id']
            and hypothesis.get('calibration_id') == view['calibration_id']
            and hypothesis.get('mask_sha256') == hashlib.sha256(_identity_silhouette(entry).tobytes()).hexdigest()
            and hypothesis.get('mesh_sha256') == geometry['mesh_sha256']
            and hypothesis.get('source_sha256') == record['source_sha256'])
        try:
            estimated = deepcopy(hypothesis['estimated']) if reuse else estimator.estimate_instance(view, _identity_silhouette(entry), entry['score'],
                model_id=record['model_id'], mask_capture_id=view['frame_id'], mask_camera_id=view['camera_id'], **options)
        except (ValueError, RuntimeError, OSError) as error:
            raise ValueError('cad_pose_estimation_failed:' + str(error)) from error
        if (estimated['source_sha256'] != record['source_sha256'] or estimated['model_id'] != record['model_id']
                or estimated['capture_id'] != view['frame_id'] or estimated['camera_id'] != view['camera_id']
                or estimated['robot_generation'] != capture['generation']
                or estimated['calibration_id'] != view['calibration_id']):
            raise ValueError('cad_pose_provenance_mismatch')
        source_from_canonical = validate_pose(estimated['T_source_canonical_m'])
        if not np.allclose(source_from_canonical, geometry['T_source_canonical_m'], atol=1e-6):
            raise ValueError('cad_pose_source_frame_mismatch')
        pose = validate_pose(estimated['T_world_object'])
        source_pose = pose @ np.linalg.inv(source_from_canonical)
        if not np.allclose(source_pose, validate_pose(estimated['T_world_source']), atol=1e-6):
            raise ValueError('cad_pose_world_source_mismatch')
        validation = {'status': 'accepted_by_policy', 'accepted': True,
            'method': 'foundationpose_direct', 'post_registration_geometry_check': False,
            'assumption': 'user_authorized_foundationpose_pose_is_accurate',
            'surface': {'full_point_count': len(target['points_world_m']),
                'full_point_sha256': point_cloud_fingerprint(target['points_world_m'])['sha256']},
            'limitations': ['pose_not_rechecked_against_mask_or_depth', 'hidden_surfaces_are_model_prior']}
        return estimated, pose, source_pose, validation
