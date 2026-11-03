"""Observed-instance CAD matching and measured validation of pose hypotheses."""
import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps

from tianji_foundationpose.contracts import validate_pose

from .cad_paths import cad_resource_root
from .pose_estimation import CadPoseEstimator
from .visual_api import VisualInventoryError
from .visual_inventory import VisualInventory, choice, image_part

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
    """Depth/table filtering narrows geometry, not the identity-bound RGB subject."""
    geometry = np.asarray(entry['mask'], dtype=bool)
    silhouette = np.asarray(entry.get('silhouette_mask', geometry), dtype=bool)
    if (silhouette.ndim != 2 or silhouette.shape != geometry.shape
            or np.any(geometry & ~silhouette)):
        raise ValueError('cad_mask_provenance_invalid')
    return silhouette


def _mask_evidence(entry):
    silhouette = _identity_silhouette(entry)
    geometry = np.asarray(entry['mask'], dtype=bool)
    return {'camera_id': entry['camera_id'],
            'source': 'identity_bound_silhouette' if 'silhouette_mask' in entry else 'provided_geometry_mask',
            'silhouette_pixels': int(np.count_nonzero(silhouette)),
            'geometry_pixels': int(np.count_nonzero(geometry)),
            'geometry_filtered_pixels': int(np.count_nonzero(silhouette & ~geometry)),
            'silhouette_sha256': hashlib.sha256(np.ascontiguousarray(silhouette).tobytes()).hexdigest(),
            'geometry_mask_sha256': hashlib.sha256(np.ascontiguousarray(geometry).tobytes()).hexdigest()}


def _display_exposure(rgb, mask):
    """Display-only uniform RGB gain; original RGBD and segmentation stay intact."""
    level = float(np.quantile(np.max(rgb[mask], axis=1), .95))
    gain = min(16., max(1., 128. / max(1., level)))
    exposed = np.clip(np.rint(rgb.astype(float) * gain), 0, 255).astype(np.uint8)
    return np.where(mask[..., None], exposed, np.full_like(rgb, 96)), gain


def _comparison_sheet(rgb, mask, references):
    """Keep subject/reference roles inside the image seen by the visual model."""
    if mask.shape != rgb.shape[:2] or not mask.any():
        raise ValueError('cad_match_mask_invalid')
    rows, columns = np.nonzero(mask)
    masked = np.where(mask[..., None], rgb, np.full_like(rgb, 96))
    crop = masked[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    sheet = Image.new('RGB', (768, 260 + 254 * len(references)), 'white')
    draw = ImageDraw.Draw(sheet)

    def paste(pixels, box):
        x0, y0, x1, y1 = box
        tile = ImageOps.contain(Image.fromarray(pixels), (x1 - x0, y1 - y0))
        sheet.paste(tile, (x0 + (x1 - x0 - tile.width) // 2, y0 + (y1 - y0 - tile.height) // 2))

    paste(rgb, (0, 24, 384, 260))
    exposed, gain = _display_exposure(rgb, mask)
    if gain >= 2:
        exposed_crop = exposed[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
        paste(crop, (384, 44, 576, 260))
        paste(exposed_crop, (576, 44, 768, 260))
        draw.text((394, 25), 'S: RAW RGB', fill='black', font_size=14)
        draw.text((582, 25), f'S: EXPOSURE x{gain:.2f}', fill='black', font_size=14)
    else:
        paste(crop, (384, 44, 768, 260))
        draw.text((394, 25), 'S: RAW RGB', fill='black', font_size=14)
    draw.text((8, 6), 'CURRENT SCENE - context only', fill='black', font_size=16)
    draw.text((394, 6), 'S: CURRENT MASKED SUBJECT', fill='black', font_size=16)
    for index, (label, preview) in enumerate(references):
        top = 260 + 254 * index
        draw.text((8, top + 8), label + ': CAD CATALOG REFERENCE', fill='black', font_size=16)
        if preview is not None:
            paste(preview, (0, top + 26, 768, top + 254))
        else:
            draw.text((8, top + 40), 'Preview unavailable; description supplied separately',
                      fill='black', font_size=16)
    return np.asarray(sheet)


class CadCandidateProposal(TypedDict):
    status: Literal['candidate', 'no_catalog_candidate', 'unknown']
    asset_id: str | None
    evidence: dict


class CadModelMatcher(VisualInventory):
    """Retrieve a model proposal for the identity-bound subject before FoundationPose."""

    minimum_confidence = None

    def match(self, capture, group, target, catalog, *, instruction):
        choices, references, descriptions = {}, [], []
        for index, record in enumerate(catalog.records.values(), 1):
            label = 'R' + str(index)
            choices[label] = {key: record.get(key) for key in ('asset_id', 'source_sha256', 'preview_sha256')}
            preview = catalog.preview(record) if record.get('preview_sha256') else None
            references.append((label, preview))
            geometry = catalog.geometry(record['mesh_key'])
            descriptions.append({'reference': label, 'display_name': record['display_name'],
                'asset_id': record['asset_id'], 'segmentation_aliases': record.get('segmentation_aliases', []),
                'description': record['description'],
                'canonical_extent_m': np.ptp(geometry['vertices_m'], axis=0).tolist()})
        points = np.asarray(target['points_world_m'], dtype=float)
        measured = {**point_cloud_fingerprint(points),
                    'world_axis_extent_m': np.ptp(points, axis=0).tolist(),
                    'limitations': 'Partial visible surface only; world-axis extents depend on pose. '
                                   'Occluded dimensions are lower bounds, not complete object dimensions.'}
        content = [{'type': 'input_text', 'text': json.dumps({
            'purpose': 'match_current_mask_to_catalog_model_before_foundationpose',
            'observation_id': capture['observation_id'],
            'generation': capture['generation'], 'entity_id': target['id'],
            'observed_appearance': {key: target[key] for key in (
                'kind', 'color', 'description', 'appearance_keywords', 'description_provenance') if key in target},
            'measured_geometry': measured,
            'sam_query_evidence': [{'camera_id': entry['camera_id'],
                'queries': list(dict.fromkeys([entry.get('prompt', ''), *entry.get('prompt_proposals', [])]))}
                for entry in group],
            'catalog_references': descriptions,
            'rules': 'Propose at most one plausible CAD candidate for S, the CURRENT MASKED SUBJECT '
                     'at the upper right of each sheet. Select its CAD model, not its pose or task destination. '
                      'Use the current visible rigid shape, subject description and measured dimensions. '
                      'SAM query words may be synonyms, fallback terms or broad classes; they need not '
                      'equal the catalog display name. Catalog aliases and SAM query evidence help interpret '
                      'the words, but the CURRENT masked subject determines the asset candidate. '
                     'Descriptions are entity proposals linked through confirmed identity, not CAD proof. '
                     'The upper left is CURRENT RGB scene context. R1, R2 and any further R labels below '
                     'are static CAD CATALOG reference views, not subjects. Gray pixels around S are excluded '
                     'or occluded regions. RAW RGB preserves the measured colours. An optional EXPOSURE '
                     'copy shows the SAME pixels multiplied by its labeled uniform RGB gain and clipped '
                     'to [0,255]; it adds no detail, is not another observation, and is not evidence of '
                     'the original material colour. Compare visible shape, allowing differences in viewpoint and '
                     'clipping at the image boundary. All sheets show the same identity-confirmed subject '
                     'from different cameras. Colour or an appearance label alone does not establish a match. '
                     'Do not infer hidden dimensions from RGB. Select the matching reference from visible shape '
                     'and size; FoundationPose estimates its pose separately. Report uncertainty honestly; '
                     'the chosen asset is a model proposal, not a numerical fit certificate. Choose no_catalog_candidate when the observed '
                     'subject is visibly inconsistent with all catalog shapes. Choose unknown when evidence '
                     'is too ambiguous to propose one candidate or to exclude the catalog. Do not force '
                     'a reference from colour, task goal, or the absence of other choices.'}, ensure_ascii=False)}]
        for entry in group:
            camera = entry['camera_id']
            view = capture['views'][camera]
            rgb = np.asarray(view['rgb'])
            mask = _identity_silhouette(entry)
            content.extend([{'type': 'input_text', 'text': 'CURRENT SUBJECT S, camera ' + camera},
                            image_part(_comparison_sheet(rgb, mask, references))])
        question = choice('cad_model', 'Which single CAD reference is a plausible retrieval candidate for S '
                          '(the CURRENT MASKED SUBJECT in the upper right)? '
                          'This selects the model for FoundationPose; it does not estimate pose. '
                          'Choose no_catalog_candidate if all catalog shapes visibly differ, '
                          'or unknown if retrieval is ambiguous.', list(choices) + ['no_catalog_candidate', 'unknown'])
        payload = {'model': 'gpt-6-luna', 'input': [{'role': 'user', 'content': content}], 'questions': [question]}

        def parse(body, evidence):
            answers = body.get('answers')
            if not isinstance(answers, list) or len(answers) != 1:
                raise VisualInventoryError('cad_match_invalid_response')
            answer = answers[0]
            probability = answer.get('confidence')
            value = answer.get('choice')
            if (answer.get('name') != 'cad_model' or answer.get('type') != 'choice'
                    or type(probability) not in (float, int) or not math.isfinite(probability)
                    or not 0 <= probability <= 1 or value not in [item['value'] for item in question['choices']]):
                raise VisualInventoryError('cad_match_invalid_response')
            status = 'candidate' if value in choices else value
            asset_id = choices[value]['asset_id'] if status == 'candidate' else None
            return CadCandidateProposal(status=status, asset_id=asset_id, evidence={
                    **evidence, 'answers': answers, 'role': 'cad_candidate_proposal', 'cad_confirmed': False,
                    'known_facts': ['only_the_explicit_choice_is_proposed',
                                    'current_foundationpose_required_before_cad_publication'],
                    'diagnostic_checks': [{'name': 'cad_model', 'status': status,
                        'choice': value, 'reported_confidence': probability,
                        'confidence_role': 'retrieval_uncertainty_not_physical_fact_acceptance'}],
                    'catalog_choices': choices,
                    'camera_ids': list(dict.fromkeys(entry['camera_id'] for entry in group)),
                    'presentation_version': 'subject_reference_sheet_v2'})

        return self.request(payload, capture, 'match_cad_model', parse)


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
    def __init__(self, catalog, matcher, pose_estimator=None, *, tolerance_m=.015):
        self.catalog, self.matcher, self.pose_estimator = catalog, matcher, pose_estimator
        self.tolerance_m = tolerance_m
        self.confirmed_models = {}
        self.unregistered_models = {}

    @classmethod
    def from_environment(cls, *, audit=None):
        return cls(CadCatalog.from_file(), CadModelMatcher.from_environment(audit=audit),
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

    def resolve(self, capture, group, target, *, instruction):
        if not group or any(entry.get('entity_id') != target['id'] for entry in group):
            raise ValueError('cad_identity_binding_required')
        for entry in group:
            view = capture['views'][entry['camera_id']]
            if (view['frame_id'] != capture['observation_id'] or view['camera_id'] != entry['camera_id']
                    or view['simulation_generation'] != capture['generation']):
                raise ValueError('cad_capture_mismatch')
        generation = capture['generation']
        policy = target.get('geometry_policy', {})
        if policy.get('mode') == 'unknown_obb':
            quote = policy.get('instruction_quote')
            if (policy.get('source') != 'user_instruction_interpreted_by_responses'
                    or policy.get('instance_id') != target['id'] or not isinstance(quote, str)
                    or not quote.strip() or quote not in instruction):
                raise ValueError('geometry_policy_binding_invalid')
            if (generation, target['id']) in self.confirmed_models:
                raise ValueError('geometry_policy_conflicts_with_validated_cad')
            result = deepcopy(target)
            result['cad_match'] = {'status': 'not_requested', 'geometry_policy': 'explicit_unknown_obb',
                'evidence': deepcopy(policy), 'observation_id': capture['observation_id'],
                'generation': generation, 'identity_source': target.get('identity_source', 'preconfirmed_group')}
            return result
        self.confirmed_models = {key: value for key, value in self.confirmed_models.items()
                                 if key[0] == generation}
        cache_key = (generation, target['id'])
        catalog_fingerprint = hashlib.sha256(json.dumps(self.catalog.records, sort_keys=True).encode()).hexdigest()
        self.unregistered_models = {key: value for key, value in self.unregistered_models.items()
            if key[0] == generation and value['catalog_fingerprint'] == catalog_fingerprint}
        saved = self.confirmed_models.get(cache_key)
        unregistered = self.unregistered_models.get(cache_key)
        if saved and (saved['asset_id'] not in self.catalog.records or
                      self.catalog.record(saved['asset_id'])['source_sha256'] != saved['source_sha256']):
            self.confirmed_models.pop(cache_key)
            saved = None
        if saved:
            match = {'status': 'candidate', 'asset_id': saved['asset_id'], 'evidence': deepcopy(saved['evidence'])}
        elif unregistered:
            match = {'status': 'no_catalog_candidate', 'asset_id': None,
                     'evidence': deepcopy(unregistered['evidence'])}
        else:
            match = self.matcher.match(capture, group, target, self.catalog, instruction=instruction)
        result = deepcopy(target)
        if match['status'] == 'unknown':
            error = ValueError('cad_retrieval_uncertain')
            error.details = {'status': 'unknown', 'instance_id': target['id'],
                'observation_id': capture['observation_id'], 'generation': generation,
                'geometry_policy': 'unresolved_catalog_retrieval', 'match_evidence': match['evidence']}
            raise error
        if match['status'] == 'no_catalog_candidate':
            evidence = match['evidence']
            answers = evidence.get('answers', [])
            answer = answers[0] if len(answers) == 1 else {}
            probability = next((row.get('probability') for row in answer.get('probabilities', [])
                                if row.get('value') == 'no_catalog_candidate'), None)
            confidence = answer.get('confidence')
            # Bind a confident no-catalog judgment to the immutable physical ID,
            # just as a positive model binding is retained across viewpoints.
            # Low-confidence retrieval proposals and unknowns remain uncached.
            if (not unregistered and evidence.get('source') == 'openai_decisions'
                    and evidence.get('request_hash') and evidence.get('observation_id') == capture['observation_id']
                    and answer.get('type') == 'choice' and answer.get('choice') == 'no_catalog_candidate'
                    and type(confidence) in (int, float) and .8 <= confidence <= 1
                    and type(probability) in (int, float) and .8 <= probability <= 1):
                self.unregistered_models[cache_key] = {'catalog_fingerprint': catalog_fingerprint,
                                                        'evidence': deepcopy(evidence)}
            result['cad_match'] = {'status': 'no_catalog_candidate', 'evidence': deepcopy(evidence),
                'geometry_policy': 'measured_obb_approximation', 'catalog_fingerprint': catalog_fingerprint,
                'historical_match_reused': unregistered is not None,
                'match_source_observation_id': evidence.get('observation_id'),
                'current_identity_binding': {'entity_id': target['id'], 'observation_id': capture['observation_id'],
                    'generation': generation, 'source': target.get('identity_source', 'preconfirmed_group'),
                    'evidence': deepcopy(target.get('identity_evidence', []))}}
            return result
        if match['status'] != 'candidate' or match['asset_id'] not in self.catalog.records:
            raise ValueError('cad_candidate_proposal_invalid')
        record = self.catalog.record(match['asset_id'])
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
        ranked = sorted(candidates.values(), key=lambda item: (-item['valid_depth_count'], item['camera_id']))
        eligible = [entry for entry in ranked if entry['valid_depth_count'] > 32]
        if not eligible:
            raise ValueError('cad_pose_depth_insufficient')
        view_estimation = {'policy': 'best_depth_view_foundationpose_direct',
            'eligible_camera_ids': [entry['camera_id'] for entry in eligible],
            'selected_camera_id': None, 'attempts': []}
        seed = saved.get('pose_seed') if saved else None
        progress = capture.get('cad_motion_context')
        motion_context = deepcopy(progress.get(target['id'], {})) if isinstance(progress, dict) else None
        tracking_seed = None
        if (seed and motion_context is not None
                and seed.get('motion_context') == motion_context
                and seed['mesh_sha256'] == geometry['mesh_sha256']
                and seed['observation_id'] != capture['observation_id']
                and seed['capture_time'] < capture['time']):
            tracking_seed = {**seed, 'instance_id': target['id'], 'source_sha256': record['source_sha256']}
        entry = eligible[0]
        estimated, pose, source_pose, validation = self._estimate_view(
            capture, target, entry, record, geometry, tracking_seed=tracking_seed)
        view_estimation['attempts'].append({'camera_id': entry['camera_id'],
            'valid_depth_count': entry['valid_depth_count'], 'status': validation['status'],
            'pose_evidence': deepcopy(estimated), 'pose_validation': deepcopy(validation)})
        view_estimation['selected_camera_id'] = entry['camera_id']
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
        try:
            estimated = estimator.estimate_instance(view, _identity_silhouette(entry), entry['score'],
                model_id=record['model_id'], mask_capture_id=view['frame_id'], mask_camera_id=view['camera_id'], **options)
        except (ValueError, RuntimeError, OSError) as error:
            raise ValueError('cad_pose_estimation_failed:' + str(error)) from error
        if (estimated['source_sha256'] != record['source_sha256'] or estimated['model_id'] != record['model_id']
                or estimated['capture_id'] != view['frame_id'] or estimated['camera_id'] != view['camera_id']
                or estimated['simulation_generation'] != capture['generation']
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
