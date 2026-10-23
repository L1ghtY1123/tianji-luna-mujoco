import json
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from luna_mujoco.carrying import carrying_geometry
from luna_mujoco.planning_settings import PlanningSettings, load_settings


def test_attachment_hard_limit_covers_dense_cloud_without_voxel_search(monkeypatch):
    from test_visible_surface_gpu_contract import measured_binding
    points = np.random.default_rng(7).uniform([-.06, -.025, -.02], [.06, .025, .02], (20000, 3))
    binding = replace(measured_binding(), points_tool_m=tuple(map(tuple, points)), sphere_budget=31)
    def no_repeated_voxel_sort(*args, **kwargs):
        pytest.fail('attachment must use bounded spatial splits, not repeated voxel sorts')
    monkeypatch.setattr(np, 'unique', no_repeated_voxel_sort)
    geometry = carrying_geometry(binding, maximum_spheres=31)
    spheres = np.asarray(geometry['spheres'])
    assert 1 <= len(spheres) <= 31
    distances = np.linalg.norm(points[:, None] - spheres[None, :, :3], axis=2) - spheres[None, :, 3]
    assert np.all(distances.min(axis=1) <= -binding.margin_m - binding.uncertainty_m + 1e-12)


def test_attachment_budget_schema_and_legacy_saved_settings(tmp_path):
    assert PlanningSettings().attachment.sphere_budget == 24
    with pytest.raises(ValidationError):
        PlanningSettings(attachment={'sphere_budget': 32})
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'attachment': {'sphere_budget': 256}, 'place_release_height_m': .01}))
    with pytest.warns(UserWarning, match='migrated to 24'):
        settings = load_settings(path)
    assert settings.attachment.sphere_budget == 24
    assert settings.place_release_height_m == .01


class Assistance:
    enabled = True

    def __init__(self, status):
        self.status = status
        self.calls = []

    def inspect(self, question, capture, facts, **kwargs):
        self.calls.append((question, facts))
        return {'success': self.status == 'confirmed', 'status': self.status,
                'source': 'openai_visual', 'observation_id': capture['observation_id']}


@pytest.mark.parametrize('status', ['confirmed', 'contradicted', 'unknown'])
async def test_visual_result_cannot_override_unknown_live_gripper_feedback(tmp_path, status):
    from test_current_grasp_attachment import fixture
    runner, context, current, prepared, checkpoints, _ = fixture(tmp_path)
    context['holding']['sphere_budget'] = 16
    current['arms']['A']['gripper_position_samples'] = []
    runner.visual_assistance = Assistance(status)
    result = await runner.verify_grasp(context)
    assert result['status'] == 'unknown'
    assert not runner.visual_assistance.calls and not prepared
    assert runner.state['facts']['grasp_evidence_pending'] is False
    from luna_mujoco.skills import Registry
    with pytest.raises(ValueError, match='unknown_skill'):
        Registry().validate_selection('inspect_visual_result', runner.state['facts'])
    context['visual_result'] = {'status': status}
    result = await runner.verify_grasp(context)
    assert runner.state['facts']['grasp_evidence_pending'] is False
    assert result['status'] == 'unknown'
    assert not runner.visual_assistance.calls
    assert not checkpoints
    assert prepared == []


@pytest.mark.parametrize('closed_fraction,expected', [(.6, 'held'), (1., 'empty')])
async def test_decisive_gripper_skips_all_visual_calls(tmp_path, closed_fraction, expected):
    from test_current_grasp_attachment import fixture
    runner, context, _, _, checkpoints, _ = fixture(tmp_path, closed_fraction=closed_fraction)
    context['holding']['sphere_budget'] = 16
    runner.visual_assistance = Assistance('confirmed')
    result = await runner.verify_grasp(context)
    assert result['status'] == expected
    assert not runner.visual_assistance.calls and not checkpoints


@pytest.mark.parametrize('ordinary', ['unknown', 'confirmed', 'contradicted'])
async def test_release_standard_does_not_use_old_visual_verdicts(tmp_path, ordinary):
    from test_release_verification import release_case

    runner, context = release_case(tmp_path)
    runner.visual_assistance = Assistance('unknown')
    context['place_visual_evidence'] = {'status': ordinary}
    result = await runner.verify_place(context)
    assert result['status'] == 'confirmed'
    assert result['verification_scope'] == 'release_only'
    assert not runner.visual_assistance.calls


def test_visual_service_loads_shared_config_without_exporting_secrets(tmp_path, monkeypatch):
    from luna_mujoco.visual_assistance import VisualAssistance
    for name in ['LUNA_VISUAL_ASSISTANCE_ENABLED', 'LUNA_VISION_MODEL', 'OPENAI_API_KEY', 'OPENAI_BASE_URL']:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / 'provider.json'
    path.write_text(json.dumps({'api_key': 'private-test-key', 'base_url': 'https://example.invalid/v1', 'model': 'existing-model'}))
    monkeypatch.setenv('OPENAI_CONFIG_FILE', str(path))
    service = VisualAssistance.from_environment()
    assert service.enabled and service.model == 'existing-model'
    assert service.base_url == 'https://example.invalid/v1'
    monkeypatch.setenv('LUNA_VISUAL_ASSISTANCE_ENABLED', 'false')
    assert not VisualAssistance.from_environment().enabled


def test_environment_credentials_never_mix_with_another_provider_config(tmp_path, monkeypatch):
    from luna_mujoco.visual_assistance import VisualAssistance
    for name in ['LUNA_VISUAL_ASSISTANCE_ENABLED', 'OPENAI_API_KEY', 'OPENAI_BASE_URL']:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / 'provider.json'
    path.write_text(json.dumps({'api_key': 'gateway-key', 'base_url': 'https://gateway.invalid/v1', 'model': 'existing-model'}))
    monkeypatch.setenv('OPENAI_CONFIG_FILE', str(path))
    monkeypatch.setenv('OPENAI_API_KEY', 'official-key')
    service = VisualAssistance.from_environment()
    assert service.api_key == 'official-key' and service.base_url == 'https://api.openai.com/v1'
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://different.invalid/v1')
    assert not VisualAssistance.from_environment().enabled
