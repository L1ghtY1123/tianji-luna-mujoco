import json

import pytest
from pydantic import BaseModel, ConfigDict, Field

from luna_mujoco.skills import Registry, SkillSelectionError


class TurnArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    angle_deg: float = Field(ge=-90, le=90)


def catalogue(tmp_path):
    skill = tmp_path / 'turn_fixture'
    skill.mkdir()
    (skill / 'contract.json').write_text(json.dumps({
        'id': 'turn_fixture', 'description': 'Turn the configured fixture.',
        'inputs': ['current_feedback'], 'requires': {'fixture_ready': True},
        'provides': ['fixture_turned'],
    }), encoding='utf-8')
    (skill / 'SKILL.md').write_text('A bounded fixture rotation.', encoding='utf-8')
    return Registry(tmp_path)


async def test_new_skill_is_described_validated_and_dispatched_without_central_list(tmp_path):
    registry = catalogue(tmp_path)
    calls = []

    async def turn(context, arguments):
        calls.append((context, arguments.angle_deg))
        return {'executed': True}

    registry.register('turn_fixture', TurnArguments, turn)
    schema = registry.describe()['turn_fixture']['arguments']
    assert registry.describe()['turn_fixture']['outputs'] == ['fixture_turned']
    assert 'results' not in registry.describe()['turn_fixture']
    assert schema['properties']['angle_deg']['maximum'] == 90
    assert schema['additionalProperties'] is False
    result = await registry.execute('turn_fixture', {'fixture_ready': True}, {'station': 1}, {'angle_deg': 45})
    assert result == {'executed': True}
    assert calls == [({'station': 1}, 45)]


@pytest.mark.parametrize('arguments', [{'angle_deg': 91}, {'angle_deg': '45'}, {'other': 1}, {}])
async def test_invalid_arguments_never_reach_handler(tmp_path, arguments):
    registry = catalogue(tmp_path)

    async def forbidden(*args):
        pytest.fail('invalid arguments reached action')

    registry.register('turn_fixture', TurnArguments, forbidden)
    with pytest.raises(SkillSelectionError, match='invalid_skill_arguments'):
        await registry.execute('turn_fixture', {'fixture_ready': True}, {}, arguments)


async def test_unregistered_skill_and_failed_preconditions_never_dispatch(tmp_path):
    registry = catalogue(tmp_path)
    with pytest.raises(SkillSelectionError, match='skill_handler_not_registered'):
        await registry.execute('turn_fixture', {'fixture_ready': True}, {}, {'angle_deg': 0})

    async def forbidden(*args):
        pytest.fail('unmet precondition reached action')

    registry.register('turn_fixture', TurnArguments, forbidden)
    with pytest.raises(SkillSelectionError, match='skill_contract_not_satisfied'):
        await registry.execute('turn_fixture', {}, {}, {'angle_deg': 0})
    with pytest.raises(ValueError, match='duplicate_skill_handler'):
        registry.register('turn_fixture', TurnArguments, forbidden)
    with pytest.raises(SkillSelectionError, match='unknown_skill'):
        registry.register('unknown', TurnArguments, forbidden)
