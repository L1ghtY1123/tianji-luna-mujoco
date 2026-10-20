import pytest

from jev_mujoco import task_request
from jev_mujoco.decisions import Decisions


def test_literal_groups_preserve_source_text_and_do_not_assign_environment_roles():
    assert hasattr(task_request, 'quantity_group_candidates')
    groups = task_request.quantity_group_candidates('搬两个红色和一个绿色，旁边三个黑色障碍不要动')
    assert [group['source_text'] for group in groups] == ['搬两个红色', '一个绿色', '旁边三个黑色障碍不要动']
    result = task_request.normalize_request_scope({'quantity': '3', 'quantity_group_0': '2',
        'quantity_group_1': '1', 'quantity_group_2': 'not_target'},
        instruction='搬两个红色和一个绿色，旁边三个黑色障碍不要动')
    assert [(group['source_text'], group['requested_count']) for group in result['quantity_groups']] == [
        ('搬两个红色', 2), ('一个绿色', 1)]


@pytest.mark.asyncio
async def test_assessment_returns_group_membership_with_identity_confidence_policy(monkeypatch):
    calls = []

    async def ask(self, kind, state, questions, **kwargs):
        calls.append((questions, kwargs))
        return {'progress': 'continue', 'member_0': 'required', 'member_1': 'required',
                'group_0_member_0': 'required', 'group_0_member_1': 'excluded',
                'group_1_member_0': 'excluded', 'group_1_member_1': 'required'}

    monkeypatch.setattr(Decisions, 'ask', ask)
    state = {'instruction': '一个红色和一个绿色', 'task_progress': {
        'inventory_finalized': False, 'membership_pending_ids': ['red', 'green'],
        'unresolved_membership': [{'id': 'red'}, {'id': 'green'}],
        'quantity_groups': [{'group_id': 'group_0', 'source_text': '一个红色', 'requested_count': 1},
                            {'group_id': 'group_1', 'source_text': '一个绿色', 'requested_count': 1}]}}
    result = await Decisions(None, lambda *args: None).assess_task(state)
    assert result['group_membership'] == {
        'group_0': {'red': 'required', 'green': 'excluded'},
        'group_1': {'red': 'excluded', 'green': 'required'}}
    assert calls[0][1]['require_confidence'] is True


@pytest.mark.asyncio
async def test_low_confidence_membership_and_completion_remain_unknown(monkeypatch):
    from test_decisions import install_agent
    install_agent(monkeypatch, {'progress': 'continue', 'member_0': 'required'},
                  {'progress': .4, 'member_0': .2})
    result = await Decisions(None, lambda *args: None).assess_task({'instruction': '搬红色', 'task_progress': {
        'membership_pending_ids': ['red'], 'unresolved_membership': [{'id': 'red'}]}})
    assert result == {'judgment': 'unknown', 'membership': {'red': 'unknown'}}
