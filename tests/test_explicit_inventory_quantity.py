"""Task selection admits consistent ID counts without rewriting model intent."""
import pytest
from test_agent_model_layer import task_data

from luna_mujoco.agent.schema import TaskSpec


@pytest.mark.parametrize('refs,quantity', [(['pink_package'],1),(['first','second'],2),(['first'],None)])
def test_explicit_ids_allow_their_consistent_count(refs,quantity):
    source=task_data() | {'scope':'ids','target_refs':refs,'quantity':quantity}
    result=TaskSpec.model_validate(source)
    assert result.target_refs==tuple(refs) and result.quantity==quantity and result.scope=='ids'


@pytest.mark.parametrize('scope,refs,quantity', [
    ('ids',['first'],2),('ids',['first','second'],1),('all',['first'],1),('count',['first'],None)])
def test_conflicting_scope_count_cannot_be_silently_normalized(scope,refs,quantity):
    with pytest.raises(ValueError,match='task_count_scope_conflict:'):
        TaskSpec.model_validate(task_data() | {'scope':scope,'target_refs':refs,'quantity':quantity})
