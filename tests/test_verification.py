"""One tri-state interface replaces the old CAD-pose success test."""
import pytest

from jev_mujoco.verification import combine_holding_evidence


@pytest.mark.parametrize('position, visual, expected', [
    ('held','unknown','held'), ('empty','unknown','empty'),
    ('unknown','held','held'), ('unknown','empty','empty'),
    ('held','empty','unknown'), ('empty','held','unknown'),
    ('unknown','unknown','unknown')])
def test_fuse_independent_evidence_without_interpreting_absence_as_failure(position, visual, expected):
    result = combine_holding_evidence({'status': position}, {'status': visual})
    assert result['status'] == expected
    assert result['success'] == (expected == 'held')
