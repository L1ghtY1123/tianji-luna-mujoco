"""Acceptance layouts remain explicit inputs, separate from controller evidence."""

import pytest
from continuous_transfer_smoke import fixture


def test_bin_source_layout_can_isolate_container_width_at_a_reference_position():
    objects, instruction = fixture(
        "bin_to_marker", bin_source_xy_m=(.455, .009),
        bin_source_size_m=(.28, .28, .10), marker_xy_m=(.508, .324))
    source, container, marker = objects
    assert source["position"] == pytest.approx([.455, .009, .635])
    assert container["position"] == pytest.approx([.455, .009, .65])
    assert container["size"] == [.28, .28, .10]
    assert marker["position"] == pytest.approx([.508, .324, .6001])
    assert "左臂" in instruction


def test_original_narrow_bin_failure_layout_remains_the_default():
    objects, _ = fixture("bin_to_marker")
    assert objects[0]["position"] == [.43, .15, .635]
    assert objects[1]["size"] == [.18, .18, .10]


def test_bin_source_height_changes_with_container_height_but_keeps_floor_contact():
    objects, _ = fixture("bin_to_marker", bin_source_size_m=(.28, .28, .08))
    assert objects[0]["position"][2] == pytest.approx(.635)
    assert objects[1]["position"][2] == pytest.approx(.64)
