from types import SimpleNamespace

from jev_mujoco.perception_mapping import CaptureSegmenter, PerceptionMapper


def test_new_gripper_feedback_replaces_robot_filter_without_discarding_mapper_allocations():
    old, new = SimpleNamespace(robot_spheres=lambda joints: ['open']), SimpleNamespace(robot_spheres=lambda joints: ['closed'])
    value = PerceptionMapper.__new__(PerceptionMapper)
    value.segmenter_boundary = old
    value.capture_segmenter = CaptureSegmenter(old)
    value.capture_segmenter.begin_capture({'old': True})
    value.mapper = SimpleNamespace(reset_capture=lambda: None)
    # Updating the owned filter wrapper must not require a CUDA installation.
    value.builder = SimpleNamespace(table_cache=object())
    mapper = value.mapper
    builder = value.builder
    table_cache = builder.table_cache
    value.update_segmenter(new, scene_type=object)
    assert value.mapper is mapper
    assert value.segmenter_boundary is new
    assert value.capture_segmenter.robot_spheres({}) == ['closed']
    assert value.capture_segmenter.capture is None
    assert value.builder is builder and value.builder.table_cache is table_cache
    assert value.builder.robot_spheres({}) == ['closed']
