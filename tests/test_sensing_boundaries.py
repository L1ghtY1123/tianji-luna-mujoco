import pytest
from test_perception import capture_fixture

from luna_mujoco.demo_perception import DemoPerception
from luna_mujoco.perception import Perception, PerceptionError


@pytest.mark.parametrize('perception_type', [Perception, DemoPerception])
def test_sensing_requires_controller_inventory_without_model_planning(perception_type):
    class Detector:
        def segment(self, *args):
            pytest.fail('segmentation must not start without controller inventory')
    engine = perception_type(Detector(), visual_inventory=object())
    with pytest.raises(PerceptionError, match='controller_inventory_required'):
        engine.observe(capture_fixture()[0], source_hint={'visual_decision': {
            'source': engine.visual_plan_source, 'purpose': 'initialize_inventory'}})
    assert not hasattr(engine, 'visual_discovery')
