"""Replay recorded two-stage HTTP fixtures without pretending they are new protocol evidence."""
from luna_mujoco.demo_perception import DemoVisualInventory


class ArchivedTwoStageInventory(DemoVisualInventory):
    def __init__(self, **kwargs):
        super().__init__(legacy_identity_protocol=True, **kwargs)
