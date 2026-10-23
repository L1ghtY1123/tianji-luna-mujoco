"""Project clearance range over unchanged Tianji request types.

The copied wrapper's 10 mm lower limit is an application policy. The public
cuRobo configurations and hard world checker accept a distance in metres.
Keep all copied structural validation, but validate clearance with this
project's 10 mm minimum. No solver or vendor module is patched.
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
from tianji_curobo import contracts, dual_contracts

from .planning_settings import PlanningSettings


def _profile(clearance, activation=.03):
    clearance = contracts._finite_float(clearance, name="trajectory_collision_clearance_m")
    activation = contracts._finite_float(activation, name="optimizer_collision_activation_distance_m")
    settings = PlanningSettings(trajectory_collision_clearance_m=clearance,
                                optimizer_collision_activation_distance_m=activation)
    return settings.trajectory_collision_clearance_m, settings.optimizer_collision_activation_distance_m


class _ProjectClearance:
    def __post_init__(self):
        has_optimizer = hasattr(self, "optimizer_collision_activation_distance_m")
        clearance, activation = _profile(self.trajectory_collision_clearance_m,
            getattr(self, "optimizer_collision_activation_distance_m", .03))
        # Let the copied constructor validate every other field using its
        # supported policy value. Restore our separately validated distance
        # before the request can reach a backend or dataclass serialization.
        object.__setattr__(self, "trajectory_collision_clearance_m", max(.01, clearance))
        super().__post_init__()
        object.__setattr__(self, "trajectory_collision_clearance_m", clearance)
        if has_optimizer:
            object.__setattr__(self, "optimizer_collision_activation_distance_m", activation)


class PlanPlaceRequest(_ProjectClearance, contracts.PlanPlaceRequest):
    pass


class PlanGraspBatchRequest(_ProjectClearance, contracts.PlanGraspBatchRequest):
    pass


class SolveIkBatchRequest(_ProjectClearance, contracts.SolveIkBatchRequest):
    pass


class _ProjectDualClearance:
    @classmethod
    def from_mapping(cls, values):
        clearance, activation = _profile(values.get("trajectory_collision_clearance_m", .01),
            values.get("optimizer_collision_activation_distance_m", .03))
        request = super().from_mapping({**values, "trajectory_collision_clearance_m": max(.01, clearance),
                                       "optimizer_collision_activation_distance_m": activation})
        return replace(request, trajectory_collision_clearance_m=clearance)


class DualPlanRequest(_ProjectDualClearance, dual_contracts.DualPlanRequest):
    pass


class DualValidationRequest(_ProjectDualClearance, dual_contracts.DualValidationRequest):
    pass
