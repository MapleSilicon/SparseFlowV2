"""SparseFlow v0.2 reproducible optimization and evidence workflow."""

from sparseflow.config import BenchmarkConfig
from sparseflow.gate2_pass import ResNet18Gate2PruningPass
from sparseflow.measurement import MeasurementRecipe
from sparseflow.passes import ChannelPruningPass, NoOpPass, OptimizationPass
from sparseflow.planning import (
    ChannelSelection,
    LayerPruningDecision,
    MemberChannelSelection,
    PruningPlan,
    PruningPlanValidationError,
    ResidualCouplingGroupPlan,
    build_resnet18_pruning_plan,
    validate_resnet18_pruning_plan,
)
from sparseflow.resnet_pruning import TransactionalPruningError

__all__ = [
    "BenchmarkConfig",
    "ChannelPruningPass",
    "ChannelSelection",
    "LayerPruningDecision",
    "MeasurementRecipe",
    "MemberChannelSelection",
    "NoOpPass",
    "OptimizationPass",
    "PruningPlan",
    "PruningPlanValidationError",
    "ResidualCouplingGroupPlan",
    "ResNet18Gate2PruningPass",
    "TransactionalPruningError",
    "build_resnet18_pruning_plan",
    "validate_resnet18_pruning_plan",
]
__version__ = "0.2.0"
