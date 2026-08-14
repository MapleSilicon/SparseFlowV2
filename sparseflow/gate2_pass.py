"""SparseFlow V1 Gate 2 optimization pass for the supported ResNet-18 reference model."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from sparseflow.dependencies import analyze_resnet18_dependencies
from sparseflow.passes import GraphChange, OptimizationPass, OptimizationResult
from sparseflow.planning import build_resnet18_pruning_plan
from sparseflow.resnet_pruning import apply_resnet18_pruning_plan
from sparseflow.serialization import sha256_json


class ResNet18Gate2PruningPass(OptimizationPass):
    """Physically prune ResNet-18 only after a complete plan validates."""

    def __init__(self, ratio: float = 0.125, seed: int = 1234) -> None:
        if not isinstance(ratio, (float, int)) or not 0.0 < float(ratio) <= 0.5:
            raise ValueError("Gate 2 pruning ratio must be in (0.0, 0.5]")
        self._ratio = float(ratio)
        self._seed = int(seed)

    @property
    def name(self) -> str:
        return "channel_prune"

    def config(self) -> dict[str, Any]:
        return {
            "ratio": self._ratio,
            "dependency_aware": True,
            "target_op_types": ["Conv2d", "BatchNorm2d", "Linear"],
            "ranking": "l1_weight_magnitude",
            "tie_breaker": "ascending_channel_index",
            "supported_pattern": "resnet18_stage_coupled_residual_groups",
            "transactional": True,
            "physical_removal": True,
        }

    @staticmethod
    def _attributes(module: nn.Module) -> dict[str, Any]:
        if isinstance(module, nn.Conv2d):
            return {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "weight_shape": list(module.weight.shape),
            }
        if isinstance(module, nn.BatchNorm2d):
            return {
                "num_features": module.num_features,
                "weight_shape": list(module.weight.shape) if module.affine else None,
            }
        if isinstance(module, nn.Linear):
            return {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_shape": list(module.weight.shape),
            }
        raise ValueError(f"unsupported Gate 2 module {type(module).__name__}")

    @staticmethod
    def _module_by_path(root: nn.Module, path: str) -> nn.Module:
        current: nn.Module = root
        for part in path.split("."):
            current = current[int(part)] if part.isdigit() else getattr(current, part)  # type: ignore[index]
        return current

    def apply(self, model: nn.Module) -> OptimizationResult:
        if getattr(model, "identifier", None) != "resnet18-reference":
            raise ValueError("ResNet18Gate2PruningPass requires resnet18-reference")
        dependency_graph = analyze_resnet18_dependencies(model)
        plan = build_resnet18_pruning_plan(model, self._ratio, seed=self._seed)
        optimized, transaction = apply_resnet18_pruning_plan(model, plan)

        changes: list[GraphChange] = []
        executed = transaction["execution_validation"]["changes"]
        for decision, executed_change in zip(plan.layer_decisions, executed, strict=True):
            before_module = self._module_by_path(model, decision.module_path)
            after_module = self._module_by_path(optimized, decision.module_path)
            before = self._attributes(before_module)
            after = self._attributes(after_module)
            if before == after:
                continue
            changes.append(
                GraphChange(
                    node=decision.module_path,
                    op_type=decision.module_type,
                    change_type=(
                        "residual_group_output_prune"
                        if decision.dependency_group
                        else "dependency_repair"
                    ),
                    before=before,
                    after=after,
                    metadata={
                        "dependency_group": decision.dependency_group,
                        "kept_input_channel_indices": list(decision.kept_input_channel_indices),
                        "kept_output_channel_indices": list(decision.kept_output_channel_indices),
                        "removed_input_channel_indices": list(decision.removed_input_channel_indices),
                        "removed_output_channel_indices": list(decision.removed_output_channel_indices),
                        "ranking": "l1_weight_magnitude",
                        "reason": decision.reason,
                        "executed_matches_plan": executed_change["matches_plan"],
                    },
                )
            )

        execution_payload = transaction["execution_validation"]
        execution_hash = sha256_json(execution_payload)
        return OptimizationResult(
            model=optimized.eval(),
            metadata={
                "gate": "resnet18_gate2_transactional_physical_pruning",
                "scope": "supported ResNet-18 reference architecture only",
                "dependency_summary": dependency_graph.summary(),
                "dependency_analysis": dependency_graph.as_dict(),
                "dependency_graph_hash": plan.dependency_graph_hash,
                "pruning_plan": plan.as_dict(),
                "pruning_plan_hash": plan.plan_hash,
                "coupling_group_validation": transaction["plan_validation"],
                "plan_vs_execution_validation": {
                    **execution_payload,
                    "execution_hash": execution_hash,
                },
                "transaction": transaction["transaction"],
                "physical_pruning_performed": True,
            },
            graph_changes=changes,
        )
