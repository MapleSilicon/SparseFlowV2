"""Transactional physical channel pruning for the supported ResNet-18 reference model."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from sparseflow.graph_diff import model_state_hash, structural_hash
from sparseflow.models import ResNet18Reference
from sparseflow.planning import PruningPlan, validate_resnet18_pruning_plan


class TransactionalPruningError(RuntimeError):
    """Raised when a validated pruning transaction cannot be completed safely."""


def _module_by_path(root: nn.Module, path: str) -> nn.Module:
    current: nn.Module = root
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)  # type: ignore[index]
    return current


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parts = path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)  # type: ignore[index]
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = replacement  # type: ignore[index]
    else:
        setattr(parent, leaf, replacement)


def _new_conv(
    conv: nn.Conv2d,
    input_indices: tuple[int, ...],
    output_indices: tuple[int, ...],
) -> nn.Conv2d:
    if conv.groups != 1:
        raise TransactionalPruningError("grouped/depthwise Conv2d is unsupported")
    replacement = nn.Conv2d(
        len(input_indices),
        len(output_indices),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    input_index = torch.tensor(input_indices, dtype=torch.long, device=conv.weight.device)
    output_index = torch.tensor(output_indices, dtype=torch.long, device=conv.weight.device)
    with torch.no_grad():
        replacement.weight.copy_(
            conv.weight.detach().index_select(0, output_index).index_select(1, input_index)
        )
        replacement.weight.requires_grad_(conv.weight.requires_grad)
        if conv.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(conv.bias.detach().index_select(0, output_index))
            replacement.bias.requires_grad_(conv.bias.requires_grad)
    replacement.train(conv.training)
    return replacement


def _new_batch_norm(batch_norm: nn.BatchNorm2d, indices: tuple[int, ...]) -> nn.BatchNorm2d:
    tensor = batch_norm.weight if batch_norm.affine else batch_norm.running_mean
    device = tensor.device if tensor is not None else None
    dtype = tensor.dtype if tensor is not None and tensor.is_floating_point() else None
    replacement = nn.BatchNorm2d(
        len(indices),
        eps=batch_norm.eps,
        momentum=batch_norm.momentum,
        affine=batch_norm.affine,
        track_running_stats=batch_norm.track_running_stats,
        device=device,
        dtype=dtype,
    )
    index = torch.tensor(indices, dtype=torch.long, device=device)
    with torch.no_grad():
        if batch_norm.affine:
            replacement.weight.copy_(batch_norm.weight.detach().index_select(0, index))
            replacement.bias.copy_(batch_norm.bias.detach().index_select(0, index))
            replacement.weight.requires_grad_(batch_norm.weight.requires_grad)
            replacement.bias.requires_grad_(batch_norm.bias.requires_grad)
        if batch_norm.track_running_stats:
            replacement.running_mean.copy_(batch_norm.running_mean.detach().index_select(0, index))
            replacement.running_var.copy_(batch_norm.running_var.detach().index_select(0, index))
            replacement.num_batches_tracked.copy_(batch_norm.num_batches_tracked.detach())
    replacement.train(batch_norm.training)
    return replacement


def _new_linear(linear: nn.Linear, input_indices: tuple[int, ...]) -> nn.Linear:
    replacement = nn.Linear(
        len(input_indices),
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    index = torch.tensor(input_indices, dtype=torch.long, device=linear.weight.device)
    with torch.no_grad():
        replacement.weight.copy_(linear.weight.detach().index_select(1, index))
        replacement.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(linear.bias.detach())
            replacement.bias.requires_grad_(linear.bias.requires_grad)
    replacement.train(linear.training)
    return replacement


def _actual_dimensions(module: nn.Module) -> dict[str, int]:
    if isinstance(module, nn.Conv2d):
        return {"in_channels": module.in_channels, "out_channels": module.out_channels}
    if isinstance(module, nn.BatchNorm2d):
        return {"num_features": module.num_features}
    if isinstance(module, nn.Linear):
        return {"in_features": module.in_features, "out_features": module.out_features}
    raise TransactionalPruningError(f"unsupported module type {type(module).__name__}")


def validate_executed_plan(
    original: nn.Module, candidate: nn.Module, plan: PruningPlan
) -> dict[str, Any]:
    """Independently compare every executed module dimension against the plan."""

    changes: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for decision in plan.layer_decisions:
        before = _module_by_path(original, decision.module_path)
        after = _module_by_path(candidate, decision.module_path)
        expected_input = len(decision.kept_input_channel_indices)
        expected_output = len(decision.kept_output_channel_indices)
        if isinstance(after, nn.Conv2d):
            matches = after.in_channels == expected_input and after.out_channels == expected_output
        elif isinstance(after, nn.BatchNorm2d):
            matches = after.num_features == expected_output
        elif isinstance(after, nn.Linear):
            matches = after.in_features == expected_input and after.out_features == expected_output
        else:
            matches = False
        if not matches:
            mismatches.append(decision.module_path)
        changes.append(
            {
                "module_path": decision.module_path,
                "module_type": decision.module_type,
                "before": _actual_dimensions(before),
                "after": _actual_dimensions(after),
                "kept_input_channel_indices": list(decision.kept_input_channel_indices),
                "kept_output_channel_indices": list(decision.kept_output_channel_indices),
                "removed_input_channel_indices": list(decision.removed_input_channel_indices),
                "removed_output_channel_indices": list(decision.removed_output_channel_indices),
                "dependency_group": decision.dependency_group,
                "matches_plan": matches,
            }
        )
    if mismatches:
        raise TransactionalPruningError(
            "executed pruning differs from validated plan for: " + ", ".join(mismatches)
        )
    return {
        "validated": True,
        "executed_as_planned": True,
        "planned_change_count": len(plan.layer_decisions),
        "executed_change_count": len(changes),
        "changes": changes,
    }


def apply_resnet18_pruning_plan(
    model: nn.Module, plan: PruningPlan, input_shape: tuple[int, ...] = (1, 3, 64, 64)
) -> tuple[ResNet18Reference, dict[str, Any]]:
    """Validate first, mutate only a deep copy, and fail without partial artifacts."""

    if not isinstance(model, ResNet18Reference):
        raise TransactionalPruningError("Gate 2 mutation requires ResNet18Reference")
    original_model_hash = model_state_hash(model)
    original_structure_hash = structural_hash(model)
    plan_validation = validate_resnet18_pruning_plan(plan, model)
    if model_state_hash(model) != original_model_hash or structural_hash(model) != original_structure_hash:
        raise TransactionalPruningError("plan validation mutated the original model")

    candidate = copy.deepcopy(model)
    try:
        for decision in plan.layer_decisions:
            module = _module_by_path(candidate, decision.module_path)
            if isinstance(module, nn.Conv2d):
                replacement = _new_conv(
                    module,
                    decision.kept_input_channel_indices,
                    decision.kept_output_channel_indices,
                )
            elif isinstance(module, nn.BatchNorm2d):
                replacement = _new_batch_norm(module, decision.kept_output_channel_indices)
            elif isinstance(module, nn.Linear):
                replacement = _new_linear(module, decision.kept_input_channel_indices)
            else:
                raise TransactionalPruningError(
                    f"unsupported planned module {decision.module_path}"
                )
            _replace_module(candidate, decision.module_path, replacement)

        candidate.eval()
        dtype = next(candidate.parameters()).dtype
        device = next(candidate.parameters()).device
        with torch.no_grad():
            output = candidate(torch.zeros(input_shape, dtype=dtype, device=device))
        if output.shape != (input_shape[0], 1000):
            raise TransactionalPruningError(
                f"candidate output shape {tuple(output.shape)} is not {(input_shape[0], 1000)}"
            )
        if not torch.isfinite(output).all():
            raise TransactionalPruningError("candidate output contains non-finite values")
        execution_validation = validate_executed_plan(model, candidate, plan)
    except Exception as exc:
        if model_state_hash(model) != original_model_hash or structural_hash(model) != original_structure_hash:
            raise TransactionalPruningError("failed transaction mutated original model") from exc
        if isinstance(exc, TransactionalPruningError):
            raise
        raise TransactionalPruningError("ResNet-18 pruning transaction failed") from exc

    if model_state_hash(model) != original_model_hash or structural_hash(model) != original_structure_hash:
        raise TransactionalPruningError("successful transaction mutated original model")
    return candidate, {
        "plan_validation": plan_validation,
        "execution_validation": execution_validation,
        "transaction": {
            "validated_before_mutation": True,
            "mutated_copy_only": True,
            "original_model_preserved": True,
            "committed": True,
        },
    }
