"""Optimization pass contracts and physical channel pruning."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn


@dataclass(frozen=True)
class GraphChange:
    node: str
    op_type: str
    change_type: str
    before: dict[str, Any]
    after: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class OptimizationResult:
    model: nn.Module
    metadata: dict[str, Any]
    graph_changes: list[GraphChange]


class OptimizationPass(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable machine-readable pass name."""

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Return canonical JSON-serializable configuration."""

    @abstractmethod
    def apply(self, model: nn.Module) -> OptimizationResult:
        """Return an optimized model and structured transformation evidence."""


class NoOpPass(OptimizationPass):
    @property
    def name(self) -> str:
        return "noop"

    def config(self) -> dict[str, Any]:
        return {"type": "no_op"}

    def apply(self, model: nn.Module) -> OptimizationResult:
        return OptimizationResult(
            model=copy.deepcopy(model).eval(),
            metadata={"transformation": "none"},
            graph_changes=[],
        )


def _conv_attributes(conv: nn.Conv2d) -> dict[str, Any]:
    return {
        "in_channels": conv.in_channels,
        "out_channels": conv.out_channels,
        "kernel_size": list(conv.kernel_size),
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "dilation": list(conv.dilation),
        "groups": conv.groups,
        "bias": conv.bias is not None,
        "weight_shape": list(conv.weight.shape),
        "bias_shape": list(conv.bias.shape) if conv.bias is not None else None,
    }


def _batch_norm_attributes(batch_norm: nn.BatchNorm2d) -> dict[str, Any]:
    return {
        "num_features": batch_norm.num_features,
        "affine": batch_norm.affine,
        "track_running_stats": batch_norm.track_running_stats,
        "weight_shape": list(batch_norm.weight.shape) if batch_norm.affine else None,
        "running_mean_shape": (
            list(batch_norm.running_mean.shape) if batch_norm.running_mean is not None else None
        ),
        "running_var_shape": (
            list(batch_norm.running_var.shape) if batch_norm.running_var is not None else None
        ),
    }


def _rank_channels(conv: nn.Conv2d, keep_count: int) -> list[int]:
    scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    ranked = torch.argsort(scores, descending=True, stable=True)[:keep_count]
    return torch.sort(ranked).values.tolist()


def _new_conv(conv: nn.Conv2d, input_indices: list[int], output_indices: list[int]) -> nn.Conv2d:
    replacement = nn.Conv2d(
        in_channels=len(input_indices),
        out_channels=len(output_indices),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(conv.weight.detach()[output_indices][:, input_indices])
        replacement.weight.requires_grad_(conv.weight.requires_grad)
        if conv.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(conv.bias.detach()[output_indices])
            replacement.bias.requires_grad_(conv.bias.requires_grad)
    return replacement.eval()


def _new_batch_norm(batch_norm: nn.BatchNorm2d, indices: list[int]) -> nn.BatchNorm2d:
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
    with torch.no_grad():
        if batch_norm.affine:
            replacement.weight.copy_(batch_norm.weight.detach()[indices])
            replacement.bias.copy_(batch_norm.bias.detach()[indices])
            replacement.weight.requires_grad_(batch_norm.weight.requires_grad)
            replacement.bias.requires_grad_(batch_norm.bias.requires_grad)
        if batch_norm.track_running_stats:
            replacement.running_mean.copy_(batch_norm.running_mean.detach()[indices])
            replacement.running_var.copy_(batch_norm.running_var.detach()[indices])
            replacement.num_batches_tracked.copy_(batch_norm.num_batches_tracked.detach())
    return replacement.eval()


class ChannelPruningPass(OptimizationPass):
    """Prune the sequential Conv-BN-Conv dependency pattern in ReferenceCNN.

    Channels are ranked by L1 weight magnitude. The last convolution is kept at
    its original output width so the classifier remains compatible. Residual,
    concatenation, grouped, depthwise, and arbitrary branched graphs are not
    supported and are rejected rather than being modified speculatively.
    """

    def __init__(self, ratio: float = 0.375) -> None:
        if not isinstance(ratio, (float, int)) or not 0.0 <= float(ratio) < 1.0:
            raise ValueError("ratio must be a number in the range [0.0, 1.0)")
        self._ratio = float(ratio)

    @property
    def name(self) -> str:
        return "channel_prune"

    def config(self) -> dict[str, Any]:
        return {
            "ratio": self._ratio,
            "dependency_aware": True,
            "target_op_types": ["Conv2d"],
            "ranking": "l1_weight_magnitude",
            "supported_pattern": "sequential_conv_batchnorm_conv",
        }

    def _validate_model(self, model: nn.Module) -> tuple[nn.Sequential, int]:
        try:
            graph_nodes = len(list(fx.symbolic_trace(model).graph.nodes))
        except Exception as exc:
            raise ValueError("model must be torch.fx traceable") from exc
        if not isinstance(getattr(model, "features", None), nn.Sequential):
            raise ValueError("channel pruning requires a sequential 'features' module")
        features = model.features
        convolutions = [module for module in features if isinstance(module, nn.Conv2d)]
        if len(convolutions) < 2:
            raise ValueError("channel pruning requires at least two Conv2d modules")
        if any(conv.groups != 1 for conv in convolutions):
            raise ValueError("grouped and depthwise Conv2d modules are not supported")
        return features, graph_nodes

    def apply(self, model: nn.Module) -> OptimizationResult:
        optimized = copy.deepcopy(model).eval()
        features, graph_nodes = self._validate_model(optimized)
        conv_positions = [
            position for position, module in enumerate(features) if isinstance(module, nn.Conv2d)
        ]
        original_convs = {position: features[position] for position in conv_positions}

        kept_outputs: dict[int, list[int]] = {}
        for position in conv_positions:
            conv = original_convs[position]
            is_last = position == conv_positions[-1]
            keep_count = conv.out_channels if is_last else max(
                1, round(conv.out_channels * (1.0 - self._ratio))
            )
            kept_outputs[position] = _rank_channels(conv, keep_count)

        changes: list[GraphChange] = []
        previous_outputs: list[int] | None = None
        previous_path: str | None = None
        for position in conv_positions:
            path = f"features.{position}"
            conv = original_convs[position]
            input_indices = previous_outputs or list(range(conv.in_channels))
            output_indices = kept_outputs[position]
            replacement = _new_conv(conv, input_indices, output_indices)
            features[position] = replacement

            if len(input_indices) != conv.in_channels:
                removed_inputs = sorted(set(range(conv.in_channels)) - set(input_indices))
                changes.append(
                    GraphChange(
                        node=path,
                        op_type="Conv2d",
                        change_type="dependency_repair",
                        before=_conv_attributes(conv),
                        after=_conv_attributes(replacement),
                        metadata={
                            "source_node": previous_path,
                            "removed_input_channels": len(removed_inputs),
                            "removed_input_channel_indices": removed_inputs,
                            "dependency_relationship": "previous_conv_output_to_conv_input",
                            "reason": "match consumer inputs to physically retained producer outputs",
                        },
                    )
                )

            if len(output_indices) != conv.out_channels:
                removed_outputs = sorted(set(range(conv.out_channels)) - set(output_indices))
                changes.append(
                    GraphChange(
                        node=path,
                        op_type="Conv2d",
                        change_type="prune_output_channels",
                        before=_conv_attributes(conv),
                        after=_conv_attributes(replacement),
                        metadata={
                            "removed_output_channels": len(removed_outputs),
                            "removed_output_channel_indices": removed_outputs,
                            "kept_output_channel_indices": output_indices,
                            "ranking": "l1_weight_magnitude",
                            "dependency_relationship": "producer_to_batchnorm_and_next_conv",
                            "reason": "remove lowest L1 weight-magnitude output channels",
                        },
                    )
                )

            batch_norm_position = position + 1
            if batch_norm_position < len(features) and isinstance(
                features[batch_norm_position], nn.BatchNorm2d
            ):
                batch_norm = features[batch_norm_position]
                if batch_norm.num_features != len(output_indices):
                    replacement_batch_norm = _new_batch_norm(batch_norm, output_indices)
                    features[batch_norm_position] = replacement_batch_norm
                    removed_features = sorted(
                        set(range(batch_norm.num_features)) - set(output_indices)
                    )
                    changes.append(
                        GraphChange(
                            node=f"features.{batch_norm_position}",
                            op_type="BatchNorm2d",
                            change_type="dependency_repair",
                            before=_batch_norm_attributes(batch_norm),
                            after=_batch_norm_attributes(replacement_batch_norm),
                            metadata={
                                "source_node": path,
                                "removed_feature_indices": removed_features,
                                "dependency_relationship": "conv_output_to_batchnorm_features",
                                "reason": "match BatchNorm features to retained Conv2d outputs",
                            },
                        )
                    )

            previous_outputs = output_indices
            previous_path = path

        return OptimizationResult(
            model=optimized.eval(),
            metadata={
                "dependency_graph_nodes": graph_nodes,
                "changed_nodes": len({change.node for change in changes}),
                "graph_change_count": len(changes),
                "scope": "sequential Conv2d-BatchNorm2d dependency chains",
            },
            graph_changes=changes,
        )
