"""Deterministic ResNet-18 channel-dependency analysis for V1 Gate 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from sparseflow.models import BasicBlock, ResNet18Reference


@dataclass(frozen=True)
class DependencyNode:
    path: str
    op_type: str
    role: str
    output_channels: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "op_type": self.op_type,
            "role": self.role,
            "output_channels": self.output_channels,
        }


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    dependency_type: str
    channels: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "dependency_type": self.dependency_type,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class ResidualGroup:
    block_name: str
    main_branch: tuple[str, ...]
    skip_branch: tuple[str, ...]
    projection_exists: bool
    constrained_channels: int
    downstream_consumers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_name": self.block_name,
            "main_branch": list(self.main_branch),
            "skip_branch": list(self.skip_branch),
            "projection_exists": self.projection_exists,
            "constrained_channels": self.constrained_channels,
            "downstream_consumers": list(self.downstream_consumers),
        }


@dataclass(frozen=True)
class ChannelConstraint:
    constraint_id: str
    constraint_type: str
    channels: int
    members: tuple[str, ...]
    downstream_consumers: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "channels": self.channels,
            "members": list(self.members),
            "downstream_consumers": list(self.downstream_consumers),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClassifierDependency:
    producer: str
    pool: str
    pooled_feature_width: int
    linear: str
    linear_input_features: int
    dependency_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "pool": self.pool,
            "pooled_feature_width": self.pooled_feature_width,
            "linear": self.linear,
            "linear_input_features": self.linear_input_features,
            "dependency_type": self.dependency_type,
        }


@dataclass(frozen=True)
class ResNetDependencyGraph:
    architecture_identifier: str
    nodes: tuple[DependencyNode, ...]
    edges: tuple[DependencyEdge, ...]
    residual_groups: tuple[ResidualGroup, ...]
    channel_constraints: tuple[ChannelConstraint, ...]
    conv_bn_dependencies: tuple[DependencyEdge, ...]
    classifier_dependency: ClassifierDependency

    def summary(self) -> dict[str, Any]:
        projections = sum(group.projection_exists for group in self.residual_groups)
        return {
            "architecture_identifier": self.architecture_identifier,
            "residual_group_count": len(self.residual_groups),
            "projection_shortcut_count": projections,
            "identity_shortcut_count": len(self.residual_groups) - projections,
            "conv_bn_dependency_count": len(self.conv_bn_dependencies),
            "classifier_dependency_count": 1,
            "channel_constraint_count": len(self.channel_constraints),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture_identifier": self.architecture_identifier,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
            "residual_groups": [group.as_dict() for group in self.residual_groups],
            "channel_constraints": [
                constraint.as_dict() for constraint in self.channel_constraints
            ],
            "conv_bn_dependencies": [
                dependency.as_dict() for dependency in self.conv_bn_dependencies
            ],
            "classifier_dependency": self.classifier_dependency.as_dict(),
            "summary": self.summary(),
        }


def _downstream_consumers(stage: int, block: int) -> tuple[str, ...]:
    if block == 0:
        return (f"layer{stage}.1.conv1", f"layer{stage}.1.add")
    if stage < 4:
        return (
            f"layer{stage + 1}.0.conv1",
            f"layer{stage + 1}.0.downsample.0",
            f"layer{stage + 1}.0.add",
        )
    return ("avgpool", "fc")


def _conv_bn_pairs(model: ResNet18Reference) -> list[tuple[str, str, int]]:
    pairs = [("conv1", "bn1", model.conv1.out_channels)]
    for stage_number in range(1, 5):
        stage = getattr(model, f"layer{stage_number}")
        for block_number, block in enumerate(stage):
            prefix = f"layer{stage_number}.{block_number}"
            pairs.extend(
                [
                    (f"{prefix}.conv1", f"{prefix}.bn1", block.conv1.out_channels),
                    (f"{prefix}.conv2", f"{prefix}.bn2", block.conv2.out_channels),
                ]
            )
            if block.downsample is not None:
                projection = block.downsample[0]
                pairs.append(
                    (
                        f"{prefix}.downsample.0",
                        f"{prefix}.downsample.1",
                        projection.out_channels,
                    )
                )
    return pairs


def analyze_resnet18_dependencies(model: nn.Module) -> ResNetDependencyGraph:
    """Describe every ResNet-18 residual and channel-width dependency."""

    if not isinstance(model, ResNet18Reference):
        raise ValueError("dependency analysis requires ResNet18Reference")

    blocks = [module for module in model.modules() if isinstance(module, BasicBlock)]
    if len(blocks) != 8:
        raise ValueError("ResNet-18 dependency analysis requires exactly 8 BasicBlocks")

    nodes: dict[str, DependencyNode] = {}
    conv_bn_dependencies: list[DependencyEdge] = []
    edges: list[DependencyEdge] = []
    for conv_path, batch_norm_path, channels in _conv_bn_pairs(model):
        nodes[conv_path] = DependencyNode(conv_path, "Conv2d", "channel_producer", channels)
        nodes[batch_norm_path] = DependencyNode(
            batch_norm_path,
            "BatchNorm2d",
            "channel_consumer",
            channels,
        )
        edge = DependencyEdge(
            conv_path,
            batch_norm_path,
            "conv_output_to_batchnorm_features",
            channels,
        )
        conv_bn_dependencies.append(edge)
        edges.append(edge)

    residual_groups: list[ResidualGroup] = []
    constraints: list[ChannelConstraint] = []
    for stage_number in range(1, 5):
        stage = getattr(model, f"layer{stage_number}")
        for block_number, block in enumerate(stage):
            prefix = f"layer{stage_number}.{block_number}"
            add_path = f"{prefix}.add"
            main_branch = (
                f"{prefix}.conv1",
                f"{prefix}.bn1",
                f"{prefix}.relu",
                f"{prefix}.conv2",
                f"{prefix}.bn2",
            )
            projection_exists = block.downsample is not None
            skip_branch = (
                (f"{prefix}.downsample.0", f"{prefix}.downsample.1")
                if projection_exists
                else (f"{prefix}.identity",)
            )
            downstream = _downstream_consumers(stage_number, block_number)
            channels = block.bn2.num_features
            skip_output = skip_branch[-1]

            nodes[add_path] = DependencyNode(add_path, "add", "residual_merge", channels)
            if not projection_exists:
                nodes[skip_output] = DependencyNode(
                    skip_output,
                    "identity",
                    "skip_branch",
                    channels,
                )
            edges.extend(
                [
                    DependencyEdge(
                        main_branch[-1],
                        add_path,
                        "main_branch_to_residual_add",
                        channels,
                    ),
                    DependencyEdge(
                        skip_output,
                        add_path,
                        "skip_branch_to_residual_add",
                        channels,
                    ),
                ]
            )
            edges.extend(
                DependencyEdge(
                    add_path,
                    consumer,
                    "residual_output_to_downstream_consumer",
                    channels,
                )
                for consumer in downstream
            )
            residual_groups.append(
                ResidualGroup(
                    block_name=prefix,
                    main_branch=main_branch,
                    skip_branch=skip_branch,
                    projection_exists=projection_exists,
                    constrained_channels=channels,
                    downstream_consumers=downstream,
                )
            )
            constraints.append(
                ChannelConstraint(
                    constraint_id=f"{prefix}.residual_add_channels",
                    constraint_type="equal_channel_width_at_add",
                    channels=channels,
                    members=(main_branch[-1], skip_output, add_path),
                    downstream_consumers=downstream,
                    reason="main and skip branches must expose identical channel widths",
                )
            )

    nodes["avgpool"] = DependencyNode("avgpool", "AdaptiveAvgPool2d", "pool", 512)
    nodes["fc"] = DependencyNode("fc", "Linear", "classifier", model.fc.out_features)
    classifier_dependency = ClassifierDependency(
        producer="layer4.1.add",
        pool="avgpool",
        pooled_feature_width=512,
        linear="fc",
        linear_input_features=model.fc.in_features,
        dependency_type="pooled_features_to_linear_input",
    )
    edges.append(
        DependencyEdge(
            "avgpool",
            "fc",
            "pooled_features_to_linear_input",
            model.fc.in_features,
        )
    )
    return ResNetDependencyGraph(
        architecture_identifier=model.architecture_identifier,
        nodes=tuple(nodes[path] for path in sorted(nodes)),
        edges=tuple(edges),
        residual_groups=tuple(residual_groups),
        channel_constraints=tuple(constraints),
        conv_bn_dependencies=tuple(conv_bn_dependencies),
        classifier_dependency=classifier_dependency,
    )
