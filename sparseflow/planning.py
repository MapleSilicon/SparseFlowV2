"""Deterministic, immutable pruning-plan contracts and ResNet-18 plan validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sparseflow.dependencies import ResNetDependencyGraph, analyze_resnet18_dependencies
from sparseflow.models import BasicBlock, ResNet18Reference
from sparseflow.serialization import sha256_json


PLAN_SCHEMA_VERSION = "0.2.0"


class PruningPlanValidationError(ValueError):
    """Raised when a pruning plan cannot be proven safe before mutation."""


@dataclass(frozen=True)
class ChannelSelection:
    dependency_group: str
    kept_channel_indices: tuple[int, ...]
    removed_channel_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dependency_group": self.dependency_group,
            "kept_channel_indices": list(self.kept_channel_indices),
            "removed_channel_indices": list(self.removed_channel_indices),
        }


@dataclass(frozen=True)
class MemberChannelSelection:
    module_path: str
    role: str
    kept_channel_indices: tuple[int, ...]
    removed_channel_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "role": self.role,
            "kept_channel_indices": list(self.kept_channel_indices),
            "removed_channel_indices": list(self.removed_channel_indices),
        }


@dataclass(frozen=True)
class ResidualCouplingGroupPlan:
    group_id: str
    stage: int
    original_channels: int
    canonical_kept_channel_indices: tuple[int, ...]
    canonical_removed_channel_indices: tuple[int, ...]
    members: tuple[MemberChannelSelection, ...]

    def as_dict(self) -> dict[str, Any]:
        member_sets = [member.kept_channel_indices for member in self.members]
        return {
            "group_id": self.group_id,
            "stage": self.stage,
            "original_channels": self.original_channels,
            "canonical_kept_channel_indices": list(self.canonical_kept_channel_indices),
            "canonical_removed_channel_indices": list(self.canonical_removed_channel_indices),
            "members": [member.as_dict() for member in self.members],
            "all_members_match": bool(member_sets)
            and all(indices == self.canonical_kept_channel_indices for indices in member_sets),
        }


@dataclass(frozen=True)
class LayerPruningDecision:
    module_path: str
    module_type: str
    original_input_channels: int | None
    original_output_channels: int | None
    kept_input_channel_indices: tuple[int, ...]
    kept_output_channel_indices: tuple[int, ...]
    removed_input_channel_indices: tuple[int, ...]
    removed_output_channel_indices: tuple[int, ...]
    dependency_group: str | None
    source_dependency: str | None
    target_dependency: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "module_type": self.module_type,
            "original_input_channels": self.original_input_channels,
            "original_output_channels": self.original_output_channels,
            "kept_input_channel_indices": list(self.kept_input_channel_indices),
            "kept_output_channel_indices": list(self.kept_output_channel_indices),
            "removed_input_channel_indices": list(self.removed_input_channel_indices),
            "removed_output_channel_indices": list(self.removed_output_channel_indices),
            "dependency_group": self.dependency_group,
            "source_dependency": self.source_dependency,
            "target_dependency": self.target_dependency,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PruningPlan:
    """Serializable pruning intent. Validation must complete before mutation."""

    dependency_groups: tuple[str, ...]
    channel_selections: tuple[ChannelSelection, ...]
    ranking_metadata: tuple[tuple[str, str], ...]
    schema_version: str = "0.1.0"
    architecture_identifier: str | None = None
    pruning_ratio: float = 0.0
    deterministic_seed: int = 1234
    minimum_channels: int = 1
    alignment: int = 1
    excluded_layers: tuple[str, ...] = ()
    dependency_graph_hash: str | None = None
    layer_decisions: tuple[LayerPruningDecision, ...] = ()
    residual_coupling_groups: tuple[ResidualCouplingGroupPlan, ...] = ()

    def __post_init__(self) -> None:
        if not self.dependency_groups or len(set(self.dependency_groups)) != len(
            self.dependency_groups
        ):
            raise ValueError("dependency groups must be non-empty and unique")
        if tuple(sorted(self.dependency_groups)) != self.dependency_groups:
            raise ValueError("dependency groups must use canonical sorted order")
        selection_groups = tuple(
            selection.dependency_group for selection in self.channel_selections
        )
        if selection_groups != self.dependency_groups:
            raise ValueError(
                "channel selections must match dependency groups in canonical order"
            )
        for selection in self.channel_selections:
            self._validate_selection(selection)
        metadata_keys = tuple(key for key, _ in self.ranking_metadata)
        if (
            not metadata_keys
            or len(set(metadata_keys)) != len(metadata_keys)
            or tuple(sorted(metadata_keys)) != metadata_keys
        ):
            raise ValueError("ranking metadata keys must be non-empty, unique, and sorted")
        if not 0.0 <= float(self.pruning_ratio) <= 0.5:
            raise ValueError("ResNet-18 pruning ratio must be in [0.0, 0.5]")
        if self.minimum_channels <= 0:
            raise ValueError("minimum_channels must be positive")
        if self.alignment <= 0:
            raise ValueError("alignment must be positive")
        decision_paths = tuple(decision.module_path for decision in self.layer_decisions)
        if len(set(decision_paths)) != len(decision_paths):
            raise ValueError("layer pruning decisions must use unique module paths")
        group_ids = tuple(group.group_id for group in self.residual_coupling_groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("residual coupling group ids must be unique")

    @staticmethod
    def _validate_selection(selection: ChannelSelection) -> None:
        if not selection.dependency_group:
            raise ValueError("dependency group must be non-empty")
        kept = selection.kept_channel_indices
        removed = selection.removed_channel_indices
        _validate_index_partition(kept, removed, label=selection.dependency_group)

    def hash_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dependency_groups": list(self.dependency_groups),
            "channel_selections": [
                selection.as_dict() for selection in self.channel_selections
            ],
            "ranking_metadata": {key: value for key, value in self.ranking_metadata},
        }
        if self.schema_version != "0.1.0" or self.layer_decisions or self.residual_coupling_groups:
            payload.update(
                {
                    "schema_version": self.schema_version,
                    "architecture_identifier": self.architecture_identifier,
                    "pruning_ratio": self.pruning_ratio,
                    "deterministic_seed": self.deterministic_seed,
                    "minimum_channels": self.minimum_channels,
                    "alignment": self.alignment,
                    "excluded_layers": list(self.excluded_layers),
                    "dependency_graph_hash": self.dependency_graph_hash,
                    "layer_decisions": [
                        decision.as_dict() for decision in self.layer_decisions
                    ],
                    "residual_coupling_groups": [
                        group.as_dict() for group in self.residual_coupling_groups
                    ],
                }
            )
        return payload

    @property
    def plan_hash(self) -> str:
        return sha256_json(self.hash_payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self.hash_payload(), "plan_hash": self.plan_hash}


def _validate_index_partition(
    kept: tuple[int, ...], removed: tuple[int, ...], *, label: str
) -> None:
    for kind, indices in (("kept", kept), ("removed", removed)):
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in indices
        ):
            raise ValueError(f"{label} {kind} channel indices must be non-negative integers")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError(f"{label} {kind} channel indices must be sorted and unique")
    if set(kept) & set(removed):
        raise ValueError(f"{label} kept and removed channel indices must not overlap")


def _module_by_path(model: nn.Module, path: str) -> nn.Module:
    current: nn.Module = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def _rank_conv_outputs(conv: nn.Conv2d, keep_count: int) -> tuple[int, ...]:
    scores = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    ranked = sorted(
        range(conv.out_channels),
        key=lambda index: (-float(scores[index].item()), index),
    )[:keep_count]
    return tuple(sorted(ranked))


def _rank_aggregate_outputs(convs: tuple[nn.Conv2d, ...], keep_count: int) -> tuple[int, ...]:
    if not convs:
        raise PruningPlanValidationError("cannot rank an empty residual coupling group")
    width = convs[0].out_channels
    if any(conv.out_channels != width for conv in convs):
        raise PruningPlanValidationError("coupled producers must have identical output widths")
    scores = torch.zeros(width, dtype=torch.float64)
    for conv in convs:
        scores += conv.weight.detach().abs().sum(dim=(1, 2, 3)).cpu().to(torch.float64)
    ranked = sorted(range(width), key=lambda index: (-float(scores[index].item()), index))[
        :keep_count
    ]
    return tuple(sorted(ranked))


def _removed(width: int, kept: tuple[int, ...]) -> tuple[int, ...]:
    kept_set = set(kept)
    return tuple(index for index in range(width) if index not in kept_set)


def _aligned_keep_count(width: int, ratio: float, minimum: int, alignment: int) -> int:
    if ratio == 0.0:
        return width
    raw = max(minimum, int(round(width * (1.0 - ratio))))
    if alignment > 1:
        raw = max(minimum, (raw // alignment) * alignment)
    return min(width, max(1, raw))


def _stage_output_producers(model: ResNet18Reference, stage: int) -> tuple[str, ...]:
    paths: list[str] = []
    if stage == 1:
        paths.append("conv1")
    stage_module = getattr(model, f"layer{stage}")
    for block_index, block in enumerate(stage_module):
        prefix = f"layer{stage}.{block_index}"
        paths.append(f"{prefix}.conv2")
        if block.downsample is not None:
            paths.append(f"{prefix}.downsample.0")
    return tuple(paths)


def _stage_bn_members(model: ResNet18Reference, stage: int) -> tuple[str, ...]:
    paths: list[str] = []
    if stage == 1:
        paths.append("bn1")
    stage_module = getattr(model, f"layer{stage}")
    for block_index, block in enumerate(stage_module):
        prefix = f"layer{stage}.{block_index}"
        paths.append(f"{prefix}.bn2")
        if block.downsample is not None:
            paths.append(f"{prefix}.downsample.1")
    return tuple(paths)


def _dependency_graph_hash(graph: ResNetDependencyGraph) -> str:
    return sha256_json(graph.as_dict())


def build_resnet18_pruning_plan(
    model: nn.Module,
    ratio: float,
    *,
    seed: int = 1234,
    minimum_channels: int = 8,
    alignment: int = 8,
) -> PruningPlan:
    """Build deterministic stage-coupled pruning intent without mutating the model."""

    if not isinstance(model, ResNet18Reference):
        raise PruningPlanValidationError("Gate 2 planning requires ResNet18Reference")
    if not isinstance(ratio, (float, int)) or not 0.0 <= float(ratio) <= 0.5:
        raise PruningPlanValidationError("ResNet-18 Gate 2 ratio must be in [0.0, 0.5]")
    ratio = float(ratio)
    dependency_graph = analyze_resnet18_dependencies(model)

    stage_kept: dict[int, tuple[int, ...]] = {}
    stage_removed: dict[int, tuple[int, ...]] = {}
    coupling_groups: list[ResidualCouplingGroupPlan] = []
    simple_selections: list[ChannelSelection] = []

    for stage, width in ((1, 64), (2, 128), (3, 256), (4, 512)):
        keep_count = _aligned_keep_count(width, ratio, minimum_channels, alignment)
        producer_paths = _stage_output_producers(model, stage)
        producers = tuple(_module_by_path(model, path) for path in producer_paths)
        if not all(isinstance(module, nn.Conv2d) for module in producers):
            raise PruningPlanValidationError(f"stage {stage} producer is not Conv2d")
        kept = _rank_aggregate_outputs(producers, keep_count)  # type: ignore[arg-type]
        removed = _removed(width, kept)
        stage_kept[stage] = kept
        stage_removed[stage] = removed
        group_id = f"stage{stage}.residual_output_channels"
        members: list[MemberChannelSelection] = []
        for path in producer_paths:
            members.append(MemberChannelSelection(path, "producer_output", kept, removed))
        for path in _stage_bn_members(model, stage):
            members.append(MemberChannelSelection(path, "batchnorm_features", kept, removed))
        coupling_groups.append(
            ResidualCouplingGroupPlan(
                group_id=group_id,
                stage=stage,
                original_channels=width,
                canonical_kept_channel_indices=kept,
                canonical_removed_channel_indices=removed,
                members=tuple(members),
            )
        )
        simple_selections.append(ChannelSelection(group_id, kept, removed))

    decisions: list[LayerPruningDecision] = []

    # Stem output must share the stage-1 canonical set because layer1 uses identity shortcuts.
    stem = model.conv1
    decisions.append(
        LayerPruningDecision(
            module_path="conv1",
            module_type="Conv2d",
            original_input_channels=stem.in_channels,
            original_output_channels=stem.out_channels,
            kept_input_channel_indices=tuple(range(stem.in_channels)),
            kept_output_channel_indices=stage_kept[1],
            removed_input_channel_indices=(),
            removed_output_channel_indices=stage_removed[1],
            dependency_group="stage1.residual_output_channels",
            source_dependency="input",
            target_dependency="bn1/layer1.0",
            reason="stage-1 residual identity requires one canonical output channel set",
        )
    )
    decisions.append(
        LayerPruningDecision(
            module_path="bn1",
            module_type="BatchNorm2d",
            original_input_channels=stem.out_channels,
            original_output_channels=stem.out_channels,
            kept_input_channel_indices=stage_kept[1],
            kept_output_channel_indices=stage_kept[1],
            removed_input_channel_indices=stage_removed[1],
            removed_output_channel_indices=stage_removed[1],
            dependency_group="stage1.residual_output_channels",
            source_dependency="conv1",
            target_dependency="layer1.0",
            reason="BatchNorm features follow retained stem channels",
        )
    )

    previous_stage_kept = stage_kept[1]
    for stage in range(1, 5):
        stage_module = getattr(model, f"layer{stage}")
        current_kept = stage_kept[stage]
        current_removed = stage_removed[stage]
        if stage > 1:
            previous_stage_kept = stage_kept[stage - 1]
        for block_index, block in enumerate(stage_module):
            if not isinstance(block, BasicBlock):
                raise PruningPlanValidationError("ResNet-18 stage contains non-BasicBlock")
            prefix = f"layer{stage}.{block_index}"
            input_kept = previous_stage_kept if block_index == 0 and stage > 1 else current_kept
            input_width = block.conv1.in_channels
            input_removed = _removed(input_width, input_kept)
            internal_keep_count = _aligned_keep_count(
                block.conv1.out_channels, ratio, minimum_channels, alignment
            )
            internal_kept = _rank_conv_outputs(block.conv1, internal_keep_count)
            internal_removed = _removed(block.conv1.out_channels, internal_kept)

            decisions.extend(
                [
                    LayerPruningDecision(
                        f"{prefix}.conv1",
                        "Conv2d",
                        block.conv1.in_channels,
                        block.conv1.out_channels,
                        input_kept,
                        internal_kept,
                        input_removed,
                        internal_removed,
                        None,
                        f"stage{stage - 1 if stage > 1 and block_index == 0 else stage}.residual_output_channels",
                        f"{prefix}.bn1",
                        "prune internal BasicBlock channels after repairing residual-stage inputs",
                    ),
                    LayerPruningDecision(
                        f"{prefix}.bn1",
                        "BatchNorm2d",
                        block.bn1.num_features,
                        block.bn1.num_features,
                        internal_kept,
                        internal_kept,
                        internal_removed,
                        internal_removed,
                        None,
                        f"{prefix}.conv1",
                        f"{prefix}.conv2",
                        "BatchNorm features follow retained internal conv1 channels",
                    ),
                    LayerPruningDecision(
                        f"{prefix}.conv2",
                        "Conv2d",
                        block.conv2.in_channels,
                        block.conv2.out_channels,
                        internal_kept,
                        current_kept,
                        internal_removed,
                        current_removed,
                        f"stage{stage}.residual_output_channels",
                        f"{prefix}.bn1",
                        f"{prefix}.add",
                        "residual main branch emits the stage canonical channel set",
                    ),
                    LayerPruningDecision(
                        f"{prefix}.bn2",
                        "BatchNorm2d",
                        block.bn2.num_features,
                        block.bn2.num_features,
                        current_kept,
                        current_kept,
                        current_removed,
                        current_removed,
                        f"stage{stage}.residual_output_channels",
                        f"{prefix}.conv2",
                        f"{prefix}.add",
                        "residual main-branch BatchNorm follows canonical channels",
                    ),
                ]
            )
            if block.downsample is not None:
                projection = block.downsample[0]
                projection_bn = block.downsample[1]
                projection_input_kept = previous_stage_kept
                projection_input_removed = _removed(projection.in_channels, projection_input_kept)
                decisions.extend(
                    [
                        LayerPruningDecision(
                            f"{prefix}.downsample.0",
                            "Conv2d",
                            projection.in_channels,
                            projection.out_channels,
                            projection_input_kept,
                            current_kept,
                            projection_input_removed,
                            current_removed,
                            f"stage{stage}.residual_output_channels",
                            f"stage{stage - 1}.residual_output_channels",
                            f"{prefix}.add",
                            "projection shortcut must emit the same canonical set as conv2",
                        ),
                        LayerPruningDecision(
                            f"{prefix}.downsample.1",
                            "BatchNorm2d",
                            projection_bn.num_features,
                            projection_bn.num_features,
                            current_kept,
                            current_kept,
                            current_removed,
                            current_removed,
                            f"stage{stage}.residual_output_channels",
                            f"{prefix}.downsample.0",
                            f"{prefix}.add",
                            "projection BatchNorm follows canonical shortcut channels",
                        ),
                    ]
                )

    decisions.append(
        LayerPruningDecision(
            "fc",
            "Linear",
            model.fc.in_features,
            model.fc.out_features,
            stage_kept[4],
            tuple(range(model.fc.out_features)),
            stage_removed[4],
            (),
            "stage4.residual_output_channels",
            "layer4.1.add/avgpool",
            "logits",
            "classifier input features follow the retained final-stage channels",
        )
    )

    plan = PruningPlan(
        dependency_groups=tuple(selection.dependency_group for selection in simple_selections),
        channel_selections=tuple(simple_selections),
        ranking_metadata=(
            ("method", "l1_weight_magnitude"),
            ("residual_policy", "stage_aggregate_canonical_indices"),
            ("tie_breaker", "ascending_channel_index"),
        ),
        schema_version=PLAN_SCHEMA_VERSION,
        architecture_identifier=model.architecture_identifier,
        pruning_ratio=ratio,
        deterministic_seed=seed,
        minimum_channels=minimum_channels,
        alignment=alignment,
        excluded_layers=("fc.output_features",),
        dependency_graph_hash=_dependency_graph_hash(dependency_graph),
        layer_decisions=tuple(decisions),
        residual_coupling_groups=tuple(coupling_groups),
    )
    validate_resnet18_pruning_plan(plan, model, dependency_graph)
    return plan


def validate_resnet18_pruning_plan(
    plan: PruningPlan,
    model: nn.Module,
    dependency_graph: ResNetDependencyGraph | None = None,
) -> dict[str, Any]:
    """Independently prove channel identity, bounds, topology and plan completeness."""

    if not isinstance(model, ResNet18Reference):
        raise PruningPlanValidationError("plan validation requires ResNet18Reference")
    graph = dependency_graph or analyze_resnet18_dependencies(model)
    if plan.architecture_identifier != model.architecture_identifier:
        raise PruningPlanValidationError("plan architecture identifier does not match model")
    if plan.dependency_graph_hash != _dependency_graph_hash(graph):
        raise PruningPlanValidationError("plan dependency graph hash does not match analysis")

    known_modules = dict(model.named_modules())
    decision_by_path = {decision.module_path: decision for decision in plan.layer_decisions}
    if "fc" not in decision_by_path:
        raise PruningPlanValidationError("plan is missing classifier repair")

    group_results: list[dict[str, Any]] = []
    for group in plan.residual_coupling_groups:
        canonical = group.canonical_kept_channel_indices
        removed = group.canonical_removed_channel_indices
        _validate_index_partition(canonical, removed, label=group.group_id)
        if not canonical:
            raise PruningPlanValidationError(f"{group.group_id} cannot remove all channels")
        if set(canonical) | set(removed) != set(range(group.original_channels)):
            raise PruningPlanValidationError(
                f"{group.group_id} kept/removed indices do not partition original channels"
            )
        member_matches: list[bool] = []
        for member in group.members:
            if member.module_path not in known_modules:
                raise PruningPlanValidationError(
                    f"{group.group_id} references unknown module {member.module_path}"
                )
            _validate_index_partition(
                member.kept_channel_indices,
                member.removed_channel_indices,
                label=f"{group.group_id}:{member.module_path}",
            )
            matches = (
                member.kept_channel_indices == canonical
                and member.removed_channel_indices == removed
            )
            member_matches.append(matches)
            if not matches:
                raise PruningPlanValidationError(
                    f"{group.group_id} member {member.module_path} does not use canonical indices"
                )
        if not member_matches:
            raise PruningPlanValidationError(f"{group.group_id} has no members")
        group_results.append(
            {
                "group_id": group.group_id,
                "all_members_match": all(member_matches),
                "member_count": len(member_matches),
                "canonical_kept_channel_indices": list(canonical),
                "canonical_removed_channel_indices": list(removed),
            }
        )

    for decision in plan.layer_decisions:
        module = known_modules.get(decision.module_path)
        if module is None:
            raise PruningPlanValidationError(
                f"plan references unknown module {decision.module_path}"
            )
        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise PruningPlanValidationError("grouped/depthwise Conv2d is unsupported")
            expected_in = set(range(module.in_channels))
            expected_out = set(range(module.out_channels))
        elif isinstance(module, nn.BatchNorm2d):
            expected_in = expected_out = set(range(module.num_features))
        elif isinstance(module, nn.Linear):
            expected_in = set(range(module.in_features))
            expected_out = set(range(module.out_features))
        else:
            raise PruningPlanValidationError(
                f"unsupported planned module type at {decision.module_path}"
            )
        if set(decision.kept_input_channel_indices) | set(
            decision.removed_input_channel_indices
        ) != expected_in:
            raise PruningPlanValidationError(
                f"{decision.module_path} input indices do not partition original dimension"
            )
        if set(decision.kept_output_channel_indices) | set(
            decision.removed_output_channel_indices
        ) != expected_out:
            raise PruningPlanValidationError(
                f"{decision.module_path} output indices do not partition original dimension"
            )

    # Projection groups must contain both main and skip producers. Identity stages must
    # retain a single stage-wide canonical set so the unmaterialized identity branch is safe.
    for residual in graph.residual_groups:
        stage = int(residual.block_name[5])
        group_id = f"stage{stage}.residual_output_channels"
        group = next(
            (candidate for candidate in plan.residual_coupling_groups if candidate.group_id == group_id),
            None,
        )
        if group is None:
            raise PruningPlanValidationError(f"missing residual coupling group {group_id}")
        member_paths = {member.module_path for member in group.members}
        main_path = f"{residual.block_name}.conv2"
        if main_path not in member_paths:
            raise PruningPlanValidationError(f"{group_id} missing main residual producer {main_path}")
        if residual.projection_exists:
            projection_path = f"{residual.block_name}.downsample.0"
            if projection_path not in member_paths:
                raise PruningPlanValidationError(
                    f"{group_id} missing projection residual producer {projection_path}"
                )

    fc = decision_by_path["fc"]
    final_group = next(
        group
        for group in plan.residual_coupling_groups
        if group.group_id == "stage4.residual_output_channels"
    )
    if fc.kept_input_channel_indices != final_group.canonical_kept_channel_indices:
        raise PruningPlanValidationError(
            "classifier input indices do not match final residual-stage channels"
        )

    return {
        "validated": True,
        "architecture_identifier": model.architecture_identifier,
        "dependency_graph_hash": plan.dependency_graph_hash,
        "plan_hash": plan.plan_hash,
        "coupling_groups": group_results,
        "all_members_match": all(result["all_members_match"] for result in group_results),
        "layer_decision_count": len(plan.layer_decisions),
    }
