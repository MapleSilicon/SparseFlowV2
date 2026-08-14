from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from run_bench import _optimization_pass, resolve_cli_configuration
from sparseflow.benchmark import count_parameters
from sparseflow.gate2_pass import ResNet18Gate2PruningPass
from sparseflow.graph_diff import model_state_hash, structural_hash
from sparseflow.models import build_model
from sparseflow.planning import (
    MemberChannelSelection,
    PruningPlanValidationError,
    build_resnet18_pruning_plan,
    validate_resnet18_pruning_plan,
)
from sparseflow.resnet_pruning import apply_resnet18_pruning_plan, validate_executed_plan


def _model():
    return build_model("resnet18-reference", seed=1234).eval()


def test_gate2_preset_and_cli_route_to_transactional_pass():
    resolved = resolve_cli_configuration(["--preset", "resnet18-gate2"])
    assert resolved.model_identifier == "resnet18-reference"
    assert resolved.pruning_ratio == pytest.approx(0.125)
    optimization = _optimization_pass(
        resolved.pass_name,
        resolved.pruning_ratio,
        model_identifier=resolved.model_identifier,
        seed=resolved.seed,
    )
    assert isinstance(optimization, ResNet18Gate2PruningPass)


def test_plan_is_deterministic_and_stage_coupling_uses_identical_indices():
    model = _model()
    first = build_resnet18_pruning_plan(model, 0.125)
    second = build_resnet18_pruning_plan(model, 0.125)
    assert first.plan_hash == second.plan_hash
    assert first.as_dict() == second.as_dict()
    assert len(first.residual_coupling_groups) == 4
    for group in first.residual_coupling_groups:
        assert group.canonical_kept_channel_indices
        assert group.canonical_removed_channel_indices
        for member in group.members:
            assert member.kept_channel_indices == group.canonical_kept_channel_indices
            assert member.removed_channel_indices == group.canonical_removed_channel_indices
    validation = validate_resnet18_pruning_plan(first, model)
    assert validation["validated"] is True
    assert validation["all_members_match"] is True


def test_same_width_different_indices_fail_before_mutation():
    model = _model()
    original_hash = model_state_hash(model)
    original_structure = structural_hash(model)
    plan = build_resnet18_pruning_plan(model, 0.125)
    group = plan.residual_coupling_groups[0]
    member = group.members[0]
    canonical = group.canonical_kept_channel_indices
    removed = group.canonical_removed_channel_indices
    replacement_kept = tuple(sorted((canonical[:-1] + (removed[0],))))
    replacement_removed = tuple(
        index for index in range(group.original_channels) if index not in replacement_kept
    )
    bad_member = replace(
        member,
        kept_channel_indices=replacement_kept,
        removed_channel_indices=replacement_removed,
    )
    bad_group = replace(group, members=(bad_member, *group.members[1:]))
    bad_plan = replace(
        plan,
        residual_coupling_groups=(bad_group, *plan.residual_coupling_groups[1:]),
    )
    with pytest.raises(PruningPlanValidationError, match="canonical indices"):
        apply_resnet18_pruning_plan(model, bad_plan)
    assert model_state_hash(model) == original_hash
    assert structural_hash(model) == original_structure


def test_projection_member_missing_is_rejected():
    model = _model()
    plan = build_resnet18_pruning_plan(model, 0.125)
    group = plan.residual_coupling_groups[1]
    members = tuple(
        member for member in group.members if member.module_path != "layer2.0.downsample.0"
    )
    bad_plan = replace(
        plan,
        residual_coupling_groups=(
            plan.residual_coupling_groups[0],
            replace(group, members=members),
            *plan.residual_coupling_groups[2:],
        ),
    )
    with pytest.raises(PruningPlanValidationError, match="missing projection"):
        validate_resnet18_pruning_plan(bad_plan, model)


def test_physical_pruning_is_transactional_and_compact():
    model = _model()
    original_hash = model_state_hash(model)
    original_structure = structural_hash(model)
    baseline_parameters = count_parameters(model)
    plan = build_resnet18_pruning_plan(model, 0.125)
    optimized, transaction = apply_resnet18_pruning_plan(model, plan)

    assert model_state_hash(model) == original_hash
    assert structural_hash(model) == original_structure
    assert count_parameters(optimized) < baseline_parameters
    assert transaction["plan_validation"]["all_members_match"] is True
    assert transaction["execution_validation"]["executed_as_planned"] is True
    assert transaction["transaction"]["original_model_preserved"] is True

    with torch.no_grad():
        output = optimized(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 1000)
    assert torch.isfinite(output).all()

    assert optimized.layer2[0].conv2.out_channels == optimized.layer2[0].downsample[0].out_channels
    assert optimized.layer2[0].bn2.num_features == optimized.layer2[0].downsample[1].num_features
    assert optimized.fc.in_features == optimized.layer4[1].conv2.out_channels


def test_gate2_pass_records_plan_and_machine_checked_execution():
    model = _model()
    result = ResNet18Gate2PruningPass(0.125).apply(model)
    metadata = result.metadata
    assert metadata["gate"] == "resnet18_gate2_transactional_physical_pruning"
    assert metadata["pruning_plan_hash"] == metadata["pruning_plan"]["plan_hash"]
    assert metadata["coupling_group_validation"]["all_members_match"] is True
    assert metadata["plan_vs_execution_validation"]["executed_as_planned"] is True
    assert metadata["transaction"]["mutated_copy_only"] is True
    assert result.graph_changes
    assert any(change.node == "fc" for change in result.graph_changes)


def test_execution_validator_detects_unplanned_dimension_change():
    model = _model()
    plan = build_resnet18_pruning_plan(model, 0.125)
    optimized, _ = apply_resnet18_pruning_plan(model, plan)
    tampered = deepcopy(optimized)
    decision = next(item for item in plan.layer_decisions if item.module_path == "fc")
    tampered.fc = nn.Linear(len(decision.kept_input_channel_indices) + 1, 1000)
    with pytest.raises(Exception, match="differs from validated plan"):
        validate_executed_plan(model, tampered, plan)
