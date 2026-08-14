"""Gate 2 evidence boundary: independently verify plan, coupling, and execution claims."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from sparseflow.benchmark import BenchmarkResult
from sparseflow.report import (
    EvidenceValidationError,
    build_evidence_report,
    validate_evidence_report,
)
from sparseflow.serialization import sha256_json


GATE2_SCHEMA_VERSION = "0.5.0"
GATE2_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "evidence-report-gate2.schema.json"
)


def _validate_gate2_optimization(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("gate") != "resnet18_gate2_transactional_physical_pruning":
        raise EvidenceValidationError("invalid Gate 2 metadata gate")
    plan = deepcopy(metadata.get("pruning_plan"))
    if not isinstance(plan, dict):
        raise EvidenceValidationError("missing Gate 2 pruning plan")
    reported_plan_hash = plan.pop("plan_hash", None)
    recomputed_plan_hash = sha256_json(plan)
    if reported_plan_hash != recomputed_plan_hash or metadata.get("pruning_plan_hash") != recomputed_plan_hash:
        raise EvidenceValidationError("invalid Gate 2 pruning plan hash")
    if plan.get("dependency_graph_hash") != metadata.get("dependency_graph_hash"):
        raise EvidenceValidationError("Gate 2 dependency graph hash differs from plan")

    coupling_results: list[dict[str, Any]] = []
    for group in plan.get("residual_coupling_groups", []):
        canonical_kept = group.get("canonical_kept_channel_indices")
        canonical_removed = group.get("canonical_removed_channel_indices")
        members = group.get("members", [])
        if not members:
            raise EvidenceValidationError(f"Gate 2 coupling group {group.get('group_id')} has no members")
        matches = [
            member.get("kept_channel_indices") == canonical_kept
            and member.get("removed_channel_indices") == canonical_removed
            for member in members
        ]
        if not all(matches):
            raise EvidenceValidationError(
                f"Gate 2 coupling group {group.get('group_id')} contains mismatched channel identities"
            )
        coupling_results.append(
            {
                "group_id": group.get("group_id"),
                "all_members_match": True,
                "member_count": len(members),
            }
        )

    reported_coupling = metadata.get("coupling_group_validation", {})
    if reported_coupling.get("all_members_match") is not True:
        raise EvidenceValidationError("Gate 2 pass did not provide a valid coupling result")
    if len(reported_coupling.get("coupling_groups", [])) != len(coupling_results):
        raise EvidenceValidationError("Gate 2 coupling-group evidence count differs from plan")

    decisions = {decision["module_path"]: decision for decision in plan.get("layer_decisions", [])}
    execution = metadata.get("plan_vs_execution_validation", {})
    executed_changes = execution.get("changes", [])
    if len(decisions) != len(executed_changes):
        raise EvidenceValidationError("Gate 2 executed change count differs from pruning plan")
    for change in executed_changes:
        path = change.get("module_path")
        decision = decisions.get(path)
        if decision is None:
            raise EvidenceValidationError(f"Gate 2 contains unplanned executed change {path}")
        for field in (
            "kept_input_channel_indices",
            "kept_output_channel_indices",
            "removed_input_channel_indices",
            "removed_output_channel_indices",
            "dependency_group",
        ):
            if change.get(field) != decision.get(field):
                raise EvidenceValidationError(
                    f"Gate 2 executed change {path} differs from plan at {field}"
                )
        if change.get("matches_plan") is not True:
            raise EvidenceValidationError(f"Gate 2 executed change {path} failed plan validation")
    if execution.get("executed_as_planned") is not True:
        raise EvidenceValidationError("Gate 2 execution is not identical to the validated plan")
    if metadata.get("transaction", {}).get("original_model_preserved") is not True:
        raise EvidenceValidationError("Gate 2 transaction did not preserve the original model")

    return {
        "plan_hash": recomputed_plan_hash,
        "all_members_match": True,
        "coupling_group_count": len(coupling_results),
        "executed_as_planned": True,
        "executed_change_count": len(executed_changes),
    }


def build_gate2_evidence_report(
    result: BenchmarkResult,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Build a normal calibrated report, then independently certify Gate 2 evidence."""

    # The Gate 2 schema accepts the staging 0.4 report, allowing the existing report
    # builder to run its full measurement-evidence recomputation before promotion.
    report = build_evidence_report(result, audit, schema_path=GATE2_SCHEMA_PATH)
    boundary = _validate_gate2_optimization(report["optimization"]["metadata"])
    report["schema_version"] = GATE2_SCHEMA_VERSION
    report["product"] = {
        "current_milestone": "SparseFlow v0.2",
        "positioning": (
            "SparseFlow v0.2: reproducible model optimization and evidence workflow "
            "with transactional dependency-aware ResNet-18 channel pruning."
        ),
        "demonstration_type": "resnet18_transactional_physical_channel_pruning",
        "commercial_benchmark": False,
        "development_milestone": "SparseFlow V1 Gate 2",
        "next_milestone": (
            "Real pretrained-model task-accuracy evaluation and recovery/fine-tuning evidence."
        ),
    }
    report["optimization"]["metadata"]["evidence_boundary_validation"] = boundary
    validate_evidence_report(report, GATE2_SCHEMA_PATH)
    return report
