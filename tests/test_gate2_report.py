from copy import deepcopy

import pytest

from sparseflow.gate2_report import _validate_gate2_optimization
from sparseflow.report import EvidenceValidationError
from sparseflow.serialization import sha256_json


def _metadata():
    plan = {
        "schema_version": "0.2.0",
        "architecture_identifier": "resnet18-reference-v1",
        "pruning_ratio": 0.125,
        "dependency_graph_hash": "a" * 64,
        "dependency_groups": ["stage1.residual_output_channels"],
        "channel_selections": [],
        "ranking_metadata": {},
        "deterministic_seed": 1234,
        "minimum_channels": 8,
        "alignment": 8,
        "excluded_layers": ["fc.output_features"],
        "layer_decisions": [
            {
                "module_path": "conv1",
                "module_type": "Conv2d",
                "original_input_channels": 3,
                "original_output_channels": 64,
                "kept_input_channel_indices": [0, 1, 2],
                "kept_output_channel_indices": [0, 1],
                "removed_input_channel_indices": [],
                "removed_output_channel_indices": list(range(2, 64)),
                "dependency_group": "stage1.residual_output_channels",
                "source_dependency": "input",
                "target_dependency": "bn1",
                "reason": "test",
            }
        ],
        "residual_coupling_groups": [
            {
                "group_id": "stage1.residual_output_channels",
                "stage": 1,
                "original_channels": 64,
                "canonical_kept_channel_indices": [0, 1],
                "canonical_removed_channel_indices": list(range(2, 64)),
                "members": [
                    {
                        "module_path": "conv1",
                        "role": "producer_output",
                        "kept_channel_indices": [0, 1],
                        "removed_channel_indices": list(range(2, 64)),
                    }
                ],
                "all_members_match": True,
            }
        ],
    }
    plan["plan_hash"] = sha256_json(plan)
    execution_change = {
        "module_path": "conv1",
        "module_type": "Conv2d",
        "before": {"in_channels": 3, "out_channels": 64},
        "after": {"in_channels": 3, "out_channels": 2},
        "kept_input_channel_indices": [0, 1, 2],
        "kept_output_channel_indices": [0, 1],
        "removed_input_channel_indices": [],
        "removed_output_channel_indices": list(range(2, 64)),
        "dependency_group": "stage1.residual_output_channels",
        "matches_plan": True,
    }
    return {
        "gate": "resnet18_gate2_transactional_physical_pruning",
        "dependency_graph_hash": "a" * 64,
        "pruning_plan": plan,
        "pruning_plan_hash": plan["plan_hash"],
        "coupling_group_validation": {
            "validated": True,
            "all_members_match": True,
            "coupling_groups": [{"group_id": "stage1.residual_output_channels"}],
            "plan_hash": plan["plan_hash"],
        },
        "plan_vs_execution_validation": {
            "validated": True,
            "executed_as_planned": True,
            "changes": [execution_change],
        },
        "transaction": {"original_model_preserved": True},
    }


def test_gate2_evidence_boundary_recomputes_plan_and_member_identity():
    result = _validate_gate2_optimization(_metadata())
    assert result["all_members_match"] is True
    assert result["executed_as_planned"] is True


def test_gate2_evidence_boundary_rejects_self_asserted_member_match():
    metadata = _metadata()
    metadata["pruning_plan"]["residual_coupling_groups"][0]["members"][0][
        "kept_channel_indices"
    ] = [0, 2]
    plan = metadata["pruning_plan"]
    plan.pop("plan_hash")
    plan["plan_hash"] = sha256_json(plan)
    metadata["pruning_plan_hash"] = plan["plan_hash"]
    with pytest.raises(EvidenceValidationError, match="mismatched channel identities"):
        _validate_gate2_optimization(metadata)


def test_gate2_evidence_boundary_rejects_execution_index_mismatch():
    metadata = _metadata()
    metadata["plan_vs_execution_validation"]["changes"][0][
        "kept_output_channel_indices"
    ] = [0, 2]
    with pytest.raises(EvidenceValidationError, match="differs from plan"):
        _validate_gate2_optimization(metadata)


def test_gate2_evidence_boundary_rejects_tampered_plan_hash():
    metadata = deepcopy(_metadata())
    metadata["pruning_plan_hash"] = "f" * 64
    with pytest.raises(EvidenceValidationError, match="plan hash"):
        _validate_gate2_optimization(metadata)
