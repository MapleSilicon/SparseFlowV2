import torch

from sparseflow.audit import collect_audit_metadata
from sparseflow.benchmark import BenchmarkConfig, count_parameters, run_benchmark
from sparseflow.compute import count_compute
from sparseflow.graph_diff import model_state_hash, structural_hash
from sparseflow.models import RESNET18_INPUT_SHAPE, build_resnet18_reference
from sparseflow.passes import ChannelPruningPass
from sparseflow.report import build_evidence_report, validate_evidence_report


def _audit(seed: int) -> dict:
    audit = collect_audit_metadata(".", seed=seed, threads=1)
    audit["git"] = {"commit_hash": "a" * 40, "dirty": False}
    return audit


def test_resnet18_zero_ratio_analysis_changes_nothing():
    baseline = build_resnet18_reference(seed=1234)
    result = ChannelPruningPass(ratio=0.0).apply(baseline)
    optimized = result.model

    assert result.graph_changes == []
    assert result.metadata["gate"] == "resnet18_gate1_dependency_analysis"
    assert result.metadata["transformation"] == "none"
    assert result.metadata["dependency_summary"]["residual_group_count"] == 8
    assert model_state_hash(baseline) == model_state_hash(optimized)
    assert structural_hash(baseline) == structural_hash(optimized)
    assert count_parameters(baseline) == count_parameters(optimized)
    assert count_compute(baseline, RESNET18_INPUT_SHAPE) == count_compute(
        optimized, RESNET18_INPUT_SHAPE
    )

    inputs = torch.randn(1, *RESNET18_INPUT_SHAPE[1:], generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        assert torch.equal(baseline(inputs), optimized(inputs))


def test_resnet18_nonzero_ratio_is_rejected_as_gate2_work():
    model = build_resnet18_reference(seed=1234)

    try:
        ChannelPruningPass(ratio=0.1).apply(model)
    except ValueError as exc:
        assert "Gate 1" in str(exc)
        assert "0.0" in str(exc)
    else:
        raise AssertionError("non-zero ResNet-18 pruning must not be implemented in Gate 1")


def test_gate1_benchmark_exports_onnx_runs_ort_and_builds_evidence():
    config = BenchmarkConfig(
        input_shape=RESNET18_INPUT_SHAPE,
        seed=1234,
        warmup_runs=0,
        measured_runs=1,
        threads=1,
        fidelity_batch_size=1,
        preset_name="resnet18-gate1",
        output_report_filename="report-resnet18-gate1.json",
    )
    result = run_benchmark(
        build_resnet18_reference(seed=config.seed),
        ChannelPruningPass(ratio=0.0),
        config,
    )
    report = build_evidence_report(result, _audit(config.seed))
    validate_evidence_report(report)

    assert report["product"]["development_milestone"] == "SparseFlow V1 Gate 1"
    assert report["product"]["demonstration_type"] == "resnet18_dependency_analysis_zero_ratio_validation"
    assert report["model"]["identifier"] == "resnet18-reference"
    assert report["model"]["architecture_identifier"] == "resnet18_basicblock_reference_v1"
    assert report["metrics"]["parameters"]["reduction_percent"] == 0.0
    assert report["metrics"]["compute"]["mac_reduction_percent"] == 0.0
    assert report["metrics"]["compute"]["flop_reduction_percent"] == 0.0
    assert report["graph"]["changes"] == []
    assert report["graph"]["baseline_structural_hash"] == report["graph"]["optimized_structural_hash"]
    assert report["artifacts"]["baseline"]["model_hash"] == report["artifacts"]["optimized"]["model_hash"]
    assert report["artifacts"]["baseline"]["pytorch_sha256"] == report["artifacts"]["optimized"]["pytorch_sha256"]
    assert report["artifacts"]["baseline"]["onnx_sha256"] == report["artifacts"]["optimized"]["onnx_sha256"]

    metadata = report["optimization"]["metadata"]
    assert metadata["dependency_summary"]["residual_group_count"] == 8
    assert metadata["dependency_summary"]["projection_shortcut_count"] == 3
    assert metadata["dependency_summary"]["identity_shortcut_count"] == 5
    assert metadata["dependency_summary"]["conv_bn_dependency_count"] == 20
    assert metadata["dependency_analysis"]["classifier_dependency"]["linear"] == "fc"
    assert metadata["zero_ratio_validation"]["passed"] is True
    assert all(
        value is True
        for key, value in metadata["zero_ratio_validation"].items()
        if key not in {"ratio", "validation_type"}
    )

    for artifact in report["artifacts"].values():
        assert artifact["onnx_validated"] is True
        assert artifact["pytorch_onnx_validation"]["passed"] is True
        assert artifact["pytorch_onnx_validation"]["output_shape_matches"] is True
        assert artifact["pytorch_onnx_validation"]["pytorch_output_shape"] == [1, 1000]
        assert artifact["pytorch_onnx_validation"]["onnx_output_shape"] == [1, 1000]
