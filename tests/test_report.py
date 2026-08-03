import copy
import json
from pathlib import Path

import pytest

from sparseflow.report import (
    EvidenceValidationError,
    render_result_table,
    validate_evidence_report,
    write_evidence_report,
)


def _valid_report():
    return {
        "schema_version": "0.3.0",
        "product": {
            "current_milestone": "SparseFlow V0",
            "positioning": "SparseFlow V0: reproducible optimization and evidence harness, validated with a deterministic physical channel-pruning demonstration.",
            "demonstration_type": "physical_channel_pruning_mechanism_demonstration",
            "commercial_benchmark": False,
        },
        "run": {
            "id": "run-abc123",
            "timestamp_utc": "2026-08-03T20:00:00Z",
            "seed": 1234,
            "input_shape": [1, 3, 16, 16],
            "repository": {"path": "C:/repo", "identifier": "sparseflow-v1"},
            "git": {"commit_hash": "a" * 40, "dirty": False},
        },
        "environment": {
            "hardware": {
                "operating_system": "Windows",
                "architecture": "64bit",
                "machine": "AMD64",
                "logical_cpu_count": 8,
                "cpu_model": None,
            },
            "software": {"python": "3.12", "torch": "2.7", "onnx": "1.18", "onnxruntime": "1.22", "numpy": "2.2"},
            "process": {"onnxruntime_intra_op_threads": 1, "onnxruntime_inter_op_threads": 1},
        },
        "model": {"identifier": "reference_cnn_v1", "fidelity_is_proxy": True},
        "configuration": {
            "preset": "noop-control",
            "model_identifier": "reference_cnn_v1",
            "seed": 1234,
            "input_shape": [1, 3, 16, 16],
            "pass_name": "noop",
            "pruning_ratio": 0.0,
            "warmup_runs": 1,
            "measured_runs": 2,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "output_report_filename": "report-noop.json",
        },
        "optimization": {"pass_name": "noop", "config": {}, "metadata": {}},
        "metrics": {
            "parameters": {"baseline": 10, "optimized": 10, "reduction_percent": 0.0},
            "serialized_size_bytes": {
                "pytorch_state_dict": {"baseline": 100, "optimized": 100, "reduction_percent": 0.0},
                "onnx": {"baseline": 100, "optimized": 100, "reduction_percent": 0.0},
            },
            "latency_ms": {
                "methodology": {"provider": "CPUExecutionProvider", "warmup_runs": 1, "measured_runs": 2, "intra_op_threads": 1, "inter_op_threads": 1, "clock": "time.perf_counter_ns"},
                "baseline": {"samples": [1.0, 1.1], "samples_sha256": "b" * 64, "p50": 1.05, "p95": 1.095, "minimum": 1.0, "maximum": 1.1, "mean": 1.05, "standard_deviation": 0.05, "coefficient_of_variation": 0.047619},
                "optimized": {"samples": [1.0, 1.1], "samples_sha256": "b" * 64, "p50": 1.05, "p95": 1.095, "minimum": 1.0, "maximum": 1.1, "mean": 1.05, "standard_deviation": 0.05, "coefficient_of_variation": 0.047619},
            },
            "compute": {
                "convention": "1 MAC = 2 FLOPs",
                "implementation": "counter",
                "input_shape": [1, 3, 16, 16],
                "baseline": {"macs": 10, "flops": 20, "unsupported_operators": []},
                "optimized": {"macs": 10, "flops": 20, "unsupported_operators": []},
                "mac_reduction_percent": 0.0,
                "flop_reduction_percent": 0.0,
            },
            "fidelity": {"label": "proxy", "method": "output comparison", "absolute_error_max": 0.0, "relative_l2_error": 0.0, "tolerance": {"absolute_max": 1e-6, "relative_l2_max": 1e-5}, "passed": True},
        },
        "latency_interpretation": {
            "noise_sensitive": False,
            "reasons": [],
            "commercial_speedup_claim_supported": False,
        },
        "graph": {
            "changes": [],
            "graph_diff_hash": "c" * 64,
            "baseline_structural_hash": "d" * 64,
            "optimized_structural_hash": "d" * 64,
        },
        "artifacts": {
            "baseline": {"model_hash": "e" * 64, "pytorch_sha256": "f" * 64, "onnx_sha256": "1" * 64, "onnx_validated": True},
            "optimized": {"model_hash": "e" * 64, "pytorch_sha256": "f" * 64, "onnx_sha256": "1" * 64, "onnx_validated": True},
        },
    }


@pytest.mark.parametrize(
    "path",
    [
        ("schema_version",),
        ("product",),
        ("run",),
        ("run", "id"),
        ("run", "timestamp_utc"),
        ("run", "seed"),
        ("run", "input_shape"),
        ("run", "git"),
        ("environment",),
        ("environment", "hardware"),
        ("environment", "software"),
        ("model",),
        ("model", "identifier"),
        ("configuration",),
        ("optimization",),
        ("optimization", "config"),
        ("metrics",),
        ("metrics", "parameters"),
        ("metrics", "serialized_size_bytes"),
        ("metrics", "latency_ms"),
        ("metrics", "compute"),
        ("metrics", "fidelity"),
        ("latency_interpretation",),
        ("graph",),
        ("graph", "graph_diff_hash"),
        ("artifacts",),
    ],
)
def test_each_mandatory_evidence_path_is_rejected_when_missing(path):
    report = copy.deepcopy(_valid_report())
    target = report
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]

    with pytest.raises(EvidenceValidationError) as exc_info:
        validate_evidence_report(report)

    assert ".".join(path) in str(exc_info.value)


@pytest.mark.parametrize(
    ("operation", "path", "value"),
    [
        ("remove", ("environment", "hardware"), None),
        ("remove", ("optimization", "config"), None),
        ("remove", ("graph", "graph_diff_hash"), None),
        ("set", ("run", "seed"), object()),
        ("set", ("metrics", "fidelity", "absolute_error_max"), float("nan")),
        ("set", ("metrics", "fidelity", "relative_l2_error"), float("inf")),
    ],
)
def test_invalid_report_never_creates_output_file(
    operation,
    path,
    value,
    tmp_path,
    monkeypatch,
):
    report = copy.deepcopy(_valid_report())
    target = report
    for key in path[:-1]:
        target = target[key]
    if operation == "remove":
        del target[path[-1]]
    else:
        target[path[-1]] = value
    output = tmp_path / "report.json"
    atomic_write_calls = []
    monkeypatch.setattr(
        "sparseflow.report.atomic_write_json",
        lambda *args, **kwargs: atomic_write_calls.append((args, kwargs)),
    )

    with pytest.raises(EvidenceValidationError):
        write_evidence_report(report, output)

    assert not output.exists()
    assert atomic_write_calls == []


def test_missing_schema_file_is_actionable(tmp_path):
    schema_path = tmp_path / "missing-schema.json"

    with pytest.raises(EvidenceValidationError) as exc_info:
        validate_evidence_report(_valid_report(), schema_path)

    assert "unable to load evidence schema" in str(exc_info.value)
    assert str(schema_path) in str(exc_info.value)


def test_malformed_json_schema_file_is_actionable(tmp_path):
    schema_path = tmp_path / "malformed-schema.json"
    schema_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(EvidenceValidationError) as exc_info:
        validate_evidence_report(_valid_report(), schema_path)

    assert "unable to load evidence schema" in str(exc_info.value)
    assert str(schema_path) in str(exc_info.value)


def test_syntactically_valid_but_invalid_json_schema_is_actionable(tmp_path):
    schema_path = tmp_path / "invalid-schema.json"
    schema_path.write_text(json.dumps({"type": 42}), encoding="utf-8")

    with pytest.raises(EvidenceValidationError, match="invalid checked-in report schema"):
        validate_evidence_report(_valid_report(), schema_path)


def test_report_is_schema_valid_and_deterministically_written(tmp_path):
    report = _valid_report()
    validate_evidence_report(report)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_evidence_report(report, first)
    write_evidence_report(report, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("\n")
    assert list(report) == ["schema_version", "product", "run", "environment", "model", "configuration", "optimization", "metrics", "latency_interpretation", "graph", "artifacts"]


def test_old_schema_report_fails_clearly():
    report = _valid_report()
    report["schema_version"] = "0.2.0"

    with pytest.raises(EvidenceValidationError, match="schema_version"):
        validate_evidence_report(report)


def test_console_renderer_contains_metrics_formatting_and_claim_warnings():
    report = _valid_report()
    report["latency_interpretation"]["noise_sensitive"] = True

    rendered = render_result_table(report)

    assert "SparseFlow V0 Evidence Benchmark" in rendered
    assert "Pass: noop" in rendered
    for row in (
        "Parameters",
        "ONNX bytes",
        "MACs",
        "FLOPs",
        "Latency p50 ms",
        "Latency p95 ms",
    ):
        assert row in rendered
    assert "10" in rendered
    assert "1.0500" in rendered
    assert "Fidelity proxy passed: True" in rendered
    assert "not task accuracy" in rendered
    assert "Latency noise-sensitive: True" in rendered
    assert "Commercial speedup claim supported: False" in rendered


def test_console_renderer_uses_gate1_heading_and_handles_zero_baseline():
    report = _valid_report()
    report["product"]["development_milestone"] = "SparseFlow V1 Gate 1"
    report["metrics"]["latency_ms"]["baseline"]["p50"] = 0.0
    report["metrics"]["latency_ms"]["optimized"]["p50"] = 0.0

    rendered = render_result_table(report)

    assert "SparseFlow V1 Gate 1 Evidence Benchmark" in rendered
    assert "0.0000" in rendered


@pytest.mark.parametrize(
    "filename",
    ["channel-prune-report.json", "noop-report.json"],
)
def test_curated_reference_reports_validate(filename):
    import json

    path = Path(__file__).resolve().parents[1] / "examples" / "reference-run" / filename
    report = json.loads(path.read_text(encoding="utf-8"))

    validate_evidence_report(report)
    assert report["schema_version"] == "0.3.0"
    assert report["product"]["commercial_benchmark"] is False
