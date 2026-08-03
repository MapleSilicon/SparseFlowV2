import copy
from pathlib import Path

import pytest

from sparseflow.report import EvidenceValidationError, validate_evidence_report, write_evidence_report


def _valid_report():
    return {
        "schema_version": "0.2.0",
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
        "optimization": {"pass_name": "noop", "config": {}, "metadata": {}},
        "metrics": {
            "parameters": {"baseline": 10, "optimized": 10, "reduction_percent": 0.0},
            "serialized_size_bytes": {
                "pytorch_state_dict": {"baseline": 100, "optimized": 100, "reduction_percent": 0.0},
                "onnx": {"baseline": 100, "optimized": 100, "reduction_percent": 0.0},
            },
            "latency_ms": {
                "methodology": {"provider": "CPUExecutionProvider", "warmup_runs": 1, "measured_runs": 2, "threads": 1, "clock": "time.perf_counter_ns"},
                "baseline": {"samples": [1.0, 1.1], "samples_sha256": "b" * 64, "p50": 1.05, "p95": 1.095},
                "optimized": {"samples": [1.0, 1.1], "samples_sha256": "b" * 64, "p50": 1.05, "p95": 1.095},
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
        ("run", "git", "commit_hash"),
        ("environment", "hardware"),
        ("optimization", "config"),
        ("run", "input_shape"),
        ("metrics", "latency_ms", "methodology"),
        ("graph", "graph_diff_hash"),
        ("metrics", "fidelity"),
    ],
)
def test_mandatory_evidence_is_enforced(path, tmp_path):
    report = copy.deepcopy(_valid_report())
    target = report
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]

    output = tmp_path / "report.json"
    with pytest.raises(EvidenceValidationError):
        write_evidence_report(report, output)
    assert not output.exists()


def test_report_is_schema_valid_and_deterministically_written(tmp_path):
    report = _valid_report()
    validate_evidence_report(report)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_evidence_report(report, first)
    write_evidence_report(report, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("\n")
    assert list(report) == ["schema_version", "run", "environment", "model", "optimization", "metrics", "graph", "artifacts"]
