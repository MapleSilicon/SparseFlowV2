"""Schema-validated evidence report assembly and atomic writing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from sparseflow.benchmark import BenchmarkResult
from sparseflow.serialization import atomic_write_json, sha256_json, stable_json_dumps


SCHEMA_VERSION = "0.3.0"
V0_POSITIONING = (
    "SparseFlow V0: reproducible optimization and evidence harness, validated "
    "with a deterministic physical channel-pruning demonstration."
)
V1_MILESTONE = (
    "SparseFlow V1: ResNet-18 dependency-aware physical channel pruning, ONNX "
    "Runtime CPU deployment, and measured accuracy/latency evidence on a real "
    "evaluation workload."
)
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "evidence-report.schema.json"


class EvidenceValidationError(RuntimeError):
    """Raised when mandatory official evidence is missing or invalid."""


def _reduction_percent(baseline: float, optimized: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - optimized) / baseline * 100.0


def _load_schema(schema_path: str | Path) -> dict[str, Any]:
    import json

    path = Path(schema_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceValidationError(f"unable to load evidence schema: {path}") from exc


def validate_evidence_report(
    report: dict[str, Any],
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> None:
    try:
        stable_json_dumps(report)
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError(f"report is not deterministic JSON: {exc}") from exc

    schema = _load_schema(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise EvidenceValidationError(f"invalid checked-in report schema: {exc}") from exc

    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "report"
        if error.validator == "required":
            missing = error.message.split("'")[1] if "'" in error.message else error.message
            location = f"{location}.{missing}"
        raise EvidenceValidationError(f"invalid evidence at {location}: {error.message}")


def build_evidence_report(
    result: BenchmarkResult,
    audit: dict[str, Any],
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    config = result.benchmark_config
    deterministic_run_inputs = {
        "commit_hash": audit.get("git", {}).get("commit_hash"),
        "model_identifier": result.model_identifier,
        "seed": config["seed"],
        "input_shape": config["input_shape"],
        "optimization": result.optimization,
        "configuration": result.resolved_configuration,
    }
    run_identifier = f"run-{sha256_json(deterministic_run_inputs)[:16]}"
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )

    baseline = result.baseline
    optimized = result.optimized
    baseline_compute = baseline["compute"]
    optimized_compute = optimized["compute"]
    gate1 = (
        result.optimization["metadata"].get("gate")
        == "resnet18_gate1_dependency_analysis"
    )
    product = {
        "current_milestone": "SparseFlow V0",
        "positioning": V0_POSITIONING,
        "demonstration_type": "physical_channel_pruning_mechanism_demonstration",
        "commercial_benchmark": False,
        "next_milestone": V1_MILESTONE,
    }
    if gate1:
        product = {
            **product,
            "development_milestone": "SparseFlow V1 Gate 1",
            "demonstration_type": (
                "resnet18_dependency_analysis_zero_ratio_validation"
            ),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "product": product,
        "run": {
            "id": run_identifier,
            "timestamp_utc": timestamp,
            "seed": config["seed"],
            "input_shape": config["input_shape"],
            "repository": audit.get("repository"),
            "git": audit.get("git"),
        },
        "environment": {
            "hardware": audit.get("hardware"),
            "software": audit.get("software"),
            "process": audit.get("process"),
        },
        "model": {
            "identifier": result.model_identifier,
            "architecture_identifier": result.architecture_identifier,
            "fidelity_is_proxy": True,
        },
        "configuration": result.resolved_configuration,
        "optimization": result.optimization,
        "metrics": {
            "parameters": {
                "baseline": baseline["parameter_count"],
                "optimized": optimized["parameter_count"],
                "baseline_trainable": baseline["trainable_parameter_count"],
                "optimized_trainable": optimized["trainable_parameter_count"],
                "reduction_percent": _reduction_percent(
                    baseline["parameter_count"], optimized["parameter_count"]
                ),
            },
            "serialized_size_bytes": {
                "pytorch_state_dict": {
                    "baseline": baseline["pytorch_serialized_size_bytes"],
                    "optimized": optimized["pytorch_serialized_size_bytes"],
                    "reduction_percent": _reduction_percent(
                        baseline["pytorch_serialized_size_bytes"],
                        optimized["pytorch_serialized_size_bytes"],
                    ),
                },
                "onnx": {
                    "baseline": baseline["onnx_serialized_size_bytes"],
                    "optimized": optimized["onnx_serialized_size_bytes"],
                    "reduction_percent": _reduction_percent(
                        baseline["onnx_serialized_size_bytes"],
                        optimized["onnx_serialized_size_bytes"],
                    ),
                },
            },
            "latency_ms": {
                "methodology": result.latency_methodology,
                "baseline": baseline["latency_ms"],
                "optimized": optimized["latency_ms"],
            },
            "compute": {
                "convention": "1 MAC = 2 FLOPs",
                "implementation": baseline_compute["implementation"],
                "input_shape": config["input_shape"],
                "baseline": {
                    "macs": baseline_compute["macs"],
                    "flops": baseline_compute["flops"],
                    "unsupported_operators": baseline_compute["unsupported_operators"],
                    "count_is_complete": baseline_compute["count_is_complete"],
                },
                "optimized": {
                    "macs": optimized_compute["macs"],
                    "flops": optimized_compute["flops"],
                    "unsupported_operators": optimized_compute["unsupported_operators"],
                    "count_is_complete": optimized_compute["count_is_complete"],
                },
                "mac_reduction_percent": _reduction_percent(
                    baseline_compute["macs"], optimized_compute["macs"]
                ),
                "flop_reduction_percent": _reduction_percent(
                    baseline_compute["flops"], optimized_compute["flops"]
                ),
            },
            "fidelity": result.fidelity,
        },
        "latency_interpretation": result.latency_interpretation,
        "graph": result.graph,
        "artifacts": result.artifacts,
    }
    validate_evidence_report(report, schema_path)
    return report


def write_evidence_report(
    report: dict[str, Any],
    path: str | Path,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> None:
    validate_evidence_report(report, schema_path)
    atomic_write_json(report, path)


def render_result_table(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    parameters = metrics["parameters"]
    sizes = metrics["serialized_size_bytes"]["onnx"]
    latency = metrics["latency_ms"]
    compute = metrics["compute"]
    rows = [
        ("Parameters", parameters["baseline"], parameters["optimized"], parameters["reduction_percent"]),
        ("ONNX bytes", sizes["baseline"], sizes["optimized"], sizes["reduction_percent"]),
        ("MACs", compute["baseline"]["macs"], compute["optimized"]["macs"], compute["mac_reduction_percent"]),
        ("FLOPs", compute["baseline"]["flops"], compute["optimized"]["flops"], compute["flop_reduction_percent"]),
        ("Latency p50 ms", latency["baseline"]["p50"], latency["optimized"]["p50"], _reduction_percent(latency["baseline"]["p50"], latency["optimized"]["p50"])),
        ("Latency p95 ms", latency["baseline"]["p95"], latency["optimized"]["p95"], _reduction_percent(latency["baseline"]["p95"], latency["optimized"]["p95"])),
    ]
    lines = [
        (
            "SparseFlow V1 Gate 1 Evidence Benchmark"
            if report["product"].get("development_milestone")
            else "SparseFlow V0 Evidence Benchmark"
        ),
        f"Pass: {report['optimization']['pass_name']}",
        "-" * 72,
        f"{'Metric':<22}{'Baseline':>16}{'Optimized':>16}{'Reduction':>14}",
        "-" * 72,
    ]
    for label, baseline, optimized, reduction in rows:
        if isinstance(baseline, float):
            baseline_text = f"{baseline:.4f}"
            optimized_text = f"{optimized:.4f}"
        else:
            baseline_text = f"{baseline:,}"
            optimized_text = f"{optimized:,}"
        lines.append(
            f"{label:<22}{baseline_text:>16}{optimized_text:>16}{reduction:>13.2f}%"
        )
    lines.extend(
        [
            "-" * 72,
            f"Fidelity proxy passed: {metrics['fidelity']['passed']}",
            "Fidelity is an untrained-model output proxy, not task accuracy.",
            f"Latency noise-sensitive: {report['latency_interpretation']['noise_sensitive']}",
            "Commercial speedup claim supported: False",
        ]
    )
    return "\n".join(lines)
