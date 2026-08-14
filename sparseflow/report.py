"""Schema-validated evidence report assembly and atomic writing."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from sparseflow.benchmark import BenchmarkResult
from sparseflow.measurement import (
    MeasurementRecipe,
    MeasurementValidationError,
    derive_minimum_detectable_improvement,
    summarize_numeric_samples,
)
from sparseflow.serialization import atomic_write_json, sha256_json, stable_json_dumps


SCHEMA_VERSION = "0.4.0"
SUPPORTED_SCHEMA_VERSIONS = ("0.3.0", SCHEMA_VERSION)
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


def _evidence_equal(location: str, *values: Any) -> None:
    if any(value != values[0] for value in values[1:]):
        raise EvidenceValidationError(
            f"invalid evidence at {location}: compared values differ"
        )


def _validate_summary(
    location: str,
    summary: dict[str, Any],
    *,
    require_non_negative: bool,
) -> None:
    expected = summarize_numeric_samples(
        summary["samples"], require_non_negative=require_non_negative
    )
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected_value, rel_tol=1e-12, abs_tol=1e-12
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise EvidenceValidationError(
                f"invalid evidence at {location}.{key}: value is not derived from raw samples"
            )


def _validate_pair_evidence(
    location: str,
    evidence: dict[str, Any],
    recipe: MeasurementRecipe,
) -> None:
    pairs = evidence["pairs"]
    if len(pairs) != recipe.pair_repetitions:
        raise EvidenceValidationError(
            f"invalid evidence at {location}.pairs: pair count differs from recipe"
        )
    execution_order = evidence["execution_order"]
    if execution_order != [pair["execution_order"] for pair in pairs]:
        raise EvidenceValidationError(
            f"invalid evidence at {location}.execution_order: order differs from pairs"
        )
    for expected_index, pair in enumerate(pairs):
        baseline_values = pair["baseline_samples_ms"]
        comparison_values = pair["comparison_samples_ms"]
        if (
            pair["pair_index"] != expected_index
            or len(baseline_values) != recipe.measured_runs_per_side
            or len(comparison_values) != recipe.measured_runs_per_side
        ):
            raise EvidenceValidationError(
                f"invalid evidence at {location}.pairs.{expected_index}: "
                "pair index or per-side sample count differs from recipe"
            )
        baseline_mean = sum(baseline_values) / len(baseline_values)
        comparison_mean = sum(comparison_values) / len(comparison_values)
        difference = baseline_mean - comparison_mean
        improvement = difference / baseline_mean * 100.0
        expected_values = {
            "baseline_mean_ms": baseline_mean,
            "comparison_mean_ms": comparison_mean,
            "difference_ms": difference,
            "improvement_percent": improvement,
        }
        for field, expected_value in expected_values.items():
            if not math.isclose(
                pair[field], expected_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise EvidenceValidationError(
                    f"invalid evidence at {location}.pairs.{expected_index}.{field}: "
                    "value is not derived from raw samples"
                )
    expected_samples_per_side = (
        recipe.pair_repetitions * recipe.measured_runs_per_side
    )
    baseline_samples = [
        value for pair in pairs for value in pair["baseline_samples_ms"]
    ]
    comparison_samples = [
        value for pair in pairs for value in pair["comparison_samples_ms"]
    ]
    if len(baseline_samples) != expected_samples_per_side or len(
        comparison_samples
    ) != expected_samples_per_side:
        raise EvidenceValidationError(
            f"invalid evidence at {location}.pairs: measured sample count differs from recipe"
        )
    differences = [pair["difference_ms"] for pair in pairs]
    improvements = [pair["improvement_percent"] for pair in pairs]
    _evidence_equal(
        f"{location}.paired_differences_ms.samples",
        evidence["paired_differences_ms"]["samples"],
        differences,
    )
    _evidence_equal(
        f"{location}.paired_improvement_percent.samples",
        evidence["paired_improvement_percent"]["samples"],
        improvements,
    )
    _validate_summary(
        f"{location}.paired_differences_ms",
        evidence["paired_differences_ms"],
        require_non_negative=False,
    )
    _validate_summary(
        f"{location}.paired_improvement_percent",
        evidence["paired_improvement_percent"],
        require_non_negative=False,
    )
    if "baseline_latency_ms" in evidence:
        _evidence_equal(
            f"{location}.baseline_latency_ms.samples",
            evidence["baseline_latency_ms"]["samples"],
            baseline_samples,
        )
        _evidence_equal(
            f"{location}.comparison_latency_ms.samples",
            evidence["comparison_latency_ms"]["samples"],
            comparison_samples,
        )
        _validate_summary(
            f"{location}.baseline_latency_ms",
            evidence["baseline_latency_ms"],
            require_non_negative=True,
        )
        _validate_summary(
            f"{location}.comparison_latency_ms",
            evidence["comparison_latency_ms"],
            require_non_negative=True,
        )


def _validate_environment_drift(environment_drift: dict[str, Any]) -> None:
    if environment_drift["diagnostic_only"] is not True:
        raise EvidenceValidationError(
            "invalid evidence at environment_drift.diagnostic_only: must be true"
        )
    if environment_drift["used_to_adjust_latency"] is not False:
        raise EvidenceValidationError(
            "invalid evidence at environment_drift.used_to_adjust_latency: must be false"
        )
    telemetry_fields = (
        "process_cpu_utilization_percent",
        "total_cpu_utilization_percent",
        "available_memory_bytes",
        "cpu_frequency_mhz",
    )
    for phase in ("calibration", "comparison"):
        for snapshot in environment_drift[phase].get("snapshots", []):
            for field in telemetry_fields:
                reading = snapshot.get(field)
                if reading is None:
                    continue
                if reading.get("available") is False and reading.get("value") is not None:
                    raise EvidenceValidationError(
                        f"invalid evidence at environment_drift.{phase}.{field}: "
                        "unavailable telemetry must be null"
                    )
                if reading.get("available") is True and reading.get("value") is None:
                    raise EvidenceValidationError(
                        f"invalid evidence at environment_drift.{phase}.{field}: "
                        "available telemetry must have a value"
                    )


def _validate_measurement_evidence(report: dict[str, Any]) -> None:
    try:
        recipe = MeasurementRecipe.from_dict(report["measurement_recipe"])
    except (KeyError, MeasurementValidationError) as exc:
        raise EvidenceValidationError(
            f"invalid evidence at measurement_recipe: {exc}"
        ) from exc

    methodology = report["metrics"]["latency_ms"]["methodology"]
    configuration = report["configuration"]
    run = report["run"]
    calibration = report["latency_calibration"]
    paired = report["paired_statistics"]
    _evidence_equal(
        "measurement_recipe.recipe_hash",
        recipe.recipe_hash,
        calibration["measurement_recipe_hash"],
        paired["measurement_recipe_hash"],
    )
    comparisons = {
        "provider": (recipe.provider, methodology["provider"]),
        "timer": (recipe.timer, methodology["clock"]),
        "warmup_runs": (
            recipe.warmup_runs,
            methodology["warmup_runs"],
            configuration["warmup_runs"],
        ),
        "pair_repetitions": (
            recipe.pair_repetitions,
            methodology["pair_repetitions"],
            methodology["measured_runs"],
            configuration["measured_runs"],
            calibration["pair_repetitions"],
            paired["pair_repetitions"],
        ),
        "measured_runs_per_side": (
            recipe.measured_runs_per_side,
            methodology["measured_runs_per_side"],
            calibration["measured_runs_per_side"],
            paired["measured_runs_per_side"],
        ),
        "intra_op_threads": (
            recipe.intra_op_threads,
            methodology["intra_op_threads"],
            configuration["intra_op_threads"],
        ),
        "inter_op_threads": (
            recipe.inter_op_threads,
            methodology["inter_op_threads"],
            configuration["inter_op_threads"],
        ),
        "benchmark_mode": (recipe.benchmark_mode, methodology["execution_mode"]),
        "input_shape": (
            list(recipe.input_shape),
            methodology["input_shape"],
            configuration["input_shape"],
            run["input_shape"],
        ),
        "random_seed": (
            recipe.random_seed,
            methodology["random_seed"],
            configuration["seed"],
            run["seed"],
        ),
        "order_policy": (recipe.order_policy, methodology["order_policy"]),
    }
    for field, values in comparisons.items():
        _evidence_equal(f"measurement_recipe.{field}", *values)

    identity = calibration["artifact_identity"]
    if calibration["artifact_identity_verified"] is not True or identity["verified"] is not True:
        raise EvidenceValidationError(
            "invalid evidence at latency_calibration.artifact_identity_verified: must be true"
        )
    baseline_artifact = report["artifacts"]["baseline"]
    for side in ("baseline", "comparison"):
        for key in ("model_hash", "pytorch_sha256", "onnx_sha256"):
            _evidence_equal(
                f"latency_calibration.artifact_identity.{side}.{key}",
                identity[side][key],
                identity["baseline"][key],
                baseline_artifact[key],
            )
        _evidence_equal(
            f"latency_calibration.artifact_identity.{side}.graph_hash",
            identity[side]["graph_hash"],
            identity["baseline"]["graph_hash"],
            report["graph"]["baseline_structural_hash"],
        )

    _validate_pair_evidence("latency_calibration", calibration, recipe)
    _validate_pair_evidence("paired_statistics", paired, recipe)
    _evidence_equal(
        "metrics.latency_ms.baseline.samples",
        report["metrics"]["latency_ms"]["baseline"]["samples"],
        [value for pair in paired["pairs"] for value in pair["baseline_samples_ms"]],
    )
    _evidence_equal(
        "metrics.latency_ms.optimized.samples",
        report["metrics"]["latency_ms"]["optimized"]["samples"],
        [value for pair in paired["pairs"] for value in pair["comparison_samples_ms"]],
    )
    _validate_summary(
        "metrics.latency_ms.baseline",
        report["metrics"]["latency_ms"]["baseline"],
        require_non_negative=True,
    )
    _validate_summary(
        "metrics.latency_ms.optimized",
        report["metrics"]["latency_ms"]["optimized"],
        require_non_negative=True,
    )

    expected_mdi = derive_minimum_detectable_improvement(
        calibration["paired_improvement_percent"]["samples"],
        confidence_level=calibration["confidence_level"],
    )
    for key in (
        "interval_method",
        "confidence_level",
        "mdi_derivation",
        "minimum_detectable_improvement_percent",
    ):
        _evidence_equal(f"latency_calibration.{key}", calibration[key], expected_mdi[key])
    _evidence_equal(
        "latency_calibration.null_paired_improvement_percent",
        calibration["null_paired_improvement_percent"],
        expected_mdi["null_paired_improvement_percent"],
    )
    _validate_environment_drift(report["environment_drift"])


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
    if report.get("schema_version") == SCHEMA_VERSION:
        try:
            _validate_measurement_evidence(report)
        except KeyError as exc:
            raise EvidenceValidationError(
                f"invalid evidence at measurement: missing {exc.args[0]}"
            ) from exc


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
        "measurement_recipe_hash": result.measurement_recipe["recipe_hash"],
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
        "measurement_recipe": result.measurement_recipe,
        "latency_calibration": result.latency_calibration,
        "environment_drift": result.environment_drift,
        "paired_statistics": result.paired_statistics,
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
            (
                "Calibrated MDI: "
                f"{report.get('latency_calibration', {}).get('minimum_detectable_improvement_percent', 0.0):.4f}%"
            ),
            "Commercial speedup claim supported: False",
        ]
    )
    return "\n".join(lines)
