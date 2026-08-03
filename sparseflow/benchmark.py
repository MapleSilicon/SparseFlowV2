"""SparseFlow V0 ONNX Runtime CPU evidence measurement workflow."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from sparseflow.audit import set_determinism
from sparseflow.compute import count_compute
from sparseflow.config import BenchmarkConfig
from sparseflow.graph_diff import (
    fx_graph_text,
    graph_changes_as_dict,
    graph_changes_hash,
    model_state_hash,
    structural_hash,
)
from sparseflow.passes import OptimizationPass
from sparseflow.serialization import sha256_bytes, sha256_file, sha256_json


MICROBENCHMARK_P50_THRESHOLD_MS = 0.1
NOISE_COEFFICIENT_OF_VARIATION_THRESHOLD = 0.10


@dataclass(frozen=True)
class BenchmarkResult:
    model_identifier: str
    architecture_identifier: str
    benchmark_config: dict[str, Any]
    resolved_configuration: dict[str, Any]
    optimization: dict[str, Any]
    baseline: dict[str, Any]
    optimized: dict[str, Any]
    latency_methodology: dict[str, Any]
    latency_interpretation: dict[str, Any]
    fidelity: dict[str, Any]
    graph: dict[str, Any]
    artifacts: dict[str, Any]


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def serialize_state_dict(model: nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def summarize_latency_samples(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one latency sample is required")
    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latency samples must be finite and non-negative")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    coefficient_of_variation = standard_deviation / mean if mean > 0 else 0.0
    normalized_samples = [float(value) for value in values]
    return {
        "samples": normalized_samples,
        "samples_sha256": sha256_json(normalized_samples),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": float(coefficient_of_variation),
    }


def interpret_latency(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if baseline["p50"] < MICROBENCHMARK_P50_THRESHOLD_MS:
        reasons.append("baseline p50 is below 0.1 ms")
    if baseline["coefficient_of_variation"] > NOISE_COEFFICIENT_OF_VARIATION_THRESHOLD:
        reasons.append("baseline coefficient of variation exceeds 0.10")
    if optimized["coefficient_of_variation"] > NOISE_COEFFICIENT_OF_VARIATION_THRESHOLD:
        reasons.append("optimized coefficient of variation exceeds 0.10")
    return {
        "noise_sensitive": bool(reasons),
        "reasons": reasons,
        "commercial_speedup_claim_supported": False,
        "policy": (
            "SparseFlow V0 does not infer commercial speedup from structural "
            "reductions or a single latency run."
        ),
    }


def export_onnx(
    model: nn.Module,
    input_shape: tuple[int, ...],
    path: str | Path,
    opset: int,
) -> dict[str, Any]:
    destination = Path(path)
    example = torch.zeros(input_shape, dtype=torch.float32)
    torch.onnx.export(
        model.eval(),
        example,
        str(destination),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(destination))
    onnx.checker.check_model(onnx_model)
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "validated": True,
        "opset": opset,
    }


def _create_ort_session(onnx_path: Path, config: BenchmarkConfig) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = config.threads
    options.inter_op_num_threads = config.threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"unexpected ONNX Runtime providers: {session.get_providers()}")
    return session


def validate_pytorch_onnx(
    model: nn.Module,
    onnx_path: Path,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Compare deterministic PyTorch and ONNX Runtime outputs."""

    generator = np.random.default_rng(config.seed)
    inputs = generator.standard_normal(config.input_shape).astype(np.float32)
    session = _create_ort_session(onnx_path, config)
    input_name = session.get_inputs()[0].name
    with torch.no_grad():
        pytorch_output = model.eval()(torch.from_numpy(inputs)).detach().cpu().numpy()
    onnx_output = session.run(None, {input_name: inputs})[0]

    shapes_match = pytorch_output.shape == onnx_output.shape
    difference = onnx_output - pytorch_output
    absolute_max = float(np.max(np.abs(difference)))
    denominator = max(float(np.linalg.norm(pytorch_output)), 1e-12)
    relative_l2 = float(np.linalg.norm(difference) / denominator)
    finite = bool(np.isfinite(pytorch_output).all() and np.isfinite(onnx_output).all())
    absolute_tolerance = 1e-4
    relative_tolerance = 1e-4
    return {
        "provider": "CPUExecutionProvider",
        "pytorch_output_shape": list(pytorch_output.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "output_shape_matches": shapes_match,
        "absolute_error_max": absolute_max,
        "relative_l2_error": relative_l2,
        "absolute_tolerance": absolute_tolerance,
        "relative_l2_tolerance": relative_tolerance,
        "outputs_finite": finite,
        "passed": (
            shapes_match
            and finite
            and absolute_max <= absolute_tolerance
            and relative_l2 <= relative_tolerance
        ),
    }


def _latency_samples(
    onnx_path: Path,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    session = _create_ort_session(onnx_path, config)

    generator = np.random.default_rng(config.seed)
    inputs = generator.standard_normal(config.input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name
    feed = {input_name: inputs}

    for _ in range(config.warmup_runs):
        session.run(None, feed)

    samples: list[float] = []
    for _ in range(config.measured_runs):
        started = time.perf_counter_ns()
        session.run(None, feed)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)

    return summarize_latency_samples(samples)


def _measure_model(
    model: nn.Module,
    tag: str,
    directory: Path,
    config: BenchmarkConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_payload = serialize_state_dict(model)
    onnx_path = directory / f"{tag}.onnx"
    onnx_artifact = export_onnx(model, config.input_shape, onnx_path, config.onnx_opset)
    metrics = {
        "parameter_count": count_parameters(model),
        "trainable_parameter_count": count_trainable_parameters(model),
        "pytorch_serialized_size_bytes": len(state_payload),
        "onnx_serialized_size_bytes": onnx_artifact["size_bytes"],
        "latency_ms": _latency_samples(onnx_path, config),
        "compute": count_compute(model, config.input_shape),
    }
    artifacts = {
        "model_hash": model_state_hash(model),
        "pytorch_sha256": sha256_bytes(state_payload),
        "onnx_sha256": onnx_artifact["sha256"],
        "onnx_validated": onnx_artifact["validated"],
        "onnx_opset": onnx_artifact["opset"],
        "pytorch_onnx_validation": validate_pytorch_onnx(
            model,
            onnx_path,
            config,
        ),
    }
    return metrics, artifacts


def _fidelity_proxy(
    baseline: nn.Module,
    optimized: nn.Module,
    config: BenchmarkConfig,
    require_equivalence: bool,
) -> dict[str, Any]:
    shape = (config.fidelity_batch_size, *config.input_shape[1:])
    generator = torch.Generator().manual_seed(config.seed)
    inputs = torch.randn(shape, generator=generator)
    with torch.no_grad():
        expected = baseline.eval()(inputs)
        actual = optimized.eval()(inputs)

    difference = actual - expected
    absolute_max = float(difference.abs().max().item())
    absolute_mean = float(difference.abs().mean().item())
    denominator = max(float(torch.linalg.vector_norm(expected).item()), 1e-12)
    relative_l2 = float(torch.linalg.vector_norm(difference).item() / denominator)
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())

    if require_equivalence:
        tolerance: dict[str, float | None] = {
            "absolute_max": 1e-6,
            "relative_l2_max": 1e-5,
        }
        passed = finite and absolute_max <= 1e-6 and relative_l2 <= 1e-5
    else:
        tolerance = {"absolute_max": None, "relative_l2_max": 1.0}
        passed = finite and relative_l2 <= 1.0

    return {
        "label": "untrained_reference_model_output_fidelity_proxy",
        "method": "deterministic random-input output tensor comparison",
        "batch_size": config.fidelity_batch_size,
        "absolute_error_max": absolute_max,
        "absolute_error_mean": absolute_mean,
        "relative_l2_error": relative_l2,
        "tolerance": tolerance,
        "outputs_finite": finite,
        "passed": passed,
        "is_task_accuracy": False,
    }


def run_benchmark(
    model: nn.Module,
    optimization_pass: OptimizationPass,
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    benchmark_config = config or BenchmarkConfig()
    set_determinism(benchmark_config.seed)
    baseline = model.eval()
    optimization_result = optimization_pass.apply(baseline)
    optimized = optimization_result.model.eval()

    with tempfile.TemporaryDirectory(prefix="sparseflow-benchmark-") as temporary_directory:
        directory = Path(temporary_directory)
        baseline_metrics, baseline_artifacts = _measure_model(
            baseline, "baseline", directory, benchmark_config
        )
        optimized_metrics, optimized_artifacts = _measure_model(
            optimized, "optimized", directory, benchmark_config
        )

    changes = optimization_result.graph_changes
    pass_configuration = optimization_pass.config()
    fidelity = _fidelity_proxy(
        baseline,
        optimized,
        benchmark_config,
        require_equivalence=(
            optimization_pass.name == "noop"
            or float(pass_configuration.get("ratio", -1.0)) == 0.0
        ),
    )
    baseline_structural_hash = structural_hash(baseline)
    optimized_structural_hash = structural_hash(optimized)
    metadata = dict(optimization_result.metadata)
    if metadata.get("gate") == "resnet18_gate1_dependency_analysis":
        zero_ratio_checks = {
            "model_hashes_identical": (
                baseline_artifacts["model_hash"] == optimized_artifacts["model_hash"]
            ),
            "pytorch_artifact_hashes_identical": (
                baseline_artifacts["pytorch_sha256"]
                == optimized_artifacts["pytorch_sha256"]
            ),
            "onnx_artifact_hashes_identical": (
                baseline_artifacts["onnx_sha256"]
                == optimized_artifacts["onnx_sha256"]
            ),
            "structural_hashes_identical": (
                baseline_structural_hash == optimized_structural_hash
            ),
            "parameter_counts_identical": (
                baseline_metrics["parameter_count"] == optimized_metrics["parameter_count"]
            ),
            "macs_identical": (
                baseline_metrics["compute"]["macs"]
                == optimized_metrics["compute"]["macs"]
            ),
            "flops_identical": (
                baseline_metrics["compute"]["flops"]
                == optimized_metrics["compute"]["flops"]
            ),
            "graph_changes_empty": not changes,
            "outputs_identical": (
                fidelity["absolute_error_max"] <= 1e-6
                and fidelity["relative_l2_error"] <= 1e-5
            ),
            "onnx_validation_passed": (
                baseline_artifacts["onnx_validated"]
                and optimized_artifacts["onnx_validated"]
            ),
            "pytorch_onnx_validation_passed": (
                baseline_artifacts["pytorch_onnx_validation"]["passed"]
                and optimized_artifacts["pytorch_onnx_validation"]["passed"]
            ),
        }
        zero_ratio_checks["no_tensors_changed"] = zero_ratio_checks[
            "model_hashes_identical"
        ] and zero_ratio_checks["pytorch_artifact_hashes_identical"]
        zero_ratio_checks["no_channels_removed"] = (
            zero_ratio_checks["structural_hashes_identical"]
            and zero_ratio_checks["parameter_counts_identical"]
        )
        metadata["zero_ratio_validation"] = {
            "validation_type": "dependency_analysis_without_transformation",
            "ratio": 0.0,
            **zero_ratio_checks,
            "passed": all(zero_ratio_checks.values()),
        }
    resolved_configuration = {
        "preset": benchmark_config.preset_name,
        "model_identifier": getattr(model, "identifier", model.__class__.__name__),
        "seed": benchmark_config.seed,
        "input_shape": list(benchmark_config.input_shape),
        "pass_name": optimization_pass.name.replace("_", "-"),
        "pruning_ratio": float(pass_configuration.get("ratio", 0.0)),
        "warmup_runs": benchmark_config.warmup_runs,
        "measured_runs": benchmark_config.measured_runs,
        "intra_op_threads": benchmark_config.threads,
        "inter_op_threads": benchmark_config.threads,
        "output_report_filename": benchmark_config.output_report_filename,
    }
    latency_interpretation = interpret_latency(
        baseline_metrics["latency_ms"], optimized_metrics["latency_ms"]
    )
    return BenchmarkResult(
        model_identifier=getattr(model, "identifier", model.__class__.__name__),
        architecture_identifier=getattr(
            model,
            "architecture_identifier",
            getattr(model, "identifier", model.__class__.__name__),
        ),
        benchmark_config=benchmark_config.as_dict(),
        resolved_configuration=resolved_configuration,
        optimization={
            "pass_name": optimization_pass.name,
            "config": pass_configuration,
            "metadata": metadata,
        },
        baseline=baseline_metrics,
        optimized=optimized_metrics,
        latency_methodology={
            "provider": "CPUExecutionProvider",
            "warmup_runs": benchmark_config.warmup_runs,
            "measured_runs": benchmark_config.measured_runs,
            "intra_op_threads": benchmark_config.threads,
            "inter_op_threads": benchmark_config.threads,
            "execution_mode": "ORT_SEQUENTIAL",
            "clock": "time.perf_counter_ns",
            "input_shape": list(benchmark_config.input_shape),
        },
        latency_interpretation=latency_interpretation,
        fidelity=fidelity,
        graph={
            "changes": graph_changes_as_dict(changes),
            "graph_diff_hash": graph_changes_hash(changes),
            "baseline_structural_hash": baseline_structural_hash,
            "optimized_structural_hash": optimized_structural_hash,
            "baseline_fx_graph": fx_graph_text(baseline),
            "optimized_fx_graph": fx_graph_text(optimized),
        },
        artifacts={"baseline": baseline_artifacts, "optimized": optimized_artifacts},
    )
