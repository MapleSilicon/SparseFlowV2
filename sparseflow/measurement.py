"""Deterministic paired latency measurement and calibration contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
import math
import random
import time
from typing import Any, Callable

import numpy as np
import onnxruntime as ort

from sparseflow.serialization import sha256_json

try:  # Diagnostic telemetry must never be required for valid evidence.
    import psutil as _psutil
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests
    _psutil = None


class MeasurementValidationError(ValueError):
    """Raised when latency evidence cannot be compared safely."""


@dataclass(frozen=True)
class MeasurementRecipe:
    """Complete, hashable recipe for one side of a paired measurement."""

    warmup_runs: int
    pair_repetitions: int
    measured_runs_per_side: int
    provider: str
    timer: str
    random_seed: int
    order_policy: str
    input_shape: tuple[int, ...]
    intra_op_threads: int
    inter_op_threads: int
    benchmark_mode: str

    def __post_init__(self) -> None:
        positive_fields = {
            "pair_repetitions": self.pair_repetitions,
            "measured_runs_per_side": self.measured_runs_per_side,
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
        }
        for name, value in positive_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise MeasurementValidationError(f"{name} must be a positive integer")
        if (
            not isinstance(self.warmup_runs, int)
            or isinstance(self.warmup_runs, bool)
            or self.warmup_runs < 0
        ):
            raise MeasurementValidationError("warmup_runs must be a non-negative integer")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise MeasurementValidationError("random_seed must be an integer")
        if (
            not isinstance(self.input_shape, tuple)
            or not self.input_shape
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in self.input_shape
            )
        ):
            raise MeasurementValidationError(
                "input_shape must be a non-empty tuple of positive integers"
            )
        for name in ("provider", "timer", "order_policy", "benchmark_mode"):
            if not getattr(self, name):
                raise MeasurementValidationError(f"{name} must be non-empty")

    def hash_payload(self) -> dict[str, Any]:
        return {
            "warmup_runs": self.warmup_runs,
            "pair_repetitions": self.pair_repetitions,
            "measured_runs_per_side": self.measured_runs_per_side,
            "provider": self.provider,
            "timer": self.timer,
            "random_seed": self.random_seed,
            "order_policy": self.order_policy,
            "input_shape": list(self.input_shape),
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "benchmark_mode": self.benchmark_mode,
        }

    @property
    def recipe_hash(self) -> str:
        return sha256_json(self.hash_payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self.hash_payload(), "recipe_hash": self.recipe_hash}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeasurementRecipe:
        if not isinstance(payload, dict):
            raise MeasurementValidationError("measurement recipe must be an object")
        field_names = {field.name for field in fields(cls)}
        expected_keys = field_names | {"recipe_hash"}
        actual_keys = set(payload)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise MeasurementValidationError(
                f"measurement recipe fields mismatch; missing={missing}, extra={extra}"
            )
        values = {name: payload[name] for name in field_names}
        input_shape = values.get("input_shape")
        if isinstance(input_shape, list):
            values["input_shape"] = tuple(input_shape)
        recipe = cls(**values)
        if payload["recipe_hash"] != recipe.recipe_hash:
            raise MeasurementValidationError(
                "recipe_hash does not match the serialized measurement recipe"
            )
        return recipe


def validate_recipe_compatibility(
    baseline: MeasurementRecipe,
    comparison: MeasurementRecipe,
) -> None:
    """Fail closed unless every measurement-affecting recipe field matches."""

    for field in fields(MeasurementRecipe):
        name = field.name
        if getattr(baseline, name) != getattr(comparison, name):
            raise MeasurementValidationError(
                f"measurement recipe mismatch at {name}: "
                f"{getattr(baseline, name)!r} != {getattr(comparison, name)!r}"
            )
    if baseline.recipe_hash != comparison.recipe_hash:
        raise MeasurementValidationError("measurement recipe_hash mismatch")


def randomized_pair_order(recipe: MeasurementRecipe) -> list[str]:
    if recipe.order_policy != "seeded_randomized_pair_order":
        raise MeasurementValidationError(
            f"unsupported order_policy: {recipe.order_policy}"
        )
    generator = random.Random(recipe.random_seed)
    return [
        (
            "baseline_then_comparison"
            if generator.getrandbits(1) == 0
            else "comparison_then_baseline"
        )
        for _ in range(recipe.pair_repetitions)
    ]


def _available(value: int | float | None) -> dict[str, Any]:
    if value is None or isinstance(value, bool):
        return {"value": None, "available": False}
    numeric = float(value)
    if not math.isfinite(numeric):
        return {"value": None, "available": False}
    normalized: int | float = int(value) if isinstance(value, int) else numeric
    return {"value": normalized, "available": True}


def _read_telemetry(reader: Callable[[], int | float | None]) -> dict[str, Any]:
    try:
        return _available(reader())
    except Exception:
        # Telemetry is diagnostic and must never make latency evidence unavailable.
        return {"value": None, "available": False}


def capture_environment_snapshot(execution_order: str) -> dict[str, Any]:
    """Capture diagnostic-only process data, explicitly marking unavailable data."""

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if _psutil is None:
        unavailable = {"value": None, "available": False}
        return {
            "timestamp_utc": timestamp,
            "execution_order": execution_order,
            "process_cpu_utilization_percent": dict(unavailable),
            "total_cpu_utilization_percent": dict(unavailable),
            "available_memory_bytes": dict(unavailable),
            "cpu_frequency_mhz": dict(unavailable),
        }

    try:
        process = _psutil.Process()
    except Exception:
        process = None
    frequency = _read_telemetry(
        lambda: getattr(_psutil.cpu_freq(), "current", None)
    )
    return {
        "timestamp_utc": timestamp,
        "execution_order": execution_order,
        "process_cpu_utilization_percent": _read_telemetry(
            lambda: process.cpu_percent(interval=None) if process is not None else None
        ),
        "total_cpu_utilization_percent": _read_telemetry(
            lambda: _psutil.cpu_percent(interval=None)
        ),
        "available_memory_bytes": _read_telemetry(
            lambda: _psutil.virtual_memory().available
        ),
        "cpu_frequency_mhz": frequency,
    }


def summarize_numeric_samples(
    samples: list[float],
    *,
    require_non_negative: bool = False,
) -> dict[str, Any]:
    if not samples:
        raise MeasurementValidationError("at least one numeric sample is required")
    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(values).all():
        raise MeasurementValidationError("samples must be finite")
    if require_non_negative and np.any(values < 0):
        raise MeasurementValidationError("samples must be non-negative")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    coefficient_of_variation = standard_deviation / abs(mean) if mean != 0 else 0.0
    normalized = [float(value) for value in values]
    return {
        "samples": normalized,
        "samples_sha256": sha256_json(normalized),
        "p50": float(np.percentile(values, 50)),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": float(coefficient_of_variation),
    }


def _create_ort_session(
    onnx_path: str | Path,
    recipe: MeasurementRecipe,
) -> ort.InferenceSession:
    if recipe.provider != "CPUExecutionProvider":
        raise MeasurementValidationError(f"unsupported provider: {recipe.provider}")
    if recipe.benchmark_mode != "ORT_SEQUENTIAL":
        raise MeasurementValidationError(
            f"unsupported benchmark_mode: {recipe.benchmark_mode}"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = recipe.intra_op_threads
    options.inter_op_num_threads = recipe.inter_op_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=[recipe.provider],
    )
    if session.get_providers() != [recipe.provider]:
        raise MeasurementValidationError(
            f"unexpected ONNX Runtime providers: {session.get_providers()}"
        )
    return session


def _measure_invocation(
    session: ort.InferenceSession,
    feed: dict[str, np.ndarray],
    timer: str,
) -> float:
    if timer != "time.perf_counter_ns":
        raise MeasurementValidationError(f"unsupported timer: {timer}")
    started = time.perf_counter_ns()
    session.run(None, feed)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if elapsed_ms <= 0 or not math.isfinite(elapsed_ms):
        raise MeasurementValidationError("timer produced a non-positive latency")
    return elapsed_ms


def run_paired_onnx_benchmark(
    baseline_path: str | Path,
    comparison_path: str | Path,
    baseline_recipe: MeasurementRecipe,
    comparison_recipe: MeasurementRecipe,
) -> dict[str, Any]:
    """Measure baseline/comparison invocations in deterministic randomized pairs."""

    validate_recipe_compatibility(baseline_recipe, comparison_recipe)
    order = randomized_pair_order(baseline_recipe)
    baseline_session = _create_ort_session(baseline_path, baseline_recipe)
    comparison_session = _create_ort_session(comparison_path, comparison_recipe)

    generator = np.random.default_rng(baseline_recipe.random_seed)
    inputs = generator.standard_normal(baseline_recipe.input_shape).astype(np.float32)
    baseline_feed = {baseline_session.get_inputs()[0].name: inputs}
    comparison_feed = {comparison_session.get_inputs()[0].name: inputs}
    snapshots = [capture_environment_snapshot("before_measurement")]

    for _ in range(baseline_recipe.warmup_runs):
        baseline_session.run(None, baseline_feed)
        comparison_session.run(None, comparison_feed)

    baseline_samples: list[float] = []
    comparison_samples: list[float] = []
    pair_records: list[dict[str, Any]] = []
    paired_differences: list[float] = []
    paired_improvements: list[float] = []
    for pair_index, pair_order in enumerate(order):
        pair_baseline: list[float] = []
        pair_comparison: list[float] = []
        sides = (
            ("baseline", "comparison")
            if pair_order == "baseline_then_comparison"
            else ("comparison", "baseline")
        )
        for side in sides:
            session = baseline_session if side == "baseline" else comparison_session
            feed = baseline_feed if side == "baseline" else comparison_feed
            destination = pair_baseline if side == "baseline" else pair_comparison
            for _ in range(baseline_recipe.measured_runs_per_side):
                destination.append(
                    _measure_invocation(session, feed, baseline_recipe.timer)
                )
        baseline_samples.extend(pair_baseline)
        comparison_samples.extend(pair_comparison)
        baseline_mean = float(np.mean(pair_baseline))
        comparison_mean = float(np.mean(pair_comparison))
        difference = baseline_mean - comparison_mean
        improvement = difference / baseline_mean * 100.0
        paired_differences.append(difference)
        paired_improvements.append(improvement)
        pair_records.append(
            {
                "pair_index": pair_index,
                "execution_order": pair_order,
                "baseline_samples_ms": pair_baseline,
                "comparison_samples_ms": pair_comparison,
                "baseline_mean_ms": baseline_mean,
                "comparison_mean_ms": comparison_mean,
                "difference_ms": difference,
                "improvement_percent": improvement,
            }
        )

    snapshots.append(capture_environment_snapshot("after_measurement"))
    return {
        "recipe_hash": baseline_recipe.recipe_hash,
        "execution_order": order,
        "pairs": pair_records,
        "baseline": summarize_numeric_samples(
            baseline_samples, require_non_negative=True
        ),
        "comparison": summarize_numeric_samples(
            comparison_samples, require_non_negative=True
        ),
        "paired_differences_ms": summarize_numeric_samples(paired_differences),
        "paired_improvement_percent": summarize_numeric_samples(paired_improvements),
        "environment_drift": {
            "diagnostic_only": True,
            "used_to_adjust_latency": False,
            "snapshots": snapshots,
        },
    }


def validate_calibration_identity(
    baseline_artifacts: dict[str, Any],
    comparison_artifacts: dict[str, Any],
    baseline_graph_hash: str,
    comparison_graph_hash: str,
) -> dict[str, Any]:
    """Prove calibration measured two independently loaded identical artifacts."""

    baseline_identity: dict[str, Any] = {}
    comparison_identity: dict[str, Any] = {}
    for key in ("model_hash", "pytorch_sha256", "onnx_sha256"):
        baseline_value = baseline_artifacts.get(key)
        comparison_value = comparison_artifacts.get(key)
        if not baseline_value or baseline_value != comparison_value:
            raise MeasurementValidationError(
                f"calibration requires identical artifacts; {key} differs"
            )
        baseline_identity[key] = baseline_value
        comparison_identity[key] = comparison_value
    if not baseline_graph_hash or baseline_graph_hash != comparison_graph_hash:
        raise MeasurementValidationError(
            "calibration requires identical artifacts; graph hash differs"
        )
    baseline_identity["graph_hash"] = baseline_graph_hash
    comparison_identity["graph_hash"] = comparison_graph_hash
    return {
        "verified": True,
        "baseline": baseline_identity,
        "comparison": comparison_identity,
    }


def derive_minimum_detectable_improvement(
    null_improvement_percent: list[float],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Derive MDI as the largest absolute bound of a two-sided null interval."""

    if not 0 < confidence_level < 1:
        raise MeasurementValidationError("confidence_level must be between 0 and 1")
    summary = summarize_numeric_samples(null_improvement_percent)
    values = np.asarray(null_improvement_percent, dtype=np.float64)
    tail_percent = (1.0 - confidence_level) / 2.0 * 100.0
    lower = float(np.percentile(values, tail_percent))
    upper = float(np.percentile(values, 100.0 - tail_percent))
    mdi = max(abs(lower), abs(upper))
    return {
        "interval_method": "two_sided_empirical_percentile_interval",
        "confidence_level": confidence_level,
        "null_paired_improvement_percent": summary,
        "mdi_derivation": {
            "statistic": "maximum_absolute_bound_of_empirical_paired_improvement_interval",
            "lower_percentile_percent": tail_percent,
            "upper_percentile_percent": 100.0 - tail_percent,
            "lower_bound_percent": lower,
            "upper_bound_percent": upper,
            "value_percent": mdi,
        },
        "minimum_detectable_improvement_percent": mdi,
    }
