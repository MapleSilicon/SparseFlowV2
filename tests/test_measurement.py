from dataclasses import replace

import pytest

from sparseflow.measurement import (
    MeasurementRecipe,
    MeasurementValidationError,
    capture_environment_snapshot,
    derive_minimum_detectable_improvement,
    randomized_pair_order,
    run_paired_onnx_benchmark,
    validate_calibration_identity,
    validate_recipe_compatibility,
)
from sparseflow.serialization import stable_json_dumps


def _recipe(**changes):
    values = {
        "warmup_runs": 2,
        "pair_repetitions": 8,
        "measured_runs_per_side": 1,
        "provider": "CPUExecutionProvider",
        "timer": "time.perf_counter_ns",
        "random_seed": 1234,
        "order_policy": "seeded_randomized_pair_order",
        "input_shape": (1, 3, 16, 16),
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "benchmark_mode": "ORT_SEQUENTIAL",
    }
    values.update(changes)
    return MeasurementRecipe(**values)


def _artifacts(**changes):
    values = {
        "model_hash": "a" * 64,
        "pytorch_sha256": "b" * 64,
        "onnx_sha256": "c" * 64,
    }
    values.update(changes)
    return values


def test_recipe_hash_is_deterministic_and_sensitive_to_recipe_fields():
    first = _recipe()
    second = _recipe()
    changed = _recipe(warmup_runs=3)

    assert first.recipe_hash == second.recipe_hash
    assert first.as_dict() == second.as_dict()
    assert changed.recipe_hash != first.recipe_hash
    assert MeasurementRecipe.from_dict(first.as_dict()) == first


def test_recipe_from_dict_rejects_tampered_hash():
    payload = _recipe().as_dict()
    payload["recipe_hash"] = "0" * 64

    with pytest.raises(MeasurementValidationError, match="recipe_hash"):
        MeasurementRecipe.from_dict(payload)


def test_randomized_pair_order_is_seeded_and_deterministic():
    first = randomized_pair_order(_recipe(random_seed=1234))
    second = randomized_pair_order(_recipe(random_seed=1234))
    changed = randomized_pair_order(_recipe(random_seed=5678))

    assert first == second
    assert first != changed
    assert len(first) == 8
    assert set(first) == {
        "baseline_then_comparison",
        "comparison_then_baseline",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "AlternateExecutionProvider"),
        ("timer", "alternate_timer"),
        ("input_shape", (1, 3, 32, 32)),
        ("intra_op_threads", 2),
        ("inter_op_threads", 2),
        ("pair_repetitions", 9),
        ("measured_runs_per_side", 2),
        ("warmup_runs", 3),
        ("benchmark_mode", "ORT_PARALLEL"),
        ("random_seed", 7),
        ("order_policy", "alternate_policy"),
    ],
)
def test_recipe_compatibility_rejects_each_measurement_mismatch(field, value):
    baseline = _recipe()
    comparison = replace(baseline, **{field: value})

    with pytest.raises(MeasurementValidationError, match=field):
        validate_recipe_compatibility(baseline, comparison)


@pytest.mark.parametrize(
    ("artifact_key", "graph_side"),
    [
        ("model_hash", None),
        ("pytorch_sha256", None),
        ("onnx_sha256", None),
        (None, "comparison"),
    ],
)
def test_calibration_rejects_nonidentical_artifacts(artifact_key, graph_side):
    baseline = _artifacts()
    comparison = _artifacts()
    baseline_graph_hash = "d" * 64
    comparison_graph_hash = baseline_graph_hash
    if artifact_key:
        comparison[artifact_key] = "e" * 64
    if graph_side:
        comparison_graph_hash = "f" * 64

    with pytest.raises(MeasurementValidationError, match="identical"):
        validate_calibration_identity(
            baseline,
            comparison,
            baseline_graph_hash,
            comparison_graph_hash,
        )


def test_calibration_identity_records_verified_hashes():
    identity = validate_calibration_identity(
        _artifacts(),
        _artifacts(),
        "d" * 64,
        "d" * 64,
    )

    assert identity["verified"] is True
    assert identity["baseline"] == identity["comparison"]
    assert identity["baseline"]["model_hash"] == "a" * 64
    assert identity["baseline"]["graph_hash"] == "d" * 64


def test_mdi_is_derived_from_two_sided_empirical_null_interval():
    result = derive_minimum_detectable_improvement(
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
        confidence_level=0.80,
    )

    derivation = result["mdi_derivation"]
    assert result["interval_method"] == "two_sided_empirical_percentile_interval"
    assert result["confidence_level"] == 0.80
    assert derivation["lower_percentile_percent"] == pytest.approx(10.0)
    assert derivation["upper_percentile_percent"] == pytest.approx(90.0)
    assert result["minimum_detectable_improvement_percent"] == pytest.approx(
        max(
            abs(derivation["lower_bound_percent"]),
            abs(derivation["upper_bound_percent"]),
        )
    )


def test_unavailable_environment_telemetry_is_null_and_serializable(monkeypatch):
    monkeypatch.setattr("sparseflow.measurement._psutil", None)

    snapshot = capture_environment_snapshot("baseline_then_comparison")

    for field in (
        "process_cpu_utilization_percent",
        "total_cpu_utilization_percent",
        "available_memory_bytes",
        "cpu_frequency_mhz",
    ):
        assert snapshot[field] == {"value": None, "available": False}
    stable_json_dumps(snapshot)


def test_telemetry_collection_failures_do_not_block_measurement_metadata(monkeypatch):
    class BrokenPsutil:
        @staticmethod
        def Process():
            raise RuntimeError("process telemetry unavailable")

        @staticmethod
        def cpu_freq():
            raise RuntimeError("frequency telemetry unavailable")

        @staticmethod
        def cpu_percent(interval=None):
            raise RuntimeError("cpu telemetry unavailable")

        @staticmethod
        def virtual_memory():
            raise RuntimeError("memory telemetry unavailable")

    monkeypatch.setattr("sparseflow.measurement._psutil", BrokenPsutil)

    snapshot = capture_environment_snapshot("after_measurement")

    assert snapshot["process_cpu_utilization_percent"]["value"] is None
    assert snapshot["total_cpu_utilization_percent"]["available"] is False
    assert snapshot["available_memory_bytes"]["available"] is False
    assert snapshot["cpu_frequency_mhz"]["available"] is False


def test_input_shape_mismatch_fails_before_runtime_session_creation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "sparseflow.measurement._create_ort_session",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(MeasurementValidationError, match="input_shape"):
        run_paired_onnx_benchmark(
            "baseline.onnx",
            "comparison.onnx",
            _recipe(input_shape=(1, 3, 16, 16)),
            _recipe(input_shape=(1, 3, 32, 32)),
        )

    assert calls == []
