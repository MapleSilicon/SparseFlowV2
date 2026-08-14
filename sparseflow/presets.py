"""Structured benchmark presets for SparseFlow V0 and V1 evidence gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkPreset:
    name: str
    model_identifier: str
    seed: int
    input_shape: tuple[int, ...]
    pass_name: str
    pruning_ratio: float
    warmup_runs: int
    measured_runs: int
    output_report_filename: str

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["input_shape"] = list(self.input_shape)
        return values


@dataclass(frozen=True)
class ResolvedRunConfiguration:
    preset_name: str
    model_identifier: str
    seed: int
    input_shape: tuple[int, ...]
    pass_name: str
    pruning_ratio: float
    warmup_runs: int
    measured_runs: int
    threads: int
    output_path: Path


_PRESETS = {
    "micro-demo": BenchmarkPreset(
        name="micro-demo",
        model_identifier="reference_cnn_v1",
        seed=1234,
        input_shape=(1, 3, 16, 16),
        pass_name="channel-prune",
        pruning_ratio=0.375,
        warmup_runs=5,
        measured_runs=30,
        output_report_filename="report.json",
    ),
    "noop-control": BenchmarkPreset(
        name="noop-control",
        model_identifier="reference_cnn_v1",
        seed=1234,
        input_shape=(1, 3, 16, 16),
        pass_name="noop",
        pruning_ratio=0.0,
        warmup_runs=5,
        measured_runs=30,
        output_report_filename="report-noop.json",
    ),
    "resnet18-gate1": BenchmarkPreset(
        name="resnet18-gate1",
        model_identifier="resnet18-reference",
        seed=1234,
        input_shape=(1, 3, 64, 64),
        pass_name="channel-prune",
        pruning_ratio=0.0,
        warmup_runs=1,
        measured_runs=3,
        output_report_filename="report-resnet18-gate1.json",
    ),
    "resnet18-gate2": BenchmarkPreset(
        name="resnet18-gate2",
        model_identifier="resnet18-reference",
        seed=1234,
        input_shape=(1, 3, 64, 64),
        pass_name="channel-prune",
        pruning_ratio=0.125,
        warmup_runs=2,
        measured_runs=5,
        output_report_filename="report-resnet18-gate2.json",
    ),
}


def preset_names() -> tuple[str, ...]:
    return tuple(_PRESETS)


def get_preset(name: str) -> BenchmarkPreset:
    try:
        return _PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(preset_names())
        raise ValueError(f"unknown benchmark preset {name!r}; choose from: {choices}") from exc
