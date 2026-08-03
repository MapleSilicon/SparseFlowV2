"""Validated SparseFlow V0 benchmark configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    input_shape: tuple[int, ...] = (1, 3, 16, 16)
    seed: int = 1234
    warmup_runs: int = 5
    measured_runs: int = 30
    threads: int = 1
    onnx_opset: int = 17
    fidelity_batch_size: int = 8
    preset_name: str | None = None
    output_report_filename: str = "report.json"

    def __post_init__(self) -> None:
        if len(self.input_shape) != 4 or any(
            not isinstance(dimension, int) or dimension <= 0 for dimension in self.input_shape
        ):
            raise ValueError("input_shape must contain four positive integers")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.measured_runs <= 0:
            raise ValueError("measured_runs must be positive")
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if self.onnx_opset < 13:
            raise ValueError("onnx_opset must be at least 13")
        if self.fidelity_batch_size <= 0:
            raise ValueError("fidelity_batch_size must be positive")
        if self.preset_name is not None and not self.preset_name:
            raise ValueError("preset_name cannot be empty")
        if not self.output_report_filename:
            raise ValueError("output_report_filename cannot be empty")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["input_shape"] = list(self.input_shape)
        return values
