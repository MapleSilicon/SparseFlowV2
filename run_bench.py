#!/usr/bin/env python3
"""SparseFlow V0 reproducible optimization and evidence harness CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sparseflow.audit import collect_audit_metadata
from sparseflow.benchmark import run_benchmark
from sparseflow.config import BenchmarkConfig
from sparseflow.models import build_reference_model
from sparseflow.passes import ChannelPruningPass, NoOpPass, OptimizationPass
from sparseflow.presets import (
    ResolvedRunConfiguration,
    get_preset,
    preset_names,
)
from sparseflow.report import (
    EvidenceValidationError,
    build_evidence_report,
    render_result_table,
    write_evidence_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SparseFlow V0: reproducible optimization and evidence harness, "
            "validated with a deterministic physical channel-pruning demonstration."
        )
    )
    parser.add_argument("--preset", choices=preset_names(), default="micro-demo")
    parser.add_argument(
        "--pass",
        dest="pass_name",
        choices=["noop", "channel-prune"],
        default=None,
    )
    parser.add_argument(
        "--pruning-ratio",
        "--prune-ratio",
        type=float,
        default=None,
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--measured-runs", "--iters", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", "--out", type=Path, default=None)
    return parser


def _optimization_pass(name: str, ratio: float) -> OptimizationPass:
    if name == "noop":
        return NoOpPass()
    return ChannelPruningPass(ratio=ratio)


def resolve_cli_configuration(argv: list[str] | None = None) -> ResolvedRunConfiguration:
    args = _parser().parse_args(argv)
    preset = get_preset(args.preset)
    pass_name = args.pass_name if args.pass_name is not None else preset.pass_name
    if args.pruning_ratio is not None:
        pruning_ratio = args.pruning_ratio
    elif pass_name == "noop":
        pruning_ratio = 0.0
    else:
        pruning_ratio = preset.pruning_ratio
    return ResolvedRunConfiguration(
        preset_name=preset.name,
        model_identifier=preset.model_identifier,
        seed=args.seed if args.seed is not None else preset.seed,
        input_shape=preset.input_shape,
        pass_name=pass_name,
        pruning_ratio=pruning_ratio,
        warmup_runs=(args.warmup if args.warmup is not None else preset.warmup_runs),
        measured_runs=(
            args.measured_runs if args.measured_runs is not None else preset.measured_runs
        ),
        threads=args.threads if args.threads is not None else 1,
        output_path=(
            args.output if args.output is not None else Path(preset.output_report_filename)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        resolved = resolve_cli_configuration(argv)
        if resolved.model_identifier != "reference_cnn_v1":
            raise ValueError(f"unsupported model identifier: {resolved.model_identifier}")
        config = BenchmarkConfig(
            input_shape=resolved.input_shape,
            seed=resolved.seed,
            warmup_runs=resolved.warmup_runs,
            measured_runs=resolved.measured_runs,
            threads=resolved.threads,
            preset_name=resolved.preset_name,
            output_report_filename=resolved.output_path.name,
        )
        model = build_reference_model(seed=resolved.seed)
        optimization_pass = _optimization_pass(resolved.pass_name, resolved.pruning_ratio)
        result = run_benchmark(model, optimization_pass, config)
        audit = collect_audit_metadata(
            Path.cwd(), seed=resolved.seed, threads=resolved.threads
        )
        report = build_evidence_report(result, audit)
        write_evidence_report(report, resolved.output_path)
    except (EvidenceValidationError, ValueError, RuntimeError, OSError) as exc:
        print(f"SparseFlow benchmark failed: {exc}", file=sys.stderr)
        return 2

    print(render_result_table(report))
    print(f"\nReport: {resolved.output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
