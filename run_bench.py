#!/usr/bin/env python3
"""SparseFlow benchmark command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sparseflow.audit import collect_audit_metadata
from sparseflow.benchmark import run_benchmark
from sparseflow.config import BenchmarkConfig
from sparseflow.models import INPUT_SHAPE, build_reference_model
from sparseflow.passes import ChannelPruningPass, NoOpPass, OptimizationPass
from sparseflow.report import (
    EvidenceValidationError,
    build_evidence_report,
    render_result_table,
    write_evidence_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark physical Conv2d channel pruning on ONNX Runtime CPU."
    )
    parser.add_argument(
        "--pass",
        dest="pass_name",
        choices=["noop", "channel-prune"],
        default="channel-prune",
    )
    parser.add_argument(
        "--pruning-ratio",
        "--prune-ratio",
        type=float,
        default=0.375,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured-runs", "--iters", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", "--out", type=Path, default=Path("report.json"))
    return parser


def _optimization_pass(name: str, ratio: float) -> OptimizationPass:
    if name == "noop":
        return NoOpPass()
    return ChannelPruningPass(ratio=ratio)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = BenchmarkConfig(
            input_shape=INPUT_SHAPE,
            seed=args.seed,
            warmup_runs=args.warmup,
            measured_runs=args.measured_runs,
            threads=args.threads,
        )
        model = build_reference_model(seed=args.seed)
        optimization_pass = _optimization_pass(args.pass_name, args.pruning_ratio)
        result = run_benchmark(model, optimization_pass, config)
        audit = collect_audit_metadata(Path.cwd(), seed=args.seed, threads=args.threads)
        report = build_evidence_report(result, audit)
        write_evidence_report(report, args.output)
    except (EvidenceValidationError, ValueError, RuntimeError, OSError) as exc:
        print(f"SparseFlow benchmark failed: {exc}", file=sys.stderr)
        return 2

    print(render_result_table(report))
    print(f"\nReport: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
