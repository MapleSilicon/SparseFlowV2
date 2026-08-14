"""Run the unpruned ImageNet-1K baseline against a frozen SparseFlow precommit."""

from __future__ import annotations

import argparse

from sparseflow.imagenet_eval import run_imagenet_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", required=True)
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--output", default="report-imagenet-baseline.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_imagenet_baseline(
        imagenet_root=args.imagenet_root,
        precommit_path=args.precommit,
        output_path=args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    evaluation = result["evaluation"]
    print("SparseFlow v0.3 Gate A — ImageNet baseline reproduction")
    print(f"samples: {evaluation['sample_count']}")
    print(f"top-1: {evaluation['top1_percent']:.3f}% (delta {evaluation['top1_delta_points']:+.3f})")
    print(f"top-5: {evaluation['top5_percent']:.3f}% (delta {evaluation['top5_delta_points']:+.3f})")
    print(f"baseline reproduction passed: {evaluation['baseline_reproduction_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
