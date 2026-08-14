"""Freeze ImageNet-1K validation identity and acceptance thresholds before evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from sparseflow.imagenet_eval import prepare_imagenet_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", required=True, help="ImageNet root containing the official archives/devkit")
    parser.add_argument("--output-dir", required=True, help="Directory for frozen manifest and precommit evidence")
    parser.add_argument("--top1-tolerance-points", type=float, default=0.20)
    parser.add_argument("--top5-tolerance-points", type=float, default=0.20)
    args = parser.parse_args()

    precommit = prepare_imagenet_baseline(
        imagenet_root=args.imagenet_root,
        output_dir=args.output_dir,
        top1_tolerance_points=args.top1_tolerance_points,
        top5_tolerance_points=args.top5_tolerance_points,
    )

    dataset = precommit["dataset"]
    acceptance = precommit["acceptance"]
    output = Path(args.output_dir).resolve()
    print("SparseFlow v0.3 Gate A — ImageNet baseline precommit")
    print(f"status: {precommit['status']}")
    print(f"samples: {dataset['sample_count']}")
    print(f"classes: {dataset['class_count']}")
    print(f"manifest SHA-256: {dataset['manifest_sha256']}")
    print(f"class-order SHA-256: {dataset['class_order_sha256']}")
    print(f"top-1 target: {acceptance['expected_top1_percent']:.3f}% ± {acceptance['top1_tolerance_points']:.3f} points")
    print(f"top-5 target: {acceptance['expected_top5_percent']:.3f}% ± {acceptance['top5_tolerance_points']:.3f} points")
    print("baseline measurement: NOT RUN")
    print(f"precommit: {output / 'baseline-precommit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
