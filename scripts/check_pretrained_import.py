"""Run SparseFlow v0.3 Gate A steps 1-4 and emit auditable JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparseflow.pretrained import load_and_validate_pretrained_resnet18
from sparseflow.serialization import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="report-pretrained-import.json")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    args = parser.parse_args()

    _, evidence = load_and_validate_pretrained_resnet18(
        seed=args.seed,
        batch_size=args.batch_size,
        atol=args.atol,
        rtol=args.rtol,
        progress=False,
    )
    output = Path(args.output)
    atomic_write_json(output, evidence)

    final_logits = evidence["import_validation"]["final_logits"]
    print("SparseFlow v0.3 Gate A — pretrained import checkpoint")
    print(f"torchvision: {evidence['weights']['torchvision_version']}")
    print(f"weights: {evidence['weights']['weights_enum']}")
    print(f"state-dict SHA-256: {evidence['weights']['state_dict_sha256']}")
    print(f"checkpoint SHA-256: {evidence['weights']['checkpoint_sha256']}")
    print(f"max abs logit error: {final_logits['max_abs_error']:.9g}")
    print(f"max rel logit error: {final_logits['max_rel_error']:.9g}")
    print(f"argmax agreement: {final_logits['argmax_agreement']}")
    print(f"logit equivalence: {evidence['import_validation']['logit_equivalence']}")
    print("dataset evaluation: NOT RUN (ImageNet validation data required for steps 5-7)")
    print(f"report: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
