#!/usr/bin/env python3
"""Run SparseFlow V0 release-readiness checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparseflow.release import run_release_checks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check SparseFlow V0 release readiness.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="report a dirty tree without failing (for pre-commit verification)",
    )
    args = parser.parse_args(argv)

    results = run_release_checks(REPO_ROOT, require_clean=not args.allow_dirty)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    failures = [result for result in results if not result.passed]
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
