from pathlib import Path

from sparseflow.release import (
    forbidden_current_claims,
    reference_disclaimer_errors,
    run_release_checks,
    tracked_root_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_root_report_detection():
    tracked = ["README.md", "report.json", "docs/report-noop.json", "report-repeat.json"]

    assert tracked_root_reports(tracked) == ["report-repeat.json", "report.json"]


def test_forbidden_present_tense_product_claim_is_detected():
    text = "SparseFlow V0 implements CUDA kernels and A100 acceleration."

    findings = forbidden_current_claims(text, "README.md")
    assert findings
    assert "CUDA" in findings[0]


def test_missing_reference_run_disclaimer_fails():
    errors = reference_disclaimer_errors("This report is fast and useful.")

    assert "bundled untrained micro-model" in errors
    assert "not a commercial benchmark" in errors
    assert "latency is noise-sensitive" in errors


def test_valid_repository_state_passes_when_dirty_check_is_disabled():
    results = run_release_checks(REPO_ROOT, require_clean=False)
    failures = [result for result in results if not result.passed]

    assert failures == [], "\n".join(f"{item.name}: {item.detail}" for item in failures)
