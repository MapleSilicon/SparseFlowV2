"""Non-destructive release-readiness checks for SparseFlow V0."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable

from sparseflow.report import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    V0_POSITIONING,
    EvidenceValidationError,
    validate_evidence_report,
)


REFERENCE_REPORTS = ("channel-prune-report.json", "noop-report.json")
REFERENCE_DISCLAIMERS = (
    "bundled untrained micro-model",
    "evidence mechanism",
    "not a commercial benchmark",
    "latency is noise-sensitive",
    "structural reductions do not automatically imply latency reductions",
    "machine",
    "software versions",
    "configuration",
    "commit",
)
FORBIDDEN_IMPLEMENTED_TERMS = ("2:4", "CUDA", "A100", "MLIR")


@dataclass(frozen=True)
class CheckResult:
    """One release check and its human-readable evidence."""

    name: str
    passed: bool
    detail: str


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def tracked_root_reports(tracked_files: Iterable[str]) -> list[str]:
    """Return tracked generated report names located at repository root."""

    reports = []
    for filename in tracked_files:
        path = PurePosixPath(filename.replace("\\", "/"))
        if len(path.parts) == 1 and re.fullmatch(r"report.*\.json", path.name):
            reports.append(path.name)
    return sorted(reports)


def forbidden_current_claims(text: str, source: str) -> list[str]:
    """Find present-tense V0 implementation claims for out-of-scope systems."""

    findings = []
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
        if not re.search(r"SparseFlow(?:\s+V0)?", sentence, re.IGNORECASE):
            continue
        positive_claim = re.search(
            r"SparseFlow(?:\s+V0)?\s+"
            r"(?:currently\s+)?(?:implements?|includes?|provides?|supports?|uses?|"
            r"ships?|offers?|delivers?|is\s+(?:an?\s+)?)",
            sentence,
            re.IGNORECASE,
        )
        if not positive_claim or re.search(r"\b(?:does|is|are)\s+not\b", sentence, re.IGNORECASE):
            continue
        for term in FORBIDDEN_IMPLEMENTED_TERMS:
            if term.lower() in sentence.lower():
                findings.append(f"{source}: forbidden current V0 claim mentions {term}: {sentence.strip()}")
    return findings


def reference_disclaimer_errors(text: str) -> list[str]:
    """Return required reference-run phrases missing from the supplied text."""

    lowered = text.lower()
    return [phrase for phrase in REFERENCE_DISCLAIMERS if phrase not in lowered]


def _result(name: str, passed: bool, success: str, failure: str) -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=success if passed else failure)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_release_checks(repo_path: str | Path, require_clean: bool = True) -> list[CheckResult]:
    """Evaluate the checked-out repository without changing files or Git state."""

    repo = Path(repo_path).resolve()
    results: list[CheckResult] = []

    status = _git(repo, "status", "--short")
    clean = status.returncode == 0 and not status.stdout.strip()
    results.append(
        _result(
            "git working tree",
            status.returncode == 0 and (clean or not require_clean),
            "clean" if clean else "dirty tree allowed for this run",
            status.stderr.strip() or status.stdout.strip() or "unable to inspect Git status",
        )
    )

    try:
        package = importlib.import_module("sparseflow")
        importlib.import_module("sparseflow.presets")
        import_ok = package.__version__ == "0.1.0"
        import_detail = f"sparseflow {package.__version__} and presets import successfully"
    except (ImportError, AttributeError) as exc:
        import_ok = False
        import_detail = str(exc)
    results.append(_result("package imports", import_ok, import_detail, import_detail))

    pytest_ready = (repo / "tests").is_dir() and "pytest" in (repo / "requirements.txt").read_text(
        encoding="utf-8"
    ).lower()
    results.append(
        _result(
            "test command",
            pytest_ready,
            "run: python -m pytest -q",
            "tests directory or pytest dependency is missing",
        )
    )

    readme = (repo / "README.md").read_text(encoding="utf-8")
    results.append(
        _result(
            "V0 positioning",
            V0_POSITIONING in readme,
            "README contains the exact V0 positioning",
            "README is missing the exact V0 positioning",
        )
    )

    tracked_process = _git(repo, "ls-files")
    tracked = tracked_process.stdout.splitlines() if tracked_process.returncode == 0 else []
    root_reports = tracked_root_reports(tracked)
    results.append(
        _result(
            "generated root reports",
            tracked_process.returncode == 0 and not root_reports,
            "no generated root report is tracked",
            f"tracked generated root reports: {', '.join(root_reports)}",
        )
    )

    schema = _read_json(repo / "schemas" / "evidence-report.schema.json")
    schema_versions = schema.get("properties", {}).get("schema_version", {}).get("enum", [])
    schema_matches = (
        SCHEMA_VERSION in schema_versions
        and all(version in schema_versions for version in SUPPORTED_SCHEMA_VERSIONS)
    )
    results.append(
        _result(
            "schema version",
            schema_matches,
            f"schema and report code use {SCHEMA_VERSION}",
            f"schema does not match report code version {SCHEMA_VERSION}",
        )
    )

    reference_dir = repo / "examples" / "reference-run"
    reference_readme = reference_dir / "README.md"
    disclaimer_errors = (
        reference_disclaimer_errors(reference_readme.read_text(encoding="utf-8"))
        if reference_readme.exists()
        else list(REFERENCE_DISCLAIMERS)
    )
    results.append(
        _result(
            "reference disclaimer",
            not disclaimer_errors,
            "reference-run limitations and provenance are explicit",
            f"missing phrases: {', '.join(disclaimer_errors)}",
        )
    )

    report_errors: list[str] = []
    for filename in REFERENCE_REPORTS:
        path = reference_dir / filename
        try:
            report = _read_json(path)
            validate_evidence_report(report)
            commit_hash = report["run"]["git"]["commit_hash"]
            commit_exists = _git(repo, "cat-file", "-e", f"{commit_hash}^{{commit}}").returncode == 0
            if not commit_hash or not commit_exists:
                report_errors.append(f"{filename}: commit hash is missing or unknown")
            if report["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
                report_errors.append(f"{filename}: unsupported schema version")
            product = report.get("product", {})
            if product.get("demonstration_type") != "physical_channel_pruning_mechanism_demonstration":
                report_errors.append(f"{filename}: mechanism demonstration label is missing")
            if product.get("commercial_benchmark") is not False:
                report_errors.append(f"{filename}: commercial benchmark flag must be false")
        except (EvidenceValidationError, OSError, KeyError, ValueError) as exc:
            report_errors.append(f"{filename}: {exc}")
    results.append(
        _result(
            "curated reports",
            not report_errors,
            "both curated reports validate and identify a real commit",
            "; ".join(report_errors),
        )
    )

    claim_findings: list[str] = []
    claim_paths = [repo / "README.md", repo / "pyproject.toml", repo / "run_bench.py"]
    claim_paths.extend(sorted((repo / "sparseflow").glob("*.py")))
    for path in claim_paths:
        claim_findings.extend(
            forbidden_current_claims(path.read_text(encoding="utf-8"), path.relative_to(repo).as_posix())
        )
    results.append(
        _result(
            "current product claims",
            not claim_findings,
            "no out-of-scope system is claimed as current V0 functionality",
            "; ".join(claim_findings),
        )
    )

    tracked_env = [name for name in tracked if PurePosixPath(name).name.lower().startswith(".env")]
    secret_pattern = re.compile(
        r"(?i)(?:api[_-]?key|secret|password|token)\s*[=:]\s*['\"][A-Za-z0-9_\-/+=]{16,}['\"]"
    )
    secret_findings: list[str] = []
    for name in tracked:
        path = repo / name
        if path.is_file() and path.suffix.lower() in {".py", ".toml", ".md", ".txt", ".json", ".yml", ".yaml"}:
            if secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                secret_findings.append(name)
    security_ok = not tracked_env and not secret_findings
    security_failure = f"tracked env files: {tracked_env}; possible secrets: {secret_findings}"
    results.append(
        _result(
            "tracked secrets",
            security_ok,
            "no tracked .env files or obvious hardcoded secrets detected",
            security_failure,
        )
    )
    return results
