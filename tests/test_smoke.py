import json
import subprocess
import sys
from pathlib import Path

from sparseflow.audit import collect_audit_metadata
from sparseflow.benchmark import BenchmarkConfig, run_benchmark
from sparseflow.models import build_reference_model
from sparseflow.passes import ChannelPruningPass, NoOpPass
from sparseflow.report import build_evidence_report, validate_evidence_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def _committed_audit(seed=1234):
    audit = collect_audit_metadata(REPO_ROOT, seed=seed, threads=1)
    audit["git"] = {"commit_hash": "a" * 40, "dirty": False}
    return audit


def test_benchmark_exports_valid_onnx_and_builds_official_reports():
    config = BenchmarkConfig(warmup_runs=1, measured_runs=2, threads=1)

    for optimization_pass in (NoOpPass(), ChannelPruningPass(ratio=0.375)):
        result = run_benchmark(build_reference_model(seed=config.seed), optimization_pass, config)
        report = build_evidence_report(result, _committed_audit(config.seed))
        validate_evidence_report(report)

        assert report["metrics"]["latency_ms"]["methodology"]["provider"] == "CPUExecutionProvider"
        assert report["artifacts"]["baseline"]["onnx_validated"] is True
        assert report["artifacts"]["optimized"]["onnx_validated"] is True

        if optimization_pass.name == "noop":
            assert report["metrics"]["parameters"]["reduction_percent"] == 0.0
            assert report["metrics"]["compute"]["mac_reduction_percent"] == 0.0
            assert report["metrics"]["compute"]["flop_reduction_percent"] == 0.0
            assert report["graph"]["changes"] == []
            assert report["graph"]["baseline_structural_hash"] == report["graph"]["optimized_structural_hash"]

    assert report["metrics"]["compute"]["optimized"]["macs"] < report["metrics"]["compute"]["baseline"]["macs"]


def test_cli_success_and_validation_failure(tmp_path):
    success_report = tmp_path / "success.json"
    success = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_bench.py"), "--pass", "noop", "--warmup", "0", "--measured-runs", "1", "--output", str(success_report)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert success.returncode == 0, success.stderr
    assert json.loads(success_report.read_text(encoding="utf-8"))["optimization"]["pass_name"] == "noop"

    failed_report = tmp_path / "failed.json"
    failure = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_bench.py"), "--pass", "noop", "--warmup", "0", "--measured-runs", "1", "--output", str(failed_report)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failure.returncode != 0
    assert "commit_hash" in failure.stderr
    assert not failed_report.exists()


def test_cli_delegates_official_report_serialization():
    source = (REPO_ROOT / "run_bench.py").read_text(encoding="utf-8")
    assert "write_evidence_report" in source
    assert "json.dump" not in source
