"""Reproducibility metadata and deterministic seed control."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_output(repo_path: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_git_metadata(repo_path: str | Path) -> dict[str, Any]:
    repository = Path(repo_path).resolve()
    commit_hash = _git_output(repository, "rev-parse", "HEAD")
    status = _git_output(repository, "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit_hash": commit_hash,
        "dirty": bool(status) if status is not None else None,
    }


def collect_hardware_fingerprint() -> dict[str, Any]:
    cpu_model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None
    return {
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "architecture": platform.architecture()[0],
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count() or 1,
        "cpu_model": cpu_model,
    }


def collect_software_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
    }


def collect_process_configuration(threads: int) -> dict[str, Any]:
    return {
        "onnxruntime_intra_op_threads": threads,
        "onnxruntime_inter_op_threads": threads,
        "onnxruntime_execution_mode": "ORT_SEQUENTIAL",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def collect_audit_metadata(repo_path: str | Path, seed: int, threads: int) -> dict[str, Any]:
    repository = Path(repo_path).resolve()
    return {
        "repository": {"path": str(repository), "identifier": repository.name},
        "git": collect_git_metadata(repository),
        "hardware": collect_hardware_fingerprint(),
        "software": collect_software_fingerprint(),
        "process": collect_process_configuration(threads),
        "seed": seed,
    }
