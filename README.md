# SparseFlow

## Current status: V0

SparseFlow V0: reproducible optimization and evidence harness, validated with a deterministic physical channel-pruning demonstration.

The repository is a small, offline-verifiable reference implementation. It uses a bundled untrained CNN to exercise physical model transformation, deployment export, measurement, provenance capture, and schema-validated evidence. It is not a commercial benchmark.

SparseFlow V1: ResNet-18 dependency-aware physical channel pruning, ONNX Runtime CPU deployment, and measured accuracy/latency evidence on a real evaluation workload.

V1 is the next milestone, not current functionality. See [docs/resnet18-v1-plan.md](docs/resnet18-v1-plan.md).

## What SparseFlow V0 does

- Builds the bundled `reference_cnn_v1` model deterministically without network access.
- Applies deterministic L1-ranked physical channel pruning to its supported sequential `Conv2d`-`BatchNorm2d`-`Conv2d` dependency chain.
- Provides a NoOp control with identical structure, parameters, compute counts, and structural hashes.
- Exports baseline and optimized models to ONNX and validates them with the ONNX checker.
- Executes both artifacts with ONNX Runtime `CPUExecutionProvider`.
- Records parameters, serialized sizes, MACs, FLOPs, latency distributions, fidelity proxy results, semantic graph changes, hashes, environment details, Git state, and resolved configuration.
- Validates every official report against the checked-in schema before an atomic write.

## What it proves

For the bundled micro-model, V0 proves that the supported pruning pass physically removes channels and repairs the directly dependent dense layers while preserving runnable PyTorch and ONNX models. The reference result reduces parameters from 5,068 to 2,962 (41.55%) and MACs/FLOPs by 46.85%.

It also proves that the harness can produce deterministic structural evidence: model hashes, semantic graph-diff hashes, parameter counts, compute counts, pass configuration, and NoOp neutrality reproduce for identical inputs and code.

## What it does not prove

V0 does not prove production speedup, commercial cost savings, task accuracy, or suitability for arbitrary networks. The bundled model is untrained and extremely small, so its latency is noise-sensitive and may improve or regress between runs. Parameter reduction is not the same as compute reduction, and compute reduction is not the same as latency reduction.

V0 is not a production inference optimizer, a general dependency-aware pruning framework, a ResNet-18 implementation, a quantization framework, or a commercial performance benchmark.

## Quick start

The benchmark downloads neither weights nor datasets. From PowerShell:

```powershell
cd "C:\Users\15879\Downloads\sparseflow-v1"
python -m pip install -r requirements.txt
python -m pytest -q
python run_bench.py --preset micro-demo
python run_bench.py --preset noop-control
python scripts/check_release.py
```

## Benchmark presets

`micro-demo` selects `reference_cnn_v1`, seed `1234`, input shape `[1, 3, 16, 16]`, the `channel-prune` pass at ratio `0.375`, five warmups, 30 measured runs, and `report.json`.

`noop-control` uses the same deterministic model and measurement settings with the `noop` pass, ratio `0.0`, and `report-noop.json`.

Explicit command-line values override preset values:

```powershell
python run_bench.py --preset micro-demo --seed 77 --warmup 10 --measured-runs 100 --threads 1 --output report-custom.json
```

The selected preset and final resolved values are recorded in the report's `configuration` object. There is no ResNet-18 preset because V0 does not support ResNet-18.

## Evidence report

Schema `0.3.0` adds explicit product positioning, resolved configuration, full latency summary statistics, and latency interpretation. `metrics.latency_ms` retains every sample plus a deterministic sample hash, p50, p95, minimum, maximum, arithmetic mean, population standard deviation, and coefficient of variation. It also records warmup and measured counts, provider, clock, and ONNX Runtime intra-op and inter-op thread counts.

`latency_interpretation.commercial_speedup_claim_supported` is false by policy in V0. A report is marked noise-sensitive when baseline p50 is below `0.1 ms` or either latency distribution has a coefficient of variation above `0.10`.

Report assembly and validation live in `sparseflow.report`. Missing mandatory evidence raises `EvidenceValidationError`, and no official report is written. See [docs/benchmark-interpretation.md](docs/benchmark-interpretation.md) for interpretation guidance.

## NoOp control

The NoOp pass verifies that the measurement pipeline does not invent structural changes. Its official evidence must show zero parameter and MAC/FLOP deltas, an empty graph-change list, matching baseline and optimized structural/model hashes, and passing output fidelity. Runtime samples may still vary because separate executions are measured.

## Reference run

Curated mechanism-demonstration reports are checked in at [examples/reference-run/channel-prune-report.json](examples/reference-run/channel-prune-report.json) and [examples/reference-run/noop-report.json](examples/reference-run/noop-report.json). Their provenance and limitations are documented in [examples/reference-run/README.md](examples/reference-run/README.md).

Ordinary root `report*.json`, ONNX files, and artifact directories are generated outputs and ignored by Git.

## Architecture

- `run_bench.py`: CLI, preset resolution, pass selection, and report orchestration.
- `sparseflow/models.py`: bundled deterministic reference CNN.
- `sparseflow/passes.py`: NoOp and supported physical channel-pruning transformations.
- `sparseflow/benchmark.py`: ONNX export, CPU timing, structural/runtime metrics, fidelity proxy, and hashes.
- `sparseflow/audit.py`: machine, software, process, repository, and Git metadata.
- `sparseflow/report.py`: evidence assembly, schema validation, and atomic output.
- `schemas/evidence-report.schema.json`: official report contract.
- `scripts/check_release.py`: non-destructive release-readiness checks.

## Current limitations

The pruning pass handles only the bundled model's linear dense dependency pattern. It does not repair residual additions, identity/projection branches, concatenations, grouped or depthwise convolutions, or arbitrary classifiers. Fidelity is a deterministic random-input output proxy, not task accuracy. Latency comes from one local CPU process and is not a production workload result.

## V1 roadmap

SparseFlow V1 targets ResNet-18 on x86 CPU using ONNX Runtime, with dependency-aware physical `Conv2d` channel pruning and evaluation on a real dataset or defensible fixed validation subset. Work is gated on residual dependency analysis, complete branch repair, valid ONNX execution, explicit task accuracy, repeated latency disclosure, and deterministic structural evidence. No latency improvement is promised before measurement.

## Explicitly out of scope

V0 does not implement 2:4 structured sparsity, CUDA or GPU kernels, A100 acceleration, MLIR compilation, quantization, model training, fine-tuning, or online dataset acquisition. Those terms are not part of the current product identity.

## Testing

The suite covers unit, integration, and CLI behavior with an 80% branch-coverage floor:

```powershell
python -m pytest -q
```

The release checker verifies repository state, package imports, test-command availability, positioning, tracked-report hygiene, schema consistency, curated reports, commit provenance, required disclaimers, current-product claims, and tracked secret indicators:

```powershell
python scripts/check_release.py
```

Use `--allow-dirty` only for a pre-commit check; the final release check requires a clean tree.

## Reproducibility

For identical code, preset, seed, and pass configuration, compare model hashes, graph-diff hash, resolved configuration, parameter counts, and MAC/FLOP counts. Latency samples are deliberately excluded from deterministic equality because operating-system scheduling, clocks, caches, and runtime behavior vary.

Each report captures the repository path and identifier, current commit hash, dirty-tree flag, machine and software fingerprint, process/thread settings, complete resolved configuration, graph changes, structural hashes, model hashes, and ONNX hashes. Generated reports should honestly retain `dirty: true` when run from a modified tree.
