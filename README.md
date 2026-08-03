# SparseFlow

## Current status: V0

SparseFlow V0: reproducible optimization and evidence harness, validated with a deterministic physical channel-pruning demonstration.

The repository is a small, offline-verifiable reference implementation. It uses a bundled untrained CNN to exercise physical model transformation, deployment export, measurement, provenance capture, and schema-validated evidence. It is not a commercial benchmark.

SparseFlow V1: ResNet-18 dependency-aware physical channel pruning, ONNX Runtime CPU deployment, and measured accuracy/latency evidence on a real evaluation workload.

Full V1 remains the next milestone. See [docs/resnet18-v1-plan.md](docs/resnet18-v1-plan.md).

Development status: **SparseFlow V1 Gate 1.5 is implemented on the `feat/resnet18-v1` branch.** Gate 1 adds deterministic local ResNet-18 construction, complete residual dependency analysis, zero-ratio equivalence validation, ONNX export, and ONNX Runtime CPU execution. Gate 1.5 adds calibrated paired latency measurement and evidence-integrity rules. Neither gate performs ResNet channel removal.

## What SparseFlow V0 does

- Builds the bundled `reference_cnn_v1` model deterministically without network access.
- Applies deterministic L1-ranked physical channel pruning to its supported sequential `Conv2d`-`BatchNorm2d`-`Conv2d` dependency chain.
- Provides a NoOp control with identical structure, parameters, compute counts, and structural hashes.
- Exports baseline and optimized models to ONNX and validates them with the ONNX checker.
- Executes both artifacts with ONNX Runtime `CPUExecutionProvider`.
- Measures latency in deterministic seeded A/B pairs and calibrates measurement noise with independently loaded identical artifacts.
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
python run_bench.py --preset resnet18-gate1
python scripts/check_release.py
```

## Benchmark presets

`micro-demo` selects `reference_cnn_v1`, seed `1234`, input shape `[1, 3, 16, 16]`, the `channel-prune` pass at ratio `0.375`, five warmups, 30 randomized measurement pairs, and `report.json`.

`noop-control` uses the same deterministic model and measurement settings with the `noop` pass, ratio `0.0`, and `report-noop.json`.

`resnet18-gate1` selects the local untrained `resnet18-reference`, input shape `[1, 3, 64, 64]`, the `channel-prune` pass at ratio `0.0`, one warmup, three randomized measurement pairs, and `report-resnet18-gate1.json`. Despite the pass name, Gate 1 only analyzes dependencies and validates structural neutrality; it does not prune.

Explicit command-line values override preset values:

```powershell
python run_bench.py --preset micro-demo --seed 77 --warmup 10 --measured-runs 100 --threads 1 --output report-custom.json
```

The selected preset and final resolved values are recorded in the report's `configuration` object. Explicit values continue to override preset values.

The equivalent direct Gate 1 command is:

```powershell
python run_bench.py --model resnet18-reference --pass channel-prune --pruning-ratio 0.0
```

## Latency Calibration

Gate 1.5 replaces sequential baseline-then-comparison timing with deterministic randomized pairs. For every pair, the seeded order policy chooses baseline then comparison or comparison then baseline, times one invocation per side, and records the order, raw samples, paired difference, and paired improvement percentage. The complete measurement recipe records warmups, pair count, runs per side, provider, timer, seed, order policy, input shape, thread counts, and benchmark mode. Its SHA-256 hash is deterministic for identical recipe values.

Before comparing different artifacts, SparseFlow calibrates the harness by loading the exact same ONNX artifact into two independent ONNX Runtime sessions. Calibration is accepted only when model, PyTorch, ONNX, and structural graph hashes are identical. The paired improvement values from this identical-artifact null distribution produce a two-sided empirical percentile interval. The minimum detectable improvement (MDI) is derived as the maximum absolute interval bound; the confidence level, interval bounds, statistic, null samples, and final value are all stored in the report.

SparseFlow does not report latency improvements below the calibrated minimum detectable improvement.

Provider, timer, warmup, pair count, runs per side, seed, order policy, input shape, thread, benchmark-mode, recipe-hash, or calibration-identity mismatches fail closed before report emission. CPU, memory, and frequency telemetry is diagnostic only and never adjusts latency values. Unavailable telemetry is stored as `null` with `available: false` when a reading is attempted; reports remain valid when optional telemetry fields cannot be collected.

## Evidence report

Schema `0.4.0` adds `measurement_recipe`, `latency_calibration`, `environment_drift`, and `paired_statistics`. The validator remains backward-compatible with curated `0.3.0` reports. `metrics.latency_ms` retains every sample plus a deterministic sample hash, p50, p95, minimum, maximum, arithmetic mean, population standard deviation, and coefficient of variation. It also records warmup and pair counts, provider, clock, seeded order policy, input shape, benchmark mode, and ONNX Runtime intra-op and inter-op thread counts.

`latency_interpretation.commercial_speedup_claim_supported` is false by policy in V0. A report is marked noise-sensitive when baseline p50 is below `0.1 ms` or either latency distribution has a coefficient of variation above `0.10`.

Report assembly and validation live in `sparseflow.report`. Missing mandatory evidence raises `EvidenceValidationError`, and no official report is written. See [docs/benchmark-interpretation.md](docs/benchmark-interpretation.md) for interpretation guidance.

A Gate 1 report additionally records the architecture identifier, full dependency graph, dependency summary, eight residual groups, projection and identity counts, Conv-BN links, classifier dependency, and a measured zero-ratio validation result. It also compares deterministic PyTorch and ONNX Runtime outputs. Gate 1 reports are mechanism evidence, not pruning, accuracy, or speedup evidence.

## NoOp control

The NoOp pass verifies that the measurement pipeline does not invent structural changes. Its official evidence must show zero parameter and MAC/FLOP deltas, an empty graph-change list, matching baseline and optimized structural/model hashes, and passing output fidelity. Runtime samples may still vary because separate executions are measured.

## Reference run

Curated mechanism-demonstration reports are checked in at [examples/reference-run/channel-prune-report.json](examples/reference-run/channel-prune-report.json) and [examples/reference-run/noop-report.json](examples/reference-run/noop-report.json). Their provenance and limitations are documented in [examples/reference-run/README.md](examples/reference-run/README.md).

Ordinary root `report*.json`, ONNX files, and artifact directories are generated outputs and ignored by Git.

## Architecture

- `run_bench.py`: CLI, preset resolution, pass selection, and report orchestration.
- `sparseflow/models.py`: bundled deterministic reference CNN and local ResNet-18.
- `sparseflow/dependencies.py`: JSON-serializable ResNet-18 dependency nodes, edges, residual groups, channel constraints, and classifier dependency.
- `sparseflow/passes.py`: NoOp and supported physical channel-pruning transformations.
- `sparseflow/benchmark.py`: ONNX export, CPU timing, structural/runtime metrics, fidelity proxy, and hashes.
- `sparseflow/measurement.py`: immutable recipes, paired execution, identical-artifact calibration, telemetry, and MDI derivation.
- `sparseflow/planning.py`: immutable serializable pruning-plan contracts for future Gate 2 work; no execution or mutation.
- `sparseflow/audit.py`: machine, software, process, repository, and Git metadata.
- `sparseflow/report.py`: evidence assembly, schema validation, and atomic output.
- `schemas/evidence-report.schema.json`: official report contract.
- `scripts/check_release.py`: non-destructive release-readiness checks.

## Current limitations

The V0 pruning pass handles only the bundled micro-model's linear dense dependency pattern. Gate 1 understands ResNet-18 residual additions and identity/projection dependencies but does not modify them. Gate 1.5 calibrates one local CPU process; it does not make that process a production workload or establish a commercial speedup. It does not repair channels after a ResNet transformation, prune grouped or depthwise convolutions, or alter the classifier. Fidelity is a deterministic random-input output proxy, not task accuracy.

## V1 roadmap

Gate 1 is complete: the local ResNet-18 has eight BasicBlocks across four stages, and the analyzer records eight residual groups, three projection shortcuts, five identity shortcuts, twenty Conv-BN dependencies, eight residual channel constraints, and the classifier input dependency. Ratio `0.0` proves identical tensors, structure, parameters, MACs/FLOPs, outputs, and hashes while still exporting valid ONNX and executing with ONNX Runtime CPU.

Gate 2 remains unimplemented. The broader SparseFlow V1 target is dependency-aware physical `Conv2d` channel pruning on ResNet-18 followed by real task evaluation. That requires actual branch repair, classifier repair, and measured accuracy. No physical ResNet pruning, accuracy evaluation, or latency-improvement claim is present in Gate 1.

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
