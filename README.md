# SparseFlow

## Current status: v0.2 complete; v0.3 Gate A in development

SparseFlow V0: reproducible optimization and evidence harness, validated with a deterministic physical channel-pruning demonstration.

SparseFlow **v0.2** extends that evidence-first workflow with transactional, dependency-aware physical channel pruning for the repository's local untrained ResNet-18 reference architecture. It remains an engineering/mechanism milestone, not a commercial benchmark or task-accuracy result.

The current implementation deliberately separates five claims:

1. **V0 mechanism evidence:** physical channel removal on the bundled micro-model.
2. **Gate 1:** deterministic ResNet-18 dependency analysis and zero-ratio structural neutrality.
3. **Gate 1.5:** calibrated randomized paired CPU latency measurement with a recipe hash and minimum detectable improvement (MDI).
4. **Gate 2 / v0.2:** validated pruning plan → mutation of a deep copy → executed-plan validation → ONNX/ORT evidence.
5. **v0.3 Gate A steps 1–4:** official torchvision ResNet-18 pretrained-weight import and semantic-equivalence validation before any task-accuracy evaluation.

## v0.3 Gate A — pretrained import checkpoint

Gate A is deliberately layered from cheapest/most-isolating to dataset-dependent validation:

1. **State-dict mapping:** exact torchvision/SparseFlow key order and tensor-shape equality; strict import only.
2. **Module-configuration equivalence:** Conv2d, BatchNorm2d, pooling, classifier, projection/downsample topology, and ReLU placement/type are compared. SparseFlow's `inplace=False` ReLU storage policy is intentionally allowed against torchvision's `inplace=True` because placement/type and numerical behavior are the relevant semantic contract.
3. **Per-stage equivalence:** with both models explicitly in eval mode and gradients disabled, deterministic 224×224 float32 inputs are compared at the stem/maxpool boundary and `layer1` through `layer4`.
4. **Final-logit equivalence:** measured max absolute/relative error, configured `atol`/`rtol`, and argmax agreement are recorded. A mismatch fails closed; tolerances are never auto-widened.
5. **Dataset/manifest validation:** not implemented until ImageNet-1K validation data is available.
6. **Full ImageNet baseline reproduction:** not implemented until step 5 is available.
7. **Published top-1/top-5 gate:** not implemented until step 6 is available.

The pinned pretrained reference is `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`. The evidence records the exact torchvision version, enum, resolved weight URL, checkpoint filename, checkpoint SHA-256 when present, imported state-dict SHA-256, and the published ImageNet-1K reference metrics (69.758% top-1 / 89.078% top-5). Dataset-independent steps 1–4 can therefore land and remain auditable before ImageNet data logistics are solved.

Run the import checkpoint with:

```powershell
python scripts/check_pretrained_import.py --output report-pretrained-import.json
```

A successful steps 1–4 checkpoint does **not** authorize a task-accuracy or pruning-quality claim. v0.3 optimization evidence remains blocked until baseline ImageNet evaluation reproduces the pinned reference within a declared tolerance.

## v0.2 Gate 2

Gate 2 supports only `resnet18-reference`. The policy is intentionally conservative:

- L1 weight-magnitude ranking with deterministic channel-index tie breaking.
- One canonical kept-channel set for each residual stage width.
- Identity and projection paths share the same canonical stage channel identity.
- Internal BasicBlock `conv1` channels may use their own deterministic ranking and are repaired into `conv2` inputs.
- Projection convolutions and BatchNorm modules are repaired together with the main branch.
- The final classifier input dimension follows the retained stage-4 channels; its 1000 output classes are never pruned.
- Supported Gate 2 ratio range is `(0.0, 0.5]`; the release preset uses `0.125`.

The mutation path is transactional. SparseFlow validates the complete immutable plan before mutation, deep-copies the original model, performs physical tensor surgery on the copy, executes the candidate, and independently verifies every planned module dimension. Failed validation does not intentionally mutate the original model or emit official evidence.

The official Gate 2 evidence boundary independently recomputes the pruning-plan hash, residual member channel identity, and plan-versus-execution channel-index agreement. A Boolean asserted by the optimization pass is not sufficient evidence by itself.

## What v0.2 does not prove

SparseFlow v0.2 does **not** establish task accuracy, accuracy recovery after pruning, production latency improvement, commercial cost savings, or generalized architecture support. The ResNet-18 reference model is local and untrained. Fidelity remains a deterministic random-input proxy, not task accuracy.

SparseFlow v0.2 does not implement 2:4 structured sparsity, CUDA or GPU kernels, A100 acceleration, MLIR compilation, quantization, model training, fine-tuning, or online dataset acquisition.

## Quick start

From PowerShell:

```powershell
cd "C:\Users\15879\Downloads\sparseflow-v1"
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python run_bench.py --preset noop-control
python run_bench.py --preset micro-demo
python run_bench.py --preset resnet18-gate1
python run_bench.py --preset resnet18-gate2
python scripts/check_pretrained_import.py --output report-pretrained-import.json
python scripts/check_release.py --allow-dirty
```

Direct Gate 2 invocation:

```powershell
python run_bench.py --model resnet18-reference --pass channel-prune --pruning-ratio 0.125 --output report-resnet18-gate2.json
```

Explicit CLI values override preset values and the resolved configuration is written into the evidence report.

## Benchmark presets

- `micro-demo`: bundled micro-model, ratio `0.375`, V0 physical-pruning mechanism evidence.
- `noop-control`: structural NoOp control.
- `resnet18-gate1`: local ResNet-18, ratio `0.0`, dependency analysis and neutrality evidence.
- `resnet18-gate2`: local ResNet-18, ratio `0.125`, transactional physical channel pruning.

The ResNet benchmark presets use input shape `[1, 3, 64, 64]`. MAC/FLOP values from these runs must not be compared directly with published 224×224 ResNet-18 figures. Gate A semantic equivalence uses deterministic `[B, 3, 224, 224]` inputs because it validates the pretrained reference architecture at its standard inference geometry.

## Latency calibration

SparseFlow measures baseline and comparison artifacts in deterministic seeded randomized pairs rather than measuring all baseline runs and then all optimized runs. The complete recipe records warmups, pair count, provider, timer, seed, order policy, input shape, thread counts, and benchmark mode and receives a deterministic SHA-256 hash.

Before a real comparison, the harness loads the identical baseline ONNX artifact into independent ONNX Runtime sessions and measures a null-effect distribution. The MDI is derived from the two-sided empirical percentile interval and stored with its derivation and raw samples.

SparseFlow does not report latency improvements below the calibrated minimum detectable improvement. Structural reductions do not automatically imply latency reductions. Latency is noise-sensitive, especially for short local CPU runs.

Environment telemetry is diagnostic only and never adjusts measured latency.

## Evidence

The V0/Gate 1/Gate 1.5 report contract remains `schemas/evidence-report.schema.json`. Gate 2 uses `schemas/evidence-report-gate2.schema.json` and schema version `0.5.0` after the existing measurement evidence has passed its fail-closed validation path.

Gate 2 evidence includes:

- dependency graph and hash;
- immutable pruning plan and hash;
- canonical kept/removed channel indices;
- residual coupling-group validation;
- transactional mutation status;
- executed module dimensions and channel indices;
- independent plan-versus-execution validation;
- semantic graph changes and structural hashes;
- parameter, serialized-size, MAC/FLOP, and fidelity-proxy metrics;
- ONNX checker and ONNX Runtime CPU validation;
- measurement recipe, null calibration, MDI, paired samples, and latency interpretation.

Gate A steps 1–4 emit a separate checkpoint report that records weight provenance plus measured stage/logit equivalence. It intentionally records dataset evaluation as `not_run` until ImageNet-1K validation data is available.

Generated root `report*.json` and ONNX artifacts remain ignored. Curated V0 evidence remains under `examples/reference-run/`.

## Architecture

- `run_bench.py` — CLI and evidence orchestration.
- `sparseflow/models.py` — deterministic reference CNN and local ResNet-18.
- `sparseflow/pretrained.py` — v0.3 pretrained import, module/config validation, stage hooks, logit equivalence, and weight provenance.
- `sparseflow/dependencies.py` — ResNet dependency graph and residual constraints.
- `sparseflow/planning.py` — immutable Gate 2 plans, canonical residual groups, ranking, and pre-mutation validation.
- `sparseflow/resnet_pruning.py` — transactional physical tensor surgery and executed-plan validation.
- `sparseflow/gate2_pass.py` — Gate 2 optimization-pass adapter and graph-change evidence.
- `sparseflow/measurement.py` — paired measurement, telemetry, calibration, and MDI.
- `sparseflow/benchmark.py` — ONNX export, ORT validation, metrics, and fidelity proxy.
- `sparseflow/report.py` — existing fail-closed report and measurement validation.
- `sparseflow/gate2_report.py` — independent Gate 2 plan/coupling/execution evidence boundary.
- `scripts/check_pretrained_import.py` — v0.3 Gate A steps 1–4 checkpoint runner.
- `scripts/check_release.py` — release-readiness checks.

## Testing

The repository enforces an 80% branch-aware coverage floor:

```powershell
python -m pytest -q
```

Gate 2 tests cover deterministic planning, canonical residual channel identity, same-width/different-index rejection, missing projection rejection, original-model preservation, physical parameter reduction, output execution, classifier repair, pass evidence, and executed-plan mismatch rejection.

Gate A helper tests cover measured error recording, strict shape mismatch rejection, explicit eval-mode preconditions, BatchNorm configuration provenance, ReLU storage-policy treatment, and projection/classifier module-map coverage. CI additionally runs the real official-weight import checkpoint.

A pre-push hook may run the full test suite locally. The repository also contains a CI workflow for branch/PR verification.

## Reference-run provenance

The curated reference run remains a bundled untrained micro-model evidence mechanism, not a commercial benchmark. Its documentation records machine, software versions, configuration, and commit provenance. Latency is noise-sensitive, and structural reductions do not automatically imply latency reductions.

## Next milestone

After Gate A steps 1–4 are green, the next work is data logistics only: acquire/pin an ImageNet-1K validation manifest, run the official `weights.transforms()` pipeline, reproduce the 69.758% / 89.078% baseline within a declared tolerance, and only then permit any pretrained pruning/accuracy experiment.
