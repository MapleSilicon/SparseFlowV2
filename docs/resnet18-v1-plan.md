# ResNet-18 V1 Technical Plan

## Target

ResNet-18 on x86 CPU using ONNX Runtime, with dependency-aware physical Conv2d channel pruning and evaluation on a real dataset or defensible fixed validation subset.

SparseFlow V1: ResNet-18 dependency-aware physical channel pruning, ONNX Runtime CPU deployment, and measured accuracy/latency evidence on a real evaluation workload.

This document defines gates for that milestone. Gate 1 dependency analysis and zero-ratio validation are now implemented. Gates 2 through 5 remain plans, and no latency improvement is promised.

## Engineering requirements

The transformation must model residual additions, identity branches, and projection/downsample branches as dependency groups. Every tensor entering an addition must have a consistent channel dimension. Physical output-channel removal must repair BatchNorm features, following `Conv2d` inputs, dependent residual branches, and the classifier input dimension.

Channel ranking must be deterministic. Layer exclusion rules, minimum channel constraints, and useful alignment constraints must be explicit and recorded. Transformations must preserve valid dense tensor shapes; masks or latent zero channels are not sufficient evidence of physical pruning.

Deployment evidence must include correct ONNX export, ONNX checker validation, and ONNX Runtime CPU execution. Evaluation must use a real task metric with baseline and optimized accuracy and absolute accuracy loss. That task accuracy must remain separate from the random-input fidelity proxy.

Official evidence must record the latency methodology, model-size metrics, MAC/FLOP metrics, complete semantic graph changes, and deterministic artifact hashes. The report must disclose unsupported operations or incomplete counts.

## Gate 1: Graph support

**Status: implemented.**

- Load a local ResNet-18 model without downloading during tests.
- Identify residual dependency groups.
- Perform a zero-ratio transformation safely.
- Preserve exact output equivalence.

Acceptance requires deterministic dependency-group discovery, no changed channel dimensions at zero ratio, runnable PyTorch execution, identical structural/model hashes, and fidelity within the NoOp-equivalent tolerance.

The implemented local model contains eight BasicBlocks across four residual stages. Analysis records eight residual groups, three projection shortcuts, five identity shortcuts, twenty Conv-BN dependencies, eight equal-width residual-add constraints, and the pooled 512-feature classifier dependency. All structures serialize to deterministic JSON.

At ratio `0.0`, `ChannelPruningPass` performs dependency analysis and returns an unchanged deep copy. Evidence verifies no tensor or channel change, an empty graph diff, identical model and structural hashes, identical parameter and MAC/FLOP counts, identical PyTorch outputs, valid ONNX artifacts, ONNX Runtime CPU execution, and approximate PyTorch/ONNX output equivalence.

## Gate 2: Structural pruning

**Status: not implemented.**

- Physically remove channels.
- Repair every dependent branch.
- Retain runnable PyTorch execution.
- Export valid ONNX.
- Execute with ONNX Runtime CPU.

Acceptance requires a non-zero structural reduction, valid dense tensor shapes at every addition, complete BatchNorm/convolution/classifier repair, successful ONNX validation, and successful CPU inference for baseline and optimized artifacts.

## Gate 3: Evaluation

**Status: not implemented.** No dataset, task-accuracy result, or accuracy-improvement claim is part of Gate 1.

- Measure baseline accuracy.
- Measure optimized accuracy.
- Report absolute accuracy loss.
- Keep fidelity proxy separate from task accuracy.

Acceptance requires a documented real dataset or defensible fixed validation subset, deterministic preprocessing and sample selection, explicit baseline and optimized task metrics, and no substitution of the fidelity proxy for task accuracy. No target accuracy number is assumed before measurement.

## Gate 4: Deployment evidence

**Status: not implemented as a performance gate.** Gate 1 confirms deployability and records diagnostic CPU timings, but those timings support no latency claim.

- Collect p50 and p95 latency.
- Repeat runs.
- Record thread configuration.
- Disclose regressions.
- Avoid cherry-picking.
- Compare deterministic structural artifacts independently from noisy latency.

Acceptance requires a documented run protocol, all planned repetitions retained, machine/software/process provenance, complete latency distributions, and honest reporting whether latency improved or regressed.

## Gate 5: Acceptance

**Status: not implemented.** Full V1 acceptance depends on Gates 2 through 4.

- Structural reduction must be non-zero.
- Task accuracy must be explicitly measured.
- Latency must be reported whether improved or regressed.
- All official reports must validate against the report schema.
- Every changed channel dimension must appear in the graph diff.
- Repeated identical runs must reproduce model and graph hashes.
- Baseline and optimized ONNX artifacts must pass validation and execute with ONNX Runtime CPU.
- Configuration, exclusions, minimum channel constraints, alignment constraints, dataset/subset identity, and environment provenance must be complete.

V1 is accepted only when every gate passes with measured evidence. The project will not invent target performance numbers or promise latency improvement before those measurements exist.
