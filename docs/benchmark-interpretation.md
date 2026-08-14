# Benchmark Interpretation

SparseFlow V0 records structural and runtime evidence separately. A change in one category must not be presented as proof of a change in the other.

## Structural metrics

**Parameter count** reports the number of scalar model parameters. Physical pruning must lower this count when channels are removed; masks alone would not.

**Serialized PyTorch size** is the byte length of a serialized `state_dict`. It reflects stored tensors and serialization overhead, so it need not fall in exact proportion to parameter count.

**Serialized ONNX size** is the checked ONNX artifact size. Graph metadata and encoding overhead can make its reduction differ from the PyTorch size reduction.

**MACs and FLOPs** are counted for supported `Conv2d` and `Linear` operations at the recorded input shape, using `1 MAC = 2 FLOPs`. Unsupported operators are disclosed. Parameter reduction is not equivalent to compute reduction.

**Semantic graph changes** list each physical channel-dimension change and its dependencies. Structural hashes and the graph-diff hash support deterministic comparison without treating runtime timings as deterministic.

## Runtime metrics

The harness exports baseline and optimized ONNX artifacts and measures them with ONNX Runtime `CPUExecutionProvider`. A deterministic seeded policy randomizes baseline-comparison order within each pair. The report records the recipe hash, warmup runs, pair repetitions, runs per side, execution order, intra-op and inter-op thread settings, every latency sample and its hash, paired differences, paired improvement percentages, p50, p95, minimum, maximum, arithmetic mean, population standard deviation, and coefficient of variation.

p50 describes the median observed execution. p95 exposes the slower tail. Neither is replaced by the arithmetic mean. Warmup runs execute before collection so initial setup is less likely to dominate measured samples, but warmup does not eliminate all variability.

Microbenchmark noise can come from operating-system scheduling, process contention, clock resolution, power management, caches, allocator state, operator dispatch, thread-pool behavior, tensor dimensions, and ONNX Runtime kernel selection. Smaller models may run slower despite using fewer parameters or MACs because fixed overhead and kernel efficiency can dominate useful compute.

Gate 1.5 also loads the baseline ONNX artifact into two independent sessions and measures an identical-artifact null distribution. Model, PyTorch, ONNX, and structural graph hashes must match before calibration is allowed. A two-sided empirical percentile interval is computed from paired null improvements, and the maximum absolute bound becomes the minimum detectable improvement (MDI). The derivation and raw null samples are evidence, not a hardcoded threshold.

SparseFlow does not report latency improvements below the calibrated minimum detectable improvement. It also flags latency as noise-sensitive when baseline p50 is below `0.1 ms`, when baseline or optimized coefficient of variation exceeds `0.10`, or when the observed paired median improvement does not exceed the calibrated MDI. These deterministic rules do not make an unflagged run production-representative.

## Interpreting reductions

Parameter reduction is not equivalent to compute reduction. MAC reduction is not equivalent to latency reduction. Structural reductions do not automatically imply latency reductions.

The NoOp control validates structural neutrality: parameter counts, compute counts, graph changes, and structural/model hashes remain neutral. Its baseline and optimized latency can still vary slightly because independently loaded sessions are measured in randomized pairs.

Latency claims require larger realistic workloads, repeated controlled runs, stable machine configuration, explicit thread settings, and disclosure of both improvements and regressions. A single tiny-model result cannot support production throughput or cost claims.

## What SparseFlow V0 proves

- The supported pass physically removes channels from the bundled untrained micro-model.
- Directly dependent BatchNorm and following convolution dimensions are repaired for that model.
- Baseline and optimized models execute in PyTorch and ONNX Runtime CPU.
- Structural metrics, semantic graph changes, artifacts, provenance, and resolved configuration are captured in a schema-validated report.
- Identical runs reproduce model hashes, graph-diff hashes, configuration, parameter counts, and MAC/FLOP counts.
- NoOp produces neutral structural evidence and passing fidelity.

## What SparseFlow V0 does not prove

- Production speedup, throughput, service-level performance, or commercial cost savings.
- Task accuracy, because the bundled model is untrained and fidelity is only an output proxy.
- General dependency-aware pruning across residual, grouped, depthwise, concatenated, or arbitrary networks.
- ResNet-18 support or a promise that a future measured latency result will improve.
