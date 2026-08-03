# SparseFlow

SparseFlow V0/V1 demonstrates dependency-aware physical `Conv2d` channel
pruning and emits reproducible structural and runtime evidence. It produces a
smaller dense PyTorch model, exports and validates ONNX, benchmarks with ONNX
Runtime on CPU, and writes a schema-validated evidence report.

The bundled untrained CNN demonstrates the mechanism offline. Its latency and
fidelity results are not commercial performance or task-accuracy claims.

## Current Scope

- Deterministic L1 weight-magnitude channel selection
- Physical channel removal in sequential Conv-BN-Conv dependency chains
- BatchNorm feature repair and following Conv2d input repair
- Compact dense PyTorch output without final masks
- ONNX export, checker validation, and artifact hashing
- ONNX Runtime `CPUExecutionProvider` latency samples, p50, and p95
- Parameter counts and PyTorch/ONNX serialized sizes
- Explicit Conv2d/Linear MAC counting with `1 MAC = 2 FLOPs`
- Unsupported compute operators listed rather than assigned zero cost
- Semantic graph changes and deterministic structural hashes
- Hardware, software, process, Git, and dirty-tree fingerprints
- Atomic deterministic JSON output validated against schema `0.2.0`

The reference pruning pass supports the model's linear `nn.Sequential`
Conv2d-BatchNorm2d chains. It does not claim general ResNet, residual,
concatenation, grouped-convolution, or depthwise-convolution support.

## Out Of Scope

2:4 structured sparsity is not part of current V0/V1. CUDA kernels, A100 and
other GPU results, sparse metadata encodings, and GPU runtime work belong to
archived prior research. Quantization is not implemented. MLIR is only a
possible future backend direction and is not current functionality.

## Install And Run

The benchmark does not download weights or datasets.

```bash
python -m pip install -r requirements.txt
python run_bench.py --pass channel-prune --pruning-ratio 0.375
python run_bench.py --pass noop
```

Useful reproducibility controls:

```bash
python run_bench.py --pass channel-prune --pruning-ratio 0.375 \
  --seed 1234 --warmup 5 --measured-runs 30 --threads 1 \
  --output report.json
```

Run the test suite with its configured 80 percent coverage floor:

```bash
pytest -q
```

## Evidence Report

Official output is assembled by `sparseflow.report`, validated against
`schemas/evidence-report.schema.json`, serialized with sorted keys and finite
JSON values, and written atomically. Missing mandatory evidence raises
`EvidenceValidationError`; no official report is written.

The report has this top-level structure:

```json
{
  "schema_version": "0.2.0",
  "run": {},
  "environment": {},
  "model": {},
  "optimization": {},
  "metrics": {
    "parameters": {},
    "serialized_size_bytes": {},
    "latency_ms": {},
    "compute": {},
    "fidelity": {}
  },
  "graph": {},
  "artifacts": {}
}
```

The structured optimization configuration, graph-change list, graph-diff
hash, structural hashes, model hashes, ONNX hashes, latency methodology, raw
samples, and environment fingerprint are recorded in every valid report.

## Fidelity Limitation

Fidelity compares deterministic baseline and optimized output tensors for a
proxy input. The NoOp control uses tight numerical tolerances. A pruned model
is expected to differ, so its report records absolute and relative error and a
finite-output relative-error threshold. This is explicitly not task accuracy;
a trained model and representative evaluation dataset are required for that.

## Roadmap

1. Add tested dependency handling for selected residual and concatenation patterns.
2. Add model adapters and real evaluation datasets without changing evidence semantics.
3. Add optional fine-tuning and task-metric integrations with explicit provenance.
4. Evaluate MLIR only as a future backend after the V0/V1 evidence contract is stable.
