# SparseFlow V0 Reference Run

These two curated reports use the bundled untrained micro-model. They validate the evidence mechanism: physical channel pruning and a structurally neutral NoOp control flow through the same export, measurement, audit, hashing, and schema-validation pipeline.

They are not a commercial benchmark. The model is extremely small, so latency is noise-sensitive. Structural reductions do not automatically imply latency reductions, and the recorded timings do not establish production speedup or cost savings.

The machine, software versions, complete resolved configuration, and Git commit are captured in each report, along with the dirty-tree status at generation time. The channel-pruning report demonstrates non-zero physical reduction; the NoOp report demonstrates zero structural delta. Repeated-run output is omitted because it adds no unique reference artifact.

Files:

- `channel-prune-report.json`: `micro-demo` physical channel-pruning mechanism demonstration.
- `noop-report.json`: `noop-control` structural-neutrality control.
