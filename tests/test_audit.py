import torch

from sparseflow.audit import (
    collect_audit_metadata,
    collect_hardware_fingerprint,
    set_determinism,
)


def test_hardware_fingerprint_has_mandatory_portable_fields():
    fingerprint = collect_hardware_fingerprint()

    assert fingerprint["operating_system"]
    assert fingerprint["architecture"]
    assert fingerprint["machine"]
    assert fingerprint["logical_cpu_count"] >= 1
    assert "cpu_model" in fingerprint


def test_seed_control_repeats_torch_values(tmp_path):
    set_determinism(44)
    first = torch.randn(4)
    set_determinism(44)
    second = torch.randn(4)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    audit = collect_audit_metadata(tmp_path, seed=44, threads=1)
    assert audit["repository"]["path"] == str(tmp_path.resolve())
    assert audit["hardware"]["logical_cpu_count"] >= 1
    assert audit["process"]["onnxruntime_intra_op_threads"] == 1
