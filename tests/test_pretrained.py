import pytest
import torch
import torch.nn as nn

from sparseflow.pretrained import (
    PretrainedImportValidationError,
    _assert_eval_mode,
    _compare_tensors,
    _module_signature,
    _semantic_module_map,
)
from sparseflow.models import build_resnet18_reference


def test_tensor_comparison_records_measured_errors_and_argmax():
    reference = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    candidate = reference + torch.tensor([[0.0, 5.0e-7, -5.0e-7]], dtype=torch.float32)
    result = _compare_tensors(reference, candidate, atol=1.0e-6, rtol=1.0e-5)
    assert result.close is True
    assert result.argmax_agreement is True
    assert result.max_abs_error > 0.0
    assert result.max_rel_error > 0.0


def test_tensor_comparison_rejects_shape_mismatch():
    with pytest.raises(PretrainedImportValidationError, match="tensor shape mismatch"):
        _compare_tensors(torch.zeros(1, 2), torch.zeros(1, 3), atol=1.0e-6, rtol=1.0e-5)


def test_eval_mode_is_explicit_precondition():
    model = build_resnet18_reference().train()
    with pytest.raises(PretrainedImportValidationError, match="training mode"):
        _assert_eval_mode(model, label="candidate")
    model.eval()
    _assert_eval_mode(model, label="candidate")


def test_batchnorm_signature_pins_numerical_configuration():
    bn = nn.BatchNorm2d(32, eps=1.0e-5, momentum=0.1, affine=True, track_running_stats=True)
    assert _module_signature(bn) == {
        "type": "BatchNorm2d",
        "num_features": 32,
        "eps": 1.0e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
    }


def test_relu_signature_checks_placement_type_not_inplace_storage_policy():
    assert _module_signature(nn.ReLU(inplace=False)) == {"type": "ReLU"}
    assert _module_signature(nn.ReLU(inplace=True)) == {"type": "ReLU"}


def test_reference_semantic_map_contains_projection_and_classifier_paths():
    modules = _semantic_module_map(build_resnet18_reference())
    assert modules["layer2.0.downsample.0"]["type"] == "Conv2d"
    assert modules["layer2.0.downsample.1"]["type"] == "BatchNorm2d"
    assert modules["layer3.0.downsample.0"]["stride"] == [2, 2]
    assert modules["layer4.0.downsample.0"]["stride"] == [2, 2]
    assert modules["fc"] == {
        "type": "Linear",
        "in_features": 512,
        "out_features": 1000,
        "bias": True,
    }
