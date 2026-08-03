import torch.nn as nn

from sparseflow.compute import count_compute
from sparseflow.models import INPUT_SHAPE, build_reference_model
from sparseflow.passes import ChannelPruningPass


def test_pruning_reduces_macs_and_flops():
    baseline = build_reference_model(seed=4)
    optimized = ChannelPruningPass(ratio=0.375).apply(baseline).model

    baseline_compute = count_compute(baseline, INPUT_SHAPE)
    optimized_compute = count_compute(optimized, INPUT_SHAPE)

    assert optimized_compute["macs"] < baseline_compute["macs"]
    assert optimized_compute["flops"] < baseline_compute["flops"]
    assert baseline_compute["flops"] == 2 * baseline_compute["macs"]
    assert baseline_compute["implementation"] == "sparseflow.fx_conv_linear_counter_v1"


def test_unsupported_operators_are_explicit():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Sigmoid()).eval()
    result = count_compute(model, (1, 3, 8, 8))

    assert "Sigmoid" in result["unsupported_operators"]
    assert result["count_is_complete"] is False
