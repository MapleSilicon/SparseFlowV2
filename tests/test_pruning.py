import json

import pytest
import torch
import torch.nn as nn

from sparseflow.benchmark import count_parameters
from sparseflow.compute import count_compute
from sparseflow.graph_diff import graph_changes_hash, model_state_hash
from sparseflow.models import INPUT_SHAPE, build_reference_model
from sparseflow.passes import ChannelPruningPass


def _modules(model, module_type):
    return [module for module in model.features if isinstance(module, module_type)]


def test_channel_pruning_physically_repairs_dependencies_and_runs():
    baseline = build_reference_model(seed=1234)
    result = ChannelPruningPass(ratio=0.375).apply(baseline)
    optimized = result.model

    baseline_convs = _modules(baseline, nn.Conv2d)
    optimized_convs = _modules(optimized, nn.Conv2d)
    optimized_bns = _modules(optimized, nn.BatchNorm2d)

    assert [conv.out_channels for conv in optimized_convs] == [5, 10, 24]
    assert [conv.in_channels for conv in optimized_convs] == [3, 5, 10]
    assert [bn.num_features for bn in optimized_bns] == [5, 10, 24]
    assert optimized_convs[0].weight.shape != baseline_convs[0].weight.shape
    assert count_parameters(optimized) < count_parameters(baseline)

    with torch.no_grad():
        output = optimized(torch.randn(2, 3, 16, 16))
    assert output.shape == (2, 4)

    change_types = [change.change_type for change in result.graph_changes]
    assert "prune_output_channels" in change_types
    assert "dependency_repair" in change_types
    assert all(change.before != change.after for change in result.graph_changes)
    assert all("reason" in change.metadata for change in result.graph_changes)


def test_pruning_configuration_and_artifacts_are_deterministic():
    pass_a = ChannelPruningPass(ratio=0.375)
    pass_b = ChannelPruningPass(ratio=0.375)
    result_a = pass_a.apply(build_reference_model(seed=99))
    result_b = pass_b.apply(build_reference_model(seed=99))

    assert pass_a.config() == pass_b.config()
    assert pass_a.config() != ChannelPruningPass(ratio=0.25).config()
    assert "0x" not in json.dumps(pass_a.config(), sort_keys=True)
    assert result_a.graph_changes == result_b.graph_changes
    assert graph_changes_hash(result_a.graph_changes) == graph_changes_hash(result_b.graph_changes)
    assert model_state_hash(result_a.model) == model_state_hash(result_b.model)
    assert count_parameters(result_a.model) == count_parameters(result_b.model)
    assert count_compute(result_a.model, INPUT_SHAPE) == count_compute(
        result_b.model, INPUT_SHAPE
    )


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 2.0])
def test_pruning_ratio_is_validated(ratio):
    with pytest.raises(ValueError, match="ratio"):
        ChannelPruningPass(ratio=ratio)
