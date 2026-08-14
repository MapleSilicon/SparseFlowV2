import json

import torch

from sparseflow.dependencies import analyze_resnet18_dependencies
from sparseflow.graph_diff import model_state_hash
from sparseflow.models import (
    RESNET18_INPUT_SHAPE,
    BasicBlock,
    build_resnet18_reference,
)
from sparseflow.serialization import sha256_json


def test_resnet18_reference_is_deterministic_and_has_expected_architecture():
    first = build_resnet18_reference(seed=1234)
    second = build_resnet18_reference(seed=1234)

    assert model_state_hash(first) == model_state_hash(second)
    assert sum(isinstance(module, BasicBlock) for module in first.modules()) == 8
    assert [len(first.layer1), len(first.layer2), len(first.layer3), len(first.layer4)] == [2, 2, 2, 2]
    assert sum(block.downsample is not None for block in first.modules() if isinstance(block, BasicBlock)) == 3

    with torch.no_grad():
        output = first(torch.zeros(1, *RESNET18_INPUT_SHAPE[1:]))
    assert output.shape == (1, 1000)


def test_resnet18_dependency_graph_is_complete_json_and_deterministic():
    first = analyze_resnet18_dependencies(build_resnet18_reference(seed=7))
    second = analyze_resnet18_dependencies(build_resnet18_reference(seed=7))
    payload = first.as_dict()

    assert payload == second.as_dict()
    assert sha256_json(payload) == sha256_json(second.as_dict())
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert payload["summary"] == {
        "architecture_identifier": "resnet18_basicblock_reference_v1",
        "residual_group_count": 8,
        "projection_shortcut_count": 3,
        "identity_shortcut_count": 5,
        "conv_bn_dependency_count": 20,
        "classifier_dependency_count": 1,
        "channel_constraint_count": 8,
    }

    groups = {group["block_name"]: group for group in payload["residual_groups"]}
    assert groups["layer1.0"]["projection_exists"] is False
    assert groups["layer1.0"]["skip_branch"] == ["layer1.0.identity"]
    assert groups["layer2.0"]["projection_exists"] is True
    assert groups["layer2.0"]["skip_branch"] == [
        "layer2.0.downsample.0",
        "layer2.0.downsample.1",
    ]
    assert all(group["downstream_consumers"] for group in groups.values())

    conv_bn = {
        (edge["source"], edge["target"])
        for edge in payload["conv_bn_dependencies"]
    }
    assert ("conv1", "bn1") in conv_bn
    assert ("layer4.0.downsample.0", "layer4.0.downsample.1") in conv_bn
    assert payload["classifier_dependency"] == {
        "producer": "layer4.1.add",
        "pool": "avgpool",
        "pooled_feature_width": 512,
        "linear": "fc",
        "linear_input_features": 512,
        "dependency_type": "pooled_features_to_linear_input",
    }
