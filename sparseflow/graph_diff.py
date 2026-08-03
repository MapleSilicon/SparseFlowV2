"""Deterministic model structure and transformation evidence."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn

from sparseflow.passes import GraphChange
from sparseflow.serialization import sha256_json


def _module_attributes(module: nn.Module) -> dict[str, Any]:
    if isinstance(module, nn.Conv2d):
        return {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "dilation": list(module.dilation),
            "groups": module.groups,
        }
    if isinstance(module, nn.BatchNorm2d):
        return {
            "num_features": module.num_features,
            "affine": module.affine,
            "track_running_stats": module.track_running_stats,
        }
    if isinstance(module, nn.Linear):
        return {"in_features": module.in_features, "out_features": module.out_features}
    if isinstance(module, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        return {"type": module.__class__.__name__}
    if isinstance(module, nn.ReLU):
        return {"inplace": module.inplace}
    return {"type": module.__class__.__name__}


def structural_snapshot(model: nn.Module) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        if not path or any(True for _ in module.children()):
            continue
        snapshot.append(
            {
                "path": path,
                "op_type": module.__class__.__name__,
                "attributes": _module_attributes(module),
                "parameters": {
                    name: list(parameter.shape)
                    for name, parameter in sorted(module.named_parameters(recurse=False))
                },
                "buffers": {
                    name: list(buffer.shape)
                    for name, buffer in sorted(module.named_buffers(recurse=False))
                    if buffer is not None
                },
            }
        )
    return snapshot


def structural_hash(model: nn.Module) -> str:
    return sha256_json(structural_snapshot(model))


def graph_changes_as_dict(changes: list[GraphChange]) -> list[dict[str, Any]]:
    return [
        {
            "node": change.node,
            "op_type": change.op_type,
            "change_type": change.change_type,
            "before": change.before,
            "after": change.after,
            "metadata": change.metadata,
        }
        for change in changes
    ]


def graph_changes_hash(changes: list[GraphChange]) -> str:
    return sha256_json(graph_changes_as_dict(changes))


def model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        try:
            payload = value.numpy().tobytes(order="C")
        except TypeError:
            payload = value.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(payload)
    return digest.hexdigest()


def fx_graph_text(model: nn.Module) -> str:
    return str(fx.symbolic_trace(model).graph)
