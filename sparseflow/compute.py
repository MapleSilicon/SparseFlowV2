"""Explicit MAC/FLOP counter for Conv2d and Linear operations."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
import torch.nn as nn


COUNTER_IDENTIFIER = "sparseflow.fx_conv_linear_counter_v1"


def _operator_name(target: Any) -> str:
    return getattr(target, "__name__", str(target))


def count_compute(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
    if len(input_shape) != 4 or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape must contain four positive dimensions")

    traced = fx.symbolic_trace(model.eval())
    example = torch.zeros(input_shape, dtype=torch.float32)
    ShapeProp(traced).propagate(example)
    modules = dict(traced.named_modules())
    macs = 0
    unsupported: set[str] = set()

    for node in traced.graph.nodes:
        if node.op == "call_module":
            module = modules[str(node.target)]
            tensor_meta = node.meta.get("tensor_meta")
            if tensor_meta is None:
                unsupported.add(module.__class__.__name__)
                continue
            output_elements = math.prod(tensor_meta.shape)
            if isinstance(module, nn.Conv2d):
                kernel_elements = math.prod(module.kernel_size)
                macs += output_elements * kernel_elements * (module.in_channels // module.groups)
            elif isinstance(module, nn.Linear):
                macs += output_elements * module.in_features
            else:
                unsupported.add(module.__class__.__name__)
        elif node.op in {"call_function", "call_method"}:
            unsupported.add(_operator_name(node.target))

    return {
        "input_shape": list(input_shape),
        "macs": int(macs),
        "flops": int(2 * macs),
        "convention": "1 MAC = 2 FLOPs",
        "implementation": COUNTER_IDENTIFIER,
        "unsupported_operators": sorted(unsupported),
        "count_is_complete": not unsupported,
    }
