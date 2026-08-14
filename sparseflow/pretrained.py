"""SparseFlow v0.3 Gate A steps 1-4: pretrained ResNet-18 import validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from sparseflow.models import ResNet18Reference, build_resnet18_reference


GATE_A_SCHEMA_VERSION = "0.1.0"
GATE_A_WEIGHTS_ENUM = "ResNet18_Weights.IMAGENET1K_V1"
GATE_A_EXPECTED_TOP1 = 69.758
GATE_A_EXPECTED_TOP5 = 89.078
GATE_A_ATOL = 1.0e-6
GATE_A_RTOL = 1.0e-5
GATE_A_STAGE_PATHS = ("maxpool", "layer1", "layer2", "layer3", "layer4")


class PretrainedImportValidationError(RuntimeError):
    """Raised when pretrained import evidence fails closed."""


@dataclass(frozen=True)
class TensorComparison:
    max_abs_error: float
    max_rel_error: float
    argmax_agreement: bool
    close: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "argmax_agreement": self.argmax_agreement,
            "close": self.close,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_path(weights: ResNet18_Weights) -> Path:
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _module_signature(module: nn.Module) -> dict[str, Any]:
    if isinstance(module, nn.Conv2d):
        return {
            "type": "Conv2d",
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "dilation": list(module.dilation),
            "groups": module.groups,
            "bias": module.bias is not None,
        }
    if isinstance(module, nn.BatchNorm2d):
        return {
            "type": "BatchNorm2d",
            "num_features": module.num_features,
            "eps": module.eps,
            "momentum": module.momentum,
            "affine": module.affine,
            "track_running_stats": module.track_running_stats,
        }
    if isinstance(module, nn.ReLU):
        # In-place storage behavior is intentionally excluded: SparseFlow uses
        # inplace=False while torchvision uses inplace=True. Placement/type are
        # numerically equivalent and are asserted by path/type matching.
        return {"type": "ReLU"}
    if isinstance(module, nn.MaxPool2d):
        kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        dilation = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation)
        return {
            "type": "MaxPool2d",
            "kernel_size": list(kernel_size),
            "stride": list(stride),
            "padding": list(padding),
            "dilation": list(dilation),
            "ceil_mode": module.ceil_mode,
        }
    if isinstance(module, nn.AdaptiveAvgPool2d):
        output_size = module.output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        return {"type": "AdaptiveAvgPool2d", "output_size": list(output_size)}
    if isinstance(module, nn.Linear):
        return {
            "type": "Linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
    return {"type": type(module).__name__}


def _semantic_module_map(model: nn.Module) -> dict[str, dict[str, Any]]:
    supported = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear)
    return {
        path: _module_signature(module)
        for path, module in model.named_modules()
        if path and isinstance(module, supported)
    }


def _assert_eval_mode(model: nn.Module, *, label: str) -> None:
    training_paths = [path or "<root>" for path, module in model.named_modules() if module.training]
    if training_paths:
        raise PretrainedImportValidationError(
            f"{label} contains modules in training mode: {', '.join(training_paths[:8])}"
        )


def _compare_tensors(reference: torch.Tensor, candidate: torch.Tensor, *, atol: float, rtol: float) -> TensorComparison:
    if reference.shape != candidate.shape:
        raise PretrainedImportValidationError(
            f"tensor shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}"
        )
    delta = (reference - candidate).abs()
    denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).eps)
    relative = delta / denominator
    return TensorComparison(
        max_abs_error=float(delta.max().item()),
        max_rel_error=float(relative.max().item()),
        argmax_agreement=bool(torch.equal(reference.argmax(dim=-1), candidate.argmax(dim=-1))) if reference.ndim >= 2 else True,
        close=bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
    )


def _capture_stage_outputs(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs: dict[str, torch.Tensor] = {}
    hooks = []

    modules = dict(model.named_modules())
    for path in GATE_A_STAGE_PATHS:
        module = modules.get(path)
        if module is None:
            raise PretrainedImportValidationError(f"missing stage module: {path}")

        def capture(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor, *, name: str = path) -> None:
            outputs[name] = output.detach().cpu().clone()

        hooks.append(module.register_forward_hook(capture))

    try:
        with torch.no_grad():
            logits = model(inputs).detach().cpu()
    finally:
        for hook in hooks:
            hook.remove()
    return logits, outputs


def load_and_validate_pretrained_resnet18(
    *,
    seed: int = 1234,
    batch_size: int = 2,
    atol: float = GATE_A_ATOL,
    rtol: float = GATE_A_RTOL,
    progress: bool = False,
) -> tuple[ResNet18Reference, dict[str, Any]]:
    """Import official torchvision weights and fail closed on semantic divergence.

    This is Gate A steps 1-4 only. It does not perform ImageNet evaluation.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    weights = ResNet18_Weights.IMAGENET1K_V1
    torchvision_model = resnet18(weights=weights, progress=progress).eval()
    sparseflow_model = build_resnet18_reference(seed=seed).eval()

    reference_state = torchvision_model.state_dict()
    candidate_state = sparseflow_model.state_dict()
    reference_keys = tuple(reference_state.keys())
    candidate_keys = tuple(candidate_state.keys())
    if reference_keys != candidate_keys:
        missing = sorted(set(reference_keys) - set(candidate_keys))
        unexpected = sorted(set(candidate_keys) - set(reference_keys))
        raise PretrainedImportValidationError(
            f"state-dict key mismatch: missing={missing} unexpected={unexpected}"
        )
    shape_mismatches = {
        key: (tuple(reference_state[key].shape), tuple(candidate_state[key].shape))
        for key in reference_keys
        if reference_state[key].shape != candidate_state[key].shape
    }
    if shape_mismatches:
        raise PretrainedImportValidationError(f"state-dict shape mismatch: {shape_mismatches}")

    load_result = sparseflow_model.load_state_dict(reference_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise PretrainedImportValidationError(
            f"strict state-dict import failed: missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
        )

    reference_modules = _semantic_module_map(torchvision_model)
    candidate_modules = _semantic_module_map(sparseflow_model)
    if reference_modules != candidate_modules:
        differing = sorted(
            path
            for path in set(reference_modules) | set(candidate_modules)
            if reference_modules.get(path) != candidate_modules.get(path)
        )
        raise PretrainedImportValidationError(
            f"semantic module configuration mismatch at: {', '.join(differing[:12])}"
        )

    _assert_eval_mode(torchvision_model, label="torchvision model")
    _assert_eval_mode(sparseflow_model, label="SparseFlow model")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.randn((batch_size, 3, 224, 224), generator=generator, dtype=torch.float32)

    reference_logits, reference_stages = _capture_stage_outputs(torchvision_model, inputs)
    candidate_logits, candidate_stages = _capture_stage_outputs(sparseflow_model, inputs)

    stage_results: dict[str, dict[str, Any]] = {}
    first_divergent_stage: str | None = None
    for path in GATE_A_STAGE_PATHS:
        result = _compare_tensors(reference_stages[path], candidate_stages[path], atol=atol, rtol=rtol)
        stage_results[path] = result.as_dict()
        if not result.close and first_divergent_stage is None:
            first_divergent_stage = path

    logit_result = _compare_tensors(reference_logits, candidate_logits, atol=atol, rtol=rtol)
    if first_divergent_stage is not None or not logit_result.close or not logit_result.argmax_agreement:
        raise PretrainedImportValidationError(
            "pretrained import equivalence failed: "
            f"first_divergent_stage={first_divergent_stage} "
            f"logit_max_abs_error={logit_result.max_abs_error:.9g} "
            f"argmax_agreement={logit_result.argmax_agreement}"
        )

    checkpoint = _checkpoint_path(weights)
    checkpoint_sha256 = _sha256_file(checkpoint) if checkpoint.exists() else None
    imported_state_hash = _sha256_state_dict(sparseflow_model.state_dict())
    reference_state_hash = _sha256_state_dict(reference_state)
    if imported_state_hash != reference_state_hash:
        raise PretrainedImportValidationError("imported state-dict content hash does not match torchvision")

    evidence = {
        "schema_version": GATE_A_SCHEMA_VERSION,
        "gate": "v0.3_gate_a_steps_1_4",
        "weights": {
            "torchvision_version": torchvision.__version__,
            "weights_enum": GATE_A_WEIGHTS_ENUM,
            "weights_url": weights.url,
            "checkpoint_filename": Path(urlparse(weights.url).path).name,
            "checkpoint_sha256": checkpoint_sha256,
            "state_dict_sha256": imported_state_hash,
            "expected_imagenet_top1_percent": GATE_A_EXPECTED_TOP1,
            "expected_imagenet_top5_percent": GATE_A_EXPECTED_TOP5,
        },
        "import_validation": {
            "state_dict_keys_match": True,
            "state_dict_shapes_match": True,
            "strict_load_passed": True,
            "semantic_module_configuration_match": True,
            "all_modules_eval": True,
            "fixed_input": {
                "seed": seed,
                "shape": [batch_size, 3, 224, 224],
                "dtype": "float32",
            },
            "tolerance_atol": atol,
            "tolerance_rtol": rtol,
            "per_stage": stage_results,
            "final_logits": logit_result.as_dict(),
            "first_divergent_stage": None,
            "logit_equivalence": True,
        },
        "dataset_evaluation": {
            "status": "not_run",
            "reason": "Gate A steps 5-7 require an ImageNet-1K validation dataset manifest",
            "baseline_reproduction_passed": None,
        },
        "gate_passed": True,
    }
    return sparseflow_model, evidence
