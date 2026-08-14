"""Fail-closed ImageNet-1K validation preparation and baseline evaluation.

The preparation phase is intentionally separate from measurement. It freezes the
validation manifest, WNID-to-index mapping, preprocessing identity, and accuracy
tolerances before any baseline accuracy is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import default_loader
from torchvision.models import ResNet18_Weights

from sparseflow.pretrained import (
    GATE_A_EXPECTED_TOP1,
    GATE_A_EXPECTED_TOP5,
    GATE_A_WEIGHTS_ENUM,
    load_and_validate_pretrained_resnet18,
)
from sparseflow.serialization import atomic_write_json, sha256_file, sha256_json, stable_json_dumps


IMAGENET_VAL_EXPECTED_SAMPLES = 50_000
IMAGENET_EXPECTED_CLASSES = 1_000
BASELINE_TOP1_TOLERANCE_POINTS = 0.20
BASELINE_TOP5_TOLERANCE_POINTS = 0.20
BASELINE_PRECOMMIT_SCHEMA_VERSION = "0.1.0"
BASELINE_RESULT_SCHEMA_VERSION = "0.1.0"


class ImageNetBaselineValidationError(RuntimeError):
    """Raised when ImageNet baseline preconditions or evidence fail closed."""


@dataclass(frozen=True)
class ManifestEntry:
    relative_path: str
    wnid: str
    label: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "wnid": self.wnid,
            "label": self.label,
        }


def _canonical_jsonl_bytes(entries: Iterable[ManifestEntry]) -> bytes:
    chunks: list[bytes] = []
    for entry in entries:
        # stable_json_dumps includes exactly one trailing newline.
        chunks.append(stable_json_dumps(entry.as_dict()).encode("utf-8"))
    return b"".join(chunks)


def _atomic_write_bytes(value: bytes, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _class_order_records(wnids: list[str]) -> list[dict[str, Any]]:
    return [{"index": index, "wnid": wnid} for index, wnid in enumerate(wnids)]


def assert_canonical_imagenet_class_order(dataset: ImageNet) -> list[dict[str, Any]]:
    """Validate torchvision ImageNet's WNID/index and sample-folder contract.

    torchvision.datasets.ImageNet delegates class discovery to ImageFolder,
    whose class names are sorted before indices are assigned. We make that
    behavior an explicit runtime invariant and additionally verify every sample's
    parent WNID agrees with its integer target.
    """

    if getattr(dataset, "split", None) != "val":
        raise ImageNetBaselineValidationError("ImageNet baseline requires split='val'")

    wnids = list(getattr(dataset, "wnids", []))
    wnid_to_idx = dict(getattr(dataset, "wnid_to_idx", {}))
    samples = list(getattr(dataset, "samples", []))

    if len(wnids) != IMAGENET_EXPECTED_CLASSES:
        raise ImageNetBaselineValidationError(
            f"expected {IMAGENET_EXPECTED_CLASSES} ImageNet WNIDs, found {len(wnids)}"
        )
    if len(set(wnids)) != IMAGENET_EXPECTED_CLASSES:
        raise ImageNetBaselineValidationError("ImageNet WNIDs are not unique")
    if wnids != sorted(wnids):
        raise ImageNetBaselineValidationError("ImageNet WNIDs are not in ImageFolder canonical sorted order")

    expected_map = {wnid: index for index, wnid in enumerate(wnids)}
    if wnid_to_idx != expected_map:
        raise ImageNetBaselineValidationError("ImageNet wnid_to_idx does not match canonical sorted WNID order")

    if len(samples) != IMAGENET_VAL_EXPECTED_SAMPLES:
        raise ImageNetBaselineValidationError(
            f"expected {IMAGENET_VAL_EXPECTED_SAMPLES} validation samples, found {len(samples)}"
        )

    for path_text, target in samples:
        path = Path(path_text)
        wnid = path.parent.name
        expected_target = expected_map.get(wnid)
        if expected_target is None:
            raise ImageNetBaselineValidationError(f"sample is under unknown WNID folder: {path}")
        if int(target) != expected_target:
            raise ImageNetBaselineValidationError(
                f"sample target/folder mismatch: path={path} target={target} expected={expected_target}"
            )

    return _class_order_records(wnids)


def build_imagenet_manifest(dataset: ImageNet) -> list[ManifestEntry]:
    """Build a deterministic 50k validation manifest after class-order checks."""

    class_records = assert_canonical_imagenet_class_order(dataset)
    wnid_to_idx = {record["wnid"]: record["index"] for record in class_records}
    root = Path(dataset.root).resolve()

    entries: list[ManifestEntry] = []
    for path_text, target in dataset.samples:
        path = Path(path_text).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ImageNetBaselineValidationError(f"sample is outside ImageNet root: {path}") from exc
        wnid = path.parent.name
        expected_target = wnid_to_idx[wnid]
        if int(target) != expected_target:
            raise ImageNetBaselineValidationError(
                f"manifest target mismatch: {relative} target={target} expected={expected_target}"
            )
        entries.append(ManifestEntry(relative_path=relative, wnid=wnid, label=expected_target))

    entries.sort(key=lambda entry: entry.relative_path)
    if len({entry.relative_path for entry in entries}) != IMAGENET_VAL_EXPECTED_SAMPLES:
        raise ImageNetBaselineValidationError("validation manifest contains duplicate relative paths")
    return entries


def prepare_imagenet_baseline(
    *,
    imagenet_root: str | Path,
    output_dir: str | Path,
    top1_tolerance_points: float = BASELINE_TOP1_TOLERANCE_POINTS,
    top5_tolerance_points: float = BASELINE_TOP5_TOLERANCE_POINTS,
) -> dict[str, Any]:
    """Freeze dataset and gate identity before any baseline inference occurs."""

    if top1_tolerance_points <= 0.0 or top5_tolerance_points <= 0.0:
        raise ValueError("baseline accuracy tolerances must be positive")

    root = Path(imagenet_root).resolve()
    destination = Path(output_dir).resolve()
    weights = ResNet18_Weights.IMAGENET1K_V1

    # No transform is applied while constructing the manifest: preparation is
    # about dataset identity and labels only. Evaluation consumes weights.transforms().
    dataset = ImageNet(root=str(root), split="val", transform=None)
    class_records = assert_canonical_imagenet_class_order(dataset)
    manifest_entries = build_imagenet_manifest(dataset)

    class_order_path = destination / "class-order.json"
    manifest_path = destination / "manifest.jsonl"
    precommit_path = destination / "baseline-precommit.json"

    atomic_write_json(class_records, class_order_path)
    manifest_bytes = _canonical_jsonl_bytes(manifest_entries)
    _atomic_write_bytes(manifest_bytes, manifest_path)

    class_order_sha256 = sha256_file(class_order_path)
    manifest_sha256 = sha256_file(manifest_path)
    transforms = weights.transforms()

    precommit = {
        "schema_version": BASELINE_PRECOMMIT_SCHEMA_VERSION,
        "gate": "v0.3_gate_a_imagenet_baseline_precommit",
        "status": "prepared_not_run",
        "dataset": {
            "loader": "torchvision.datasets.ImageNet",
            "split": "val",
            "sample_count": len(manifest_entries),
            "class_count": len(class_records),
            "manifest_file": manifest_path.name,
            "manifest_sha256": manifest_sha256,
            "class_order_file": class_order_path.name,
            "class_order_sha256": class_order_sha256,
            "class_order_validated": True,
        },
        "weights": {
            "weights_enum": GATE_A_WEIGHTS_ENUM,
            "weights_url": weights.url,
        },
        "preprocessing": {
            "source": "weights.transforms()",
            "repr": repr(transforms),
        },
        "acceptance": {
            "expected_top1_percent": GATE_A_EXPECTED_TOP1,
            "expected_top5_percent": GATE_A_EXPECTED_TOP5,
            "top1_tolerance_points": float(top1_tolerance_points),
            "top5_tolerance_points": float(top5_tolerance_points),
        },
        "baseline_reproduction_passed": None,
    }
    atomic_write_json(precommit, precommit_path)
    return precommit


def _load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                entry = ManifestEntry(
                    relative_path=str(payload["relative_path"]),
                    wnid=str(payload["wnid"]),
                    label=int(payload["label"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ImageNetBaselineValidationError(
                    f"invalid manifest line {line_number}: {exc}"
                ) from exc
            entries.append(entry)
    if len(entries) != IMAGENET_VAL_EXPECTED_SAMPLES:
        raise ImageNetBaselineValidationError(
            f"manifest must contain {IMAGENET_VAL_EXPECTED_SAMPLES} entries, found {len(entries)}"
        )
    return entries


class _ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, entries: list[ManifestEntry], transform: Any) -> None:
        self.root = root
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[index]
        path = (self.root / entry.relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ImageNetBaselineValidationError(f"manifest path escapes dataset root: {entry.relative_path}") from exc
        image = default_loader(str(path))
        return self.transform(image), entry.label


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    predictions = logits.topk(k, dim=1, largest=True, sorted=True).indices
    return int(predictions.eq(targets.view(-1, 1)).any(dim=1).sum().item())


def run_imagenet_baseline(
    *,
    imagenet_root: str | Path,
    precommit_path: str | Path,
    output_path: str | Path,
    batch_size: int = 64,
    workers: int = 4,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure the unpruned pretrained baseline against frozen preconditions."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")

    root = Path(imagenet_root).resolve()
    precommit_file = Path(precommit_path).resolve()
    output = Path(output_path).resolve()
    precommit = json.loads(precommit_file.read_text(encoding="utf-8"))

    if precommit.get("status") != "prepared_not_run":
        raise ImageNetBaselineValidationError("baseline precommit status must be prepared_not_run")
    if precommit.get("baseline_reproduction_passed") is not None:
        raise ImageNetBaselineValidationError("precommit must not contain a baseline verdict")

    manifest_path = precommit_file.parent / precommit["dataset"]["manifest_file"]
    class_order_path = precommit_file.parent / precommit["dataset"]["class_order_file"]
    if sha256_file(manifest_path) != precommit["dataset"]["manifest_sha256"]:
        raise ImageNetBaselineValidationError("manifest hash changed after precommit")
    if sha256_file(class_order_path) != precommit["dataset"]["class_order_sha256"]:
        raise ImageNetBaselineValidationError("class-order hash changed after precommit")

    class_records = json.loads(class_order_path.read_text(encoding="utf-8"))
    if len(class_records) != IMAGENET_EXPECTED_CLASSES:
        raise ImageNetBaselineValidationError("class-order artifact does not contain 1000 classes")
    expected_class_hash = sha256_json(class_records)
    # The file hash pins exact bytes; this semantic hash is also recorded in results
    # so future formatting-only changes are distinguishable from mapping changes.

    entries = _load_manifest(manifest_path)
    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model, import_evidence = load_and_validate_pretrained_resnet18(progress=False)
    model = model.to(device).eval()

    loader = DataLoader(
        _ManifestImageDataset(root, entries, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
    )

    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            top1_correct += _topk_correct(logits, targets, 1)
            top5_correct += _topk_correct(logits, targets, 5)
            total += int(targets.numel())

    if total != IMAGENET_VAL_EXPECTED_SAMPLES:
        raise ImageNetBaselineValidationError(
            f"evaluation consumed {total} samples; expected {IMAGENET_VAL_EXPECTED_SAMPLES}"
        )

    top1 = 100.0 * top1_correct / total
    top5 = 100.0 * top5_correct / total
    acceptance = precommit["acceptance"]
    top1_delta = top1 - float(acceptance["expected_top1_percent"])
    top5_delta = top5 - float(acceptance["expected_top5_percent"])
    top1_passed = abs(top1_delta) <= float(acceptance["top1_tolerance_points"])
    top5_passed = abs(top5_delta) <= float(acceptance["top5_tolerance_points"])
    passed = bool(top1_passed and top5_passed)

    result = {
        "schema_version": BASELINE_RESULT_SCHEMA_VERSION,
        "gate": "v0.3_gate_a_imagenet_baseline",
        "precommit": {
            "file": precommit_file.name,
            "sha256": sha256_file(precommit_file),
            "manifest_sha256": precommit["dataset"]["manifest_sha256"],
            "class_order_sha256": precommit["dataset"]["class_order_sha256"],
            "class_order_semantic_sha256": expected_class_hash,
        },
        "weights": import_evidence["weights"],
        "preprocessing": precommit["preprocessing"],
        "evaluation": {
            "sample_count": total,
            "top1_percent": top1,
            "top5_percent": top5,
            "expected_top1_percent": acceptance["expected_top1_percent"],
            "expected_top5_percent": acceptance["expected_top5_percent"],
            "top1_tolerance_points": acceptance["top1_tolerance_points"],
            "top5_tolerance_points": acceptance["top5_tolerance_points"],
            "top1_delta_points": top1_delta,
            "top5_delta_points": top5_delta,
            "top1_passed": top1_passed,
            "top5_passed": top5_passed,
            "baseline_reproduction_passed": passed,
        },
        "gate_passed": passed,
    }
    atomic_write_json(result, output)
    if not passed:
        raise ImageNetBaselineValidationError(
            "ImageNet baseline reproduction failed: "
            f"top1={top1:.3f} delta={top1_delta:+.3f}, "
            f"top5={top5:.3f} delta={top5_delta:+.3f}"
        )
    return result
