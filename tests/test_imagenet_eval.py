from pathlib import Path

import pytest

import sparseflow.imagenet_eval as imagenet_eval
from sparseflow.imagenet_eval import (
    BASELINE_TOP1_TOLERANCE_POINTS,
    BASELINE_TOP5_TOLERANCE_POINTS,
    ImageNetBaselineValidationError,
    assert_canonical_imagenet_class_order,
    build_imagenet_manifest,
    prepare_imagenet_baseline,
)
from sparseflow.serialization import sha256_file


class _FakeImageNet:
    def __init__(self, root: Path, *, corrupt_index: bool = False, corrupt_sample: bool = False):
        self.root = str(root)
        self.split = "val"
        self.wnids = [f"n{index:08d}" for index in range(1000)]
        self.wnid_to_idx = {wnid: index for index, wnid in enumerate(self.wnids)}
        if corrupt_index:
            self.wnid_to_idx[self.wnids[1]] = 9
        self.samples = []
        for index in range(50_000):
            label = index % 1000
            wnid = self.wnids[label]
            if corrupt_sample and index == 123:
                label = (label + 1) % 1000
            self.samples.append((str(root / "val" / wnid / f"ILSVRC2012_val_{index + 1:08d}.JPEG"), label))


def test_class_order_guard_accepts_canonical_torchvision_contract(tmp_path):
    dataset = _FakeImageNet(tmp_path)
    records = assert_canonical_imagenet_class_order(dataset)
    assert len(records) == 1000
    assert records[0] == {"index": 0, "wnid": "n00000000"}
    assert records[-1] == {"index": 999, "wnid": "n00000999"}


def test_class_order_guard_rejects_silent_index_permutation(tmp_path):
    dataset = _FakeImageNet(tmp_path, corrupt_index=True)
    with pytest.raises(ImageNetBaselineValidationError, match="wnid_to_idx"):
        assert_canonical_imagenet_class_order(dataset)


def test_class_order_guard_rejects_folder_target_mismatch(tmp_path):
    dataset = _FakeImageNet(tmp_path, corrupt_sample=True)
    with pytest.raises(ImageNetBaselineValidationError, match="sample target/folder mismatch"):
        assert_canonical_imagenet_class_order(dataset)


def test_manifest_is_deterministic_and_contains_exact_frozen_labels(tmp_path):
    dataset = _FakeImageNet(tmp_path)
    first = build_imagenet_manifest(dataset)
    second = build_imagenet_manifest(dataset)
    assert first == second
    assert len(first) == 50_000
    assert first[0].relative_path.startswith("val/n00000000/")
    assert first[0].label == 0
    assert first[-1].relative_path.startswith("val/")


def test_prepare_precommits_hashes_and_tolerances_before_measurement(tmp_path, monkeypatch):
    root = tmp_path / "imagenet"
    output = tmp_path / "gate"
    fake = _FakeImageNet(root)
    monkeypatch.setattr(imagenet_eval, "ImageNet", lambda *args, **kwargs: fake)

    precommit = prepare_imagenet_baseline(imagenet_root=root, output_dir=output)

    assert precommit["status"] == "prepared_not_run"
    assert precommit["baseline_reproduction_passed"] is None
    assert precommit["dataset"]["sample_count"] == 50_000
    assert precommit["dataset"]["class_count"] == 1000
    assert precommit["acceptance"]["expected_top1_percent"] == pytest.approx(69.758)
    assert precommit["acceptance"]["expected_top5_percent"] == pytest.approx(89.078)
    assert precommit["acceptance"]["top1_tolerance_points"] == BASELINE_TOP1_TOLERANCE_POINTS
    assert precommit["acceptance"]["top5_tolerance_points"] == BASELINE_TOP5_TOLERANCE_POINTS
    assert precommit["preprocessing"]["source"] == "weights.transforms()"

    manifest = output / "manifest.jsonl"
    class_order = output / "class-order.json"
    precommit_file = output / "baseline-precommit.json"
    assert manifest.exists()
    assert class_order.exists()
    assert precommit_file.exists()
    assert sha256_file(manifest) == precommit["dataset"]["manifest_sha256"]
    assert sha256_file(class_order) == precommit["dataset"]["class_order_sha256"]


def test_tolerance_is_precommitted_not_post_hoc(tmp_path, monkeypatch):
    root = tmp_path / "imagenet"
    output = tmp_path / "gate"
    fake = _FakeImageNet(root)
    monkeypatch.setattr(imagenet_eval, "ImageNet", lambda *args, **kwargs: fake)

    precommit = prepare_imagenet_baseline(
        imagenet_root=root,
        output_dir=output,
        top1_tolerance_points=0.15,
        top5_tolerance_points=0.10,
    )
    assert precommit["acceptance"]["top1_tolerance_points"] == pytest.approx(0.15)
    assert precommit["acceptance"]["top5_tolerance_points"] == pytest.approx(0.10)
