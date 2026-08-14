from pathlib import Path

import pytest

import sparseflow.imagenet_eval as imagenet_eval
from sparseflow.imagenet_eval import (
    ImageNetBaselineValidationError,
    prepare_imagenet_baseline,
    run_imagenet_baseline,
)


class _FakeImageNet:
    def __init__(self, root: Path):
        self.root = str(root)
        self.split = "val"
        self.wnids = [f"n{index:08d}" for index in range(1000)]
        self.wnid_to_idx = {wnid: index for index, wnid in enumerate(self.wnids)}
        self.samples = []
        for index in range(50_000):
            label = index % 1000
            wnid = self.wnids[label]
            self.samples.append(
                (
                    str(root / "val" / wnid / f"ILSVRC2012_val_{index + 1:08d}.JPEG"),
                    label,
                )
            )


def _prepare_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "imagenet"
    output = tmp_path / "gate"
    fake = _FakeImageNet(root)
    monkeypatch.setattr(imagenet_eval, "ImageNet", lambda *args, **kwargs: fake)
    prepare_imagenet_baseline(imagenet_root=root, output_dir=output)
    return root, output


def _fail_if_inference_starts(*args, **kwargs):
    raise AssertionError("pretrained model loading/inference must not start after precommit integrity failure")


def test_changed_manifest_halts_before_model_loading_or_inference(tmp_path, monkeypatch):
    root, output = _prepare_fixture(tmp_path, monkeypatch)
    manifest = output / "manifest.jsonl"
    precommit = output / "baseline-precommit.json"
    result = output / "baseline-result.json"

    # Change one byte after the precommit was frozen. The baseline command must
    # reject the artifact hash before parsing samples or loading pretrained weights.
    manifest.write_bytes(manifest.read_bytes() + b" ")
    monkeypatch.setattr(
        imagenet_eval,
        "load_and_validate_pretrained_resnet18",
        _fail_if_inference_starts,
    )

    with pytest.raises(ImageNetBaselineValidationError, match="manifest hash changed after precommit"):
        run_imagenet_baseline(
            imagenet_root=root,
            precommit_path=precommit,
            output_path=result,
            batch_size=1,
            workers=0,
            device="cpu",
        )

    assert not result.exists()


def test_changed_class_order_halts_before_model_loading_or_inference(tmp_path, monkeypatch):
    root, output = _prepare_fixture(tmp_path, monkeypatch)
    class_order = output / "class-order.json"
    precommit = output / "baseline-precommit.json"
    result = output / "baseline-result.json"

    class_order.write_bytes(class_order.read_bytes() + b" ")
    monkeypatch.setattr(
        imagenet_eval,
        "load_and_validate_pretrained_resnet18",
        _fail_if_inference_starts,
    )

    with pytest.raises(ImageNetBaselineValidationError, match="class-order hash changed after precommit"):
        run_imagenet_baseline(
            imagenet_root=root,
            precommit_path=precommit,
            output_path=result,
            batch_size=1,
            workers=0,
            device="cpu",
        )

    assert not result.exists()
