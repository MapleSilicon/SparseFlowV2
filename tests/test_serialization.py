from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sparseflow.serialization import (
    atomic_write_json,
    sha256_file,
    sha256_json,
    stable_json_dumps,
    to_jsonable,
)


@dataclass(frozen=True)
class _EvidenceValue:
    name: str
    dimensions: tuple[int, ...]


def test_json_conversion_supports_declared_portable_types():
    value = {
        "dataclass": _EvidenceValue("conv", (1, 2)),
        "path": Path("artifacts/model.onnx"),
        "array": np.array([1, 2], dtype=np.int64),
        "scalar": np.float32(1.5),
    }

    assert to_jsonable(value) == {
        "dataclass": {"name": "conv", "dimensions": [1, 2]},
        "path": str(Path("artifacts/model.onnx"),),
        "array": [1, 2],
        "scalar": 1.5,
    }


def test_json_conversion_rejects_non_string_mapping_keys():
    with pytest.raises(TypeError, match="keys must be strings"):
        to_jsonable({1: "invalid"})


def test_json_conversion_rejects_unsupported_value():
    with pytest.raises(TypeError, match="object.*not JSON-safe"):
        stable_json_dumps({"unsupported": object()})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_json_conversion_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError, match="NaN and Infinity"):
        stable_json_dumps({"value": value})


def test_stable_json_is_sorted_and_hashes_are_content_sensitive():
    first = {"z": 3, "a": {"y": 2, "b": 1}}
    same_content = {"a": {"b": 1, "y": 2}, "z": 3}
    changed = {"a": {"b": 1, "y": 99}, "z": 3}

    serialized = stable_json_dumps(first)

    assert serialized == stable_json_dumps(same_content)
    assert serialized.index('"a"') < serialized.index('"z"')
    assert serialized.endswith("\n")
    assert sha256_json(first) == sha256_json(same_content)
    assert sha256_json(first) != sha256_json(changed)


def test_atomic_write_creates_complete_final_file_and_file_hash(tmp_path):
    destination = tmp_path / "nested" / "report.json"
    value = {"complete": True, "items": [1, 2, 3]}

    atomic_write_json(value, destination)

    assert destination.read_text(encoding="utf-8") == stable_json_dumps(value)
    assert json.loads(destination.read_text(encoding="utf-8")) == value
    assert sha256_file(destination) == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_atomic_write_argument_order_is_value_then_path(tmp_path):
    destination = tmp_path / "report.json"
    value = {"contract": "value-first"}

    atomic_write_json(value, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == value

    with pytest.raises(TypeError):
        atomic_write_json(destination, value)

    assert json.loads(destination.read_text(encoding="utf-8")) == value


def test_atomic_write_serialization_failure_preserves_destination_and_cleans_temp(tmp_path):
    destination = tmp_path / "report.json"
    original = {"state": "complete"}
    atomic_write_json(original, destination)
    original_bytes = destination.read_bytes()

    with pytest.raises(TypeError, match="not JSON-safe"):
        atomic_write_json({"unsupported": object()}, destination)

    assert destination.read_bytes() == original_bytes
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_atomic_write_failure_does_not_create_destination_or_temp_file(tmp_path):
    destination = tmp_path / "new-report.json"

    with pytest.raises(ValueError, match="NaN and Infinity"):
        atomic_write_json({"invalid": float("nan")}, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []
