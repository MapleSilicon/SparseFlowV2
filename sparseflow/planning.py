"""Immutable pruning-plan data contracts for future V1 execution work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sparseflow.serialization import sha256_json


@dataclass(frozen=True)
class ChannelSelection:
    dependency_group: str
    kept_channel_indices: tuple[int, ...]
    removed_channel_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dependency_group": self.dependency_group,
            "kept_channel_indices": list(self.kept_channel_indices),
            "removed_channel_indices": list(self.removed_channel_indices),
        }


@dataclass(frozen=True)
class PruningPlan:
    """Serializable intent only; this type does not execute or mutate a model."""

    dependency_groups: tuple[str, ...]
    channel_selections: tuple[ChannelSelection, ...]
    ranking_metadata: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.dependency_groups or len(set(self.dependency_groups)) != len(
            self.dependency_groups
        ):
            raise ValueError("dependency groups must be non-empty and unique")
        if tuple(sorted(self.dependency_groups)) != self.dependency_groups:
            raise ValueError("dependency groups must use canonical sorted order")
        selection_groups = tuple(
            selection.dependency_group for selection in self.channel_selections
        )
        if selection_groups != self.dependency_groups:
            raise ValueError(
                "channel selections must match dependency groups in canonical order"
            )
        for selection in self.channel_selections:
            self._validate_selection(selection)
        metadata_keys = tuple(key for key, _ in self.ranking_metadata)
        if (
            not metadata_keys
            or len(set(metadata_keys)) != len(metadata_keys)
            or tuple(sorted(metadata_keys)) != metadata_keys
        ):
            raise ValueError("ranking metadata keys must be non-empty, unique, and sorted")

    @staticmethod
    def _validate_selection(selection: ChannelSelection) -> None:
        if not selection.dependency_group:
            raise ValueError("dependency group must be non-empty")
        kept = selection.kept_channel_indices
        removed = selection.removed_channel_indices
        for label, indices in (("kept", kept), ("removed", removed)):
            if any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in indices
            ):
                raise ValueError(f"{label} channel indices must be non-negative integers")
            if tuple(sorted(set(indices))) != indices:
                raise ValueError(f"{label} channel indices must be sorted and unique")
        if set(kept) & set(removed):
            raise ValueError("kept and removed channel indices must not overlap")

    def hash_payload(self) -> dict[str, Any]:
        return {
            "dependency_groups": list(self.dependency_groups),
            "channel_selections": [
                selection.as_dict() for selection in self.channel_selections
            ],
            "ranking_metadata": {
                key: value for key, value in self.ranking_metadata
            },
        }

    @property
    def plan_hash(self) -> str:
        return sha256_json(self.hash_payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self.hash_payload(), "plan_hash": self.plan_hash}
