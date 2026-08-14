from dataclasses import FrozenInstanceError

import pytest

from sparseflow.planning import ChannelSelection, PruningPlan
from sparseflow.serialization import stable_json_dumps


def _plan():
    return PruningPlan(
        dependency_groups=("layer1.block0", "layer2.block0"),
        channel_selections=(
            ChannelSelection(
                dependency_group="layer1.block0",
                kept_channel_indices=(0, 2, 3),
                removed_channel_indices=(1,),
            ),
            ChannelSelection(
                dependency_group="layer2.block0",
                kept_channel_indices=(0, 1),
                removed_channel_indices=(2, 3),
            ),
        ),
        ranking_metadata=(
            ("method", "l1_norm"),
            ("tie_breaker", "ascending_channel_index"),
        ),
    )


def test_pruning_plan_is_immutable_serializable_and_deterministic():
    plan = _plan()

    assert plan.plan_hash == _plan().plan_hash
    assert plan.as_dict()["plan_hash"] == plan.plan_hash
    stable_json_dumps(plan.as_dict())
    with pytest.raises(FrozenInstanceError):
        plan.dependency_groups = ("changed",)


@pytest.mark.parametrize(
    "selection",
    [
        ChannelSelection("group", (1, 0), (2,)),
        ChannelSelection("group", (0, 0), (1,)),
        ChannelSelection("group", (0, 1), (1, 2)),
        ChannelSelection("group", (-1, 0), (1,)),
    ],
)
def test_channel_selection_rejects_noncanonical_or_overlapping_indices(selection):
    with pytest.raises(ValueError):
        PruningPlan(
            dependency_groups=("group",),
            channel_selections=(selection,),
            ranking_metadata=(("method", "l1_norm"),),
        )


def test_pruning_plan_requires_one_selection_per_dependency_group():
    with pytest.raises(ValueError, match="dependency groups"):
        PruningPlan(
            dependency_groups=("group-a",),
            channel_selections=(ChannelSelection("group-b", (0,), (1,)),),
            ranking_metadata=(("method", "l1_norm"),),
        )
