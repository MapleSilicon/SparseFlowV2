from sparseflow.graph_diff import (
    graph_changes_hash,
    structural_hash,
    structural_snapshot,
)
from sparseflow.models import build_reference_model
from sparseflow.passes import ChannelPruningPass, NoOpPass


def test_structural_hashes_capture_real_changes():
    baseline = build_reference_model(seed=7)
    noop = NoOpPass().apply(baseline)
    pruned = ChannelPruningPass(ratio=0.375).apply(baseline)

    assert structural_snapshot(baseline) == structural_snapshot(noop.model)
    assert structural_hash(baseline) == structural_hash(noop.model)
    assert structural_hash(baseline) != structural_hash(pruned.model)
    assert graph_changes_hash(noop.graph_changes) == graph_changes_hash([])
    assert graph_changes_hash(pruned.graph_changes) != graph_changes_hash([])
