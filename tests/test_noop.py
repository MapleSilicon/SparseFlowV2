import torch

from sparseflow.benchmark import count_parameters, serialize_state_dict
from sparseflow.compute import count_compute
from sparseflow.graph_diff import model_state_hash, structural_hash
from sparseflow.models import INPUT_SHAPE, build_reference_model
from sparseflow.passes import NoOpPass


def test_noop_preserves_structure_compute_size_and_output():
    model = build_reference_model(seed=1234)
    result = NoOpPass().apply(model)
    optimized = result.model

    assert result.graph_changes == []
    assert count_parameters(model) == count_parameters(optimized)
    assert len(serialize_state_dict(model)) == len(serialize_state_dict(optimized))
    assert count_compute(model, INPUT_SHAPE) == count_compute(optimized, INPUT_SHAPE)
    assert structural_hash(model) == structural_hash(optimized)
    assert model_state_hash(model) == model_state_hash(optimized)

    generator = torch.Generator().manual_seed(22)
    sample = torch.randn((2, *INPUT_SHAPE[1:]), generator=generator)
    with torch.no_grad():
        baseline = model(sample)
        after = optimized(sample)
    torch.testing.assert_close(after, baseline, rtol=1e-6, atol=1e-7)
