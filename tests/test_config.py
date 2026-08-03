import pytest

from sparseflow.config import BenchmarkConfig


@pytest.mark.parametrize(
    ("override", "invalid_field"),
    [
        ({"input_shape": (1, 3, 16)}, "input_shape"),
        ({"input_shape": (1, 3, 0, 16)}, "input_shape"),
        ({"input_shape": (1, 3, -1, 16)}, "input_shape"),
        ({"input_shape": (1, 3, 16.0, 16)}, "input_shape"),
        ({"seed": "1234"}, "seed"),
        ({"warmup_runs": -1}, "warmup_runs"),
        ({"measured_runs": 0}, "measured_runs"),
        ({"measured_runs": -1}, "measured_runs"),
        ({"threads": 0}, "threads"),
        ({"threads": -1}, "threads"),
        ({"onnx_opset": 12}, "onnx_opset"),
        ({"fidelity_batch_size": 0}, "fidelity_batch_size"),
        ({"fidelity_batch_size": -1}, "fidelity_batch_size"),
        ({"preset_name": ""}, "preset_name"),
        ({"output_report_filename": ""}, "output_report_filename"),
    ],
)
def test_benchmark_config_rejects_each_invalid_field_independently(
    override,
    invalid_field,
):
    with pytest.raises(ValueError) as exc_info:
        BenchmarkConfig(**override)

    assert invalid_field in str(exc_info.value)


def test_benchmark_config_defaults_and_json_safe_dict_are_deterministic():
    config = BenchmarkConfig(preset_name="micro-demo")

    first = config.as_dict()
    second = config.as_dict()

    assert BenchmarkConfig()
    assert first == second
    assert first is not second
    assert first["input_shape"] == [1, 3, 16, 16]
    assert isinstance(first["input_shape"], list)
    assert first["preset_name"] == "micro-demo"
