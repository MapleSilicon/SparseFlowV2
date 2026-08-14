from pathlib import Path

from run_bench import resolve_cli_configuration
from sparseflow.presets import get_preset, preset_names


def test_micro_demo_preset_resolves_expected_configuration():
    preset = get_preset("micro-demo")
    assert preset.model_identifier == "reference_cnn_v1"
    assert preset.pass_name == "channel-prune"
    assert preset.pruning_ratio == 0.375
    assert preset.input_shape == (1, 3, 16, 16)
    assert preset.output_report_filename == "report.json"


def test_noop_control_preset_resolves_expected_configuration():
    preset = get_preset("noop-control")
    assert preset.pass_name == "noop"
    assert preset.pruning_ratio == 0.0
    assert preset.output_report_filename == "report-noop.json"
    assert preset_names() == (
        "micro-demo",
        "noop-control",
        "resnet18-gate1",
        "resnet18-gate2",
    )


def test_resnet18_gate1_preset_and_model_cli_resolve_without_pruning():
    preset = get_preset("resnet18-gate1")
    assert preset.model_identifier == "resnet18-reference"
    assert preset.input_shape == (1, 3, 64, 64)
    assert preset.pass_name == "channel-prune"
    assert preset.pruning_ratio == 0.0
    assert preset.warmup_runs == 1
    assert preset.measured_runs == 3
    assert preset.output_report_filename == "report-resnet18-gate1.json"

    resolved = resolve_cli_configuration(
        ["--model", "resnet18-reference", "--pass", "channel-prune", "--pruning-ratio", "0.0"]
    )
    assert resolved.preset_name == "resnet18-gate1"
    assert resolved.model_identifier == "resnet18-reference"
    assert resolved.input_shape == (1, 3, 64, 64)
    assert resolved.pruning_ratio == 0.0


def test_resnet18_gate2_preset_resolves_transactional_pruning_configuration():
    preset = get_preset("resnet18-gate2")
    assert preset.model_identifier == "resnet18-reference"
    assert preset.input_shape == (1, 3, 64, 64)
    assert preset.pass_name == "channel-prune"
    assert preset.pruning_ratio == 0.125
    assert preset.output_report_filename == "report-resnet18-gate2.json"


def test_explicit_cli_values_override_preset_values():
    resolved = resolve_cli_configuration(
        [
            "--preset",
            "noop-control",
            "--pass",
            "channel-prune",
            "--pruning-ratio",
            "0.25",
            "--seed",
            "77",
            "--warmup",
            "3",
            "--measured-runs",
            "9",
            "--threads",
            "2",
            "--output",
            "custom.json",
        ]
    )
    assert resolved.preset_name == "noop-control"
    assert resolved.pass_name == "channel-prune"
    assert resolved.pruning_ratio == 0.25
    assert resolved.seed == 77
    assert resolved.warmup_runs == 3
    assert resolved.measured_runs == 9
    assert resolved.threads == 2
    assert resolved.output_path == Path("custom.json")
