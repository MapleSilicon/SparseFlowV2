import pytest

from sparseflow.benchmark import interpret_latency, summarize_latency_samples


def test_latency_summary_calculates_population_coefficient_of_variation():
    summary = summarize_latency_samples([1.0, 2.0, 3.0])

    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 3.0
    assert summary["mean"] == 2.0
    assert summary["standard_deviation"] == pytest.approx(0.8164965809)
    assert summary["coefficient_of_variation"] == pytest.approx(0.4082482904)


def test_small_or_variable_latency_is_flagged_as_noise_sensitive():
    baseline = summarize_latency_samples([0.05, 0.06, 0.15, 0.04])
    optimized = summarize_latency_samples([0.04, 0.09, 0.05, 0.12])
    interpretation = interpret_latency(baseline, optimized)

    assert interpretation["noise_sensitive"] is True
    assert "baseline p50 is below 0.1 ms" in interpretation["reasons"]
    assert any("coefficient of variation" in reason for reason in interpretation["reasons"])
    assert interpretation["commercial_speedup_claim_supported"] is False


def test_stable_realistic_samples_are_not_flagged_but_make_no_commercial_claim():
    baseline = summarize_latency_samples([5.00, 5.01, 4.99, 5.00])
    optimized = summarize_latency_samples([4.50, 4.49, 4.51, 4.50])
    interpretation = interpret_latency(baseline, optimized)

    assert interpretation["noise_sensitive"] is False
    assert interpretation["reasons"] == []
    assert interpretation["commercial_speedup_claim_supported"] is False
    assert "MAC" not in " ".join(interpretation["reasons"])
