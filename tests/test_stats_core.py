import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulator import simulate_experiment
from stats_core import confidence_interval, welch_t_test


def test_detects_known_effect_with_large_sample():
    """A large, well-powered experiment with a real effect should reject the null."""
    sim = simulate_experiment(n_per_arm=5_000, true_effect=5.0, baseline_std=20.0, seed=1)
    control = sim.data.loc[sim.data["group"] == "control", "outcome"].to_numpy()
    treatment = sim.data.loc[sim.data["group"] == "treatment", "outcome"].to_numpy()

    result = welch_t_test(control, treatment)

    assert result.p_value < 0.05
    assert abs(result.effect - sim.true_effect) < 1.0


def test_confidence_interval_covers_true_effect_at_nominal_rate():
    """Across repeated experiments, a 95% CI should contain the true effect ~95% of the time."""
    n_sims = 500
    covered = 0
    true_effect = 3.0

    for seed in range(n_sims):
        sim = simulate_experiment(n_per_arm=200, true_effect=true_effect, baseline_std=20.0, seed=seed)
        control = sim.data.loc[sim.data["group"] == "control", "outcome"].to_numpy()
        treatment = sim.data.loc[sim.data["group"] == "treatment", "outcome"].to_numpy()

        result = welch_t_test(control, treatment)
        ci = confidence_interval(result, alpha=0.05)
        if ci.lower <= true_effect <= ci.upper:
            covered += 1

    coverage_rate = covered / n_sims
    assert 0.90 <= coverage_rate <= 0.99


def test_no_false_positives_beyond_nominal_alpha_under_null():
    """With no true effect, repeated experiments should reject the null at ~alpha rate, not more."""
    n_sims = 1_000
    alpha = 0.05
    false_positives = 0

    for seed in range(n_sims):
        sim = simulate_experiment(n_per_arm=200, true_effect=0.0, baseline_std=20.0, seed=seed)
        control = sim.data.loc[sim.data["group"] == "control", "outcome"].to_numpy()
        treatment = sim.data.loc[sim.data["group"] == "treatment", "outcome"].to_numpy()

        result = welch_t_test(control, treatment)
        if result.p_value < alpha:
            false_positives += 1

    false_positive_rate = false_positives / n_sims
    # Binomial sampling noise around the nominal 5% rate; allow a wide-ish
    # tolerance so the test isn't flaky, while still catching real bugs
    # (e.g. a rate of 20%+ would fail this).
    assert 0.03 <= false_positive_rate <= 0.08


def test_underpowered_experiment_often_misses_a_real_effect():
    """A tiny sample with a small effect should fail to detect it most of the time."""
    n_sims = 200
    detections = 0

    for seed in range(n_sims):
        sim = simulate_experiment(n_per_arm=10, true_effect=1.0, baseline_std=20.0, seed=seed)
        control = sim.data.loc[sim.data["group"] == "control", "outcome"].to_numpy()
        treatment = sim.data.loc[sim.data["group"] == "treatment", "outcome"].to_numpy()

        result = welch_t_test(control, treatment)
        if result.p_value < 0.05:
            detections += 1

    detection_rate = detections / n_sims
    assert detection_rate < 0.5
