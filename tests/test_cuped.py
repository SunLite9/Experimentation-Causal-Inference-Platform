import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cuped import compute_theta, cuped_adjust
from simulator import simulate_experiment
from stats_core import welch_t_test


def _apply_cuped(data, true_effect=None):
    control_mask = data["group"] == "control"
    treatment_mask = data["group"] == "treatment"

    pooled_outcome = data["outcome"].to_numpy()
    pooled_covariate = data["pre_covariate"].to_numpy()
    theta = compute_theta(pooled_outcome, pooled_covariate)
    covariate_mean = pooled_covariate.mean()

    adjusted_control = cuped_adjust(
        data.loc[control_mask, "outcome"].to_numpy(),
        data.loc[control_mask, "pre_covariate"].to_numpy(),
        theta,
        covariate_mean,
    )
    adjusted_treatment = cuped_adjust(
        data.loc[treatment_mask, "outcome"].to_numpy(),
        data.loc[treatment_mask, "pre_covariate"].to_numpy(),
        theta,
        covariate_mean,
    )
    return adjusted_control, adjusted_treatment


def test_cuped_does_not_bias_the_effect_estimate():
    """CUPED reduces variance; the point estimate of the effect should stay
    close to the true effect, same as the naive t-test."""
    true_effect = 3.0
    sim = simulate_experiment(
        n_per_arm=5_000,
        true_effect=true_effect,
        baseline_std=20.0,
        extra_noise_std=25.0,
        extra_noise_correlation=0.9,
        seed=7,
    )
    data = sim.data
    control_mask = data["group"] == "control"
    treatment_mask = data["group"] == "treatment"

    naive_result = welch_t_test(
        data.loc[control_mask, "outcome"].to_numpy(),
        data.loc[treatment_mask, "outcome"].to_numpy(),
    )

    adjusted_control, adjusted_treatment = _apply_cuped(data)
    cuped_result = welch_t_test(adjusted_control, adjusted_treatment)

    assert abs(naive_result.effect - true_effect) < 1.5
    assert abs(cuped_result.effect - true_effect) < 1.5
    # CUPED's point estimate should track the naive one, not drift away from it.
    assert abs(cuped_result.effect - naive_result.effect) < 1.0


def test_cuped_reduces_variance_when_covariate_captures_extra_noise():
    """When the pre-experiment covariate is correlated with treatment-unrelated
    noise, CUPED should substantially shrink outcome variance."""
    sim = simulate_experiment(
        n_per_arm=5_000,
        true_effect=2.0,
        baseline_std=20.0,
        extra_noise_std=30.0,
        extra_noise_correlation=0.9,
        seed=11,
    )
    data = sim.data
    adjusted_control, adjusted_treatment = _apply_cuped(data)

    raw_var = data["outcome"].var(ddof=1)
    adjusted_var = np.concatenate([adjusted_control, adjusted_treatment]).var(ddof=1)

    assert adjusted_var < raw_var
    variance_reduction_pct = 100 * (1 - adjusted_var / raw_var)
    assert variance_reduction_pct > 20
