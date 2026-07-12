import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal import naive_treatment_effect, propensity_score_matching_effect
from simulator import simulate_observational_data


def test_naive_estimate_is_biased_by_confounding():
    """A raw treated-vs-control comparison should be pulled well away from
    the true effect when treatment assignment depends on a covariate that
    also drives the outcome."""
    true_effect = 5.0
    sim = simulate_observational_data(
        n=10_000, true_effect=true_effect, confounding_strength=2.0, seed=1
    )
    data = sim.data

    naive_effect = naive_treatment_effect(data["outcome"].to_numpy(), data["treatment"].to_numpy())

    assert abs(naive_effect - true_effect) > 2.0


def test_propensity_matching_recovers_true_effect():
    """Propensity score matching should land close to the true effect and
    its confidence interval should cover it, unlike the naive comparison."""
    true_effect = 5.0
    sim = simulate_observational_data(
        n=10_000, true_effect=true_effect, confounding_strength=2.0, seed=2
    )
    data = sim.data
    outcome = data["outcome"].to_numpy()
    treatment = data["treatment"].to_numpy()
    covariates = data[["covariate_1", "covariate_2"]].to_numpy()

    result = propensity_score_matching_effect(outcome, treatment, covariates, caliper=0.05)

    assert abs(result.effect - true_effect) < 1.5
    assert result.ci_lower <= true_effect <= result.ci_upper


def test_matching_corrects_naive_bias():
    """The matched estimate should be substantially closer to the true
    effect than the naive estimate, on the same data."""
    true_effect = 5.0
    sim = simulate_observational_data(
        n=10_000, true_effect=true_effect, confounding_strength=2.0, seed=3
    )
    data = sim.data
    outcome = data["outcome"].to_numpy()
    treatment = data["treatment"].to_numpy()
    covariates = data[["covariate_1", "covariate_2"]].to_numpy()

    naive_effect = naive_treatment_effect(outcome, treatment)
    matched = propensity_score_matching_effect(outcome, treatment, covariates, caliper=0.05)

    assert abs(matched.effect - true_effect) < abs(naive_effect - true_effect)
