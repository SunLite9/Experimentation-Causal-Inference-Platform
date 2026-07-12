"""Synthetic A/B experiment data generator.

Generates unit-level experiment data with a known, configurable ground-truth
treatment effect, plus a pre-experiment covariate correlated with the outcome
(needed for variance-reduction work downstream). Because the true effect is
known by construction, any analysis method run on this data can be checked
against ground truth instead of just "does the number look plausible."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Container for a simulated experiment plus the ground truth used to build it."""

    data: pd.DataFrame
    true_effect: float
    baseline_mean: float
    baseline_std: float


@dataclass
class ObservationalSimulationResult:
    """Container for simulated observational (non-randomized) data."""

    data: pd.DataFrame
    true_effect: float


def simulate_experiment(
    n_per_arm: int = 5_000,
    true_effect: float = 0.0,
    baseline_mean: float = 100.0,
    baseline_std: float = 20.0,
    covariate_correlation: float = 0.7,
    extra_noise_std: float = 0.0,
    extra_noise_correlation: float = 0.0,
    seed: int | None = None,
) -> SimulationResult:
    """Simulate a two-arm randomized experiment.

    Each unit gets:
      - `pre_covariate`: a pre-experiment measurement of the same underlying
        metric, correlated with the outcome via `covariate_correlation`. This
        is what CUPED will later use to strip out unrelated variance.
      - `outcome`: the post-experiment metric. Control units are centered at
        `baseline_mean`; treated units are shifted by `true_effect`.

    Args:
        n_per_arm: Number of units in each of control/treatment.
        true_effect: Ground-truth additive treatment effect on the outcome.
            Set to 0 to simulate a true null (no effect) scenario.
        baseline_mean: Mean of the outcome metric in the absence of treatment.
        baseline_std: Standard deviation of the outcome metric from natural
            unit-to-unit variation (unrelated to treatment).
        covariate_correlation: Correlation between the pre-experiment
            covariate and the outcome's natural variation, in [0, 1).
            Higher values make the covariate a better CUPED predictor.
        extra_noise_std: Standard deviation of an additional noise source
            layered onto the outcome that is unrelated to treatment (e.g. a
            seasonal or cohort effect). Use this to simulate scenarios where
            a naive t-test loses power to noise a smarter analysis can remove.
        extra_noise_correlation: Correlation between the pre-experiment
            covariate and the extra noise term. Set close to 1.0 to simulate
            the case where the pre-experiment covariate happens to capture
            the extra noise source, which is exactly the case where CUPED
            recovers power that a naive t-test throws away.
        seed: Random seed for reproducibility.

    Returns:
        A SimulationResult with the unit-level DataFrame and the ground-truth
        parameters used to generate it.
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_arm * 2

    group = np.array(["control"] * n_per_arm + ["treatment"] * n_per_arm)

    # Shared latent factor drives correlation between the pre-experiment
    # covariate and the outcome's natural (non-treatment) variation.
    latent = rng.standard_normal(n_total)

    covariate_noise = rng.standard_normal(n_total)
    pre_covariate = (
        baseline_mean
        + covariate_correlation * baseline_std * latent
        + np.sqrt(max(1 - covariate_correlation**2, 0.0)) * baseline_std * covariate_noise
    )

    outcome_noise = rng.standard_normal(n_total)
    outcome = (
        baseline_mean
        + covariate_correlation * baseline_std * latent
        + np.sqrt(max(1 - covariate_correlation**2, 0.0)) * baseline_std * outcome_noise
    )

    if extra_noise_std > 0:
        extra_latent = rng.standard_normal(n_total)
        extra_component = extra_noise_correlation * extra_latent
        extra_residual = np.sqrt(max(1 - extra_noise_correlation**2, 0.0)) * rng.standard_normal(
            n_total
        )
        extra_noise = extra_noise_std * (extra_component + extra_residual)
        outcome = outcome + extra_noise
        if extra_noise_correlation != 0:
            pre_covariate = pre_covariate + extra_noise_std * extra_noise_correlation * extra_latent

    outcome = outcome + np.where(group == "treatment", true_effect, 0.0)

    data = pd.DataFrame(
        {
            "unit_id": np.arange(n_total),
            "group": group,
            "pre_covariate": pre_covariate,
            "outcome": outcome,
        }
    )

    return SimulationResult(
        data=data,
        true_effect=true_effect,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
    )


def simulate_observational_data(
    n: int = 10_000,
    true_effect: float = 5.0,
    confounding_strength: float = 2.0,
    baseline_mean: float = 50.0,
    noise_std: float = 5.0,
    seed: int | None = None,
) -> ObservationalSimulationResult:
    """Simulate non-randomized (observational) treatment/outcome data with confounding.

    Unlike `simulate_experiment`, treatment assignment here is NOT random: it
    depends on the units' own covariates, and those same covariates also
    directly affect the outcome. This is the textbook confounding setup
    where a naive treated-vs-control comparison is biased, because the units
    that self-selected (or were selected) into treatment differ systematically
    from the ones that didn't, for reasons unrelated to the treatment itself.

    Args:
        n: Total number of units.
        true_effect: Ground-truth additive treatment effect on the outcome.
        confounding_strength: How strongly `covariate_1` drives treatment
            assignment (via a logistic propensity model). Higher values
            produce more selection bias and a larger naive-estimate bias.
        baseline_mean: Outcome mean at covariate_1 = covariate_2 = 0, untreated.
        noise_std: Standard deviation of outcome noise unrelated to treatment
            or the covariates.
        seed: Random seed for reproducibility.

    Returns:
        An ObservationalSimulationResult with unit-level covariates,
        (non-random) treatment assignment, outcome, and the true effect.
    """
    rng = np.random.default_rng(seed)

    covariate_1 = rng.standard_normal(n)  # confounder: drives both treatment and outcome
    covariate_2 = rng.standard_normal(n)  # weaker confounder, same role

    propensity_logit = confounding_strength * covariate_1 + 0.5 * covariate_2
    true_propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = rng.binomial(1, true_propensity)

    outcome = (
        baseline_mean
        + 4.0 * covariate_1
        + 2.0 * covariate_2
        + true_effect * treatment
        + rng.normal(0, noise_std, n)
    )

    data = pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "covariate_1": covariate_1,
            "covariate_2": covariate_2,
            "treatment": treatment,
            "outcome": outcome,
            "true_propensity": true_propensity,
        }
    )

    return ObservationalSimulationResult(data=data, true_effect=true_effect)
