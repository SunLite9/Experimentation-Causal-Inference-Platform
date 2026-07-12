"""Core experiment-analysis statistics, implemented from first principles.

Every formula here is the textbook Welch's t-test / power-analysis formula,
not a wrapper around a library black box, so the math is inspectable and the
assumptions are explicit. Where scipy is used, it's only for the Normal/t
CDF/quantile functions (`scipy.stats.t`, `scipy.stats.norm`), not for the
test statistics themselves.

Formulas (Welch's unequal-variance t-test, the standard choice when the two
arms aren't assumed to have equal variance, which is the safer default for
real experiment data):

    t = (mean_treatment - mean_control) / SE
    SE = sqrt(var_treatment / n_treatment + var_control / n_control)
    df = SE^4 / ( (var_t/n_t)^2 / (n_t - 1) + (var_c/n_c)^2 / (n_c - 1) )   [Welch-Satterthwaite]
    p = 2 * P(T_df > |t|)

Confidence interval for the difference in means:
    (mean_t - mean_c) +/- t_crit(df, alpha/2) * SE

Sample size for a two-sample t-test (equal allocation, pooled variance
approximation, standard formula from Cohen 1988 / any experimentation
textbook):
    n_per_arm = 2 * (z_{alpha/2} + z_{beta})^2 * sigma^2 / delta^2

Minimum detectable effect (MDE), inverting the same formula for delta:
    delta = (z_{alpha/2} + z_{beta}) * sigma * sqrt(2 / n_per_arm)

Power, inverting for beta given n and delta:
    power = P(Z > z_{alpha/2} - delta / SE_pooled)
    SE_pooled = sigma * sqrt(2 / n_per_arm)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TTestResult:
    mean_control: float
    mean_treatment: float
    effect: float  # mean_treatment - mean_control
    se: float
    t_stat: float
    df: float
    p_value: float
    n_control: int
    n_treatment: int


@dataclass
class ConfidenceInterval:
    lower: float
    upper: float
    alpha: float


def welch_t_test(control: np.ndarray, treatment: np.ndarray) -> TTestResult:
    """Welch's two-sample t-test for a difference in means.

    Does not assume equal variance between arms, which is the right default
    for experiment data since treatment can itself change the variance.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = control.mean(), treatment.mean()
    var_c, var_t = control.var(ddof=1), treatment.var(ddof=1)

    se = np.sqrt(var_t / n_t + var_c / n_c)
    effect = mean_t - mean_c
    t_stat = effect / se if se > 0 else 0.0

    # Welch-Satterthwaite degrees of freedom.
    numerator = (var_t / n_t + var_c / n_c) ** 2
    denominator = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    df = numerator / denominator if denominator > 0 else n_c + n_t - 2

    p_value = 2 * stats.t.sf(np.abs(t_stat), df)

    return TTestResult(
        mean_control=mean_c,
        mean_treatment=mean_t,
        effect=effect,
        se=se,
        t_stat=t_stat,
        df=df,
        p_value=p_value,
        n_control=n_c,
        n_treatment=n_t,
    )


def confidence_interval(result: TTestResult, alpha: float = 0.05) -> ConfidenceInterval:
    """Two-sided (1 - alpha) confidence interval for the difference in means."""
    t_crit = stats.t.ppf(1 - alpha / 2, result.df)
    margin = t_crit * result.se
    return ConfidenceInterval(
        lower=result.effect - margin, upper=result.effect + margin, alpha=alpha
    )


def required_sample_size(
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Required sample size per arm to detect `mde` with the given power.

    Standard two-sample z-test approximation, equal allocation between arms:
        n_per_arm = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / mde^2
    """
    if mde == 0:
        raise ValueError("mde must be nonzero to compute a finite sample size.")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = 2 * (z_alpha + z_beta) ** 2 * baseline_std**2 / mde**2
    return int(np.ceil(n))


def minimum_detectable_effect(
    baseline_std: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest true effect detectable with the given sample size and power.

    Inversion of the sample-size formula:
        mde = (z_alpha/2 + z_beta) * sigma * sqrt(2 / n_per_arm)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    return (z_alpha + z_beta) * baseline_std * np.sqrt(2 / n_per_arm)


def statistical_power(
    baseline_std: float,
    n_per_arm: int,
    true_effect: float,
    alpha: float = 0.05,
) -> float:
    """Power to detect `true_effect` with the given sample size and std dev.

    Standard two-sample z-test power formula:
        power = P(Z > z_alpha/2 - true_effect / SE)
        SE = sigma * sqrt(2 / n_per_arm)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se = baseline_std * np.sqrt(2 / n_per_arm)
    return float(stats.norm.sf(z_alpha - true_effect / se))


@dataclass
class SRMCheckResult:
    n_control: int
    n_treatment: int
    expected_ratio: float
    chi2_stat: float
    p_value: float
    srm_detected: bool


def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5, alpha: float = 0.001) -> SRMCheckResult:
    """Sample ratio mismatch (SRM) check: is the observed control/treatment
    split consistent with the intended allocation?

    This has to be checked *before* trusting any effect estimate, not after.
    A t-test, CUPED, or a sequential test can all return a perfectly
    well-calibrated result on a sample that was never actually randomized the
    way it was supposed to be — e.g. a logging bug that drops treatment-arm
    events more often than control, or a redirect that leaks users out of one
    arm. That kind of bug doesn't announce itself in the effect estimate; it
    shows up as a split that doesn't match the intended ratio, which is a
    completely different thing to check than "is the effect significant."

    Standard two-cell chi-square goodness-of-fit test against the intended
    allocation ratio:

        chi2 = sum_i (observed_i - expected_i)^2 / expected_i,  i in {control, treatment}
        p    = P(ChiSq_1 > chi2)

    Uses a much stricter default alpha (0.001, not 0.05) for two reasons:
    under a correctly running experiment this should almost never trigger, and
    the cost of missing a real mismatch (trusting a broken experiment's
    results) is high, so it's worth tolerating a lower false-positive rate
    here than the standard significance threshold used for the effect
    estimate itself. Standard practice in production experimentation
    platforms — see Fabijan et al., "Diagnosing Sample Ratio Mismatch in
    Online Controlled Experiments," KDD 2019.

    A single fixed alpha is itself a simplification: chi-square power grows
    with sample size, so a fixed threshold gets *more* sensitive to
    practically meaningless deviations (a 50.01%/49.99% split) as an
    experiment gets larger, not less. Fabijan et al.'s own recommendation is
    to scale the threshold with sample size rather than use one constant for
    every experiment size; that scaling isn't implemented here (see
    DESIGN.md's limitations section) — 0.001 is a stricter, more defensible
    fixed default than a conventional 0.05/0.01, not a solution to that
    scale-dependence.
    """
    n_total = n_control + n_treatment
    if n_total == 0:
        raise ValueError("n_control + n_treatment must be > 0")

    expected_control = n_total * (1 - expected_ratio)
    expected_treatment = n_total * expected_ratio

    chi2_stat = (
        (n_control - expected_control) ** 2 / expected_control
        + (n_treatment - expected_treatment) ** 2 / expected_treatment
    )
    p_value = float(stats.chi2.sf(chi2_stat, df=1))

    return SRMCheckResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_ratio=expected_ratio,
        chi2_stat=float(chi2_stat),
        p_value=p_value,
        srm_detected=p_value < alpha,
    )
