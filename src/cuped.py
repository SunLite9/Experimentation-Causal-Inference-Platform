"""CUPED: Controlled-experiment Using Pre-Experiment Data.

Strips out outcome variance that's predictable from a pre-experiment
covariate (and therefore has nothing to do with the treatment) before
running the t-test, so the test has more power to detect a real effect at
the same sample size.

Formula (Deng et al., "Improving the Sensitivity of Online Controlled
Experiments by Utilizing Pre-Experiment Data", WSDM 2013):

    theta       = Cov(Y, X) / Var(X)
    Y_adjusted  = Y - theta * (X - mean(X))

`theta` is the coefficient that minimizes Var(Y_adjusted); it's exactly the
OLS slope of Y on X. Subtracting `theta * (X - mean(X))` removes the part of
Y's variance that's linearly predictable from X, while leaving E[Y_adjusted]
unchanged (in expectation) since `X - mean(X)` has mean zero within each arm.
Because the correction is applied identically to both arms, it doesn't shift
the difference in means — it only removes variance, which is why CUPED
increases power without introducing bias.
"""

from __future__ import annotations

import numpy as np


def compute_theta(outcome: np.ndarray, covariate: np.ndarray) -> float:
    """CUPED adjustment coefficient: theta = Cov(Y, X) / Var(X).

    Should be estimated on the pooled sample (both arms together) so the
    same correction is applied to control and treatment alike — using
    per-arm theta values would reintroduce bias.
    """
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)

    cov_matrix = np.cov(outcome, covariate, ddof=1)
    cov_xy = cov_matrix[0, 1]
    var_x = cov_matrix[1, 1]

    if var_x == 0:
        return 0.0
    return cov_xy / var_x


def cuped_adjust(
    outcome: np.ndarray,
    covariate: np.ndarray,
    theta: float | None = None,
    covariate_mean: float | None = None,
) -> np.ndarray:
    """Return the CUPED-adjusted outcome: Y - theta * (X - mean(X)).

    `theta` and `covariate_mean` should both be estimated once on the pooled
    control+treatment sample and passed in explicitly to each arm. Using a
    per-arm covariate mean instead would subtract a different constant from
    each arm and bias the difference in means; using the pooled mean keeps
    the adjustment mean-zero overall so it cannot shift the treatment effect
    estimate. If omitted, both are estimated from the arguments given
    (correct only when called once on the full pooled sample).
    """
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)

    if theta is None:
        theta = compute_theta(outcome, covariate)
    if covariate_mean is None:
        covariate_mean = covariate.mean()

    return outcome - theta * (covariate - covariate_mean)
