"""Sequential (always-valid) significance testing.

A plain t-test's p-value is only valid at a single, pre-committed sample
size. Checking it repeatedly as data accumulates and stopping the first time
it dips below 0.05 ("peeking") inflates the true false-positive rate well
above the nominal 5%, because each additional look is another chance for
noise to cross the threshold (the classic "we peeked early and shipped"
mistake — see `peeking_demo.py`).

This implements the **mixture sequential probability ratio test (mSPRT)**
for a difference in means (Robbins, 1970; the same construction used in
Johari, Koomen, Pekelis & Walsh, "Peeking at A/B Tests", KDD 2017 — the
method behind Optimizely's always-valid p-values). It's the more tractable
of the two standard options (vs. group-sequential alpha-spending, which
requires pre-committing to a fixed number/spacing of looks); mSPRT is valid
at *any* stopping rule, including checking after every single new data point.

Construction: model the unknown treatment effect theta under a Gaussian
mixing prior N(0, tau^2) instead of a fixed alternative. At each look t, with
current effect estimate Delta_t and estimated variance V_t of that estimate:

    Lambda_t = sqrt(V_t / (V_t + tau^2)) * exp( tau^2 * Delta_t^2 / (2 * V_t * (V_t + tau^2)) )
    p_t      = min(1, 1 / Lambda_t)

`Lambda_t` is a likelihood ratio (mixture alternative vs. null) and is a
nonnegative martingale under the null hypothesis (theta = 0) by construction,
which is what makes `p_t` valid as an "always-valid p-value": by Ville's
inequality, P(exists t: Lambda_t >= 1/alpha) <= alpha under the null,
regardless of when or how often you look. Rejecting whenever p_t <= alpha
therefore keeps the overall false-positive rate at or below alpha even under
continuous peeking — unlike the naive t-test's per-look p-value.

`tau^2` is a prior-variance tuning parameter representing the scale of
effect you actually care about detecting; it doesn't need to be exactly
right for validity to hold, but a `tau` on the same order as effects you
expect gives the best power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SequentialLook:
    n_per_arm: int
    effect: float
    likelihood_ratio: float
    p_value: float
    significant: bool


def mixture_likelihood_ratio(effect: float, variance_of_effect: float, tau2: float) -> float:
    """Lambda_t: the mSPRT likelihood ratio at a single look.

    `variance_of_effect` is Var(Delta_t), the variance of the current
    difference-in-means estimate (e.g. `2 * pooled_variance / n_per_arm`
    under equal allocation with per-unit variance `pooled_variance`).
    """
    if variance_of_effect <= 0:
        return 1.0

    v = variance_of_effect
    return math.sqrt(v / (v + tau2)) * math.exp(
        (tau2 * effect**2) / (2 * v * (v + tau2))
    )


def always_valid_p_value(likelihood_ratio: float) -> float:
    """p_t = min(1, 1 / Lambda_t)."""
    if likelihood_ratio <= 0:
        return 1.0
    return min(1.0, 1.0 / likelihood_ratio)


def sequential_look(
    n_per_arm: int,
    effect: float,
    pooled_variance: float,
    tau2: float,
    alpha: float = 0.05,
) -> SequentialLook:
    """Evaluate one look of the sequential test given the data seen so far.

    `pooled_variance` is the per-unit outcome variance (pooled across arms);
    the variance of the difference-in-means estimate under equal allocation
    with `n_per_arm` units in each arm is `2 * pooled_variance / n_per_arm`.
    """
    variance_of_effect = 2 * pooled_variance / n_per_arm if n_per_arm > 0 else float("inf")
    lam = mixture_likelihood_ratio(effect, variance_of_effect, tau2)
    p = always_valid_p_value(lam)

    return SequentialLook(
        n_per_arm=n_per_arm,
        effect=effect,
        likelihood_ratio=lam,
        p_value=p,
        significant=p <= alpha,
    )
