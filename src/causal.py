"""Propensity score matching: a causal-effect estimator for observational
data, where treatment assignment was not randomized.

Chosen over difference-in-differences because the data this platform already
generates and analyzes is cross-sectional (single time point) rather than
panel/pre-post data, so propensity matching is the more natural fit; the
underlying idea generalizes the CUPED-style intuition ("adjust for what you
can measure before comparing outcomes") to the case where the "adjustment"
has to correct for who got treated, not just reduce variance.

Method (Rosenbaum & Rubin, 1983):
1. Fit a **propensity model** — logistic regression predicting P(treatment=1
   | covariates) — on the observed covariates.
2. **Match** each treated unit to the control unit with the closest
   propensity score (nearest-neighbor matching, optionally within a caliper
   that discards poor matches).
3. Estimate the treatment effect as the mean of the matched-pair outcome
   differences, with a standard error and confidence interval computed from
   those paired differences.

Why this removes confounding bias: if treatment assignment depends only on
observed covariates (the "unconfoundedness"/"ignorability" assumption), then
conditioning on the propensity score balances the covariate distributions
between treated and matched-control units — comparing outcomes within a
matched pair approximates comparing the same unit with and without
treatment, which is what a randomized experiment would give directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


@dataclass
class PropensityMatchResult:
    n_treated: int
    n_matched: int
    effect: float
    se: float
    ci_lower: float
    ci_upper: float
    caliper: float | None


def estimate_propensity_scores(covariates: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Fit a logistic regression propensity model and return P(treatment=1 | X)."""
    model = LogisticRegression(max_iter=1000)
    model.fit(covariates, treatment)
    return model.predict_proba(covariates)[:, 1]


def match_treated_to_control(
    propensity: np.ndarray,
    treatment: np.ndarray,
    caliper: float | None = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor match each treated unit to one control unit by propensity score.

    Matching is with replacement (a control unit may be reused across
    multiple treated units), which keeps every treated unit matchable even
    when controls are scarce in some region of the propensity distribution.
    If `caliper` is set, pairs whose propensity scores differ by more than
    `caliper` are dropped rather than forced into a poor match.

    Returns:
        (treated_idx, matched_control_idx): parallel arrays of row indices
        into the original data, one control match per treated unit kept.
    """
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    control_scores = propensity[control_idx].reshape(-1, 1)
    neighbors = NearestNeighbors(n_neighbors=1).fit(control_scores)
    distances, match_positions = neighbors.kneighbors(propensity[treated_idx].reshape(-1, 1))
    distances = distances.ravel()
    matched_control_idx = control_idx[match_positions.ravel()]

    if caliper is not None:
        keep = distances <= caliper
        treated_idx = treated_idx[keep]
        matched_control_idx = matched_control_idx[keep]

    return treated_idx, matched_control_idx


def propensity_score_matching_effect(
    outcome: np.ndarray,
    treatment: np.ndarray,
    covariates: np.ndarray,
    caliper: float | None = 0.05,
    alpha: float = 0.05,
) -> PropensityMatchResult:
    """Estimate the treatment effect via propensity score matching.

    Fits the propensity model, matches treated units to controls, and
    computes the effect as the mean matched-pair outcome difference with a
    standard (paired) t-based confidence interval.
    """
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    covariates = np.asarray(covariates, dtype=float)

    propensity = estimate_propensity_scores(covariates, treatment)
    treated_idx, matched_control_idx = match_treated_to_control(propensity, treatment, caliper)

    diffs = outcome[treated_idx] - outcome[matched_control_idx]
    n_matched = len(diffs)
    mean_diff = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n_matched)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n_matched - 1)

    return PropensityMatchResult(
        n_treated=int((treatment == 1).sum()),
        n_matched=n_matched,
        effect=mean_diff,
        se=se,
        ci_lower=mean_diff - t_crit * se,
        ci_upper=mean_diff + t_crit * se,
        caliper=caliper,
    )


def naive_treatment_effect(outcome: np.ndarray, treatment: np.ndarray) -> float:
    """Unadjusted treated-vs-control mean difference, ignoring confounding.

    This is what a naive analyst would compute on observational data without
    accounting for the fact that treatment wasn't randomly assigned — used
    here purely as the biased baseline to compare the matched estimate against.
    """
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    return outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
