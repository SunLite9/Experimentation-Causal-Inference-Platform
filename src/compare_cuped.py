"""Compare the naive t-test against the CUPED-adjusted t-test, either on a
simulated experiment with extra treatment-unrelated noise (the scenario
CUPED is built for -- a pre-experiment covariate correlated with noise that
has nothing to do with the treatment) or on the real Criteo randomized
dataset, and report the resulting variance reduction and required-sample-
size reduction.

The simulated path is the primary evidence of correctness (§true_effect is
known, so the point estimate can be checked against ground truth -- see
tests/test_cuped.py). The `--source criteo` path is a plumbing/face-validity
check on real data: it confirms CUPED's variance-reduction machinery runs
correctly on a real dataset's actual covariate-outcome correlation (which is
far weaker than the simulated demo's deliberately strong 0.9 correlation),
not a correctness proof, since there's no known true effect in Criteo data
to compare against.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cuped import compute_theta, cuped_adjust
from simulator import simulate_experiment
from stats_core import confidence_interval, required_sample_size, welch_t_test


def analyze(data: pd.DataFrame, true_effect: float | None = None) -> None:
    control_mask = data["group"] == "control"
    treatment_mask = data["group"] == "treatment"

    raw_control = data.loc[control_mask, "outcome"].to_numpy()
    raw_treatment = data.loc[treatment_mask, "outcome"].to_numpy()

    naive_result = welch_t_test(raw_control, raw_treatment)
    naive_ci = confidence_interval(naive_result)

    # theta and the covariate mean are estimated once on the pooled sample,
    # then applied identically to both arms so the adjustment can't shift
    # the difference in means (see src/cuped.py).
    pooled_outcome = data["outcome"].to_numpy()
    pooled_covariate = data["pre_covariate"].to_numpy()
    theta = compute_theta(pooled_outcome, pooled_covariate)
    covariate_mean = pooled_covariate.mean()
    covariate_correlation = np.corrcoef(pooled_outcome, pooled_covariate)[0, 1]

    adjusted_control = cuped_adjust(
        raw_control, data.loc[control_mask, "pre_covariate"].to_numpy(), theta, covariate_mean
    )
    adjusted_treatment = cuped_adjust(
        raw_treatment, data.loc[treatment_mask, "pre_covariate"].to_numpy(), theta, covariate_mean
    )

    cuped_result = welch_t_test(adjusted_control, adjusted_treatment)
    cuped_ci = confidence_interval(cuped_result)

    raw_var = data["outcome"].var(ddof=1)
    adjusted_var = np.concatenate([adjusted_control, adjusted_treatment]).var(ddof=1)
    variance_reduction_pct = 100 * (1 - adjusted_var / raw_var) if raw_var > 0 else 0.0

    # With real data there's no configured true effect to target the sample
    # size calculation at, so fall back to the observed naive effect -- same
    # convention run_baseline_analysis.py uses for the same reason.
    mde_target = true_effect if true_effect not in (None, 0) else (naive_result.effect or 1e-9)
    n_naive = required_sample_size(baseline_std=np.sqrt(raw_var), mde=mde_target)
    n_cuped = required_sample_size(baseline_std=np.sqrt(adjusted_var), mde=mde_target)
    sample_size_reduction_pct = 100 * (1 - n_cuped / n_naive)

    print(f"Covariate-outcome correlation: {covariate_correlation:.4f}")

    print("\n=== Naive t-test (raw outcome) ===")
    print(f"Effect estimate:  {naive_result.effect:.4f}")
    print(f"95% CI:            [{naive_ci.lower:.4f}, {naive_ci.upper:.4f}]")
    print(f"p-value:           {naive_result.p_value:.6f}")

    print("\n=== CUPED-adjusted t-test ===")
    print(f"theta:             {theta:.4f}")
    print(f"Effect estimate:  {cuped_result.effect:.4f}")
    print(f"95% CI:            [{cuped_ci.lower:.4f}, {cuped_ci.upper:.4f}]")
    print(f"p-value:           {cuped_result.p_value:.6f}")

    if true_effect is not None:
        print(f"\nTrue effect (ground truth): {true_effect:.4f}")
    print(f"Raw outcome variance:       {raw_var:.4f}")
    print(f"CUPED-adjusted variance:    {adjusted_var:.4f}")
    print(f"Variance reduction:         {variance_reduction_pct:.2f}%")
    print(f"Sample size (naive, per arm):    {n_naive}")
    print(f"Sample size (CUPED, per arm):    {n_cuped}")
    print(f"Required sample size reduction:  {sample_size_reduction_pct:.2f}%")


def run_comparison(
    n_per_arm: int = 5_000,
    true_effect: float = 2.0,
    baseline_mean: float = 100.0,
    baseline_std: float = 20.0,
    extra_noise_std: float = 30.0,
    extra_noise_correlation: float = 0.9,
    seed: int = 42,
) -> None:
    sim = simulate_experiment(
        n_per_arm=n_per_arm,
        true_effect=true_effect,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        extra_noise_std=extra_noise_std,
        extra_noise_correlation=extra_noise_correlation,
        seed=seed,
    )
    analyze(sim.data, true_effect=sim.true_effect)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive vs. CUPED-adjusted t-tests.")
    parser.add_argument("--n-per-arm", type=int, default=5_000)
    parser.add_argument("--true-effect", type=float, default=2.0)
    parser.add_argument("--extra-noise-std", type=float, default=30.0)
    parser.add_argument("--extra-noise-correlation", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source",
        choices=["simulator", "criteo"],
        default="simulator",
        help="Use synthetic simulated data (default) or the real Criteo experiment dataset.",
    )
    args = parser.parse_args()

    if args.source == "criteo":
        from data_loader import load_criteo_experiment

        data = load_criteo_experiment()
        analyze(data, true_effect=None)
    else:
        run_comparison(
            n_per_arm=args.n_per_arm,
            true_effect=args.true_effect,
            extra_noise_std=args.extra_noise_std,
            extra_noise_correlation=args.extra_noise_correlation,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
