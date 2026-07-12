"""Compare a naive treated-vs-control comparison against propensity score
matching on simulated observational (non-randomized) data with a known true
effect, to show the naive estimate is biased by confounding while the
matched estimate recovers the truth.
"""

from __future__ import annotations

import argparse

from causal import naive_treatment_effect, propensity_score_matching_effect
from simulator import simulate_observational_data


def run_comparison(
    n: int = 10_000,
    true_effect: float = 5.0,
    confounding_strength: float = 2.0,
    caliper: float = 0.05,
    seed: int = 42,
) -> None:
    sim = simulate_observational_data(
        n=n,
        true_effect=true_effect,
        confounding_strength=confounding_strength,
        seed=seed,
    )
    data = sim.data

    outcome = data["outcome"].to_numpy()
    treatment = data["treatment"].to_numpy()
    covariates = data[["covariate_1", "covariate_2"]].to_numpy()

    naive_effect = naive_treatment_effect(outcome, treatment)
    matched = propensity_score_matching_effect(outcome, treatment, covariates, caliper=caliper)

    print("=== Naive treated-vs-control comparison (ignores confounding) ===")
    print(f"Estimated effect:  {naive_effect:.4f}")
    print(f"Bias vs. truth:    {naive_effect - true_effect:+.4f}")

    print("\n=== Propensity score matching ===")
    print(f"Treated units:            {matched.n_treated}")
    print(f"Matched pairs (caliper={matched.caliper}): {matched.n_matched}")
    print(f"Estimated effect:        {matched.effect:.4f}")
    print(f"95% CI:                   [{matched.ci_lower:.4f}, {matched.ci_upper:.4f}]")
    print(f"Bias vs. truth:           {matched.effect - true_effect:+.4f}")

    print(f"\nTrue effect (ground truth): {true_effect:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive vs. propensity-matched causal estimates.")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--true-effect", type=float, default=5.0)
    parser.add_argument("--confounding-strength", type=float, default=2.0)
    parser.add_argument("--caliper", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_comparison(
        n=args.n,
        true_effect=args.true_effect,
        confounding_strength=args.confounding_strength,
        caliper=args.caliper,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
