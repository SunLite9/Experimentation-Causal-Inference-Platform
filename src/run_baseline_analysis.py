"""Entry point: simulate an experiment (or load a real one) and run the core
stats engine end to end, printing a full analysis report.
"""

from __future__ import annotations

import argparse

from simulator import simulate_experiment
from stats_core import (
    confidence_interval,
    minimum_detectable_effect,
    required_sample_size,
    srm_check,
    statistical_power,
    welch_t_test,
)


def analyze(data, true_effect: float | None = None) -> None:
    control = data.loc[data["group"] == "control", "outcome"].to_numpy()
    treatment = data.loc[data["group"] == "treatment", "outcome"].to_numpy()

    # Checked first and separately from everything below: a sample ratio
    # mismatch means the randomization itself is broken, which makes any
    # downstream effect estimate untrustworthy regardless of its own p-value.
    srm = srm_check(n_control=len(control), n_treatment=len(treatment))
    print("=== Sample ratio mismatch (SRM) check ===")
    print(f"Observed split:                  {srm.n_control} control / {srm.n_treatment} treatment")
    print(f"Expected ratio:                  {srm.expected_ratio:.2f}")
    print(f"chi-square p-value:               {srm.p_value:.6f}")
    if srm.srm_detected:
        print("SRM DETECTED -- do not trust the results below until this is investigated.\n")
    else:
        print("No SRM detected.\n")

    result = welch_t_test(control, treatment)
    ci = confidence_interval(result, alpha=0.05)

    baseline_std = control.std(ddof=1)
    power = statistical_power(baseline_std, result.n_control, result.effect)
    mde = minimum_detectable_effect(baseline_std, result.n_control)
    n_needed = required_sample_size(baseline_std, mde=result.effect if result.effect != 0 else mde)

    print("=== Baseline experiment analysis ===")
    if true_effect is not None:
        print(f"True effect (ground truth):     {true_effect:.4f}")
    print(f"Control mean:                    {result.mean_control:.4f}  (n={result.n_control})")
    print(f"Treatment mean:                  {result.mean_treatment:.4f}  (n={result.n_treatment})")
    print(f"Estimated effect:                {result.effect:.4f}")
    print(f"95% CI:                          [{ci.lower:.4f}, {ci.upper:.4f}]")
    print(f"t-statistic:                     {result.t_stat:.4f}  (df={result.df:.1f})")
    print(f"p-value:                         {result.p_value:.6f}")
    print(f"Significant at alpha=0.05:       {result.p_value < 0.05}")
    print(f"Observed power (at this effect): {power:.4f}")
    print(f"MDE at n={result.n_control} per arm:          {mde:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline t-test analysis.")
    parser.add_argument("--n-per-arm", type=int, default=5_000)
    parser.add_argument("--true-effect", type=float, default=2.0)
    parser.add_argument("--baseline-mean", type=float, default=100.0)
    parser.add_argument("--baseline-std", type=float, default=20.0)
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
        sim = simulate_experiment(
            n_per_arm=args.n_per_arm,
            true_effect=args.true_effect,
            baseline_mean=args.baseline_mean,
            baseline_std=args.baseline_std,
            seed=args.seed,
        )
        analyze(sim.data, true_effect=sim.true_effect)


if __name__ == "__main__":
    main()
