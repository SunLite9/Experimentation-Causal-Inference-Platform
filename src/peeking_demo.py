"""Demonstrate the peeking problem: repeatedly checking a naive t-test as
data accumulates inflates the false-positive rate well above the nominal
alpha, while the mSPRT-based always-valid p-value stays close to alpha under
the same repeated-checking behavior.

Simulates many independent experiments where the null is true (no real
effect), checks significance every `checkpoint_size` new samples per arm in
each experiment, and records — across all simulated experiments — how often
each method flags significance at *some* point during the run.
"""

from __future__ import annotations

import argparse

import numpy as np

from sequential import sequential_look
from simulator import simulate_experiment
from stats_core import welch_t_test


def run_peeking_simulation(
    n_sims: int = 1_000,
    max_n_per_arm: int = 2_000,
    checkpoint_size: int = 100,
    baseline_std: float = 20.0,
    tau2: float = 4.0,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    checkpoints = list(range(checkpoint_size, max_n_per_arm + 1, checkpoint_size))

    naive_false_positives = 0
    sequential_false_positives = 0

    for sim_index in range(n_sims):
        sim_seed = int(rng.integers(0, 2**31 - 1))
        sim = simulate_experiment(
            n_per_arm=max_n_per_arm,
            true_effect=0.0,
            baseline_std=baseline_std,
            seed=sim_seed,
        )
        data = sim.data
        control = data.loc[data["group"] == "control", "outcome"].to_numpy()
        treatment = data.loc[data["group"] == "treatment", "outcome"].to_numpy()

        naive_flagged = False
        sequential_flagged = False

        for n in checkpoints:
            c_so_far = control[:n]
            t_so_far = treatment[:n]

            naive_result = welch_t_test(c_so_far, t_so_far)
            if naive_result.p_value < alpha:
                naive_flagged = True

            pooled_variance = np.concatenate([c_so_far, t_so_far]).var(ddof=1)
            look = sequential_look(
                n_per_arm=n,
                effect=naive_result.effect,
                pooled_variance=pooled_variance,
                tau2=tau2,
                alpha=alpha,
            )
            if look.significant:
                sequential_flagged = True

            # Naive peeking typically stops at the first significant look;
            # the sequential method's validity holds regardless, so both are
            # tracked over the full checkpoint schedule for a fair comparison.

        if naive_flagged:
            naive_false_positives += 1
        if sequential_flagged:
            sequential_false_positives += 1

    return {
        "n_sims": n_sims,
        "checkpoints": checkpoints,
        "naive_false_positive_rate": naive_false_positives / n_sims,
        "sequential_false_positive_rate": sequential_false_positives / n_sims,
        "nominal_alpha": alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the peeking problem.")
    parser.add_argument("--n-sims", type=int, default=1_000)
    parser.add_argument("--max-n-per-arm", type=int, default=2_000)
    parser.add_argument("--checkpoint-size", type=int, default=100)
    parser.add_argument("--tau2", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_peeking_simulation(
        n_sims=args.n_sims,
        max_n_per_arm=args.max_n_per_arm,
        checkpoint_size=args.checkpoint_size,
        tau2=args.tau2,
        alpha=args.alpha,
        seed=args.seed,
    )

    print(f"=== Peeking demonstration ({results['n_sims']} simulated null experiments) ===")
    print(f"Checks per experiment: {len(results['checkpoints'])} (every {args.checkpoint_size} samples/arm, up to {args.max_n_per_arm})")
    print(f"Nominal alpha:                          {results['nominal_alpha']:.3f}")
    print(f"Naive peeking false-positive rate:      {results['naive_false_positive_rate']:.4f}")
    print(f"Sequential (mSPRT) false-positive rate: {results['sequential_false_positive_rate']:.4f}")


if __name__ == "__main__":
    main()
