import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peeking_demo import run_peeking_simulation


def test_sequential_testing_controls_false_positive_rate_under_repeated_peeking():
    """Under the null (no true effect), checking a naive t-test repeatedly as
    data accumulates should blow past the nominal alpha; the mSPRT-based
    always-valid p-value should stay at or safely below it."""
    alpha = 0.05
    results = run_peeking_simulation(
        n_sims=400,
        max_n_per_arm=2_000,
        checkpoint_size=100,
        tau2=4.0,
        alpha=alpha,
        seed=123,
    )

    # Naive repeated peeking is expected to be clearly inflated above nominal alpha.
    assert results["naive_false_positive_rate"] > 0.15

    # mSPRT's always-valid p-value bounds the false-positive rate at or below
    # alpha by construction (Ville's inequality); with a finite number of
    # looks it's typically well below alpha, which is the safe direction.
    assert results["sequential_false_positive_rate"] <= alpha
