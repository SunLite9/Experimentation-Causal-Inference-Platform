import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulator import simulate_experiment
from stats_core import srm_check


def test_srm_check_flags_a_clear_mismatch():
    """A split that's badly off the intended 50/50 allocation must be flagged."""
    result = srm_check(n_control=400, n_treatment=600, expected_ratio=0.5)

    assert result.srm_detected
    assert result.p_value < 0.01


def test_srm_check_does_not_flag_the_intended_allocation():
    """An exact match to the intended ratio must never be flagged."""
    result = srm_check(n_control=5000, n_treatment=5000, expected_ratio=0.5)

    assert not result.srm_detected
    assert result.p_value == 1.0


def test_srm_check_false_positive_rate_matches_nominal_alpha():
    """Across many genuinely-balanced random splits, the check should flag at
    close to its own alpha rate -- not meaningfully more (miscalibrated test)
    and not meaningfully less (a test with no power to ever fire)."""
    rng = np.random.default_rng(0)
    alpha = 0.01
    n_sims = 2000
    n_total = 1000

    false_positives = 0
    for _ in range(n_sims):
        n_treatment = int(rng.binomial(n_total, 0.5))
        n_control = n_total - n_treatment
        result = srm_check(n_control, n_treatment, expected_ratio=0.5, alpha=alpha)
        if result.srm_detected:
            false_positives += 1

    false_positive_rate = false_positives / n_sims
    assert 0.003 <= false_positive_rate <= 0.02


def test_srm_check_default_alpha_is_stricter_than_conventional_005_or_001():
    """Locks in the intentionally strict default (0.001): a borderline split
    (p ~= 0.0027) that a conventional 0.01 threshold would flag must NOT be
    flagged by the default, since the default is meant to be stricter than
    that -- see stats_core.srm_check's docstring for why."""
    n_control, n_treatment = 5150, 4850  # n_total=10,000, p-value ~ 0.0027

    default_result = srm_check(n_control, n_treatment)
    looser_result = srm_check(n_control, n_treatment, alpha=0.01)

    assert 0.001 < default_result.p_value < 0.01
    assert not default_result.srm_detected
    assert looser_result.srm_detected


def test_srm_check_on_simulated_experiment_never_flags():
    """The simulator always builds an exact n_per_arm/n_per_arm split, so the
    check should never flag it -- a sanity check that real counts from the
    rest of the system flow into srm_check correctly."""
    sim = simulate_experiment(n_per_arm=2000, true_effect=0.0, seed=1)
    data = sim.data
    n_control = int((data["group"] == "control").sum())
    n_treatment = int((data["group"] == "treatment").sum())

    result = srm_check(n_control, n_treatment)

    assert not result.srm_detected
