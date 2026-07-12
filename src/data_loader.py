"""Loader for a real randomized-experiment dataset, used to sanity-check the
stats engine against something other than synthetic data.

Source: the Criteo uplift-modeling benchmark (Diemert et al., AdKDD 2018),
which is itself a real randomized ad-exposure experiment (users are randomly
withheld from a treatment) with visit/conversion outcomes and 12 anonymized
pre-treatment features. It's pulled from Criteo's own Hugging Face dataset
repository rather than the original criteo.com download link.
"""

from __future__ import annotations

import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ID = "criteo/criteo-uplift"
_FILENAME = "criteo-research-uplift-v2.1.csv.gz"


def load_criteo_experiment(
    n_rows: int | None = 200_000,
    outcome_col: str = "visit",
    covariate_col: str = "f0",
    seed: int = 0,
) -> pd.DataFrame:
    """Download (if needed, cached locally) and load the Criteo uplift dataset.

    The full file is ~25M rows / 311MB, so by default only the first `n_rows`
    are loaded to keep this fast; pass `n_rows=None` for the full dataset.

    Reshapes the raw columns into the same schema used by the simulator
    (`group`, `outcome`, `pre_covariate`) so it can be dropped straight into
    `stats_core.welch_t_test` and friends:
      - `group`: "treatment" / "control", from the raw `treatment` column.
      - `outcome`: the raw `outcome_col` (default "visit", a binary
        did-the-user-visit indicator; "conversion" is the other option).
      - `pre_covariate`: one of the anonymized pre-treatment features
        (default "f0"), used as a stand-in pre-experiment covariate.

    Rows are drawn via a random sample rather than the first `n_rows`,
    because the raw file is grouped by treatment (a naive `nrows=` prefix
    read pulls a treatment-only or control-only slice).

    Args:
        n_rows: Number of rows to load, or None for the full dataset.
        outcome_col: Which raw outcome column to use ("visit" or "conversion").
        covariate_col: Which raw feature column to use as the pre-experiment
            covariate (f0..f11).
        seed: Random seed for the row sample.

    Returns:
        DataFrame with columns [unit_id, group, pre_covariate, outcome],
        matching `simulator.simulate_experiment`'s output schema.
    """
    local_path = hf_hub_download(repo_id=_REPO_ID, filename=_FILENAME, repo_type="dataset")

    raw = pd.read_csv(
        local_path,
        compression="gzip",
        usecols=["treatment", outcome_col, covariate_col],
    )
    if n_rows is not None and n_rows < len(raw):
        raw = raw.sample(n=n_rows, random_state=seed)

    data = pd.DataFrame(
        {
            "unit_id": raw.index,
            "group": raw["treatment"].map({1: "treatment", 0: "control"}),
            "pre_covariate": raw[covariate_col],
            "outcome": raw[outcome_col],
        }
    )
    return data
