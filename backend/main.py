"""FastAPI backend for the experimentation dashboard.

A thin HTTP layer over the analysis functions in `src/` — every endpoint
here just calls the same simulator/stats_core/cuped/sequential/causal
functions used by the CLI scripts and tests, so the numbers the frontend
shows are guaranteed to match the ones already verified there.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal import naive_treatment_effect, propensity_score_matching_effect  # noqa: E402
from cuped import compute_theta, cuped_adjust  # noqa: E402
from sequential import sequential_look  # noqa: E402
from simulator import simulate_experiment, simulate_observational_data  # noqa: E402
from stats_core import confidence_interval, srm_check, welch_t_test  # noqa: E402

app = FastAPI(title="Experimentation Causal Inference Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALPHA = 0.05

# The flagship scenario: a real +1.5 effect buried under unrelated noise that
# a naive t-test misses, but that CUPED recovers because a pre-experiment
# covariate happens to capture that noise (corr = 0.92). See the "Flagship
# demo" section of the README.
FLAGSHIP_PARAMS = dict(
    n_per_arm=3000,
    true_effect=1.5,
    baseline_mean=100.0,
    baseline_std=20.0,
    extra_noise_std=25.0,
    extra_noise_correlation=0.92,
    covariate_correlation=0.7,
    seed=22,
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class SimulateRandomizedRequest(BaseModel):
    n_per_arm: int = 5_000
    true_effect: float = 2.0
    baseline_mean: float = 100.0
    baseline_std: float = 20.0
    extra_noise_std: float = 0.0
    extra_noise_correlation: float = 0.0
    covariate_correlation: float = 0.7
    seed: int = 42
    include_peeking: bool = False
    checkpoint_size: int = 100


class SimulateObservationalRequest(BaseModel):
    n: int = 10_000
    true_effect: float = 5.0
    confounding_strength: float = 2.0
    caliper: float = 0.05
    seed: int = 42


class TTestResultOut(BaseModel):
    effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool


class SRMCheckOut(BaseModel):
    n_control: int
    n_treatment: int
    expected_ratio: float
    p_value: float
    srm_detected: bool


class PeekingCheckpoint(BaseModel):
    n_per_arm: int
    naive_p_value: float
    sequential_p_value: float


class PeekingResult(BaseModel):
    checkpoints: list[PeekingCheckpoint]
    naive_first_flag_n: int | None
    sequential_first_flag_n: int | None


class RandomizedAnalysisResponse(BaseModel):
    true_effect: float | None
    srm: SRMCheckOut
    naive: TTestResultOut
    cuped: TTestResultOut
    variance_reduction_pct: float
    peeking: PeekingResult | None = None


class CausalAnalysisResponse(BaseModel):
    true_effect: float | None
    naive_effect: float
    matched_effect: float
    matched_ci_lower: float
    matched_ci_upper: float
    n_matched: int
    n_treated: int


# ---------------------------------------------------------------------------
# Analysis helpers (shared by simulate + upload paths)
# ---------------------------------------------------------------------------


def _analyze_randomized(data: pd.DataFrame, checkpoint_size: int | None) -> RandomizedAnalysisResponse:
    required = {"group", "outcome", "pre_covariate"}
    missing = required - set(data.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing required columns: {sorted(missing)}")

    control_mask = data["group"] == "control"
    treatment_mask = data["group"] == "treatment"
    raw_control = data.loc[control_mask, "outcome"].to_numpy()
    raw_treatment = data.loc[treatment_mask, "outcome"].to_numpy()

    srm = srm_check(n_control=len(raw_control), n_treatment=len(raw_treatment))

    naive_result = welch_t_test(raw_control, raw_treatment)
    naive_ci = confidence_interval(naive_result)

    pooled_outcome = data["outcome"].to_numpy()
    pooled_covariate = data["pre_covariate"].to_numpy()
    theta = compute_theta(pooled_outcome, pooled_covariate)
    covariate_mean = pooled_covariate.mean()

    adj_control = cuped_adjust(
        raw_control, data.loc[control_mask, "pre_covariate"].to_numpy(), theta, covariate_mean
    )
    adj_treatment = cuped_adjust(
        raw_treatment, data.loc[treatment_mask, "pre_covariate"].to_numpy(), theta, covariate_mean
    )
    cuped_result = welch_t_test(adj_control, adj_treatment)
    cuped_ci = confidence_interval(cuped_result)

    raw_var = data["outcome"].var(ddof=1)
    adjusted_var = np.concatenate([adj_control, adj_treatment]).var(ddof=1)
    variance_reduction_pct = 100 * (1 - adjusted_var / raw_var) if raw_var > 0 else 0.0

    peeking = None
    if checkpoint_size:
        n_per_arm = min(len(raw_control), len(raw_treatment))
        checkpoints = list(range(checkpoint_size, n_per_arm + 1, checkpoint_size))
        rows: list[PeekingCheckpoint] = []
        naive_first_flag_n = None
        sequential_first_flag_n = None
        for n in checkpoints:
            c_so_far, t_so_far = raw_control[:n], raw_treatment[:n]
            look_naive = welch_t_test(c_so_far, t_so_far)
            pooled_variance = np.concatenate([c_so_far, t_so_far]).var(ddof=1)
            look_seq = sequential_look(n_per_arm=n, effect=look_naive.effect, pooled_variance=pooled_variance, tau2=4.0, alpha=ALPHA)

            if look_naive.p_value < ALPHA and naive_first_flag_n is None:
                naive_first_flag_n = n
            if look_seq.significant and sequential_first_flag_n is None:
                sequential_first_flag_n = n

            rows.append(
                PeekingCheckpoint(n_per_arm=n, naive_p_value=look_naive.p_value, sequential_p_value=look_seq.p_value)
            )

        peeking = PeekingResult(
            checkpoints=rows,
            naive_first_flag_n=naive_first_flag_n,
            sequential_first_flag_n=sequential_first_flag_n,
        )

    return RandomizedAnalysisResponse(
        true_effect=None,
        srm=SRMCheckOut(
            n_control=srm.n_control,
            n_treatment=srm.n_treatment,
            expected_ratio=srm.expected_ratio,
            p_value=srm.p_value,
            srm_detected=srm.srm_detected,
        ),
        naive=TTestResultOut(
            effect=naive_result.effect,
            ci_lower=naive_ci.lower,
            ci_upper=naive_ci.upper,
            p_value=naive_result.p_value,
            significant=naive_result.p_value < ALPHA,
        ),
        cuped=TTestResultOut(
            effect=cuped_result.effect,
            ci_lower=cuped_ci.lower,
            ci_upper=cuped_ci.upper,
            p_value=cuped_result.p_value,
            significant=cuped_result.p_value < ALPHA,
        ),
        variance_reduction_pct=variance_reduction_pct,
        peeking=peeking,
    )


def _analyze_observational(data: pd.DataFrame, caliper: float) -> CausalAnalysisResponse:
    if "treatment" not in data.columns or "outcome" not in data.columns:
        raise HTTPException(status_code=422, detail="CSV must have 'treatment' and 'outcome' columns")
    covariate_cols = [c for c in data.columns if c.startswith("covariate_")]
    if not covariate_cols:
        raise HTTPException(status_code=422, detail="CSV must have at least one 'covariate_*' column")

    outcome = data["outcome"].to_numpy()
    treatment = data["treatment"].to_numpy()
    covariates = data[covariate_cols].to_numpy()

    naive_effect = naive_treatment_effect(outcome, treatment)
    matched = propensity_score_matching_effect(outcome, treatment, covariates, caliper=caliper)

    return CausalAnalysisResponse(
        true_effect=None,
        naive_effect=naive_effect,
        matched_effect=matched.effect,
        matched_ci_lower=matched.ci_lower,
        matched_ci_upper=matched.ci_upper,
        n_matched=matched.n_matched,
        n_treated=matched.n_treated,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/randomized/simulate", response_model=RandomizedAnalysisResponse)
def simulate_randomized(req: SimulateRandomizedRequest) -> RandomizedAnalysisResponse:
    sim = simulate_experiment(
        n_per_arm=req.n_per_arm,
        true_effect=req.true_effect,
        baseline_mean=req.baseline_mean,
        baseline_std=req.baseline_std,
        extra_noise_std=req.extra_noise_std,
        extra_noise_correlation=req.extra_noise_correlation,
        covariate_correlation=req.covariate_correlation,
        seed=req.seed,
    )
    checkpoint_size = req.checkpoint_size if req.include_peeking else None
    result = _analyze_randomized(sim.data, checkpoint_size)
    result.true_effect = sim.true_effect
    return result


@app.post("/api/randomized/upload", response_model=RandomizedAnalysisResponse)
async def upload_randomized(file: UploadFile = File(...), checkpoint_size: int | None = None) -> RandomizedAnalysisResponse:
    contents = await file.read()
    data = pd.read_csv(io.BytesIO(contents))
    return _analyze_randomized(data, checkpoint_size)


@app.post("/api/observational/simulate", response_model=CausalAnalysisResponse)
def simulate_observational(req: SimulateObservationalRequest) -> CausalAnalysisResponse:
    sim = simulate_observational_data(
        n=req.n,
        true_effect=req.true_effect,
        confounding_strength=req.confounding_strength,
        seed=req.seed,
    )
    result = _analyze_observational(sim.data, req.caliper)
    result.true_effect = sim.true_effect
    return result


@app.post("/api/observational/upload", response_model=CausalAnalysisResponse)
async def upload_observational(file: UploadFile = File(...), caliper: float = 0.05) -> CausalAnalysisResponse:
    contents = await file.read()
    data = pd.read_csv(io.BytesIO(contents))
    return _analyze_observational(data, caliper)


@app.get("/api/flagship", response_model=RandomizedAnalysisResponse)
def flagship_demo() -> RandomizedAnalysisResponse:
    sim = simulate_experiment(**FLAGSHIP_PARAMS)
    result = _analyze_randomized(sim.data, checkpoint_size=None)
    result.true_effect = sim.true_effect
    return result


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
