# Experimentation Causal Inference Platform

[![tests](https://github.com/SunLite9/experimentation-causal-inference-platform/actions/workflows/tests.yml/badge.svg)](https://github.com/SunLite9/experimentation-causal-inference-platform/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A tool that analyzes A/B experiments the way a careful data scientist actually does: checking the randomization first, reducing noise with CUPED, keeping early peeking statistically safe, and correcting for confounding when randomization was not possible at all.

## Problem and motivation

The common way to analyze an experiment, split users into two groups and run a t test, answers a narrower question than most people realize. A metric's variance often has nothing to do with the treatment, which can hide a real effect. Teams check dashboards early and often, which quietly inflates the true false positive rate. Randomization itself can silently break (a logging bug, a bad redirect) and a normal looking p value will never say so. And sometimes there is no randomization to begin with. Each of these needs its own fix, and getting any of them wrong looks exactly like getting them right unless the fix is checked against data with a known answer. This project exists to implement each fix properly and prove it works, rather than assume it does.

## Key features

- **Sample ratio check** that runs before anything else and withholds every downstream verdict if the randomization looks broken.
- **CUPED variance reduction** using a covariate measured before the experiment, verified to not bias the effect estimate.
- **Sequential testing (mSPRT)** for always valid p values, safe under continuous peeking.
- **Propensity score matching** for observational data that was never randomized.
- Every method verified against simulated data with a known true effect, not just trusted at face value.
- A flagship demo where a plain t test says don't ship and CUPED correctly says ship, on the same data with the same true effect.
- A dashboard for simulating or uploading experiment data and comparing every method side by side.

## Architecture and workflow

```
Experiment data (simulated or uploaded CSV)
        ↓
Sample ratio check      did the randomization actually work?
        ↓
t test                  effect size, confidence interval, p value
        ↓
CUPED                   removes noise unrelated to treatment, more power
        ↓
Sequential test         safe to check results early and often
        ↓
Dashboard                pick a scenario, see every verdict side by side
```

When treatment was not randomized:

```
Observational data → propensity matching → effect corrected for confounding
```

## Tech stack

- **Analysis engine**: Python, numpy, pandas, scipy, scikit learn.
- **API**: FastAPI, served with uvicorn.
- **Frontend**: React, TypeScript, Vite.
- **Testing**: pytest, with a GitHub Actions workflow running the suite on every push.
- **Real world data check**: the Criteo uplift benchmark, loaded via `huggingface_hub`.

## Installation and setup

```bash
git clone https://github.com/SunLite9/experimentation-causal-inference-platform.git
cd experimentation-causal-inference-platform
pip install -r requirements.txt
pytest tests/
```

For the dashboard, two terminals:

```bash
# Terminal 1, backend at http://localhost:8000
uvicorn backend.main:app --reload --port 8000

# Terminal 2, frontend at http://localhost:5173, proxies /api to the backend
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`. No environment variables or external services are required; everything runs locally, and the Criteo dataset (only needed for the real data commands below) downloads and caches automatically on first use.

## Usage examples

Baseline analysis on a simulated experiment:

```bash
python src/run_baseline_analysis.py --true-effect 2.0 --n-per-arm 5000
```

CUPED comparison, reporting variance and required sample size reduction:

```bash
python src/compare_cuped.py --true-effect 2.0 --n-per-arm 5000 --extra-noise-std 30 --extra-noise-correlation 0.9
```

Peeking demonstration, naive versus sequential false positive rate:

```bash
python src/peeking_demo.py --n-sims 1000 --max-n-per-arm 2000 --checkpoint-size 100
```

Causal method comparison, naive versus propensity matched estimate:

```bash
python src/compare_causal.py --true-effect 5.0 --n 10000 --confounding-strength 2.0
```

In the dashboard, click "Load flagship demo" to see:

```
True effect: 1.5
Naive:  effect=1.1758  p=0.1544  significant=False   -> DON'T SHIP
CUPED:  effect=1.5870  p=0.0041  significant=True    -> SHIP  (variance reduction: 55.1%)
```

Backend endpoints, all under `/api`: `POST /randomized/simulate`, `POST /randomized/upload`, `POST /observational/simulate`, `POST /observational/upload`, `GET /flagship`, `GET /health`.

## Results and metrics

Every number below comes from actually running the corresponding script against simulated data with a known true effect.

| Method | Result |
|---|---|
| Core stats engine | 5,000 per arm, true effect 2.0: detected effect 2.07, 95% CI [1.28, 2.86], p < 0.001. Across 500 simulations, CI coverage matched the nominal 95%. Across 1,000 null simulations, the false positive rate matched the nominal 5%. |
| Sample ratio check | Flags a 400/600 split against an intended 50/50 every time, never flags an exact match, fires at close to its nominal alpha across thousands of balanced random splits. |
| CUPED | 57.7% variance reduction and 57.6% required sample size reduction on simulated data with a strong (0.9) covariate correlation. On real Criteo data with an unengineered covariate (correlation −0.13): 1.8% variance reduction, the realistic end of the range. |
| Sequential testing | Naive repeated peeking: 23.1% false positive rate (nominal 5%). The always valid p value stays at 1.1%, within the theoretical bound. |
| Propensity matching | Naive estimate: 10.32 (true effect 5.0, biased by confounding). Matched estimate: 4.54, 95% CI [4.34, 4.74], covering the true effect. |
| Flagship demo | Same 3,000 per arm dataset, true effect 1.5: naive not significant (p = 0.154), CUPED significant (p = 0.0041), recovering the true effect after a 55.1% variance reduction. |

## Testing

```bash
pytest tests/
```

15 tests, one file per method (`test_stats_core.py`, `test_srm.py`, `test_cuped.py`, `test_sequential.py`, `test_causal.py`), all of them calibration or correctness checks against a known true effect rather than smoke tests: does the confidence interval actually cover the true effect at the stated rate, does the false positive rate actually match the nominal alpha, does the matched estimate actually land closer to the truth than the naive one. A GitHub Actions workflow runs this suite on every push and pull request to `main`. There is no automated test suite for the frontend; the dashboard was verified manually during development by comparing rendered numbers against the same functions called directly.

## Project structure

```
src/
  simulator.py              synthetic experiment generator with a known true effect
  stats_core.py              t test, confidence intervals, power and sample size
  cuped.py                    CUPED variance reduction
  compare_cuped.py            naive vs CUPED comparison, measured variance and sample size reduction
  sequential.py                always valid p values for safe early peeking
  peeking_demo.py              naive vs sequential false positive rate under repeated peeking
  causal.py                     propensity score matching for data that was not randomized
  compare_causal.py             naive vs matched estimate on confounded data
  data_loader.py               loads the real Criteo experiment dataset
  run_baseline_analysis.py   simulate or load an experiment, print the analysis
tests/                        correctness and calibration tests for every method
backend/
  main.py                       FastAPI app, a thin layer over src/
frontend/
  src/App.tsx                  React and TypeScript dashboard
  src/components/              verdict badges, results panels, peeking chart
notebooks/                    scratch analysis, not part of the shipped project
data/                         cached datasets, not tracked in git
DESIGN.md                     the full reasoning behind every decision, tradeoffs, results, and known limitations
```

## Design decisions and tradeoffs

- **Welch's t test** everywhere, not Student's pooled variance version, since a treatment can change variance as well as the mean, and Welch's is a safe default in both cases.
- **Formulas implemented by hand**, not called from `scipy.stats.ttest_ind`, so the mechanism the rest of the system builds on is fully understood and independently verified.
- **mSPRT chosen over group sequential alpha spending** for sequential testing, since it does not require committing to a look schedule in advance.
- **Propensity matching chosen over difference in differences or uplift modeling**, since it fits the project's cross sectional data shape and is directly testable against a known confounding strength.
- **FastAPI and React chosen over the original Streamlit version**, once the interface became a first class concern, for a typed API boundary and a conventional client and server split.

Every alternative considered, why it was rejected, and the specific cost accepted for each choice is documented in full in [DESIGN.md](DESIGN.md).

## Limitations and future improvements

- Power, MDE, and sample size use a normal approximation, measurably off at very small sample sizes.
- The sample ratio check's alpha is a fixed constant, not scaled to sample size.
- Propensity matching's confidence interval uses a simplified formula that does not fully account for matching with replacement.
- CUPED and the sequential test are not composed into a single variance reduced, peeking safe pipeline.
- No interference or SUTVA detection; every method assumes one unit's outcome is unaffected by another unit's treatment assignment.
- No authentication, persistence, or production deployment path; this is a local, single user analysis tool.

The full list, with the reasoning behind each one and concrete future work, is in [DESIGN.md](DESIGN.md).

## Deployment

This project runs locally and is not currently deployed anywhere. There is no Dockerfile, no hosted instance, and no environment based configuration for a non localhost backend URL. The GitHub Actions workflow in `.github/workflows/tests.yml` runs the test suite on every push and pull request to `main`, which is the only automated pipeline that exists today. Standing up a real deployment (a single build step serving the frontend from the backend, containerized, with authentication and rate limiting) is listed as future work in DESIGN.md.

## Contributing and license

This is a personal portfolio project. Issues and pull requests are welcome but there is no formal contribution process. Licensed under the [MIT License](LICENSE).
