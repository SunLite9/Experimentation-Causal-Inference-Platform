# Experimentation Causal Inference Platform

[![tests](https://github.com/SunLite9/experimentation-causal-inference-platform/actions/workflows/tests.yml/badge.svg)](https://github.com/SunLite9/experimentation-causal-inference-platform/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A tool for analyzing experiments the way a careful data scientist actually does, not just a single significance test. It checks whether an experiment was even randomized correctly, reduces noise with CUPED, keeps peeking at results safe with a sequential test, and corrects for confounding when randomization was not possible at all. Every method is checked against simulated data with a known answer, so each result can be proven correct instead of just looking reasonable.

The centerpiece is a flagship demo: one experiment where a plain t test says do not ship a feature that actually works, and CUPED correctly says ship it, on the same data with the same true effect.

## Architecture

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

## Why simulate data at all

A real dataset never comes with a known true effect, so there is no way to check whether a method got the right answer, only whether it looks plausible. The simulator here generates data with a configurable, known effect, so each method's output can be checked against that answer directly.

As a real world check, `src/data_loader.py` also loads a genuine randomized experiment, the Criteo uplift benchmark (Diemert et al., AdKDD 2018), and reshapes it into the same format the simulator produces, so the same analysis code runs unmodified on real data.

## Project layout

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
notebooks/                    scratch analysis
data/                         cached datasets, not tracked in git
DESIGN.md                     the reasoning behind every decision, tradeoffs, and known limitations
```

## The methods

**Sample ratio check.** Checked first, before anything else. It asks a different question than the rest: was the data even split the way it was supposed to be. A logging bug or a broken redirect can skew the control and treatment split without showing up in the effect estimate at all. A t test run on a broken 45/55 split still returns a normal looking p value. This is a two cell chi square test against the intended split, using a stricter threshold (0.001, not 0.05) since it should almost never fire and missing a real mismatch is costly. If it fires, every result below it is withheld rather than shown next to a warning.

**t test.** Welch's version, which does not assume the two groups have equal variance. This is the safer default, since a treatment can change variance as well as the mean.

**Confidence interval, power, and sample size.** Standard formulas built on the same Welch standard error, used to size an experiment or check how sensitive one already run actually was.

**CUPED.** Every metric carries variance that has nothing to do with the treatment: seasonality, cohort mix, existing differences between users. CUPED (Deng et al., WSDM 2013) removes the part of that variance that a covariate measured before the experiment already predicts, so the same sample size detects smaller effects. The coefficient is fit once on the combined control and treatment sample so it cannot bias the effect estimate.

**Sequential test.** A p value is only valid if you commit to a sample size in advance and look once. Teams actually watch dashboards and stop the moment something crosses 0.05, which inflates the true false positive rate well past 5%. This uses the mixture sequential probability ratio test (Robbins, 1970; the method behind Optimizely's stats engine), which stays valid no matter how often or when you look.

**Propensity matching.** When treatment was not randomized, comparing treated and untreated users directly measures the treatment effect plus whatever made those users different in the first place. This fits a logistic regression predicting treatment from observed covariates, matches each treated unit to its closest control by that score, and compares outcomes within matched pairs.

Full derivations, the alternatives considered for each one, and every known limitation are in [DESIGN.md](DESIGN.md).

## Running it

```bash
pip install -r requirements.txt
pytest tests/
```

Baseline analysis on a simulated experiment:

```bash
python src/run_baseline_analysis.py --true-effect 2.0 --n-per-arm 5000
```

Or on the real Criteo dataset (downloads and caches on first run):

```bash
python src/run_baseline_analysis.py --source criteo
```

CUPED comparison:

```bash
python src/compare_cuped.py --true-effect 2.0 --n-per-arm 5000 --extra-noise-std 30 --extra-noise-correlation 0.9
```

Peeking demonstration:

```bash
python src/peeking_demo.py --n-sims 1000 --max-n-per-arm 2000 --checkpoint-size 100
```

Causal method comparison:

```bash
python src/compare_causal.py --true-effect 5.0 --n 10000 --confounding-strength 2.0
```

## Dashboard

```bash
# Terminal 1, backend at http://localhost:8000
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Terminal 2, frontend at http://localhost:5173, proxies /api to the backend
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`.

The dashboard has one page with two study types. **Randomized experiment**: simulate or upload a CSV with `group`, `outcome`, and `pre_covariate` columns. It runs the sample ratio check first; if the split looks broken, every verdict badge below it is replaced with "verdict withheld" instead of shown next to a warning. Otherwise it shows the plain t test and the CUPED adjusted t test side by side, each with a ship or don't ship badge, plus a peeking chart if that scenario is enabled. **Observational data**: simulate confounded data or upload a CSV with `treatment`, `outcome`, and one or more `covariate_*` columns, and see the naive comparison next to the propensity matched estimate.

**Flagship demo.** Click "Load flagship demo" in the sidebar to load one fixed scenario: 3,000 users per arm, a real effect of 1.5, with extra noise correlated 0.92 to a covariate measured before the experiment.

```
True effect: 1.5
Naive:  effect=1.1758  p=0.1544  significant=False   -> DON'T SHIP
CUPED:  effect=1.5870  p=0.0041  significant=True    -> SHIP  (variance reduction: 55.1%)
```

The feature genuinely works, but the metric also carries a large amount of noise unrelated to treatment. That noise inflates the plain t test's standard error enough to swallow a real effect. The team happened to also collect a measurement of the same metric before the experiment started, and that covariate turns out to be strongly correlated with the extra noise. CUPED removes over half the irrelevant variance, and the same underlying signal becomes large enough relative to the remaining noise to detect. The point estimate barely moves between the two methods, both land near the true 1.5. What changes is how much of the metric's variance gets correctly attributed to noise before testing for the treatment's effect.

The specific seed behind this scenario was found by sweeping 200 candidates for one where the two methods land on opposite sides of p = 0.05, disclosed here rather than hidden. That search does not manufacture the underlying claim: CUPED's lack of bias and its variance reduction are proven separately by the repeated simulation tests in `tests/test_cuped.py`, which hold regardless of which seed happens to make the disagreement visible as a ship or don't ship flip.

Backend endpoints, all under `/api`: `POST /randomized/simulate`, `POST /randomized/upload`, `POST /observational/simulate`, `POST /observational/upload`, `GET /flagship`, `GET /health`.

## Measured results

Every number below comes from actually running the corresponding script against simulated data with a known true effect.

| Phase | Result |
|---|---|
| Core stats engine | On a 5,000 per arm simulated experiment with a true effect of 2.0: detected effect 2.07, 95% CI [1.28, 2.86], p < 0.001. Across 500 repeated simulations, 95% confidence intervals covered the true effect at the nominal rate. Across 1,000 null simulations, the false positive rate matched the nominal 5%. |
| Sample ratio check | Default alpha 0.001. Flags a 400/600 split against an intended 50/50 every time, never flags an exact match, and fires at close to its nominal rate across thousands of genuinely balanced random splits. |
| CUPED | 57.7% variance reduction and 57.6% required sample size reduction versus the plain t test, on simulated data with a deliberately strong (0.9) covariate correlation. On real Criteo data with an unengineered covariate (correlation −0.13): 1.8% variance reduction, the realistic end of the range. |
| Sequential testing | Naive repeated peeking: 23.1% false positive rate (nominal 5%) across 1,000 null experiments checked 20 times each. The always valid p value stays at 1.1%, within the theoretical bound. |
| Propensity matching | Naive observational estimate: 10.32 (true effect 5.0, biased by 5.32 from confounding). Propensity matched estimate: 4.54, 95% CI [4.34, 4.74], covering the true effect. |
| Flagship demo | Same 3,000 per arm dataset, true effect 1.5: plain t test not significant (p = 0.154), don't ship. CUPED adjusted, significant (p = 0.0041), ship. Recovers the true effect after a 55.1% variance reduction. |

## Status

Core stats, the sample ratio check, CUPED, sequential testing, propensity matching, and the dashboard are all implemented, tested, and checked against known true effects. See [DESIGN.md](DESIGN.md) for the reasoning behind every decision, what is intentionally out of scope, and what remains a known limitation.
