# Experimentation Causal Inference Platform: Design Document

A rigorous experiment analysis platform that runs the statistics a careful data scientist actually performs before shipping a change: sample ratio mismatch detection, variance reduced effect estimation (CUPED), peeking safe sequential testing, and confounding corrected causal inference. Every method is verified against a known true effect rather than trusted at face value.

---

## 1. Executive Overview

This system exists to answer one question honestly: when a statistical method used to evaluate an experiment claims something, is that claim actually true, or does it just look plausible. It implements four corrections to the naive "split into two groups and run a t test" approach, a sample ratio mismatch check that catches a broken randomization before any effect estimate is trusted, CUPED variance reduction that recovers statistical power a naive test throws away, an always valid sequential test that keeps early peeking honest, and propensity score matching for the case where randomization was never possible at all, and it proves each one works by running it against simulated data with a known, configured answer rather than asserting correctness from the formula alone. The centerpiece is a flagship demonstration where a plain t test says a real feature is not worth shipping and CUPED, run on the exact same data, correctly says it is, which is reproduced in full in §16.6. Everything downstream of that claim, the architecture (§5), every component (§7 through §9), how to actually run it (§13), what running it produced (§16), and what remains incomplete (§21), is documented in enough depth that this document alone should be sufficient to explain, defend, and probe the system without access to the code, the README, or the person who built it.

## 2. Problem Definition and Context

The naive way to analyze an experiment is to split users into two groups, compute the mean of each, run a t test, and check if p < 0.05. That is not wrong, exactly. It is a correct answer to a much narrower question than the one people think they are asking. Four specific ways it goes wrong, all of which this system exists to catch:

1. **Noise that is not about the treatment still counts against you.** A metric's variance is a mix of "caused by the treatment" and "everything else": seasonality, cohort mix, pre existing user differences. A t test's standard error does not distinguish between them. A real effect can be statistically invisible purely because the "everything else" bucket is large, even though some of that bucket was removable in principle, since a unit's own behavior before the experiment often predicts a good chunk of it.
2. **Checking more than once changes the odds.** A p value's validity is conditional on committing to a sample size in advance and looking exactly once. Real teams watch dashboards and stop the moment something crosses the line. Each additional look is another independent roll of the dice for noise to cross 0.05, so the true probability of a false positive somewhere across a monitored experiment climbs well past the nominal 5%, silently, because nothing about the individual p values looks wrong in isolation.
3. **You cannot always randomize.** Sometimes treatment assignment is a business decision, a self selection, or a policy rollout, not a coin flip. Comparing treated versus untreated users directly then measures the treatment effect plus whatever made those users different, and more likely to be treated, in the first place. The two are inseparable without an explicit adjustment for what is known about why they differ.
4. **The randomization itself can silently be broken, and no p value will tell you.** A logging bug that drops treatment arm events more often than control, a redirect that leaks users out of one arm, a caching layer that serves the wrong variant to some fraction of requests: none of these announce themselves in the effect estimate. A t test run on a broken 45/55 split instead of the intended 50/50 still returns a p value that looks like a normal p value. The only way to catch this class of bug is to check the allocation itself against what was intended, which is a completely different check from anything a significance test does (§9.2).

Each failure mode needs a different correction. Variance reduction, sequential testing, causal adjustment, and an allocation check are not interchangeable, and a system that only implements some of them leaves the others as blind spots. The harder problem underneath all four is knowing your correction is actually correct, as opposed to merely plausible. A textbook formula transcribed into code can still have a sign flipped, a wrong degree of freedom, or a subtle bias nobody notices because everything downstream still "looks reasonable." The answer adopted here is to test every method against data where the right answer is known in advance (§15). That single decision shapes almost everything else in this document.

### 2.1 Assumptions this system rests on

Every method here, the t test, CUPED, the sequential test, propensity matching, shares one assumption that is easy to state and easy to forget: **SUTVA** (the Stable Unit Treatment Value Assumption), meaning one unit's outcome does not depend on which arm any other unit was assigned to. This holds by construction in the simulator, since each unit's outcome is generated independently, but it is not a given on real data. A referral program, a marketplace with shared inventory, and a social feed with visible interactions all violate it, because a treated user's behavior can then spill over into a control user's outcome, or the reverse, which biases the effect estimate in a direction and magnitude this system has no way to detect (§21.1). This system does not check for or warn about interference; it assumes the input data already comes from a setting where SUTVA is reasonable, the same way it assumes a CSV's `outcome` column is actually numeric. The causal inference branch has one additional assumption layered on top, **ignorability** (treatment depends only on observed covariates, §9.5), which is likewise unchecked and simply assumed of whatever data is provided.

## 3. Goals, Success Criteria, and Scope

**Goal.** Build an experiment analysis system where every statistical claim it makes can be independently checked, not just trusted, and package it behind both a scriptable command line interface and a dashboard so the four corrections can be compared side by side against a naive baseline on the same data.

**Success criteria.** A method counts as done only once it clears a specific, falsifiable bar, not once it runs without error:

- The core t test's 95% confidence interval covers a known true effect at close to 95% across repeated simulations, and its false positive rate under the null matches the nominal alpha (§16.1).
- The sample ratio check flags a genuinely mismatched split every time and a genuine match never, and its own false positive rate matches its nominal alpha (§16.2).
- CUPED's point estimate stays unbiased relative to the naive estimate while measurably reducing variance when the covariate is informative (§16.3).
- The sequential test's false positive rate under continuous, repeated peeking stays at or below its nominal alpha, in a regime where the naive test's rate is confirmed to be well above it (§16.4).
- Propensity matching's estimate lands closer to a known true effect than the naive comparison on the same confounded data, with a confidence interval that covers the truth (§16.5).
- The flagship scenario reproducibly shows the naive and CUPED verdicts disagreeing on the same data, with the disagreement traced to a specific, explainable mechanism rather than noise (§16.6).

**In scope.** A synthetic data simulator with a configurable, known treatment effect for both randomized and observational designs; the four statistical methods above; a plumbing level check against one real, public randomized dataset (Criteo, §8.3, §15.2); a FastAPI backend that is a thin pass through to the same analysis code the tests use; a React and TypeScript dashboard covering both study types and the flagship demo; and a test suite that checks calibration and correctness against known answers, not just that the code runs.

**Explicitly out of scope**, decided rather than merely unfinished, with the reasoning for each recorded where the decision was made: difference in differences and uplift modeling as causal methods (§11.3), group sequential alpha spending as an alternative to mSPRT (§11.2), a machine learned CUPED adjustment (CUPAC, §11.1), composing CUPED with the sequential test into one pipeline (§9.4), multi metric correction for simultaneously monitored metrics, and any production deployment concern, authentication, persistence, containerization, monitoring (§19). Using a real dataset as the *primary* evidence for correctness was also considered and rejected up front, because no public dataset ships with a known true effect to check against (§11.4); this is the constraint that makes simulation the backbone of the entire system (§15.1).

## 4. Requirements and Constraints

**Functional requirements**, derived directly from the success criteria in §3:

- Implement a two sample significance test with a confidence interval, and power and sample size calculators, from first principles rather than a library call, so the mechanism is fully inspectable (§9.1).
- Implement a check on the randomization mechanism itself, independent of the effect estimate, that runs before any other result is trusted (§9.2).
- Implement a variance reduction method that provably does not bias the effect estimate (§9.3).
- Implement a significance test that remains valid under repeated, uncommitted peeking (§9.4).
- Implement a causal effect estimator for data where treatment was not randomized, that corrects a demonstrably biased naive comparison (§9.5).
- Expose every method through both a scriptable CLI and an HTTP API backing a browser dashboard, with the API containing no statistical logic of its own (§7.6).
- Verify every claim above against simulated data with a known, configured answer, not just unit test that functions execute (§14, §15).

**Constraints:**

- **No infrastructure budget.** The system runs entirely on a local machine, so no database, no cloud service, and no paid dataset could be assumed or required. This forced the "no persistence" and "no authentication" boundaries in §21.3 and pushed data acquisition toward simulation plus one free, public dataset (§8.3).
- **Single developer, single machine.** There was no second reviewer to catch a sign error in a formula by inspection, which is the direct motivation for verifying every method against a known answer instead of relying on code review as the correctness check (§15).
- **Windows as the primary development environment.** This shaped some concrete implementation choices, most notably discovered and worked through during construction (§12) rather than planned for up front.
- **Language and framework choice.** Python for the analysis engine, since numpy, pandas, and scipy are the standard tools for this kind of numerical work and scikit learn's logistic regression and nearest neighbor implementations are well tested building blocks for the causal method (§9.5); FastAPI and React with TypeScript for the interface, chosen over the project's original Streamlit implementation once the interface became a first class concern rather than a wrapper (§11.5, §12.2).
- **No real dataset with both known confounding and a known true effect exists publicly**, which is why propensity matching's correctness evidence is entirely simulation based, and why it is not exercised against the one real dataset the project does use (§8.3).

## 5. System Architecture

```
Experiment data (simulated with a known true effect, or a CSV upload)
        │
        ▼
Welch's t test                                    effect, CI, p value            (§9.1)
        │
        ▼
CUPED adjustment (covariate measured before the experiment)
                                                    lower variance effect, CI      (§9.3)
        │
        ▼
Sequential test (mSPRT, always valid p value)      peeking safe verdict           (§9.4)
        │
        ▼
Observational data → propensity score matching     confounding corrected effect   (§9.5)
        │
        ▼
FastAPI backend (thin HTTP layer, no duplicated logic) → React and TypeScript dashboard (§7.6, §7.7)
```

The pipeline is drawn as a straight line, but it is not really one. CUPED and the sequential test are two independent corrections applied to the same randomized data problem, not sequential stages of it (§9.4 explains why they are not composed together yet). The causal inference branch is a separate problem, data that was never randomized, with its own correction. The dashboard is a thin presentation layer over all three; it contains no statistics of its own.

**The guiding principle is that every claim has to be checkable.** Every method in this system is checked against synthetic data with a known, configured answer: a known effect size, a known false positive rate, a known amount of confounding, so "does this code work" has an actual yes or no answer instead of a judgment call. A secondary, real world dataset (the Criteo uplift benchmark, §8.3) is used only as a plumbing and face validity check, specifically because it lacks a known true effect and therefore cannot validate correctness, only that the code does not fall over on real world data shapes (§15.2).

The project is organized so that every layer calls straight down into the one below it, with nothing duplicated:

```
src/
  simulator.py              synthetic experiment generator with a known true effect     (§8)
  stats_core.py              t test, confidence intervals, power and sample size          (§9.1, §9.2)
  cuped.py                    CUPED variance reduction                                     (§9.3)
  compare_cuped.py            naive vs CUPED comparison script                              (§13, §16)
  sequential.py                always valid p values for safe early peeking                  (§9.4)
  peeking_demo.py              naive vs sequential false positive rate script                 (§13, §16)
  causal.py                     propensity score matching                                      (§9.5)
  compare_causal.py             naive vs matched estimate script                                 (§13, §16)
  data_loader.py               loads the real Criteo experiment dataset                          (§8.3)
  run_baseline_analysis.py   CLI entry point: simulate or load, then print the analysis           (§13)
tests/                        one test file per method, all checked against known true effects   (§14)
backend/
  main.py                       FastAPI app, a thin layer over src/                                (§7.6)
frontend/
  src/App.tsx                  React and TypeScript dashboard                                       (§7.7)
  src/components/              verdict badges, results panels, peeking chart                         (§7.7)
notebooks/                    scratch analysis, not part of the shipped system
data/                         cached datasets, not tracked in git
```

## 6. End-to-End System Flow

**A command line analysis run, start to finish:** the user invokes a script such as `run_baseline_analysis.py` with parameters describing the scenario to simulate (§13.2). The script calls `simulator.simulate_experiment`, which returns a unit level table and the true effect used to generate it (§8.1). The script runs `stats_core.srm_check` first and prints its result before anything else; if it fires, the script still proceeds but flags the result as untrustworthy (§9.2). It then runs `stats_core.welch_t_test` and `stats_core.confidence_interval` on the raw outcome, and prints the effect, confidence interval, t statistic, p value, and power, directly comparable against the true effect that was configured. `compare_cuped.py`, `peeking_demo.py`, and `compare_causal.py` follow the same shape, simulate or load data, run the naive method, run the corrected method, print both side by side (§13.2, §16).

**A dashboard session, start to finish:** the user opens the frontend at `localhost:5173` and either adjusts simulation parameters or uploads a CSV (§7.7). After a short debounce, the frontend issues a request to one of the backend's simulate or upload endpoints (§10). The backend calls `simulator.simulate_experiment` or parses the uploaded CSV into the same schema, runs `srm_check` first, and if it fires, every subsequent field in the response is still computed but the frontend renders every verdict badge below it as "verdict withheld" rather than showing a normal looking result next to a warning (§9.2, §7.7). Otherwise the backend runs the naive t test and, using `theta` and the covariate mean computed once on the pooled sample, the CUPED adjusted t test, and returns both along with the measured variance reduction (§9.3, §7.6). If the peeking scenario is enabled, the backend additionally re runs both the naive and sequential test at each checkpoint as the simulated data accumulates and returns the full series for the peeking chart (§9.4). The frontend renders naive and corrected results side by side with ship or don't ship badges, and the peeking chart if applicable (§7.7).

**An observational data session:** the same pattern, but the backend calls `simulator.simulate_observational_data` or parses an uploaded CSV with `treatment`, `outcome`, and `covariate_*` columns, computes the naive treated versus control difference (no confidence interval, §9.5) and the propensity matched estimate with its confidence interval, and returns both for the frontend to render side by side.

**The flagship demo:** a single click calls `GET /api/flagship`, which runs the same randomized data path above against one fixed, pre chosen set of parameters instead of whatever the user has configured (§13.5), and the frontend renders the result the same way it would render any other randomized scenario, so the disagreement between naive and CUPED verdicts is visible using the exact same code path as every other run, not a special cased demo (§16.6).

## 7. Component-Level Design

### 7.1 Simulator (`src/simulator.py`)

Generates unit level data for both study designs with a known, configured true effect baked in, so every downstream method can be checked against an answer key. Exposes `simulate_experiment` for the randomized design and `simulate_observational_data` for the confounded, non randomized design. Full generating formulas and the reasoning behind their specific construction are in §8.

### 7.2 Core statistics (`src/stats_core.py`)

The shared foundation every other method builds on: Welch's t test, its confidence interval, power and sample size calculators, and the sample ratio mismatch check. Every other component either calls into this module directly (CUPED, the sequential test's per look statistic) or is checked against the same correctness standard this module was checked against (§9.1, §9.2, §14).

### 7.3 CUPED (`src/cuped.py`)

Two small, pure functions, `compute_theta` and `cuped_adjust`, that transform a raw outcome array into a variance reduced one using a covariate measured before the experiment. Deliberately minimal surface area: it does not know about arms, groups, or treatment, callers (`compare_cuped.py`, `backend/main.py`) are responsible for pooling correctly across arms before calling it (§9.3).

### 7.4 Sequential testing (`src/sequential.py`)

Computes one always valid p value per look, given the current effect estimate, its variance, and the mixing prior variance `tau^2`. Stateless by design, each look is evaluated independently from the current cumulative data, which is what makes it safe to call at an arbitrary, uncommitted cadence (§9.4).

### 7.5 Causal inference (`src/causal.py`)

Fits a propensity model, matches treated units to control units by nearest propensity score, and estimates the effect from the matched pairs. Also exposes the naive, unadjusted comparison as a separate function specifically so the biased baseline and the corrected estimate can be computed and displayed side by side without either script or the backend duplicating that logic (§9.5).

### 7.6 Backend (`backend/main.py`)

A FastAPI app containing no statistical logic of its own. Every endpoint calls directly into the same `simulator`, `stats_core`, `cuped`, `sequential`, and `causal` functions used by the CLI scripts and the test suite. This was a hard constraint, not just a preference. If the API recomputed or reimplemented any of the analysis, the dashboard's numbers could silently drift from the numbers the tests actually verify, and a discrepancy between what the tests proved correct and what the interface shows would defeat the entire premise this system is built on (§5). Full endpoint and schema reference in §10.

### 7.7 Frontend (`frontend/src/App.tsx` and `src/components/`)

A React and TypeScript dashboard built with Vite. Two design choices are specific to this component rather than the system as a whole. First, the frontend re runs the analysis automatically, a short debounce interval after any parameter change, rather than requiring an explicit submit action, which gives a tighter feedback loop for exploring how a parameter affects the verdict (drag a slider, watch the badge flip); the accepted cost is a new backend request roughly every 250 milliseconds of inactivity while adjusting controls, a non issue for a local single user tool but not deployment safe as is (§18). Second, the peeking check visualization is a hand written SVG component rather than a charting library, since for exactly one chart type used in exactly one place, a library's bundle size and API surface cost more than it saves, and hand rolling it gave direct control over the specific interaction and accessibility details (legend, hover tooltip, a dashed alpha threshold line, colorblind safe series colors).

## 8. Data Design

The simulator is the primary data source for this entire system; the reasoning for why simulation rather than a real dataset is the backbone of correctness is a validation methodology question addressed fully in §15.1, not repeated here.

### 8.1 Randomized experiment data (`simulate_experiment`)

Each unit gets a `pre_covariate` (a stand in for a user's baseline behavior before the experiment started) and an `outcome`. Both are built from a shared latent factor:

```
pre_covariate = baseline_mean + r * sigma * latent + sqrt(1 - r^2) * sigma * noise_1
outcome       = baseline_mean + r * sigma * latent + sqrt(1 - r^2) * sigma * noise_2 + treatment_effect
```

This construction, rather than drawing the covariate as a noisy copy of the outcome directly, was chosen specifically so the correlation coefficient `r` between covariate and outcome is a single, direct knob, `covariate_correlation`, with no second order effects on the outcome's own mean or variance. That matters because CUPED's whole story is that variance reduction scales with correlation, and that claim needed to be testable by turning one dial, not by re deriving what a change in the covariate's generating process does to the outcome's marginal distribution.

An extra noise mechanism (`extra_noise_std`, `extra_noise_correlation`), off by default, was added on top specifically to build the CUPED demonstration. It injects a second noise term into the outcome that is unrelated to treatment but correlated with the covariate, exactly the scenario where CUPED has something to remove. Without this knob, demonstrating CUPED's value would require hand picking a real dataset that happened to have this property, which is unfalsifiable; you cannot prove that is why CUPED helped, only that it did on that one dataset. Every parameter here is exposed as a CLI flag and a dashboard control (§13.2, §13.3).

The resulting table has columns `unit_id`, `group` (`control` or `treatment`), `pre_covariate`, and `outcome`, which is also the exact schema expected for a CSV upload to the randomized study type (§10.3).

### 8.2 Observational data (`simulate_observational_data`)

For the causal inference branch, treatment assignment cannot be a coin flip; that would defeat the point. Instead, two covariates drive both the propensity to be treated (through a logistic model) and the outcome directly:

```
propensity = sigmoid(confounding_strength * covariate_1 + 0.5 * covariate_2)
treatment  ~ Bernoulli(propensity)
outcome    = baseline + 4 * covariate_1 + 2 * covariate_2 + true_effect * treatment + noise
```

This is confounding by construction: a unit's `covariate_1` is simultaneously why it is more likely to be treated, and why its outcome would have been different regardless. `confounding_strength` is a single knob controlling how badly a naive comparison gets it wrong, which made it possible to demonstrate, not just assert, that bias grows with confounding strength and that matching removes it regardless (§16.5).

The resulting table has columns `unit_id`, `covariate_1`, `covariate_2`, `treatment`, `outcome`, and `true_propensity`. A CSV upload for the observational study type needs `treatment`, `outcome`, and one or more `covariate_*` columns (§10.3); any number of covariate columns is accepted, not just two.

### 8.3 Real world data (`data_loader.py`)

A secondary loader pulls the Criteo uplift benchmark, a genuine randomized ad exposure experiment (Diemert et al., AdKDD 2018), from its Hugging Face mirror on first use and caches it locally, and reshapes it into the same schema as §8.1 so the exact same analysis code runs on it unmodified. What this check does and does not prove, and why it only exercises some of the four methods, is a validation methodology question covered in full in §15.2.

### 8.4 Statelessness

Every simulate, upload, or analyze call, whether through the CLI or the API, is stateless. There is no database and nothing is persisted between calls; closing the browser tab or ending the script loses everything computed. This was a direct consequence of the "no infrastructure budget" constraint in §4 and is recorded as a known limitation, not an oversight, in §21.3.

## 9. Algorithms, Models, and Technical Methods

### 9.1 Welch's t test, confidence interval, power, and sample size (`stats_core.py`)

Student's original t test assumes both arms have equal variance. That assumption is often false in an experiment; sometimes the entire point of the treatment is that it changes variance, not just the mean (a feature that helps some users a lot and others not at all, for example). **Welch's t test** does not require that assumption, at the cost of a slightly more involved degree of freedom calculation:

```
t  = (mean_treatment - mean_control) / SE
SE = sqrt(var_treatment / n_treatment + var_control / n_control)
df = SE^4 / ( (var_t/n_t)^2 / (n_t - 1) + (var_c/n_c)^2 / (n_c - 1) )   [Welch Satterthwaite]
p  = 2 * P(T_df > |t|)
```

Welch's was chosen as the default rather than checking variance equality first and switching tests, because that check is itself a hypothesis test with its own false positive rate, and running one test to decide which second test to run is a well known way to quietly distort the overall error rate. Welch's is a safe default in both the equal and unequal variance case, so there was no real reason not to use it everywhere; the alternative and its rejection are recorded formally in §11.1.

The confidence interval for the difference in means is built from the same standard error and degrees of freedom: `(mean_t - mean_c) +/- t_crit(df, alpha/2) * SE`.

`scipy.stats.t` is used only to evaluate the Student t and normal CDFs; the test statistic, standard error, and degrees of freedom are hand written. This was deliberate for two reasons. First, a system whose entire purpose is to catch other people's statistical mistakes should not itself be a black box nobody in the project can explain line by line. Second, hand writing the formula and then independently verifying it against known true effects (§14.1) is a stronger correctness claim than trusting a well known library, because it exercises understanding of why the formula is shaped the way it is, which is what lets the rest of this system's corrections be built correctly on top of it.

Power, minimum detectable effect, and required sample size use the standard normal approximation rather than an iterative solve against the exact noncentral t distribution:

```
n_per_arm = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / mde^2                 # sample size
mde       = (z_alpha/2 + z_beta) * sigma * sqrt(2 / n_per_arm)            # minimum detectable effect
power     = P(Z > z_alpha/2 - true_effect / SE),  SE = sigma * sqrt(2 / n_per_arm)
```

This is the textbook standard approach (Cohen, 1988) and is accurate once each arm has a reasonable number of observations, roughly 30 or more as a rule of thumb, which covers every scenario this system is actually used for. It is a known, accepted inaccuracy at very small sample sizes; see §21.1.

### 9.2 Sample ratio mismatch check (`srm_check`)

Every method above assumes the data it is given actually reflects the intended random allocation. `srm_check` is the thing that checks that assumption instead of silently trusting it: a two cell chi square goodness of fit test comparing the observed control and treatment split against the intended ratio (default 50/50):

```
chi2 = sum_i (observed_i - expected_i)^2 / expected_i,  i in {control, treatment}
p    = P(ChiSq_1 > chi2)
```

It is deliberately the first thing printed in `run_baseline_analysis.py`'s output and the first field in the API's `RandomizedAnalysisResponse` (§10.2), checked and surfaced before any effect estimate, because a mismatched split makes every downstream number untrustworthy regardless of how significant or well calibrated it looks in isolation (§2, failure mode 4).

Three design choices worth stating explicitly:

- **A much stricter alpha than the effect test (0.001, not 0.05).** Under a correctly running experiment this check should almost never fire, and the cost of missing a real mismatch, trusting results from a broken randomization, is high enough to justify tolerating a lower false positive rate here than the standard significance threshold used for the actual effect estimate. This follows Fabijan et al., "Diagnosing Sample Ratio Mismatch in Online Controlled Experiments," KDD 2019, standard practice in production experimentation platforms.
- **A fixed alpha is itself a known simplification, not a solved problem.** Chi square power grows with sample size, so a fixed threshold gets more sensitive to practically meaningless deviations as an experiment gets larger, the exact failure mode Fabijan et al. warn about; their own recommendation is to scale the threshold with sample size rather than use one constant everywhere. That scaling is not implemented here (§21.1).
- **It is a pure counts based check** (`n_control`, `n_treatment`), not something that inspects the outcome data at all. Sample ratio mismatch is a statement about the mechanism that assigned people to arms, and conflating it with anything about the outcome would risk the check itself becoming a second, redundant hypothesis test about the effect rather than the categorically different question it is supposed to answer.

Detecting a mismatch visibly disables the verdict, not just the trust: every badge below a mismatch warning is replaced with an explicit "verdict withheld" state rather than a warning banner shown next to an otherwise confident looking result (§7.7). Verified in `tests/test_srm.py` (§14.2); results in §16.2.

### 9.3 CUPED (`cuped.py`)

```
theta      = Cov(Y, X) / Var(X)
Y_adjusted = Y - theta * (X - mean(X))
```

`theta` is the ordinary least squares slope of the outcome on the covariate measured before the experiment, the value that minimizes the variance of `Y_adjusted`. Subtracting `theta * (X - mean(X))` removes the part of `Y`'s variance that is linearly predictable from `X`, while leaving the mean of `Y_adjusted` unchanged in each arm, because `X - mean(X)` has mean zero by construction. This is Deng et al., "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre Experiment Data," WSDM 2013.

**Why theta and mean(X) are computed on the pooled sample, not per arm.** This was a specific decision, not an oversight, and getting it wrong would have silently reintroduced the exact bias CUPED exists to avoid. If `theta`, or the covariate mean used to center it, were estimated separately within each arm, the adjustment would subtract a different constant from each arm's outcomes, and that difference would land directly in the estimated treatment effect, biasing it by however much the two arms' covariate distributions happen to differ by chance in a given sample. Pooling `theta` and `mean(X)` across both arms and applying the identical values to each guarantees the adjustment's effect on the difference in means is exactly zero in expectation. `tests/test_cuped.py` checks this directly (§14.3): on data with a known effect and injected unrelated noise, both the naive and CUPED adjusted point estimates must land close to the true effect and close to each other. The rejected alternative, per arm theta, is recorded in §11.1.

**Why pooling does not leak treatment information into the adjustment.** A natural objection is whether averaging in the treated arm's data to compute `theta` lets the treatment contaminate the correction applied to control. It does not, because `X`, the covariate measured before the experiment, is measured before treatment assignment happens, and assignment is random. `X`'s distribution is therefore independent of treatment by construction, and `theta` is purely a statement about how `Y` covaries with `X` in general, not about the treatment effect. This is why the covariate must genuinely be measured before treatment; see §21.1 for what breaks if it is not. The deferred nonlinear alternative (CUPAC) is recorded in §11.1.

### 9.4 Sequential testing (`sequential.py`)

A p value's guarantee, that under the null it is significant only 5% of the time, is a statement about a single, pre committed test. Checking it repeatedly and stopping at the first significant result is a different procedure with a much higher true false positive rate, because each look is an additional opportunity for noise to cross the threshold. This system implements the **mixture sequential probability ratio test (mSPRT)**, also called always valid p values (Robbins, 1970; Johari, Koomen, Pekelis and Walsh, "Peeking at A/B Tests," KDD 2017, the method behind Optimizely's stats engine), chosen over Bonferroni correction and group sequential alpha spending for the reasons recorded in §11.2.

Instead of testing a single fixed alternative, mSPRT places a Gaussian mixing prior `N(0, tau^2)` over the possible treatment effect. At each look t, with current effect estimate `Delta_t` and its variance `V_t`:

```
Lambda_t = sqrt(V_t / (V_t + tau^2)) * exp( tau^2 * Delta_t^2 / (2 * V_t * (V_t + tau^2)) )
p_t      = min(1, 1 / Lambda_t)
```

`Lambda_t` is a nonnegative martingale under the null by construction, so by Ville's inequality, `P(exists t: Lambda_t >= 1/alpha) <= alpha`. The probability of ever falsely flagging significance across the whole sequence of looks is bounded by alpha, not just at any single look.

`tau^2` represents the scale of effect the test is tuned to detect well. It does not need to be exactly right for the alpha guarantee to hold, but it affects power. The implemented test's measured false positive rate under repeated peeking came in well below the nominal 5%, around 1% (§16.4), which is expected rather than a bug: Ville's inequality is an upper bound, not an exact calibration target, and with a small, finite number of discrete looks (20 in the demonstration) the bound is loose. Chasing exact 5% calibration would mean moving to a different mathematical construction, such as a tighter boundary designed for a specific, known number of looks, which reintroduces the pre commitment rigidity mSPRT was chosen to avoid; the investigation that confirmed this is structural rather than a tuning mistake is documented in §12.3.

The sequential test operates on one metric's raw outcome stream and is not currently composed with CUPED; there is no "CUPED adjusted always valid p value" path, even though combining the two would in principle give both benefits, peeking safety and variance reduction, simultaneously. They are demonstrated independently: CUPED runs against a final, fixed size dataset, the sequential test runs against the raw outcome as it accumulates. Recorded as a scope boundary in §3 and a limitation in §21.1.

### 9.5 Propensity score matching (`causal.py`)

When treatment assignment depends on a covariate that also affects the outcome, `E[Y | treated] - E[Y | untreated]` is not the treatment effect; it is the treatment effect plus the average outcome difference that would have existed between these two groups even with no treatment at all. The two are inseparable without explicitly conditioning on what is known about why the groups differ. **Propensity score matching** (Rosenbaum and Rubin, 1983) was chosen over difference in differences and uplift modeling for the reasons recorded in §11.3: fit a propensity model, match treated units to similar propensity control units, compare outcomes within matched pairs. Its correctness claim, that matching balances the covariate distribution between treated and matched control so comparing within a pair approximates the missing counterfactual, is directly testable against the simulator's known confounding strength (§8.2, §16.5).

**Matching mechanism.** Each treated unit is matched to the closest propensity control unit, with replacement: the same control unit can be reused across multiple treated units. This was chosen over matching without replacement because without replacement, treated units can run out of nearby, unused controls when the treated and control propensity distributions do not overlap perfectly, silently degrading match quality for whichever units happen to be processed last. With replacement, every treated unit gets its best available match regardless of processing order. The cost, stated plainly: the confidence interval currently reported is computed from a simple paired difference t interval over the matched pairs, which treats each matched pair as independent, an assumption not exactly true when a control unit has been reused. The correct variance estimator for matching with replacement (Abadie and Imbens, 2006) is not implemented; see §21.1.

**Propensity model.** A plain logistic regression on the observed covariates. This is well specified here, because the simulator's own treatment assignment mechanism is itself a logistic function of the covariates (§8.2), so the model can in principle recover the true propensity function exactly. On real world, non simulated data the true assignment mechanism is unknown and may not be linear in the logit; this risk is recorded in §21.1.

**Caliper.** Treated units whose best available match is farther than `caliper` away in propensity score distance are dropped from the estimate entirely, rather than forced into a poor match, which is the right call for estimate quality but means the reported effect is technically the average effect among the matchable treated population, not literally every treated unit in the dataset (§21.1).

**Honest interface choice.** The naive treated versus control comparison is shown for comparison purposes but deliberately without a confidence interval or a significance verdict, since a naive comparison on confounded data does not have a valid confidence interval in the first place, its standard error formula assumes no confounding to correct for, and showing one anyway would imply a rigor the naive method does not have.

## 10. APIs, Interfaces, and Data Contracts

### 10.1 Endpoints

All under `/api`, implemented in `backend/main.py` with no statistical logic of its own (§7.6):

| Endpoint | Purpose |
|---|---|
| `POST /randomized/simulate` | Simulate a randomized experiment from request parameters and return the full analysis |
| `POST /randomized/upload` | Same analysis, from an uploaded CSV instead of a simulation |
| `POST /observational/simulate` | Simulate confounded observational data and return naive plus matched estimates |
| `POST /observational/upload` | Same analysis, from an uploaded CSV instead of a simulation |
| `GET /flagship` | Load the fixed flagship demo scenario (§13.5, §16.6) |
| `GET /health` | Liveness check |

### 10.2 Request and response contracts

`SimulateRandomizedRequest`: `n_per_arm` (default 5000), `true_effect` (2.0), `baseline_mean` (100.0), `baseline_std` (20.0), `extra_noise_std` (0.0), `extra_noise_correlation` (0.0), `covariate_correlation` (0.7), `seed` (42), `include_peeking` (false), `checkpoint_size` (100).

`SimulateObservationalRequest`: `n` (10000), `true_effect` (5.0), `confounding_strength` (2.0), `caliper` (0.05), `seed` (42).

`RandomizedAnalysisResponse`: `true_effect`, `srm` (an `SRMCheckOut`: counts, expected ratio, p value, `srm_detected`), `naive` and `cuped` (each a `TTestResultOut`: effect, CI bounds, p value, `significant`), `variance_reduction_pct`, and an optional `peeking` block (a list of per checkpoint naive and sequential p values, plus the sample size at which each method first flagged significance, if it did).

`CausalAnalysisResponse`: `true_effect`, `naive_effect`, `matched_effect`, `matched_ci_lower`, `matched_ci_upper`, `n_matched`, `n_treated`.

### 10.3 CSV upload contracts

Randomized upload requires columns `group` (values `control` or `treatment`), `outcome`, and `pre_covariate`, matching the simulator's own output schema exactly (§8.1). Observational upload requires `treatment`, `outcome`, and at least one column named `covariate_*` (§8.2). Column presence is checked and returns a 422 error listing what is missing; column types and value ranges are not validated further (§17.2).

### 10.4 Command line interface

Every CLI script exposes its simulation parameters as flags, listed in full with example invocations in §13.2; there is no separate configuration file or environment variable surface, since the system requires none (§4).

## 11. Design Decisions, Alternatives, and Tradeoffs

### 11.1 Alternatives considered within the statistics engine and CUPED

- **Student's pooled variance t test**, checked conditionally by testing variance equality first. Rejected because that check is itself a hypothesis test with its own false positive rate, and test then test procedures are a well known way to quietly distort the overall error rate; Welch's is a safe default in both cases (§9.1).
- **CUPED theta estimated separately per arm.** Rejected for the bias reason worked through analytically in §9.3, before any code was written, rather than discovered by debugging a biased result after the fact.
- **Nonlinear or machine learned CUPED adjustment (CUPAC)**, replacing the linear coefficient with a learned prediction of the outcome from richer covariates. Deliberately not implemented: the simulator's own covariate to outcome relationship is linear by construction, so a linear adjustment is already optimal on this system's own data, and a nonlinear model would add real implementation and validation complexity (its own train and test split discipline, to avoid overfitting the adjustment itself) for no measurable benefit against this system's data generating process. It would matter on real world data with genuinely nonlinear relationships; that is real scope left on the table, not a hidden flaw (§21.2).

### 11.2 Alternatives considered for sequential testing

- **Bonferroni correction on the naive test**, dividing alpha by the number of planned looks. Rejected: it requires committing to an exact number of looks in advance, exactly the rigidity that makes fixed horizon testing impractical in the first place, and it is needlessly conservative because it does not use the correlation between consecutive looks at the same accumulating dataset.
- **Group sequential design with alpha spending** (O'Brien Fleming boundaries, for example). This is what most production experimentation platforms actually use, and is more statistically efficient than the mixture approach when the look schedule is known in advance, but it requires pre specifying that schedule, or at minimum a maximum number of looks, and solving for boundary values that are more involved to implement and verify correctly. Deferred as future scope (§21.2).
- **mSPRT (chosen).** Does not require pre committing to a look schedule at all, and stays valid even under continuous peeking, checked after every single new data point, which matches how teams actually behave more closely than a group sequential design does (§9.4).

### 11.3 Alternatives considered for the causal inference component

- **Difference in differences.** Rejected for this system specifically because it requires panel data, pre and post period outcomes for the same units, which the rest of this system's data model does not have (the randomized experiment side is single period by design, §8.1). Adding it would have meant maintaining two incompatible data shapes throughout the simulator, the API, and the frontend for one causal method, judged not worth the complexity given propensity matching covers the same underlying idea, adjust for what you can observe before comparing outcomes, on data already in the system's existing shape. Deferred as future scope (§21.2).
- **Uplift modeling** (directly modeling how the treatment effect varies per unit, rather than a single average effect). Rejected because it answers a different question, for whom does this work rather than does this work once corrected for confounding, and evaluating it properly needs its own validation methodology (Qini curves, for example) distinct from the confidence interval based verification used everywhere else in this system. Deferred as future scope (§21.2), not rejected on merit.
- **A flexible propensity model** (gradient boosting, for example) instead of logistic regression. Would be more robust to an unknown, nonlinear real world assignment mechanism, at the cost of being harder to inspect and more prone to overfitting the propensity score itself, which paradoxically can make matching worse, a well known issue in the causal inference literature. Logistic regression was chosen to match the simulator's own generating process rather than to be maximally robust to an unknown one (§21.1).

### 11.4 Alternative considered and rejected: real data as the primary evidence source

The project parameters this system was scoped against explicitly allowed either a public dataset or a simulator as the primary source (§3). Public datasets were rejected as the *primary* source for one specific reason: none of them ship with a known treatment effect, so a method that recovers a plausible looking number on real data has not been shown to be correct, only "not obviously broken." Simulation was the only option that let every method be checked against an answer key; this decision is the foundation of the validation methodology in §15.

### 11.5 Interface: FastAPI and React chosen over Streamlit

A conventional client and server split with a typed HTTP boundary is legible to a wider range of reviewers than a Streamlit script, which reads as one Python file with inline widgets to anyone unfamiliar with that specific framework, and separating the analysis API from its presentation forces the API contract in §7.6 to be explicit and enforced by the TypeScript types on the frontend, rather than implicit in how a Streamlit script happens to call Python functions inline. The story of how and when this replacement happened, and how it was verified to change nothing about what was being shown, is in §12.2.

### 11.6 Key tradeoffs at a glance

| Decision | Rejected alternative(s) | Why | Cost accepted |
|---|---|---|---|
| Welch's t test everywhere | Student's pooled variance t test, applied conditionally | Avoids a pre test for variance equality, which has its own error rate cost | Marginally more complex degree of freedom calculation |
| t test, CI, and power formulas hand written | `scipy.stats.ttest_ind` directly | Forces understanding of the exact mechanism the rest of the system builds on, independently verified against known true effects | More code to maintain and keep correct |
| Normal approximation for power, MDE, and sample size | Exact noncentral t iterative solve | Standard textbook approach, simple, accurate at realistic sample sizes | Inaccurate at very small n (§21.1) |
| CUPED theta pooled across arms | Theta estimated separately per arm | Per arm estimation reintroduces bias into the effect estimate | None significant, pooling is strictly better here |
| Linear CUPED adjustment | Machine learned residualization (CUPAC) | Already optimal given the simulator's linear data generating process | Would not capture nonlinear relationships on real data |
| mSPRT always valid p values | Bonferroni correction; group sequential alpha spending | No pre committed look schedule required, valid under continuous peeking | Empirically conservative, measured about 1% versus nominal 5%, not maximally powerful |
| Propensity score matching | Difference in differences; uplift modeling | Fits existing cross sectional data shape, directly testable against known confounding | Questions specific to those methods left unanswered (§21.2) |
| Nearest neighbor matching with replacement | Matching without replacement | Avoids match starvation when propensity distributions do not overlap well | Reported CI is a simplified approximation, not the fully rigorous with replacement estimator (§21.1) |
| Logistic regression propensity model | A flexible classifier such as gradient boosting | Well specified given the simulator's own logistic assignment mechanism | Risk of misspecification on real world, non simulated data |
| FastAPI and React interface | Streamlit, kept as the first version then replaced | Typed API boundary, conventional client and server split, broader legibility | A full rewrite of the interface layer, two frontend histories to reconcile |
| Debounced auto refresh in the interface | An explicit "run analysis" button | Tighter feedback loop for exploring parameter effects | Frequent backend requests while adjusting controls, not deployment safe as is |
| Hand built SVG chart | A charting library such as Recharts or Plotly | No extra dependency for one chart type, full control over interaction details | More component code to maintain directly |
| Simulation as the primary evidence source | A public dataset as the primary evidence source | No public dataset ships with a known true effect to check against | Real world data quirks are only caught by the secondary Criteo check (§8.3, §15.2) |

## 12. Implementation and Project Evolution

### 12.1 What broke during construction, and how it was diagnosed

**The Criteo loader silently pulled a treatment only slice.** Loading the first `n_rows` of the Criteo CSV as a quick sample produced a dataset where `treatment.value_counts()` showed 100% treatment, 0% control. The file is grouped by treatment status, not shuffled, so any prefix read is a slice of one arm only. This was caught by inspecting the value counts directly rather than trusting the loader, right after `run_baseline_analysis --source criteo` crashed downstream. Fixed by reading only the needed columns for the entire file (fast enough at roughly 19 seconds given the reduced column set) and taking a true random sample afterward, rather than relying on row order.

**A pre existing outer git repository.** Running `git status` from the project directory returned results scoped to a git repository rooted at the user's home directory, not the project folder, meaning a prior, unrelated `git init` had at some point been run one level up, tracking the entire home directory. Rather than committing into or otherwise disturbing that repository, a separate, freshly initialized repository was created scoped specifically to the project folder, and the outer repository was left untouched throughout.

**A division by zero in the power calculation.** `run_baseline_analysis.py --source criteo` crashed inside `statistical_power` with a division by zero. Root cause: `n_control` was 0, which traced directly back to the treatment only slice bug above. With zero control arm rows, every downstream per arm statistic broke. Fixed together with the sampling fix above; the crash was actually a useful early signal that something upstream was wrong, well before anyone inspected the value counts directly.

**Investigating the sequential test's conservativeness.** The first end to end run of the peeking demonstration showed the mSPRT test's empirical false positive rate at roughly 1%, well under the nominal 5% target. Before accepting that as correct, `tau^2` was swept across a range of values (1, 2, 4, 8, 20, 50, 100, 200) to rule out a single poorly chosen constant as the cause. The rate stayed in the same low single digit range across the whole sweep, which pointed at something structural, Ville's inequality being a loose upper bound with only 20 discrete looks, rather than a tuning mistake. This is recorded as an accepted, understood property (§9.4) rather than something "fixed" by further tuning, because chasing exact 5% calibration would have meant abandoning the specific guarantee, validity under arbitrary, uncommitted peeking, that motivated choosing mSPRT in the first place.

**Finding the flagship demo scenario systematically, not by hand.** The flagship demo needed a specific combination of sample size, true effect, extra noise, and random seed that actually produces a naive versus CUPED disagreement; most parameter combinations do not. Rather than hand adjusting numbers until one run happened to look right, which would have been indistinguishable from cherry picking after the fact, a small script swept 200 candidate random seeds against a fixed set of parameters and searched for one where the naive p value cleared 0.05 while the CUPED p value did not. The resulting numbers (naive p = 0.154, CUPED p = 0.0041) were then independently re verified by calling the exact same analysis functions the dashboard and backend use, not just trusted from the search script's own output. This invites an obvious objection, addressed directly in §15.3: is searching 200 seeds for a favorable outcome just cherry picking, dressed up with the word "systematic"?

**Verifying the interface without a human clicking through it.** Neither the Streamlit version nor the React version could be visually confirmed by a human clicking around during development. Both were verified instead by starting the actual server processes, driving them with headless Chrome screenshots at specific application states, and checking that the numbers rendered in the screenshot matched the numbers independently computed by calling the same underlying analysis functions directly in a script. This substitutes for manual QA but is not automated; see §21.4.

### 12.2 The interface's two generations

The dashboard was first built as a single process Streamlit app, then rebuilt as a FastAPI backend with a separate React and TypeScript frontend. Both decisions were made for real reasons, not by default. Streamlit came first because it let every other method in the system get a working, visual demonstration with minimal additional code, and because a single process Python app with no separate build step meant the interface layer could iterate as fast as the analysis code underneath it while that code was still the primary focus. It was replaced once the interface itself became a first class concern rather than a wrapper, for the reasons recorded as a design decision in §11.5. The rewrite kept the same feature set (simulate or upload, naive and corrected panels side by side, ship or don't ship verdicts, one click flagship demo) and was verified against the same numbers the Streamlit version and the CLI scripts produced, specifically to confirm the interface swap changed nothing about what was actually being shown.

## 13. Operational Guide

### 13.1 Setup and tests

```bash
pip install -r requirements.txt
pytest tests/
```

`requirements.txt` pins numpy, pandas, scipy, scikit learn, huggingface_hub, pytest, fastapi, uvicorn, and python multipart. A CI workflow (`.github/workflows/tests.yml`) runs the same test command on every push and pull request to `main` (§14.4).

### 13.2 Command line scripts

Baseline analysis on a simulated experiment:

```bash
python src/run_baseline_analysis.py --true-effect 2.0 --n-per-arm 5000
```

The same pipeline against the real Criteo dataset instead (downloads and caches from Hugging Face on first run, subsequent runs use the local cache):

```bash
python src/run_baseline_analysis.py --source criteo
```

CUPED comparison:

```bash
python src/compare_cuped.py --true-effect 2.0 --n-per-arm 5000 --extra-noise-std 30 --extra-noise-correlation 0.9
python src/compare_cuped.py --source criteo
```

Peeking demonstration:

```bash
python src/peeking_demo.py --n-sims 1000 --max-n-per-arm 2000 --checkpoint-size 100
```

Causal method comparison:

```bash
python src/compare_causal.py --true-effect 5.0 --n 10000 --confounding-strength 2.0
```

### 13.3 Dashboard

```bash
# Terminal 1, backend at http://localhost:8000
uvicorn backend.main:app --reload --port 8000

# Terminal 2, frontend at http://localhost:5173, proxies /api to the backend
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173`. No environment variables or external services are required.

### 13.4 CSV upload formats

Randomized study type: `group` (`control` or `treatment`), `outcome`, `pre_covariate` (§8.1, §10.3). Observational study type: `treatment`, `outcome`, one or more `covariate_*` columns (§8.2, §10.3).

### 13.5 The flagship demo, step by step

Clicking "Load flagship demo" in the dashboard sidebar calls `GET /api/flagship`, which runs `simulate_experiment` with one fixed set of parameters (3,000 users per arm, true effect 1.5, baseline standard deviation 20, extra noise standard deviation 25 correlated 0.92 with the covariate, seed 22), then runs the same randomized data analysis path as any other simulated scenario (§6). Reproducible from the command line with the same parameters via `compare_cuped.py`. What it shows and why is in §16.6; how the specific seed was chosen is in §12.1 and §15.3.

## 14. Testing and Validation

### 14.1 How correctness is actually established

None of this system's methods are trusted on the strength of "the formula looks right transcribed." Every core stats claim is checked against simulated data with a known answer:

- A well powered simulation with a real, known effect must have the t test reject the null and recover an effect estimate within a small tolerance of the true value.
- A simulation with no true effect, repeated 1,000 times, must reject the null at close to the nominal 5% rate: not meaningfully more, which would mean the p value calculation is wrong, and not meaningfully less, which would mean the test has no power at all.
- A 95% confidence interval, repeated across 500 simulations, must contain the true effect close to 95% of the time, checking the calibration of the interval, not just its formula.
- A deliberately underpowered simulation (tiny sample, small effect) must fail to detect the effect most of the time, confirming the power calculation is internally consistent with what the t test actually does, not just self consistent in isolation.

This is the pattern every other method in the system follows: implement the formula, then independently verify it against a scenario where the right answer is known, rather than treating "it compiled and returned a number" as evidence of correctness.

### 14.2 Test suite structure

One file per method, 15 tests total, in `tests/`:

- `test_stats_core.py`: the four checks in §14.1, applied to the core t test, CI, and power calculators.
- `test_srm.py`: a clearly mismatched split (400/600 against an intended 50/50) is flagged, an exact match is never flagged, and across thousands of genuinely balanced random splits, the check fires at close to its own nominal alpha.
- `test_cuped.py`: on data with a known effect and injected unrelated noise, both the naive and CUPED adjusted point estimates land close to the true effect and close to each other (no bias); variance drops substantially when the covariate captures the injected noise.
- `test_sequential.py`: across hundreds of simulated null experiments checked repeatedly, the naive t test's false positive rate is confirmed well above nominal, while the always valid p value stays at or below it.
- `test_causal.py`: on simulated confounded data, the raw treated versus control difference is confirmed meaningfully biased, and the matched estimate is confirmed closer to the known true effect and covered by its confidence interval.

### 14.3 What is not covered by automated tests

The frontend has no automated test suite; correctness was verified manually via headless browser screenshots at specific points in development (§12.1), not via tests that run on every change (§21.4). The real data (Criteo) commands are run and checked by hand, not wired into the test suite or CI, since a roughly 300 megabyte network download with third party availability outside this project's control is a poor fit for a test that needs to run quickly and deterministically on every change (§21.3).

### 14.4 Continuous integration

`.github/workflows/tests.yml` installs `requirements.txt` and runs `pytest tests/` on every push and pull request to `main`. This is the only automated gate in the project; there is no automated frontend build check or lint step in CI (§21.3).

## 15. Experimental Methodology

### 15.1 Why simulation is the primary validation method

A method run on real data can produce a number that looks plausible and still be wrong; there is no way to tell from the outside. A method run on data with a known, configured effect either recovers that number within sampling noise, or it does not. That is the entire justification for building a simulator before anything else: it turns "I believe this is correct" into "I can show you it is correct," and it is the only way to test the two false positive shaped failure modes (peeking, confounding) at all, since those require running the same generating process thousands of times to measure an empirical rate, which no static real world dataset can provide (§11.4).

### 15.2 The Criteo real data check, and what it does and does not validate

Running the baseline pipeline against Criteo data (§8.3) is a check that the plumbing, CSV parsing, column reshaping, arm splitting, holds up against real world data shapes: different scale, different missingness behavior, different column types than anything a simulator would generate, the kind of thing code that only ever sees its own synthetic output can silently get wrong. It is explicitly not a correctness check on the statistics themselves, because there is no known true effect in the Criteo data to compare against.

Exactly which methods this exercises, stated precisely:

- **The baseline t test and the sample ratio check** run against Criteo directly.
- **CUPED** also runs against Criteo, using one of the dataset's anonymized pre treatment features (`f0` by default) as the covariate. This is a genuinely useful, humbling data point: the real covariate to outcome correlation on Criteo is around −0.13, checked across several available features ranging roughly ±0.03 to ±0.28, nowhere near the 0.9+ correlation the simulated flagship demo deliberately uses. Real variance reduction on this dataset comes out to roughly 2%, not the simulated demo's 55 to 58% (§16.3). That gap is the point: the flagship numbers demonstrate the mechanism at a deliberately favorable correlation, not a claim about what CUPED delivers on an arbitrary real metric (§21.1).
- **The sequential test does not run against Criteo**, because the dataset has no timestamp or arrival order column; peeking only means something against data that actually accumulates over time.
- **Propensity matching does not run against Criteo**, a category fit issue rather than a missing feature: Criteo's treatment is itself randomized (assembled from real incrementality tests), so there is no confounding in it for matching to correct. No public dataset with both known confounding and a known true effect was available, which is exactly why the simulator is the primary evidence for this method (§11.4).

### 15.3 The flagship demo's experimental design, and the cherry picking objection addressed directly

The seed behind the flagship scenario was found by sweeping 200 candidates against a fixed set of parameters for one where the naive and CUPED verdicts land on opposite sides of p = 0.05 (§12.1). Is searching 200 seeds for a favorable outcome just cherry picking, dressed up with the word "systematic"? It would be, if the seed search were the evidence that CUPED works. It is not. That evidence is the repeated simulation calibration tests in `tests/test_cuped.py` (§14.2), which check the general, seed independent claims, that the point estimate stays close to the true effect and that variance drops substantially when the covariate captures injected noise, across many runs, none of them cherry picked. What the seed search found is a single instance where that already proven general property happens to land on opposite sides of the p < 0.05 line for the two methods, useful for building intuition in one vivid, legible example, not a substitute for the aggregate proof. Run the flagship parameters at a different seed and the two methods will usually agree, both significant or both not, and that is expected and fine, because the general claim was never that CUPED always flips the verdict. It is that CUPED reduces variance and does not bias the estimate, which holds regardless of which seed makes that fact visible as a ship or don't ship flip.

### 15.4 The sequential test's tau sweep as an experimental design

To rule out a single poorly chosen `tau^2` as the explanation for the sequential test's observed conservativeness (§9.4, §12.1), the peeking demonstration was re run with `tau^2` swept across eight values spanning two orders of magnitude (1, 2, 4, 8, 20, 50, 100, 200). The empirical false positive rate stayed in the same low single digit range across the entire sweep, which is what let the conclusion move from "maybe this constant is wrong" to "this is a structural property of a loose upper bound with a finite number of looks" (§16.4).

## 16. Results and Observed Behavior

Every number below comes from actually running the corresponding script or test against simulated data with a known true effect, using the exact commands in §13. Nothing here is a projection or an expected value; it is what the code produced when it was run.

### 16.1 Core statistics engine

On a 5,000 per arm simulated experiment with a true effect of 2.0 (`run_baseline_analysis.py --true-effect 2.0 --n-per-arm 5000`): detected effect 2.07, 95% CI [1.28, 2.86], p < 0.001, matching the configured true effect within sampling noise. Across 500 repeated simulations, 95% confidence intervals covered the true effect at close to the nominal 95% rate. Across 1,000 null simulations (no true effect), the t test rejected the null at close to the nominal 5% false positive rate. A deliberately underpowered configuration (small n, small effect) failed to detect the effect most of the time, confirming the power calculation is internally consistent with what the t test actually does.

### 16.2 Sample ratio mismatch check

Default alpha 0.001. A clearly mismatched split, 400/600 against an intended 50/50, is flagged every time. An exact 50/50 match is never flagged. Across thousands of genuinely balanced random splits, the check fires at close to its own nominal alpha, not more, which would mean a broken check, and not less, which would mean a check with no power to ever fire.

### 16.3 CUPED

On simulated data with a deliberately strong (0.9) covariate correlation and injected extra noise (`compare_cuped.py --true-effect 2.0 --n-per-arm 5000 --extra-noise-std 30 --extra-noise-correlation 0.9`):

| | Naive | CUPED adjusted |
|---|---|---|
| Outcome variance | 1306.66 | 553.21 |
| Required sample size per arm (80% power) | 5,128 | 2,172 |

Variance reduction 57.7%, required sample size reduction 57.6%. This is a demonstration number at a deliberately engineered 0.9 correlation, not a realistic expectation (§21.1). On the real Criteo dataset, using feature `f0` as the covariate (real correlation with outcome around −0.13, `compare_cuped.py --source criteo`):

| | Naive | CUPED adjusted |
|---|---|---|
| Outcome variance | 0.0449 | 0.0441 |
| Variance reduction | N/A | 1.8% |

That is the honest range: CUPED's real world payoff depends entirely on how predictive whatever pre experiment covariate is actually available, and has to be checked per metric.

### 16.4 Sequential testing

`peeking_demo.py --n-sims 1000 --max-n-per-arm 2000 --checkpoint-size 100`, 1,000 simulated null experiments, checked at 20 points as data accumulates:

| | Naive (repeated peeking) | Sequential (mSPRT) |
|---|---|---|
| False positive rate | 23.1% | 1.1% |
| Nominal alpha | 5% | 5% |

Naive peeking inflates the false positive rate to roughly 4.6x the nominal 5%. The always valid p value stays at 1.1%, comfortably within the theoretical bound. The sequential rate running below alpha rather than exactly at it is expected, not a bug (§9.4, §15.4).

### 16.5 Propensity matching

`compare_causal.py --true-effect 5.0 --n 10000 --confounding-strength 2.0`, 10,000 simulated units where a covariate drives both treatment assignment and outcome:

| | Naive (unadjusted) | Propensity matching |
|---|---|---|
| Estimated effect | 10.32 | 4.54 |
| Bias versus true effect (5.0) | +5.32 | −0.46 |
| 95% CI covers true effect? | N/A (no valid CI, §9.5) | Yes: [4.34, 4.74] |

The naive comparison overstates the effect by more than 2x purely from confounding. Propensity matching corrects for that and lands within 0.46 of the true effect, with a confidence interval that covers it.

### 16.6 The flagship demo

The fixed scenario described operationally in §13.5 (3,000 users per arm, true effect 1.5, extra noise correlated 0.92 with the covariate, seed 22):

```
True effect: 1.5
Naive:  effect=1.1758  p=0.1544  significant=False   -> DON'T SHIP
CUPED:  effect=1.5870  p=0.0041  significant=True    -> SHIP  (variance reduction: 55.1%)
```

**The setup.** A team runs an A/B test on a new feature, 3,000 users per arm. The standard t test on the primary metric comes back not statistically significant, the feature looks like a wash, and a reasonable team kills it.

**What actually happened.** The simulated ground truth behind this scenario has a real, positive treatment effect of 1.5, the feature genuinely works. The naive t test missed it not because the effect is not real, but because the outcome metric also carries a large amount of variance that has nothing to do with the treatment (an unrelated noise source with standard deviation 25, versus a baseline outcome standard deviation of 20). That extra variance inflates the standard error enough to swallow a real, meaningful effect. The team also happened to be collecting a measurement of the same metric before the experiment for each user, and that covariate turns out to be strongly correlated (0.92) with the extra noise, because both trace back to the same source, a user's baseline engagement level, for example.

**Why they disagree.** Both tests are looking at the same underlying treatment effect. The naive test's standard error includes noise the covariate measured before the experiment could have explained away; CUPED's does not. The point estimate barely moves (1.18 to 1.59, both near the true 1.5), CUPED does not invent an effect that was not there. What changes is the noise around the estimate: with over half the irrelevant variance removed (55.1%), the same underlying signal is now large relative to the uncertainty, and the test correctly detects it.

**The takeaway.** A naive significance test is not wrong on its own terms, it is a correct answer to a weaker question: is the effect detectable given everything counted as noise, including noise that could have been removed. Ignoring available data collected before the experiment means leaving statistical power on the table, and in the case of a real effect, that can be the difference between shipping a feature that works and shelving it because the analysis was not sensitive enough to see it.

**Is this cherry picked?** See §15.3 for the full account of how the seed was found and why that search does not undermine the general, seed independent claim this demo illustrates.

## 17. Security, Reliability, and Failure Handling

This is a local, single user analysis tool, not a production service, so the security and reliability posture reflects that scope rather than an oversight; the scope decision itself is recorded in §3 and §4.

### 17.1 Security

There is no authentication, authorization, or rate limiting on the FastAPI backend. CORS is restricted to the local Vite dev origin, but the API itself has no concept of a user or a request budget. It is not something safe to expose on a shared or public network as is; this is the explicit boundary drawn in §21.3.

### 17.2 Input validation and failure handling

CSV upload endpoints check column presence and return a 422 error listing what is missing, but do not validate column types, value ranges, or row counts; a malformed upload, a non numeric `outcome` column, for example, will fail with a raw exception rather than a clear error message (§10.3, §21.3).

### 17.3 The sample ratio check as a domain specific reliability mechanism

Distinct from general software error handling, the sample ratio mismatch check (§9.2) is this system's one purpose built defense against a real, specific failure mode: a broken randomization mechanism that would otherwise silently corrupt every downstream result while looking completely normal. It is a reliability mechanism for the experiment itself, not for the software.

### 17.4 Bugs encountered as reliability lessons

Two of the concrete bugs found during construction were, in effect, missing input validation surfacing downstream as a crash rather than a clear error: the Criteo loader's treatment only slice and the resulting division by zero in the power calculation (§12.1). Both were caught by a crash rather than a silent wrong answer, which was fortunate rather than designed; there is no systematic defense in this codebase against a silent wrong answer from a similarly shaped data quality problem, which is the same category of risk as the CSV validation gap in §17.2.

## 18. Performance, Scalability, and Cost

No formal performance or scale requirements were set for this system, since it is a local, single user tool by design (§3, §4); the observations below describe what exists, not a benchmarking exercise.

**Performance.** The heaviest simulated scenarios used in this project's own scripts and tests are in the tens of thousands of rows (10,000 units for the causal comparison, thousands of repeated simulations of 5,000 rows each for calibration tests), and every core statistics and CUPED computation is a small number of vectorized numpy operations over an array of that size, fast enough that no script in this project takes more than a few seconds to run end to end aside from the one time Criteo download. Propensity matching's nearest neighbor step uses scikit learn's `NearestNeighbors`, which is more than adequate at these sizes; no attempt was made to test or optimize behavior at a scale beyond what this project's own demonstrations require.

**Scalability.** The debounced auto refresh in the frontend (§7.7) is the one place scale was an explicit design consideration: a new backend request roughly every 250 milliseconds of inactivity is a non issue for one local user, but would need rate limiting or a "commit" interaction step before this pattern would be appropriate behind a shared, multi user deployment (§19). Nothing about the backend or the analysis code was load tested against concurrent users, since none are expected in this system's current scope.

**Cost.** Zero infrastructure cost; everything runs on a local machine with no cloud service, database, or paid API in the loop. The one external cost is a roughly 300 megabyte one time download of the Criteo dataset from its Hugging Face mirror, cached locally after the first run (§8.3).

## 19. Deployment, Monitoring, and Maintenance

**Deployment.** This project runs locally and is not currently deployed anywhere. There is no Dockerfile, no hosted instance, and no environment based configuration for a non localhost backend URL. The Vite dev server proxies `/api` requests to the FastAPI backend during development; there is no production build and serve unification, the built frontend is not currently served by the backend or bundled into a single deployable artifact (§21.3).

**Monitoring.** None exists. There is no logging infrastructure, no error tracking, and no metrics collection beyond whatever a developer sees in the terminal running the backend or frontend dev server. For a stateless, local, single user tool this has not mattered in practice, but it is a real gap relative to anything meant to run unattended.

**Maintenance.** The one automated maintenance gate is the CI workflow (`.github/workflows/tests.yml`, §14.4), which runs the Python test suite on every push and pull request to `main`. There is no automated check on the frontend build, no dependency update automation, and no automated check on the real data (Criteo) path (§14.3). Standing up a real deployment, a single build step serving the frontend from the backend, containerized, with environment based configuration, authentication, and rate limiting, is listed as concrete future work in §21.5.

## 20. Interpretation and Lessons Learned

The single biggest lesson from building this system is that "the formula looks right" and "the formula is right" are different claims, and the gap between them is exactly where the sequential test's conservativeness investigation (§12.1, §15.4) and the Criteo loader's treatment only slice bug (§12.1) both lived: neither would have been caught by code review alone, and both were caught by comparing what the code produced against an independently known answer. That is the practice this entire project is organized around, and it generalizes past this specific codebase: any statistical method is a claim, and a claim that has not been checked against a scenario with a known answer is not yet a fact, no matter how standard the underlying formula is.

A second lesson is that honesty about a method's limits is not in tension with demonstrating that it works, it is what makes the demonstration credible. The flagship demo (§16.6) is the clearest example: disclosing exactly how the seed was found (§15.3), rather than presenting the number without that context, is what makes the surrounding claim, that CUPED reduces variance without biasing the estimate, believable, because the aggregate proof (§14.2) and the illustrative example are kept explicitly separate instead of letting the second one stand in for the first. The same pattern shows up in the Criteo comparison (§15.2, §16.3): reporting the real world 1.8% variance reduction next to the engineered 57.7% demonstration number is more convincing than reporting either one alone, because it shows the mechanism is real while being honest about where its payoff actually depends on data that was not controlled.

A third lesson is narrower but concrete: an interface rewrite (§12.2) is only actually safe to trust if it is checked against the same ground truth numbers the old interface produced, not just visually compared. Verifying that the FastAPI and React rewrite changed nothing about what was being shown used the exact same "compare against an independently computed answer" discipline as the statistical methods themselves, which suggests that discipline is a general property of how this project approached correctness, not something specific to statistics.

## 21. Limitations, Known Issues, Technical Debt, and Future Work

### 21.1 Statistical approximations accepted as is

- **Power, MDE, and sample size formulas use a normal approximation**, not an exact noncentral t solve. Accurate at realistic sample sizes; measurably off at very small per arm n, roughly under 30. Not currently flagged in the interface when a configuration is small enough for this to matter.
- **The sample ratio check's alpha is a single fixed constant (0.001), not scaled to sample size.** Chi square power grows with n, so the same fixed threshold is conservative for a small experiment and increasingly trigger happy on practically meaningless deviations for a very large one, the specific problem Fabijan et al., the paper this check is based on, recommend solving by scaling the threshold with sample size. That scaling is not implemented (§9.2); a fixed, stricter than conventional constant was chosen as a reasonable default for this system's realistic sample sizes, not as a substitute for the more rigorous approach.
- **Propensity matching confidence intervals use a simplified paired t formula** (technical debt) that treats matched pairs as independent, which is not exactly true under matching with replacement, since a reused control correlates the pairs it appears in. The fully rigorous approach (Abadie and Imbens, 2006) is not implemented. The simplification is unverified against a calibration test analogous to the one built for the plain t test (§14.1); this is a real gap, not just a stylistic simplification.
- **The logistic regression propensity model is well specified against this system's own simulator, not against arbitrary real world assignment mechanisms.** A CSV upload with a genuinely nonlinear treatment assignment process would silently get a worse propensity fit with no warning.
- **The caliper can silently shrink the effective estimand** (§9.5), with no automated warning when the dropped unit fraction gets large.
- **CUPED and the sequential test are not composed.** There is no path to get a variance reduced and peeking safe estimate simultaneously; they are demonstrated as two independent corrections to two independent problems, not a combined pipeline. A real production stats engine would want both at once.
- **The sequential test handles one metric at a time.** Monitoring multiple metrics simultaneously, extremely common in practice, needs its own multiple testing correction on top of the sequential correction; that composition is not implemented or even modeled.
- **No interference or SUTVA detection.** Every method assumes one unit's outcome is unaffected by another unit's treatment assignment (§2.1). If that is false, a referral loop, shared marketplace inventory, a visible social feed, every effect estimate in this system is biased in a direction and magnitude nothing here would surface. There is no cluster randomization support, no exposure modeling, and no diagnostic that would even hint the assumption is being violated.
- **The flagship demo's variance reduction (55 to 58%) is not representative of typical real world CUPED gains.** It demonstrates the mechanism at a deliberately strong, hand chosen covariate correlation (0.9+). Running the same code against a real covariate on the Criteo dataset (§8.3, §16.3) gives a correlation around −0.13 and roughly 2% variance reduction, a much more honest expectation for an arbitrary real metric.

### 21.2 Methods deliberately not implemented

- **Difference in differences** and **uplift modeling**, both considered for the causal inference component and set aside in favor of propensity matching (§11.3), not because they are worse methods, but because they answer different questions and each would need its own data model and validation approach to do properly.
- **Group sequential or alpha spending sequential testing**, the main alternative to mSPRT, is not implemented; only one sequential testing approach exists in this system (§11.2).
- **Multiple testing correction across simultaneously monitored metrics** is out of scope entirely (§21.1).
- **Nonlinear or machine learned CUPED (CUPAC)**, considered and deferred (§11.1). A real world metric with a genuinely nonlinear relationship to its available covariates would see smaller gains from the linear CUPED implemented here than a CUPAC style approach would deliver.

### 21.3 Engineering gaps

- **No authentication, authorization, or rate limiting** on the FastAPI backend (§17.1). It is a local, single user analysis tool, not something safe to expose on a shared or public network as is.
- **No persistence.** Every simulate, upload, or analyze call is stateless; there is no saved history of past analyses, no session concept, and no database (§8.4).
- **No production deployment path** (§19). No Dockerfile, no environment based backend URL configuration for a non localhost deployment, and no CI pipeline running the frontend build automatically (a CI workflow does run the Python test suite on every push, §14.4).
- **CSV upload validation is minimal** (§17.2). Only column presence is checked, not column types, value ranges, or row counts.
- **The real data (Criteo) validation runs are manual, not automated** (§14.3). A roughly 300 megabyte network download with third party availability and licensing terms outside this project's control is a poor fit for a test that needs to run quickly and deterministically on every change.

### 21.4 Product shape gaps

- **No automated interface regression testing** (§14.3). The dashboard's correctness was verified manually via headless browser screenshots at specific points in development (§12.1), not via an automated test suite that runs on every change.
- **The flagship demo is a single fixed scenario.** There is no mechanism to define, save, or share additional "interesting disagreement" scenarios beyond the one hard coded set of parameters.
- **No sensitivity analysis or robustness checks are surfaced to the user**, for example there is no built in way to see how the propensity matching estimate changes as the caliper varies.

### 21.5 Concrete future work

1. Implement the Abadie and Imbens variance estimator for matching with replacement, and add a calibration test for it analogous to the one that already exists for the plain t test.
2. Compose CUPED and the sequential test into a single variance reduced, peeking safe pipeline.
3. Add a flexible, non logistic propensity model option, with a diagnostic such as a propensity score overlap plot surfaced in the dashboard.
4. Add difference in differences as a second causal method, which requires extending the simulator and API to support panel (pre and post) data alongside the existing cross sectional shape.
5. Add a production deployment path: a single build step that serves the built frontend from the FastAPI backend, containerized, with environment based configuration instead of hard coded localhost URLs.
6. Add authentication and per session rate limiting before considering any shared or public deployment.
7. Replace the manual headless screenshot verification process (§12.1) with an automated interface test suite that runs against the same flagship and peeking scenarios on every change.
8. Surface a caliper sensitivity view in the dashboard so a user can see how the matched estimate and matched pair count move together as the caliper changes, rather than only seeing the result at one fixed caliper value.
9. Scale the sample ratio check's alpha with sample size instead of using one fixed constant (§9.2, §21.1), per Fabijan et al.'s own recommendation, so the check stays well calibrated at both small and very large sample sizes rather than becoming oversensitive as n grows.

## 22. Conclusion

This system exists because a single significance test answers a narrower question than the one people actually care about, and every component here corrects for one specific way that gap shows up: noise the treatment did not cause (CUPED, §9.3), the odds changing when results get checked more than once (sequential testing, §9.4), assignment that was never random in the first place (propensity matching, §9.5), and a randomization mechanism that can be silently broken with no p value ever announcing it (the sample ratio check, §9.2). None of these corrections were trusted on the strength of a formula transcribed correctly. Each one was implemented from the underlying statistics, then checked against simulated data with a known, configured answer, and §16 is what that checking actually produced, not what the formulas were expected to produce. The tradeoffs in §11 and the limitations in §21 are the other half of that same honesty: every deliberate simplification, from a normal approximation at small sample sizes to an unimplemented Abadie and Imbens variance estimator, is recorded next to the specific reason it was accepted, not discovered later by someone else reading the code. What this adds up to is a system that is narrow by design, one dataset shape per problem, one method per failure mode, no production deployment story, but correct within that scope in a way that can be demonstrated by running §13's commands and reproducing §16's numbers, rather than a system whose correctness has to be taken on faith. That is the standard this document was written to be judged against: not whether the system does everything, but whether everything it does and does not do is written down clearly enough that reading this document once is enough to explain, defend, and probe the system as fluently as the person who built it.
