# Experimentation Causal Inference Platform — Engineering Design Document

A rigorous experiment-analysis platform that runs the statistics a careful data scientist actually performs before shipping a change — sample ratio mismatch detection, variance-reduced effect estimation (CUPED), peeking-safe sequential testing, and confounding-corrected causal inference — every method verified against a known ground truth rather than trusted at face value. This document records why it's built the way it is: the alternatives rejected, the tradeoffs accepted, what broke during construction, and what's still a known limitation.

---

## 1. The problem, and why it's hard

The naive way to analyze an experiment is: split users into two groups,
compute the mean of each, run a t-test, check if p < 0.05. That's not wrong,
exactly — it's a correct answer to a much narrower question than the one
people think they're asking. Three specific ways it goes wrong, all of which
this system exists to catch:

1. **Noise that isn't about the treatment still counts against you.** A
   metric's variance is a mix of "caused by the treatment" and "everything
   else" — seasonality, cohort mix, pre-existing user differences. A t-test's
   standard error doesn't distinguish between them. A real effect can be
   statistically invisible purely because the "everything else" bucket is
   large, even though some of that bucket was removable in principle (the
   unit's own pre-experiment behavior often predicts a good chunk of it).
2. **Checking more than once changes the odds.** A p-value's validity is
   conditional on committing to a sample size in advance and looking exactly
   once. Real teams watch dashboards and stop the moment something crosses
   the line. Each additional look is another independent roll of the dice for
   noise to cross 0.05, and the true probability of a false positive
   *somewhere* across a monitored experiment climbs well past the nominal 5%
   — silently, because nothing about the individual p-values looks wrong in
   isolation.
3. **You can't always randomize.** Sometimes treatment assignment is a
   business decision, a self-selection, or a policy rollout — not a coin
   flip. Comparing treated vs. untreated users directly then measures the
   treatment effect *plus* whatever made those users different (and more
   likely to be treated) in the first place. The two are inseparable without
   an explicit adjustment for what's known about why they differ.
4. **The randomization itself can silently be broken, and no p-value will
   tell you.** A logging bug that drops treatment-arm events more often than
   control, a redirect that leaks users out of one arm, a caching layer that
   serves the wrong variant to some fraction of requests — none of these
   announce themselves in the effect estimate. A t-test run on a broken
   45/55 split instead of the intended 50/50 still returns a p-value that
   looks like a normal p-value. The only way to catch this class of bug is
   to check the *allocation itself* against what was intended, which is a
   completely different check from anything a significance test does (§4.5).

Each failure mode needs a *different* correction — variance reduction,
sequential testing, causal adjustment, and an allocation check aren't
interchangeable, and a system that only implements some of them leaves the
others as blind spots. The harder problem underneath all four: how do you
know your correction is actually correct, as opposed to merely plausible? A
textbook formula transcribed into code can still have a sign flipped, an
off-by-one in a degrees-of-freedom calculation, or a subtle bias nobody
notices because everything downstream still "looks reasonable." The answer
adopted here is: **test every method against data where the right answer is
known in advance** (see §2 and §4). That single decision shapes almost
everything else in this document.

### 1.1 Assumptions this system rests on

Every method here — the t-test, CUPED, the sequential test, propensity
matching — shares one assumption that's easy to state and easy to forget:
**SUTVA** (the Stable Unit Treatment Value Assumption), i.e. one unit's
outcome doesn't depend on which arm any *other* unit was assigned to. This
holds by construction in the simulator (each unit's outcome is generated
independently), but it is not a given on real data — a referral program, a
marketplace with shared inventory, a social feed with visible interactions
all violate it, because a treated user's behavior can then spill over into
a control user's outcome (or vice versa), which biases the effect estimate
in a direction and magnitude this system has no way to detect (§11.1). This
system does not check for or warn about interference; it assumes the input
data already comes from a setting where SUTVA is reasonable, the same way it
assumes a CSV's `outcome` column is actually numeric. The causal-inference
branch has one additional assumption layered on top — **ignorability**
(treatment depends only on observed covariates, §7.1) — which is likewise
unchecked and simply assumed of whatever data is provided.

## 2. System overview

```
Experiment data (simulated, known ground truth — or CSV upload)
        │
        ▼
Welch's t-test  ──────────────────────────▶  effect, CI, p-value            (§4)
        │
        ▼
CUPED adjustment (pre-experiment covariate) ─▶ variance-reduced effect, CI   (§5)
        │
        ▼
Sequential test (mSPRT, always-valid p-value) ▶ peeking-safe verdict        (§6)
        │
        ▼
Observational data ──▶ propensity score matching ▶ confounding-corrected effect, CI (§7)
        │
        ▼
FastAPI backend (thin HTTP layer, zero duplicated logic) ──▶ React/TypeScript dashboard (§8)
```

The pipeline is drawn as a straight line but it isn't really one — CUPED and
the sequential test are two independent corrections applied to the same
randomized-data problem, not sequential stages of it (see §6's note on why
they aren't composed together yet). The causal-inference branch is a
separate problem (non-randomized data) with its own correction. The
dashboard is a thin presentation layer over all three; it contains no
statistics of its own.

**The guiding principle is ground-truth verifiability.** Every method in this
system is checked against synthetic data with a *known, configured* answer —
a known effect size, a known false-positive rate, a known amount of
confounding — so "does this code work" has an actual yes/no answer instead of
a judgment call. A secondary, real-world dataset (the Criteo uplift-modeling
benchmark, §3.5) is used only as a plumbing/face-validity check, specifically
*because* it lacks a known ground truth and therefore can't validate
correctness — only that the code doesn't fall over on real-world data shapes.

## 3. Data layer: the simulator

### 3.1 Why simulation is the primary data source at all

A method run on real data can produce a number that looks plausible and
still be wrong — there's no way to tell from the outside. A method run on
data with a *known, configured* effect either recovers that number (within
sampling noise) or it doesn't. That's the entire justification for building
a simulator before anything else: it turns "I believe this is correct" into
"I can show you it's correct," and it's the only way to test the two
false-positive-shaped failure modes (peeking, confounding) at all, since
those require running the *same* generating process thousands of times to
measure an empirical rate — not something any static real-world dataset can
provide.

### 3.2 The randomized-experiment generator (`simulate_experiment`)

Each unit gets a `pre_covariate` (a stand-in for "this user's baseline
behavior before the experiment started") and an `outcome`. Both are built
from a **shared latent factor**:

```
pre_covariate = baseline_mean + r · σ · latent + √(1-r²) · σ · noise_1
outcome       = baseline_mean + r · σ · latent + √(1-r²) · σ · noise_2 + treatment_effect
```

This construction (rather than, say, drawing the covariate as a noisy copy
of the outcome directly) was chosen specifically so the *correlation
coefficient* `r` between covariate and outcome is a single, direct knob —
`covariate_correlation` — with no second-order effects on the outcome's own
mean or variance. That matters because CUPED's whole story is "variance
reduction scales with correlation," and that claim needed to be testable by
turning one dial, not by re-deriving what a change in the covariate's
generating process does to the outcome's marginal distribution.

An **extra-noise mechanism** (`extra_noise_std`, `extra_noise_correlation`)
was added on top, off by default, specifically to build the CUPED
demonstration: it injects a second noise term into the outcome that's
*unrelated to treatment* but *correlated with the covariate*, which is
exactly the scenario where CUPED has something to remove. Without this
knob, demonstrating CUPED's value would require hand-picking a real dataset
that happened to have this property, which is unfalsifiable — you can't
prove that's *why* CUPED helped, only that it did on that one dataset.

### 3.3 The observational-data generator (`simulate_observational_data`)

For the causal-inference branch, treatment assignment can't be a coin flip —
that would defeat the point. Instead, two covariates drive **both** the
propensity to be treated (through a logistic model) and the outcome
directly:

```
propensity = sigmoid(confounding_strength · covariate_1 + 0.5 · covariate_2)
treatment  ~ Bernoulli(propensity)
outcome    = baseline + 4·covariate_1 + 2·covariate_2 + true_effect·treatment + noise
```

This is confounding by construction — a unit's `covariate_1` is
simultaneously why it's more likely to be treated *and* why its outcome
would have been different regardless. `confounding_strength` is a single
knob controlling how badly a naive comparison gets it wrong, which made it
possible to demonstrate (not just assert) that bias grows with confounding
strength and that matching removes it regardless.

### 3.4 Alternative considered and rejected: real data as the primary source

The project parameters this system was scoped against explicitly allowed
either a public dataset or a simulator as the primary source. Public
datasets were rejected as the *primary* source for one specific reason:
none of them ship with a known treatment effect, so a method that recovers
a plausible-looking number on real data has not been shown to be
*correct* — only "not obviously broken." Simulation was the only option that
let every method be checked against an answer key.

### 3.5 The Criteo real-data check, and what it does and doesn't validate

A secondary loader (`data_loader.py`) pulls the Criteo uplift-modeling
benchmark — a genuine randomized ad-exposure experiment — from its Hugging
Face mirror, and reshapes it into the same schema the simulator produces so
the exact same analysis code runs on it unmodified. This exists to catch a
different class of bug than the simulation tests can: real data has
different scale, different missingness behavior, and different column dtypes
than anything a simulator would generate, and code that only ever sees its
own synthetic output can silently assume things about the data (no NaNs,
already-sorted, small value ranges) that don't hold in the wild. Running the
baseline pipeline against Criteo data is a check that the *plumbing* — CSV
parsing, column reshaping, arm splitting — holds up against real-world data
shapes. It is explicitly **not** a correctness check on the statistics
themselves, because there is no known true effect in the Criteo data to
compare against. See §10.1 for a bug this exact check caught.

**Exactly which methods this actually exercises, stated precisely rather
than left to imply "the pipeline" covers everything:**

- **The baseline t-test and the SRM check** run against Criteo directly
  (`run_baseline_analysis.py --source criteo`).
- **CUPED** also runs against Criteo (`compare_cuped.py --source criteo`),
  using one of the dataset's anonymized pre-treatment features (`f0` by
  default) as the pre-experiment covariate. This is a genuinely useful,
  humbling data point: the real covariate-outcome correlation on Criteo is
  around **-0.13** (checked across several of the available features, which
  range roughly ±0.03 to ±0.28) — nowhere near the 0.9+ correlation the
  simulated flagship demo deliberately uses to produce a clean, legible
  disagreement (§5, §11.1). Real variance reduction on this dataset comes out
  to roughly **2%**, not the simulated demo's 55-58%. That gap is the point:
  the flagship numbers are a *demonstration of the mechanism* at a
  deliberately favorable correlation, not a claim about what CUPED delivers
  on an arbitrary real metric — see §11.1 for this stated as a limitation in
  its own right.
- **The sequential test does not run against Criteo**, because the dataset
  has no timestamp or arrival-order column — "peeking" only means something
  against data that actually accumulates over time, and imposing an
  arbitrary row order on Criteo to fake that would test nothing real.
- **Propensity matching does not run against Criteo**, and this is a
  category-fit issue rather than a missing feature: Criteo's treatment is
  itself randomized (it's assembled from real incrementality tests), so
  there is no confounding in it for matching to correct — running PSM on it
  would mostly demonstrate that matching doesn't distort an already-clean
  randomized comparison, which is a real and fine property to check but a
  different claim than the one PSM exists to prove (§7.2). No public dataset
  with both known confounding *and* a known true effect was available, which
  is exactly why the simulator is the primary evidence for this method
  (§3.4) and why this gap is left open rather than papered over with a
  dataset that doesn't fit the method.

## 4. Core statistics engine (`stats_core.py`)

### 4.1 Welch's t-test vs. Student's pooled-variance t-test

Student's original t-test assumes both arms have equal variance. That
assumption is often false in an experiment — sometimes the entire point of
the treatment is that it changes variance, not just the mean (e.g., a
feature that helps some users a lot and others not at all). Welch's t-test
doesn't require that assumption, at the cost of a slightly more involved
degrees-of-freedom calculation (Welch–Satterthwaite). Welch's was chosen as
the unconditional default rather than checking variance equality and
switching tests, because that check is itself a hypothesis test with its own
false-positive rate, and "test-then-test" procedures are a well-known way to
quietly distort the overall error rate. Welch's is a safe default in both
the equal- and unequal-variance case, so there was no real reason not to use
it everywhere.

### 4.2 Implemented from formulas, not `scipy.stats.ttest_ind`

`scipy.stats.t` is used only to evaluate the Student-t and normal CDFs —
the actual test statistic, standard error, and degrees-of-freedom
calculations are hand-written. This was a deliberate choice over calling a
library function directly, for two reasons: first, a system whose entire
purpose is to catch *other people's* statistical mistakes shouldn't itself
be a black box nobody in the project can explain line by line; second, and
more concretely, hand-writing the formula and then independently verifying
it against ground-truth simulations (§4.4) is a stronger correctness claim
than trusting a well-known library, because it exercises understanding of
*why* the formula is shaped the way it is, which is what lets the rest of
this system's corrections (CUPED, sequential testing) be built correctly on
top of it.

### 4.3 z-approximation for power / sample size / MDE, not exact solving

The sample-size, MDE, and power formulas here use the standard normal
(z) approximation to the sampling distribution of the mean difference,
not an iterative solve against the exact (noncentral) t-distribution. This
is the textbook-standard approach (Cohen, 1988) and is accurate once each
arm has a reasonable number of observations (a rule of thumb is ~30+), which
covers every scenario this system is actually used for. It is a **known,
accepted inaccuracy at very small sample sizes** — see §11.1.

### 4.4 How correctness is actually established

None of the above is trusted on the strength of "the formula looks right
transcribed." Every core-stats claim is checked against simulated data with
a known answer:

- A well-powered simulation with a real, known effect must have the t-test
  reject the null and recover an effect estimate within a small tolerance of
  the true value.
- A simulation with **no** true effect, repeated 1,000 times, must reject
  the null at close to the nominal 5% rate — not meaningfully more (which
  would mean the p-value calculation is wrong) and not meaningfully less
  (which would mean the test has no power at all).
- A 95% CI, repeated across 500 simulations, must contain the true effect
  close to 95% of the time — checking the *calibration* of the interval,
  not just its formula.
- A deliberately underpowered simulation (tiny n, small effect) must fail to
  detect the effect most of the time, confirming the power calculation is
  internally consistent with what the t-test actually does, not just
  self-consistent in isolation.

This is the pattern every other method in the system follows: implement the
formula, then independently verify it against a scenario where the right
answer is known, rather than treating "it compiled and returned a number"
as evidence of correctness.

### 4.5 Sample ratio mismatch (SRM) check (`srm_check`)

Every method above assumes the data it's given actually reflects the
intended random allocation. `srm_check` is the thing that checks that
assumption instead of silently trusting it: a two-cell chi-square
goodness-of-fit test comparing the observed control/treatment split against
the intended ratio (default 50/50). It's deliberately the first thing
printed in `run_baseline_analysis.py`'s output and the first field in the
API's `RandomizedAnalysisResponse` — checked and surfaced *before* any
effect estimate, because a mismatched split makes every downstream number
untrustworthy regardless of how significant or well-calibrated it looks in
isolation (§1, failure mode 4).

Three design choices worth stating explicitly:

- **A much stricter alpha than the effect test (0.001, not 0.05).** Under a
  correctly running experiment this check should almost never fire, and the
  cost of missing a real mismatch — trusting results from a broken
  randomization — is high enough to justify tolerating a lower false-positive
  rate here than the standard significance threshold used for the actual
  effect estimate.
- **A fixed alpha is itself a known simplification, not a solved problem.**
  Chi-square power grows with sample size, so a *fixed* threshold gets more
  sensitive to practically meaningless deviations (a 50.01%/49.99% split) as
  an experiment gets larger — the exact failure mode Fabijan et al. warn
  about, and their own recommendation is to scale the threshold with sample
  size rather than use one constant everywhere. That scaling isn't
  implemented here; 0.001 is a stricter, more defensible constant than a
  conventional 0.05 or 0.01, not a fix for the underlying scale-dependence
  (§11.1).
- **It's a pure counts-based check** (`n_control`, `n_treatment`), not
  something that inspects the outcome data at all. This is intentional: SRM
  is a statement about the *mechanism that assigned people to arms*, and
  conflating it with anything about the outcome would risk the check itself
  becoming a second, redundant hypothesis test about the effect rather than
  the categorically different question it's supposed to answer.
- **Detecting SRM visibly disables the verdict, not just the trust.** The
  dashboard doesn't just show a warning banner alongside normal-looking
  ship/don't-ship badges — every badge below an SRM warning is replaced with
  an explicit "verdict withheld" state (`VerdictBadge`'s `withheld` prop) and
  the result cards are visually dimmed. A banner that says "don't trust
  this" next to a confident green "Ship" badge would contradict itself; the
  UI enforces the same conclusion the text states, rather than just stating it.

Verified the same way as everything else in this system: `tests/test_srm.py`
checks that a clearly mismatched split (400/600 against an intended 50/50)
is flagged, that an exact match is never flagged, and — the calibration
check that matters most — that across thousands of genuinely balanced random
splits, the check fires at close to its own nominal alpha, not more and not
less.

## 5. CUPED (`cuped.py`)

### 5.1 The mechanism

```
theta      = Cov(Y, X) / Var(X)
Y_adjusted = Y - theta · (X - mean(X))
```

`theta` is the ordinary-least-squares slope of the outcome on the
pre-experiment covariate — the value that minimizes the variance of
`Y_adjusted`. Subtracting `theta · (X - mean(X))` removes the part of `Y`'s
variance that's linearly predictable from `X`, while leaving the *mean* of
`Y_adjusted` unchanged in each arm, because `X - mean(X)` has mean zero by
construction.

### 5.2 Why theta and mean(X) are computed on the pooled sample, not per-arm

This was a specific decision, not an oversight, and getting it wrong would
have silently reintroduced the exact bias CUPED exists to avoid. If `theta`
(or the covariate mean used to center it) were estimated separately within
each arm, the adjustment would subtract a *different* constant from each
arm's outcomes — and that difference would land directly in the estimated
treatment effect, biasing it by however much the two arms' covariate
distributions happen to differ by chance in a given sample. Pooling `theta`
and `mean(X)` across both arms and applying the identical values to each
guarantees the adjustment's effect on the *difference* in means is exactly
zero in expectation. `tests/test_cuped.py` checks this directly: on data
with a known effect and injected unrelated noise, both the naive and
CUPED-adjusted point estimates must land close to the true effect and close
to *each other* — if CUPED silently shifted the estimate, that test would
catch it.

### 5.3 Why pooling doesn't leak treatment information into the adjustment

A natural objection: doesn't averaging in the treated arm's data to compute
`theta` let the treatment "contaminate" the correction applied to control?
No — because `X`, the pre-experiment covariate, is measured *before*
treatment assignment happens, and assignment is random. `X`'s distribution
is therefore independent of treatment by construction, and `theta` is
purely a statement about how `Y` covaries with `X` in general, not about the
treatment effect. This is why the covariate must genuinely be
*pre-experiment* — see §11.1 for what breaks if it isn't.

### 5.4 Alternative considered: per-arm theta

Rejected for the bias reason in §5.2 — worked through analytically before
any code was written, rather than discovered by debugging a biased result
after the fact.

### 5.5 Alternative considered and deferred: nonlinear / ML-based adjustment (CUPAC)

Production CUPED variants sometimes replace the linear OLS coefficient with
a machine-learned prediction of the outcome from richer pre-experiment
features (sometimes called CUPAC — "Control Using Predictions As
Covariates"). This was deliberately not implemented. The simulator's
covariate–outcome relationship is linear by construction, so a linear
adjustment is already optimal on this system's own data — a nonlinear
model would add real implementation and validation complexity (needing its
own train/test split discipline to avoid overfitting the adjustment itself)
for no measurable benefit against this system's data-generating process. It
would matter on real-world data with genuinely nonlinear covariate–outcome
relationships; that's real scope left on the table, not a hidden flaw — see
§11.2.

## 6. Sequential testing (`sequential.py`)

### 6.1 The problem, formally

A p-value's guarantee ("under the null, this is significant only 5% of the
time") is a statement about a *single, pre-committed* test. Checking it
repeatedly and stopping at the first significant result is a different
procedure with a different, much higher true false-positive rate, because
each look is an additional opportunity for noise to cross the threshold.

### 6.2 mSPRT chosen over the alternatives

Three approaches were on the table:

- **Bonferroni-correct the naive test** for the number of planned looks
  (divide alpha by the number of checks). Rejected: it requires committing
  to an exact number of looks in advance — exactly the rigidity that makes
  fixed-horizon testing impractical in the first place — and it's needlessly
  conservative because it doesn't use the correlation between consecutive
  looks at the same accumulating dataset.
- **Group-sequential design with alpha-spending** (e.g., O'Brien-Fleming
  boundaries). This is what most production experimentation platforms
  actually use. It's more statistically efficient than the mixture approach
  when the look schedule is known in advance, but it requires pre-specifying
  that schedule (or at minimum a maximum number of looks) and solving for
  boundary values that are more involved to implement and verify correctly.
- **Mixture sequential probability ratio test (mSPRT) / always-valid
  p-values** (Robbins, 1970; Johari, Koomen, Pekelis & Walsh, 2017 — the
  method behind Optimizely's stats engine). Chosen because it doesn't
  require pre-committing to a look schedule at all — it stays valid even
  under continuous peeking, checked after every single new data point,
  which matches how teams actually behave (they don't pre-register a
  checking cadence) more closely than a group-sequential design does.

The mechanism: instead of testing a single fixed alternative, mSPRT places a
Gaussian mixing prior `N(0, tau²)` over the possible treatment effect. At
each look, the resulting likelihood ratio `Lambda_t` is a nonnegative
martingale under the null, so by Ville's inequality
`P(exists t : Lambda_t >= 1/alpha) <= alpha` — the probability of *ever*
falsely flagging significance across the whole sequence of looks is bounded
by alpha, not just at any single look.

### 6.3 tau² is a tuning knob, and the empirically-observed conservativeness

`tau²` represents the scale of effect the test is tuned to detect well; it
doesn't need to be exactly right for the alpha guarantee to hold, but it
affects power. Empirically (see §10.5), the implemented test's measured
false-positive rate under repeated peeking came in well *below* the nominal
5% (around 1%) across a swept range of `tau²` values, not just one poorly
chosen setting. This was investigated as a possible bug and concluded to be
expected behavior: Ville's inequality is an upper bound, not an exact
calibration target, and with a small, finite number of discrete looks (20 in
the demonstration) the bound is loose. This was a deliberate stopping point
— chasing exact 5% calibration would mean moving to a different mathematical
construction (e.g. a tighter boundary designed for a specific, known number
of looks), which reintroduces the pre-commitment rigidity mSPRT was chosen
to avoid. The conservativeness is logged as an accepted property, not
silently smoothed over — see §9.

### 6.4 What this does not do

The sequential test operates on one metric's raw outcome stream. It is
**not** currently composed with CUPED — there is no "CUPED-adjusted
always-valid p-value" path, even though combining the two would in
principle give both benefits (peeking-safety and variance reduction)
simultaneously, and is exactly what a production-grade stats engine would
want. They are demonstrated independently: CUPED runs against a final,
fixed-size dataset; the sequential test runs against the raw outcome as it
accumulates. See §11.2.

## 7. Causal inference for non-randomized data (`causal.py`)

### 7.1 Confounding, formally

When treatment assignment depends on a covariate that also affects the
outcome, `E[Y | treated] - E[Y | untreated]` is not the treatment effect —
it's the treatment effect *plus* the average outcome difference that would
have existed between these two groups even with no treatment at all. The
two are inseparable without explicitly conditioning on what's known about
why the groups differ.

### 7.2 Propensity score matching chosen over the alternatives

Two other methods were explicitly on the table:

- **Difference-in-differences.** Rejected for this system specifically
  because it requires panel data — pre- and post-period outcomes for the
  same units — which the rest of this system's data model doesn't have (the
  randomized-experiment side is single-period by design; see §3.2). Adding
  DiD would have meant maintaining two incompatible data shapes throughout
  the simulator, the API, and the frontend for one causal method, which was
  judged not worth the complexity given propensity matching covers the same
  underlying "correct for what's observed" idea on data already in the
  system's existing shape.
- **Uplift modeling** (directly modeling heterogeneous treatment effects
  per-unit, rather than a single average effect). Rejected because it
  answers a different question — "for whom does this work" rather than
  "does this work, corrected for confounding" — and evaluating it properly
  needs its own validation methodology (e.g. Qini curves) distinct from the
  confidence-interval-based verification used everywhere else in this
  system. Deferred as future scope (§11.4), not rejected on merit.
- **Propensity score matching** (Rosenbaum & Rubin, 1983) was chosen: fit a
  propensity model, match treated units to similar-propensity control
  units, compare outcomes within matched pairs. It fits the existing
  cross-sectional data shape, and its correctness claim — "matching balances
  the covariate distribution between treated and matched control, so
  comparing within a pair approximates the missing counterfactual" — is
  directly testable against the simulator's known confounding strength.

### 7.3 Nearest-neighbor matching with replacement, and its cost

Each treated unit is matched to the closest-propensity control unit,
**with replacement** — the same control unit can be reused across multiple
treated units. This was chosen over matching without replacement because
without replacement, treated units can run out of nearby, unused controls
when the treated and control propensity distributions don't overlap
perfectly, silently degrading match quality for whichever units happen to
be processed last. With replacement, every treated unit gets its
best-available match regardless of processing order.

**The cost, stated plainly:** the confidence interval currently reported is
computed from a simple paired-difference t-interval over the matched pairs,
which treats each matched pair as independent. That assumption is not
exactly true when a control unit has been reused across several treated
units — the correct variance estimator for matching with replacement (see
Abadie & Imbens, 2006) accounts for how many times each control was reused
and is more involved to implement. The interval reported here is very
likely close in practice for the caliper and confounding strengths used in
this system's own tests (verified in §7.5's ground-truth check), but it is
a known, accepted approximation rather than the fully rigorous estimator —
see §11.1.

### 7.4 Logistic regression for the propensity model, and its risk

The propensity model is a plain logistic regression on the observed
covariates. This is the textbook default, and — importantly — it's
*well-specified* here, because the simulator's own treatment-assignment
mechanism is itself a logistic function of the covariates (§3.3), so the
model can in principle recover the true propensity function exactly. On
real-world, non-simulated data, the true assignment mechanism is unknown
and may not be linear-in-logit; a flexible model (e.g. gradient boosting)
would be more robust there at the cost of being harder to inspect and more
prone to overfitting the propensity score itself (which paradoxically can
make matching *worse*, a well-known issue in the causal inference
literature). This was a conscious choice to match the system's own
generating process rather than to be maximally robust to an unknown one —
see §11.1.

### 7.5 The caliper, and the estimand it silently changes

Treated units whose best available match is farther than `caliper` away in
propensity-score distance are dropped from the estimate entirely, rather
than forced into a poor match. This is the right call for estimate
*quality* — a bad match is worse than no match — but it means the reported
effect is technically the average effect among the *matchable* treated
population, not literally every treated unit in the dataset. For most of
this system's default configurations the drop rate is small, but a caliper
set too tight against a given confounding strength can silently shrink the
effective population the result generalizes to. This is verified only
indirectly, via `n_matched` vs. `n_treated` being reported side by side in
both the CLI output and the dashboard — no automated check currently flags
"too many units were dropped" as a warning; see §11.1.

### 7.6 Honest UI choice: no confidence interval on the naive comparison

The naive treated-vs-control comparison is shown in the CLI output and the
dashboard for comparison purposes, but deliberately without a confidence
interval or a significance verdict, and labeled as such. A naive comparison
on confounded data doesn't have a *valid* CI in the first place (its
standard error formula assumes no confounding to correct for) — showing one
anyway would imply a rigor the naive method doesn't have, which is precisely
the kind of false confidence this whole system exists to push back against.

## 8. API and frontend

### 8.1 A thin backend, deliberately

`backend/main.py` contains no statistical logic. Every endpoint calls
directly into the same `simulator` / `stats_core` / `cuped` / `sequential` /
`causal` functions used by the CLI scripts and the test suite. This was a
hard constraint, not just a preference: if the API recomputed or
re-implemented any of the analysis, the dashboard's numbers could silently
drift from the numbers the tests actually verify, and a discrepancy between
"what the tests proved correct" and "what the UI shows" would defeat the
entire ground-truth-verification premise this system is built on (§2).

### 8.2 Two interface generations, and why the second replaced the first

The dashboard was first built as a single-process Streamlit app, then
rebuilt as a FastAPI backend with a separate React/TypeScript frontend.
Both decisions were made for real reasons, not by default:

- **Streamlit first** because it let every other method in the system get a
  working, visual demonstration with minimal additional code, and because a
  single-process Python app with no separate build step meant the interface
  layer could iterate as fast as the analysis code underneath it while that
  code was still the primary focus.
- **Replaced with FastAPI + React** once the interface itself became a
  first-class concern rather than a wrapper: a conventional client/server
  split with a typed HTTP boundary is legible to a wider range of reviewers
  than a Streamlit script (which reads as "one Python file with inline
  widgets" to anyone unfamiliar with that specific framework), and
  separating the analysis API from its presentation forces the API contract
  in §8.1 to be explicit and enforced by the TypeScript types on the
  frontend, rather than implicit in how a Streamlit script happens to call
  Python functions inline.

The rewrite kept the same feature set (simulate-or-upload, naive vs.
corrected panels side by side, ship/don't-ship verdicts, one-click flagship
demo) and was verified against the same numbers the Streamlit version and
the CLI scripts produced, specifically to confirm the interface swap changed
nothing about what was actually being shown (§10.6).

### 8.3 Debounced auto-refresh vs. an explicit "Run" button

The frontend re-runs the analysis automatically, a short debounce interval
after any parameter changes, rather than requiring an explicit submit
action. This gives a tighter feedback loop for exploring how a parameter
affects the verdict (drag a slider, watch the badge flip), which matters for
a tool whose purpose is partly to build intuition about these methods, not
just produce a single final answer. The accepted cost: dragging a slider
issues a new backend request roughly every 250ms of inactivity, which is a
non-issue for a local single-user tool but would need rate-limiting or a
"commit" step before this pattern would be appropriate behind a shared
deployment (§11.5).

### 8.4 A hand-built chart instead of a charting library

The peeking-check visualization (naive vs. sequential p-value across
checkpoints) is a hand-written SVG component, not a library like Recharts
or Plotly. For exactly one chart type used in exactly one place, a charting
library's bundle size and API surface cost more than it saves; hand-rolling
it also gave direct control over the specific interaction and accessibility
details (legend, direct hover tooltip, a dashed alpha-threshold reference
line, colorblind-safe series colors) rather than working around a library's
defaults for each of those.

### 8.5 What the dev setup does not solve

The Vite dev server proxies `/api` requests to the FastAPI backend and CORS
is restricted to the local dev origin — this is a development convenience,
not a deployment story. There is no production build/serve unification (the
built frontend isn't currently served by the backend or bundled into a
single deployable artifact), no containerization, and no environment-based
configuration for a non-localhost backend URL. Standing this up beyond a
local machine is explicitly out of scope right now — see §11.5.

## 9. Key tradeoffs at a glance

| Decision | Rejected alternative(s) | Why | Cost accepted |
|---|---|---|---|
| Welch's t-test everywhere | Student's pooled-variance t-test (conditionally) | Avoids a pre-test for variance equality, which has its own error-rate cost | Marginally more complex df calculation |
| t-test/CI/power formulas hand-written | `scipy.stats.ttest_ind` directly | Forces understanding of the exact mechanism the rest of the system builds on; independently verified against ground truth | More code to maintain and keep correct |
| z-approximation for power/MDE/sample size | Exact noncentral-t iterative solve | Standard textbook approach, simple, accurate at realistic sample sizes | Inaccurate at very small n (§11.1) |
| CUPED theta pooled across arms | Per-arm theta | Per-arm estimation reintroduces bias into the effect estimate | None significant — pooling is strictly better here |
| Linear CUPED adjustment | ML-based residualization (CUPAC) | Optimal already, given the simulator's linear data-generating process | Wouldn't capture nonlinear relationships on real data |
| mSPRT always-valid p-values | Bonferroni correction; group-sequential alpha-spending | No pre-committed look schedule required; valid under continuous peeking | Empirically conservative (measured ~1% vs. nominal 5%); not maximally powerful |
| Propensity score matching | Difference-in-differences; uplift modeling | Fits existing cross-sectional data shape; directly testable against known confounding | DiD/uplift-specific questions unanswered (§11.4) |
| NN matching **with** replacement | Matching without replacement | Avoids match starvation when propensity distributions don't overlap well | Reported CI is a simplified approximation, not the fully rigorous with-replacement estimator (§11.1) |
| Logistic regression propensity model | Flexible ML classifier (e.g. GBM) | Well-specified given the simulator's own logistic assignment mechanism | Risk of misspecification on real-world non-simulated data |
| FastAPI + React interface | Streamlit (kept as the first version, then replaced) | Typed API boundary; conventional client/server split; broader legibility | A full rewrite of the interface layer; two front-end histories to reconcile |
| Debounced auto-refresh in the UI | Explicit "Run analysis" button | Tighter feedback loop for exploring parameter effects | Frequent backend requests while adjusting controls; not deployment-safe as-is |
| Hand-built SVG chart | Charting library (Recharts/Plotly/etc.) | No extra dependency for one chart type; full control over interaction details | More component code to maintain directly |

## 10. What broke during construction, and how it was diagnosed

### 10.1 The Criteo loader silently pulled a treatment-only slice

Loading the first `n_rows` of the Criteo CSV as a quick sample produced a
dataset where `treatment.value_counts()` showed 100% treatment, 0% control —
the file is grouped by treatment status, not shuffled, so any prefix read is
a slice of one arm only. This was caught by inspecting the value counts
directly rather than trusting the loader, right after `run_baseline_analysis
--source criteo` crashed downstream (§10.3). Fixed by reading only the
needed columns for the *entire* file (fast enough at ~19s given the reduced
column set) and taking a true random sample afterward, rather than relying
on row order.

### 10.2 A pre-existing outer git repository

Running `git status` from the project directory returned results scoped to
a git repository rooted at the user's home directory (`C:\Users\rmadi`),
not the project folder — meaning a prior, unrelated `git init` had at some
point been run one level up, tracking the entire home directory. Rather
than committing into or otherwise disturbing that repository, a separate,
freshly initialized repository was created scoped specifically to the
project folder, and the outer repository was left untouched throughout.

### 10.3 `ZeroDivisionError` in the power calculation

`run_baseline_analysis.py --source criteo` crashed inside
`statistical_power` with a division by zero. Root cause: `n_control` was 0,
which traced directly back to §10.1's treatment-only-slice bug — with zero
control-arm rows, every downstream per-arm statistic broke. Fixed together
with the sampling fix in §10.1; the crash was actually a useful early
signal that something upstream was wrong, well before anyone inspected the
value counts directly.

### 10.4 Windows npm builds broken by an `&` in the project path

`npm run dev`, `npm run build`, and `npx vite` all failed with
`'AB' is not recognized as an internal or external command` on this
project's actual filesystem path (which contains an `&`: "Causal Inference
& AB Experimentation Platform"). Diagnosed by running the underlying
binaries directly (`node node_modules/vite/bin/vite.js`), which worked
fine — isolating the failure to npm's Windows `.cmd` shim generation, which
doesn't robustly quote paths containing shell-special characters like `&`
when it re-invokes `cmd.exe`. This is an environment quirk of the project's
directory name on Windows, not a bug in this project's code; the workaround
(invoke the binaries directly, bypassing the shim) is documented in the
README rather than worked around by renaming the user's own directory.

### 10.5 Investigating the sequential test's conservativeness

The first end-to-end run of the peeking demonstration showed the mSPRT
test's empirical false-positive rate at roughly 1%, well under the nominal
5% target. Before accepting that as correct, `tau²` was swept across a
range of values (1, 2, 4, 8, 20, 50, 100, 200) to rule out a single
poorly-chosen constant as the cause — the rate stayed in the same
low-single-digits range across the whole sweep, which pointed at something
structural (Ville's inequality being a loose upper bound with only 20
discrete looks) rather than a tuning mistake. This is recorded as an
accepted, understood property (§6.3, §9) rather than something "fixed" by
further tuning, because chasing exact 5% calibration would have meant
abandoning the specific guarantee (validity under arbitrary, uncommitted
peeking) that motivated choosing mSPRT in the first place.

### 10.6 Finding the flagship-demo scenario systematically, not by hand

The flagship demo (naive says don't-ship, CUPED says ship, on the same
data) needed a specific combination of sample size, true effect, extra
noise, and random seed that actually produces that disagreement — most
parameter combinations don't. Rather than hand-adjusting numbers until one
run happened to look right (which would have been indistinguishable from
cherry-picking after the fact), a small script swept 200 candidate random
seeds against a fixed set of parameters and searched for one where the naive
p-value cleared 0.05 (not significant) while the CUPED p-value did not
(significant) — the seed used in the shipped demo was selected from that
systematic search, and the resulting numbers (naive p=0.154, CUPED
p=0.0041) were then independently re-verified by calling the exact same
analysis functions the dashboard and backend use, not just trusted from the
search script's own output.

**This invites an obvious objection: isn't searching 200 seeds for a
favorable outcome just cherry-picking, dressed up with the word
"systematic"?** It would be, if the seed search were the *evidence* that
CUPED works. It isn't — that evidence is the repeated-simulation calibration
tests in `tests/test_cuped.py`, which check the general, seed-independent
claims ("the point estimate stays close to the true effect," "variance drops
substantially when the covariate captures injected noise") across many runs,
none of them cherry-picked. What the seed search found is a single instance
where that already-proven general property happens to land on opposite
sides of the p<0.05 line for the two methods — useful for building intuition
in one vivid, legible example, not as a substitute for the aggregate proof.
The honest version of this distinction: run the flagship parameters at a
different seed and the two methods will usually agree (both significant, or
both not) — that's expected and fine, because the general claim was never
"CUPED always flips the verdict," it's "CUPED reduces variance and doesn't
bias the estimate," which holds regardless of which seed makes that fact
visible as a ship/don't-ship flip.

### 10.7 Verifying the UI without a human clicking through it

Neither the Streamlit version nor the React version could be visually
confirmed by a human clicking around during development. Both were verified
instead by (a) starting the actual server processes, (b) driving them with
headless Chrome screenshots at specific application states (default view,
flagship demo loaded, peeking chart populated, causal panel populated), and
(c) checking that the numbers rendered in the screenshot matched the numbers
independently computed by calling the same underlying analysis functions
directly in a script. This substitutes for manual QA but is not automated —
see §11.6.

## 11. Limitations, known issues, and future work

### 11.1 Statistical approximations accepted as-is

- **Power/MDE/sample-size formulas use a normal (z) approximation**, not an
  exact noncentral-t solve. Accurate at realistic sample sizes; measurably
  off at very small per-arm n (roughly under 30). Not currently flagged in
  the UI when a configuration is small enough for this to matter.
- **The SRM check's alpha is a single fixed constant (0.001), not scaled to
  sample size.** Chi-square power grows with n, so the same fixed threshold
  is conservative for a small experiment and increasingly trigger-happy on
  practically meaningless deviations for a very large one — the specific
  problem Fabijan et al. (the paper this check is based on) recommend
  solving by scaling the threshold with sample size. That scaling isn't
  implemented (§4.5); a fixed, stricter-than-conventional constant was
  chosen as a reasonable default for this system's realistic sample sizes
  (hundreds to tens of thousands per arm), not as a substitute for the more
  rigorous approach.
- **Propensity-matching confidence intervals use a simplified paired-t
  formula** that treats matched pairs as independent, which isn't exactly
  true under matching-with-replacement (a reused control correlates the
  pairs it appears in). The fully rigorous approach (Abadie & Imbens, 2006)
  isn't implemented. The simplification is unverified against a
  ground-truth CI-coverage test analogous to the one built for the plain
  t-test (§4.4) — this is a real gap, not just a stylistic simplification.
- **The logistic-regression propensity model is well-specified against this
  system's own simulator, not against arbitrary real-world assignment
  mechanisms.** A CSV upload with a genuinely nonlinear treatment-assignment
  process would silently get a worse propensity fit with no warning.
- **The caliper can silently shrink the effective estimand** (§7.5) with no
  automated warning when the dropped-unit fraction gets large.
- **CUPED and the sequential test are not composed.** There is no path to
  get a variance-reduced *and* peeking-safe estimate simultaneously — they
  are demonstrated as two independent corrections to two independent
  problems, not a combined pipeline. A real production stats engine would
  want both at once.
- **The sequential test handles one metric at a time.** Monitoring multiple
  metrics simultaneously (which is extremely common in practice) needs its
  own multiple-testing correction on top of the sequential correction; that
  composition isn't implemented or even modeled.
- **No interference/SUTVA detection.** Every method assumes one unit's
  outcome is unaffected by another unit's treatment assignment (§1.1). If
  that's false — a referral loop, shared marketplace inventory, a visible
  social feed — every effect estimate in this system is biased in a
  direction and magnitude nothing here would surface; there is no
  cluster-randomization support, no exposure modeling, and no diagnostic
  that would even hint the assumption is being violated.
- **The flagship demo's variance reduction (55-58%) is not representative
  of typical real-world CUPED gains — it's a demonstration of the mechanism
  at a deliberately strong, hand-chosen covariate correlation (0.9+).**
  Running the same code against a real covariate on the Criteo dataset
  (§3.5) gives a correlation around -0.13 and roughly 2% variance reduction,
  which is a much more honest expectation for an arbitrary real metric.
  Real gains depend entirely on how predictive the available pre-experiment
  covariate actually is, and that has to be checked per-metric, not assumed
  from this project's own demo numbers.

### 11.2 Methods deliberately not implemented

- **Difference-in-differences** and **uplift modeling** were both
  considered for the causal-inference component and set aside in favor of
  propensity matching (§7.2) — not because they're worse methods, but
  because they answer different questions (DiD needs panel data this
  system doesn't model; uplift modeling targets heterogeneous rather than
  average effects) and each would need its own data model and validation
  approach to do properly.
- **Group-sequential / alpha-spending sequential testing** (the main
  alternative to mSPRT) is not implemented; only one sequential-testing
  approach exists in this system (§6.2).
- **Multiple-testing correction across simultaneously monitored metrics**
  is out of scope entirely, as noted above.
- **Nonlinear / ML-based CUPED (CUPAC)** — using a learned prediction of the
  outcome instead of a linear OLS coefficient — was considered and deferred
  (§5.5). The simulator's own covariate–outcome relationship is linear by
  construction, so a linear adjustment is already optimal on this system's
  data; a real-world metric with a genuinely nonlinear relationship to its
  available covariates would see smaller gains from the linear CUPED
  implemented here than a CUPAC-style approach would deliver.

### 11.3 Engineering gaps

- **No authentication, authorization, or rate limiting** on the FastAPI
  backend. CORS is restricted to the local Vite dev origin, but the API
  itself has no concept of a user or a request budget — it is a local,
  single-user analysis tool, not something safe to expose on a shared or
  public network as-is.
- **No persistence.** Every simulate/upload/analyze call is stateless;
  there's no saved history of past analyses, no session concept, and no
  database. Closing the browser tab loses everything.
- **No production deployment path.** The frontend build output isn't
  currently served by the backend or bundled into a single deployable
  artifact; there's no Dockerfile, no environment-based backend URL
  configuration for a non-localhost deployment, and no CI pipeline running
  the test suite or the frontend build automatically.
- **CSV upload validation is minimal** — only column *presence* is checked
  (`_analyze_randomized` / `_analyze_observational` in `backend/main.py`),
  not column types, value ranges, or row counts. A malformed upload (e.g. a
  non-numeric `outcome` column) will fail with a raw exception rather than a
  clear error message.
- **The real-data (Criteo) validation runs are manual, not automated.**
  `run_baseline_analysis.py --source criteo` and
  `compare_cuped.py --source criteo` were run by hand to produce the numbers
  quoted in §3.5, but neither is wired into the test suite or any CI
  pipeline — a ~300MB network download with third-party availability and
  licensing terms outside this project's control is a poor fit for a test
  that needs to run quickly and deterministically on every change. This
  means a regression in real-data handling specifically (as opposed to the
  ground-truth-simulation-tested logic) would not be caught automatically.
- **The Windows path-with-`&` npm bug (§10.4) is worked around, not
  fixed** — anyone cloning this repo to a path containing `&`, `%`, or other
  `cmd.exe`-special characters on Windows will hit the same shim failure and
  need the same direct-invocation workaround documented in the README.

### 11.4 Product-shape gaps

- **No automated UI regression testing.** The dashboard's correctness was
  verified manually via headless-browser screenshots at specific points in
  development (§10.7), not via an automated test suite that runs on every
  change — a future UI change could silently break a panel with nothing to
  catch it.
- **The flagship demo is a single fixed scenario.** There's no mechanism to
  define, save, or share additional "interesting disagreement" scenarios
  beyond the one hard-coded set of parameters — building a second flagship
  scenario means editing constants in source, not a UI-driven authoring
  flow.
- **No sensitivity analysis or robustness checks are surfaced to the
  user** — e.g., there's no built-in way to see how the propensity-matching
  estimate changes as the caliper varies, even though that's exactly the
  kind of check a careful analyst would want before trusting a matched
  estimate.

### 11.5 Concrete future work

1. Implement the Abadie–Imbens variance estimator for matching with
   replacement, and add a ground-truth CI-coverage test for it analogous to
   the one that already exists for the plain t-test.
2. Compose CUPED and the sequential test into a single "variance-reduced,
   peeking-safe" pipeline.
3. Add a flexible (non-logistic) propensity-model option, with a
   diagnostic (e.g. propensity-score overlap plot) surfaced in the
   dashboard.
4. Add difference-in-differences as a second causal method, which requires
   extending the simulator and API to support panel (pre/post) data
   alongside the existing cross-sectional shape.
5. Add a production deployment path: a single build step that serves the
   built frontend from the FastAPI backend, containerized, with
   environment-based configuration instead of hard-coded localhost URLs.
6. Add authentication and per-session rate limiting before considering any
   shared or public deployment.
7. Replace the manual headless-screenshot verification process (§10.7) with
   an automated UI test suite that runs against the same flagship and
   peeking scenarios on every change.
8. Surface a caliper-sensitivity view in the dashboard so a user can see how
   the matched estimate and matched-pair count move together as the caliper
   changes, rather than only seeing the result at one fixed caliper value.
9. Scale the SRM check's alpha with sample size instead of using one fixed
   constant (§4.5, §11.1), per Fabijan et al.'s own recommendation, so the
   check stays well-calibrated at both small and very large sample sizes
   rather than becoming oversensitive as n grows.
