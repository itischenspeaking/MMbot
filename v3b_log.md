# v3b — learning hidden toxicity

## 1. Model / Theory

v3a assumed phi known and fixed. v3b removes that:

    v3a:  phi known         -> h*(phi)
    v3b:  phi_t hidden       -> realized markouts -> phi_hat_t -> h_t

Environment: phi schedule 0 -> 1 -> 0, 1500 ticks per segment, sigma=0.3,
k=0 (v2 skew excluded from the main experiment — see Experiment 1).

Estimator, inverting v3a's calibration m(phi) = phi*sigma*sqrt(2/pi):

    phi_hat_t = clip( mean(last N fill markouts) / (sigma*sqrt(2/pi)), 0, 1 )

Only fills update the window; N is counted in fills, not ticks. phi_hat is
NaN until N fills have been observed. Adaptive spread reuses v3a's rule
directly, no new theory:

    h_t = 1/kappa + phi_hat_t * sigma * sqrt(2/pi)

Two predictions made before running anything:

    response lag     ~ N/2 fills
    estimator noise  ~ 1/sqrt(N)

Stage 1 sanity (not detailed here): a constant phi_schedule reproduces
InformedFlow byte-identically (RNG draw budget is fixed regardless of phi),
the exogenous price path is unaffected by the schedule, and per-regime
markout matches phi*sigma*sqrt(2/pi). Mechanism confirmed clean before any
estimation work started.

## 2. Experiment 1 — is there enough value to adapt to?

**Question.** If the economic value of knowing phi is negligible, online
learning isn't worth building.

**Hypothesis.** Oracle (spread set to the true per-regime optimum) beats
Best Fixed (single best spread for the whole schedule), but the size of the
gap depends on sigma.

**Method.** Analytical comparison, k=0, A=0.4, kappa=1, phi: 0->1->0. Start
at sigma=0.1; raise sigma only if the gap is too small, per pre-specified
rule — do not also widen the phi contrast (already at 0/1).

**Results.**

| sigma | h*(low) | h*(high) | h_fixed* | Oracle − Fixed (4500 ticks) |
|---|---|---|---|---|
| 0.1 | 1.000 | 1.080 | 1.027 | 0.45 |
| 0.3 | 1.000 | 1.239 | 1.080 | 3.795 |

sigma=0.1's gap is too small to survive Monte Carlo noise (a same-seed
diagnostic at the largest static mismatch this model can produce, h=1.00 vs
1.08 at phi=1 constant, gave t=1.14 — noise dominates). sigma=0.3 gives
t=4.15 on the same diagnostic.

sigma=0.4 was also checked (gap=6.52, t=4.98) but not adopted: the marginal
gain in t over 0.3 was small, high-regime fill count drops further
(160 vs 174 fills/segment), and 0.3 is the less amplified setting that still
clears the noise floor.

**Conclusion.** sigma=0.1 gives too little economic value to adaptation;
sigma=0.3 makes the Oracle-vs-Fixed opportunity large enough to study while
remaining moderate. Locked for the main experiment: sigma=0.3, phi 0->1->0,
seg_len=1500, k=0.

## 3. Experiment 2 — can markouts recover hidden toxicity?

**Question.** The maker never sees phi. Can it infer regime changes purely
from its own realized fills?

**Hypothesis.** phi_hat tracks 0->1->0. Small N: fast, noisy. Large N:
slow, stable.

**Method.** Fixed maker (h=1, k=0) — quoting held constant so estimation is
decoupled from any quoting feedback. N in {20, 50, 100}, 300 seeds/N.

**Noise** (high-regime steady state, cleanest of clip effects):

| N | noise (std) | predicted ratio | observed ratio |
|---|---|---|---|
| 20 | 0.0939 | 1.000 | 1.000 |
| 50 | 0.0593 | 0.632 | 0.631 |
| 100 | 0.0410 | 0.447 | 0.437 |

Matches 1/sqrt(N) closely.

**Response lag** (fills after boundary to cross 0.5; seeds already on the
wrong side at the boundary excluded, reported separately):

| N | predicted | up (mean/median) | down (mean/median) | false-side rate |
|---|---|---|---|---|
| 20 | ~10 | 9.5 / 10.0 | 10.3 / 9.0 | up 3.7%, down 0% |
| 50 | ~25 | 23.9 / 24.0 | 24.9 / 24.0 | 0% |
| 100 | ~50 | 48.8 / 49.0 | 49.0 / 49.0 | 0% |

Prediction reproduced almost exactly.

**Clipping bias.** At true phi=0, mean phi_hat is not 0:

| N | mean phi_hat (phi=0 truth) |
|---|---|
| 20 | 0.110 |
| 50 | 0.070 |
| 100 | 0.048 |

Not a calibration bug: clip(phi_hat, 0, 1) truncates negative noise at 0 but
keeps positive noise, producing a one-sided upward bias at the low boundary
(mirrored downward at phi=1). Bias shrinks with N as predicted, since larger
N means less noise to truncate.

**Choosing N.** N=20 is fast but noisy (3.7% false-side rate, largest bias).
A typical regime at h=1 holds ~1500*0.4*e^-1 ≈ 221 fills. N=100's ~49-fill
lag consumes ~22% of that; N=50's ~24-fill lag consumes only ~11%, a better
responsiveness–stability compromise. Noise is small enough at N=50 that the
resulting h swings ~1.3%, no false-side events, bias mild. N=50 selected
for Experiment 3.

**Conclusion.** Signed markouts recover the hidden regime. The
responsiveness–stability trade-off is measurable and matches the predicted
scaling and lag almost exactly. N=50 is the practical compromise for this
segment length.

## 4. Experiment 3 — does adaptation create value?

**Question.** Experiment 2 shows the maker can tell when the market turned
toxic. Does acting on that estimate — widening/narrowing the spread —
actually help?

**Strategies**, same-seed CRN (identical price path and RNG draws; fills
diverge only because quotes differ):

- **Best Fixed** — h=1.080 always.
- **Adaptive** — quotes h_warmup=1.080 (= Best Fixed) until N=50 fills
  accumulate, byte-identical to Fixed until then. After warm-up,
  h_t = 1/kappa + phi_hat_t*sigma*sqrt(2/pi), using only phi_hat available
  from fills strictly before t (no look-ahead — the estimator updates after
  the tick's delta_S is applied, and the update only affects the next
  tick's quote).
- **Oracle** — h_t from the true phi_t. Upper benchmark only, not a
  realizable strategy.

**Mechanism check.** Adaptive avg h by regime vs. Oracle's target:

| regime | Oracle target | Adaptive |
|---|---|---|
| low | 1.000 | 1.033 |
| high | 1.239 | 1.203 |
| low | 1.000 | 1.045 |

Adaptive widens in the toxic regime and narrows back — qualitatively
correct, quantitatively short of Oracle by the estimation lag and noise.

**Closed-loop response lag** (from the Adaptive run's own fill sequence,
not the fixed-quote offline estimate in Experiment 2):

    0->1:  24.4 fills / 175 ticks
    1->0:  25.4 fills / 205 ticks

Fill-space lag (~24-25) matches Experiment 2's offline estimate (~24) —
the estimator's statistical behavior is unchanged by closing the loop.
Tick-space lag is the visible second-order effect: wider h in the high
regime means fewer fills, so new information arrives more slowly in wall-
clock time even though it arrives at the same rate per fill.

**PnL**, 3000 seeds, same-seed paired differences:

| | mean PnL |
|---|---|
| Fixed | 617.19 |
| Adaptive | 619.86 |
| Oracle | 621.62 |

| paired difference | mean | 95% CI |
|---|---|---|
| Adaptive − Fixed | 2.672 | [−0.719, 6.064] |
| Oracle − Adaptive | 1.752 | [−0.870, 4.373] |
| Oracle − Fixed | 4.424 | [0.243, 8.605] |

Oracle − Fixed is positive and resolved (CI excludes 0), consistent with
Experiment 1's analytical gap of 3.795 — the available economic opportunity
in this environment is real but small, on the order of 4 PnL over 4500
ticks.

Adaptive − Fixed has a positive point estimate but the 95% CI crosses
zero: **the PnL improvement is not statistically resolved at 3000 seeds.**
The available opportunity is small relative to path-level PnL variance
(driven by uncontrolled k=0 inventory): even with 3000 paired seeds, the
Adaptive − Fixed estimate has SE ≈ 1.73. Resolving an improvement smaller
than the opportunity itself requires either a much larger sample or a
design that isolates the execution-edge signal from inventory noise (out of
scope here — k was fixed at 0 by design, to isolate toxicity learning from
v2's inventory problem).

**Capture fraction.** Point estimate only, not a precise rate:

    2.672 / 4.424 (sample Oracle-Fixed)   ≈ 0.60
    2.672 / 3.795 (analytical opportunity) ≈ 0.70

Both versions agree Adaptive captures roughly 60-70% of the available
improvement. Both ratios inherit substantial uncertainty from the sampled
Adaptive − Fixed numerator; the sample Oracle − Fixed denominator (vs. the
deterministic analytical one) makes the first ratio noisier still — this is
the same effect that produced the >100% ratio noted below.

**Sampling note.** An earlier 1000-seed run gave Adaptive − Fixed
mean = 6.045, which divided by the analytical opportunity (3.795) exceeds
100% — apparently capturing more than the theoretical upper bound. This
is not a real effect: Oracle − Fixed at 1000 seeds (7.975) was itself
running high of its analytical value (3.795) from sampling noise, and both
sample estimates moved back toward the analytical scale at 3000 seeds
(Oracle − Fixed: 4.424). The >100% ratio was Monte Carlo noise in a small
sample, not a violation of the theoretical bound.

**RMSE note.** Full-path closed-loop phi_hat RMSE (0.271) is much worse
than Experiment 2's steady-state RMSE (~0.06-0.12). The two are not
computed on the same basis: Experiment 2 explicitly excluded transition
periods from its steady-state metric, while the Stage 3 full-path RMSE
includes both regime transitions in full. This is not evidence of
closed-loop estimator degradation — it hasn't been re-measured on a matched
steady-state basis, so no claim is made either way.

## 5. Conclusion

v3b removed the assumption that toxicity is directly observable. Rolling
signed markouts recovered hidden toxicity, with the predicted
noise–response-lag trade-off matching analytical scaling closely. Feeding
the estimate into v3a's spread rule made quotes widen and narrow in the
correct regimes and produced a positive but statistically unresolved PnL
improvement over the best fixed strategy — the available opportunity itself
is small (~4 PnL/4500 ticks), and inventory noise from k=0 makes it hard to
resolve at this sample size. The gap to Oracle is consistent with
finite-window estimation noise and detection lag, compounded by a
second-order effect where wider spreads in toxic regimes slow the arrival
of new information.

**Next step (out of v3b scope):** replacing the hard rolling window with a
smoother or faster online estimator, or isolating the execution-edge signal
from k=0 inventory noise to better resolve the Adaptive-vs-Fixed gap. Both
belong to a later version, not v3b.
