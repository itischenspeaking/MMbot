# v4 — from mechanisms to a decision-making bot

v2 controls inventory. v3 controls toxicity. Both in isolation. v4 asks:

> Can a market maker jointly manage inventory risk and adverse selection,
> learn hidden toxicity online, and prove its policy generalizes beyond the
> one environment it was tuned on?

Five sub-projects, each answering one question, each with a fixed spread
definition and no scope creep once a sub-project starts. This document is
the plan; results for each sub-project are written up in their own log as
they complete.

## v4pre — integrated baseline

**Question.** Can v2's inventory skew and v3b's toxicity-adaptive spread
run in the same bot without corrupting each other? No new theory.

**Spread definition.** An earlier draft doubled up toxicity compensation by
setting the adaptive base spread to a fixed reference (h=1.08, which
already contains the time-averaged toxicity premium) and then adding
phi_hat on top. Corrected:

    center_t = S_t - k*q_t

    warm-up (fewer than N fills observed):
        h_t = h_warmup = 1.08

    after warm-up:
        h_t = 1/kappa + phi_hat_t * sigma * sqrt(2/pi)     (lambda = 1, locked)

1.08 is the warm-up fallback only. Once phi_hat is defined, the adaptive
base is 1/kappa = 1.00, not 1.08 — the fixed reference does not reappear
in the formula. Locked parameters: k=0.04, N=50, lambda=1.

**Implementation.** `IntegratedMaker` (strategy.py) — v3b's `AdaptiveMaker`
with the center shifted by -k*q. No new mechanism, no new estimator.

**Regression tests** (`sanity_v4.py`):

1. k=0 recovers v3b's AdaptiveMaker byte-identically.
2. Toxicity-width module disabled (h fixed) recovers v2's inventory-only
   maker byte-identically.
3. k=0 and toxicity disabled recovers a static maker byte-identically.
4. IntegratedMaker never reads phi_true.
5. markout at tick t affects only the quote at tick t+1, never t itself
   (no look-ahead, inherited from v3b's observe() timing).
6. Same seed, different k: S and delta_S must be byte-identical. Catches
   any strategy-dependent randomness leaking into the exogenous market
   path during integration.

**Graduation.** All six regressions pass; inventory RMS is well below
toxicity-only; spread widens in the toxic regime and narrows back; no PnL
catastrophe under default parameters; one baseline parameter set frozen.
This bot is kept permanently — even if the rest of v4 stalls, it stands on
its own.

## v4a — inventory x toxicity interaction

**Question.** Do inventory control and toxicity control complement each
other, or fight? High inventory wants to trade out fast; high toxicity
wants to trade less. Do these pull in opposite directions?

**Method.** Analytical first: derive how h*(phi) shifts under k>0 (skew
breaks bid/ask symmetry). Predicted order of magnitude ~O(k^2 * E[q^2]),
kept as a hypothesis — q is endogenous and feeds back into fills, so a
closed form may not fall out cleanly; if it doesn't, the experiment
proceeds anyway. Then a small factorial: {Fixed, Inventory-only,
Toxicity-only, Integrated} x {inventory pressure: q0=0 / q0=+Q} x
{toxicity: constant low / constant high / 0->1->0}. Metrics beyond PnL:
RMS inventory, max |inventory|, liquidation speed, fills, markout.

**Standing prediction, not a graduation gate.** k>0 should suppress the
uncontrolled inventory variance that kept v3b's Adaptive-Fixed CI crossing
zero, which may improve the statistical resolution of that comparison. This
is recorded as a hypothesis to check, not a target to hit — the graduation
criterion is quantifying the effect on SE and explaining it, not forcing
significance. Chasing significance by re-tuning k or adding seeds after
seeing a null result is exactly the practice v3b's own log argued against.

**Graduation.** A clear trade-off is documented: quantified interaction
between toxicity protection and liquidation speed, and the k>0 effect on
statistical resolution measured and explained either way. Integrated is not
required to win PnL in every cell.

## v4b.1 — online learning under random regime switching (k=0)

**Question.** The v3b schedule (0->1->0) was a fixed script. Can the maker
still track toxicity when it switches at random, unknown times? Does a
smarter estimator beat the plain rolling window?

**Method.** Environment: two-state Markov toxicity (own RNG stream, fixed
draw budget). Three estimators compared: rolling-N (v3b baseline), EWMA,
and a two-state HMM filter with a *calibrated* transition-rate prior
p_model (not the true p — the maker doesn't get to see the generating
process, only a belief about it). A misspecification test (p_model != p_true)
is run explicitly. p is not learned online — that is a separate estimation
problem, deliberately left alone. Kept at k=0 so the emission model stays
simple: with symmetric quotes, no-fill ticks carry no information about
phi (fill probability doesn't depend on phi), so the HMM only updates on
fills. This simplification is only valid at k=0 — see v4b.2.

**Graduation.** HMM posterior passes a calibration sanity check on
synthetic data. Either the HMM measurably beats rolling/EWMA on the
noise-lag frontier at some switching speed, or a clean null result is
recorded with the reason. Misspecified-p robustness characterized.

## v4b.2 — the same estimator, back inside k>0

**Question.** v4b.1's simplification (fills-only updates) assumed
symmetric quoting. Once center = S - k*q reappears, bid and ask sit at
different distances from fair value, so their fill probabilities differ.
Conditional on the direction of the last price move, "no fill this tick"
is no longer necessarily uninformative — a real gap in the v4b.1 design
worth checking rather than assuming away.

**Method.** Either correct the emission model to use the full observation
(price-move sign, trade/no-trade) rather than fills only, or explicitly
measure how much the fills-only approximation costs once k>0.

**Graduation.** The chosen estimator's attribution is clean under k>0: the
approximation error is quantified, or the corrected emission model is
implemented and validated.

## v4c — from estimation to decision

**Question.** Does the quote actually need phi_hat and sigma_hat as
separate estimated quantities, or is the observable markout itself already
the sufficient statistic?

**Motivating fact.** Away from the clip boundary, phi_hat = m_hat /
(sigma*sqrt(2/pi)) substituted into h = 1/kappa + phi_hat*sigma*sqrt(2/pi)
collapses to h = 1/kappa + m_hat — sigma cancels. sigma only re-enters
through the clip bounds, i.e. h = 1/kappa + clip(m_hat, 0, sigma*c). Low-
toxicity regimes sit near the (sigma-free) lower bound; only high-toxicity
regimes, sitting near the upper bound sigma*c, are where an estimated
sigma_hat can actually bite.

**Method.** Compare h_phi = 1/kappa + phi_hat*sigma_hat*sqrt(2/pi) against
the direct policy h_m = 1/kappa + clip(m_hat, 0, sigma_hat*c). Then test
estimated-sigma vs known-sigma, with the residual effect measured
separately by regime — predicted to concentrate in the high-toxicity
segment, where clipping actually engages sigma_hat.

**Graduation.** The cancellation is confirmed in simulation. The residual
effect of sigma estimation is quantified and, per the motivating fact,
expected to be small and regime-concentrated rather than a first-order
performance driver.

## v4d — robust tuning and generalization

**Question.** Does the final policy generalize past the one environment it
was built in?

**Method.** A small environment matrix (a handful of volatility x toxicity
x regime-persistence combinations, not a full factorial). Seeds split into
development / validation / held-out test — test seeds run exactly once,
after parameters are frozen, never re-tuned against. Coarse grid search
only (a handful of values per parameter: h0, k, memory, and lambda —
demoted from its own chapter to a v4d tuning dimension: h = 1/kappa +
lambda*m_hat, dev-set tuned, held-out validated). Stress tests on frozen
parameters: constant-low, constant-high, fast switching, slow switching,
high volatility, initial inventory shock.

Note for interpretation: if lambda* < 1 emerges, it may be partly
correcting the clip-boundary bias documented in v3b's Experiment 2 (low-
regime phi_hat has a positive bias) rather than purely shrinking noise —
these two explanations should be separated before writing up the finding.

**Graduation.** Parameters frozen before test-seed evaluation. No
catastrophic failure mode on unseen regimes. Final ablation table (Fixed ->
+inventory -> +toxicity -> +better estimator -> +shrinkage -> Oracle) shows
what each layer bought. The final policy explains in two minutes: inventory
controls the quote center; markouts estimate toxicity; toxicity controls
the width; the response to a noisy estimate is shrunk because the estimate
is noisy.

## Version tree

    v4pre  Integrated baseline
           inventory center-skew + toxicity spread, no new theory
                |
    v4a    Inventory x toxicity interaction
           do the two controls complement or conflict?
                |
    v4b.1  Online learning under random regime switching (k=0)
           rolling window vs EWMA vs HMM, unknown switch timing
                |
    v4b.2  Same estimator, k>0
           does asymmetric quoting make no-fill informative?
                |
    v4c    From estimation to decision
           does the policy need phi_hat and sigma_hat, or does
           markout alone suffice?
                |
    v4d    Robust tuning and generalization
           dev/validation/test split, frozen parameters, stress
           tests, final ablation
                |
              STOP

## Explicitly out of scope

RL, deep learning, general Bayesian/particle/Kalman filtering (v4b uses
only a two-state HMM with a calibrated prior, not a general filtering
framework), online learning of the Markov transition rate p, stochastic
volatility, price impact, queue position, full limit-order-book simulation,
multi-asset, options Greeks, latency modeling, order cancellation, multi-
level quoting.
