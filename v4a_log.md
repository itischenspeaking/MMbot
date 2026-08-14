# v4a — inventory x toxicity interaction

v4pre showed inventory skew and toxicity adaptation can be combined
operationally. v4a asks whether they actually interfere with each other —
in pricing, in estimation, and in economics. No parameter tuning, no new
controller. Frozen throughout:

    k = 0.04, N = 50, sigma = 0.3, kappa = 1, h_warmup = 1.08, lambda = 1

Architecture recap: `center_t = S_t - k*q_t` controls inventory,
`h_t = 1/kappa + phi_hat_t*sigma*sqrt(2/pi)` controls toxicity-dependent
width. One moves the center, the other the width — but that doesn't
automatically mean they're independent: inventory skew changes the
bid/ask distances and fill probabilities, which could shift the optimal
spread, change what the estimator sees, and ultimately change the
economic value of toxicity adaptation. Three checks, in order:
**pricing -> estimation -> economics**.

## Predictions

- **P1.** Inventory skew's effect on optimal spread should be mostly
  second-order, not a large first-order O(kq) correction.
- **P2.** The fill-based toxicity estimator should stay roughly calibrated
  at k=0.04.
- **P3.** Inventory control should sharply cut inventory risk, possibly at
  the cost of some mean PnL.
- **P4.** Inventory control may lower the paired SE of the toxicity
  treatment effect: SE(IT-I) < SE(T-F). This predicts an SE reduction
  only — not that either effect becomes significant, and not a target to
  chase by re-tuning if it doesn't happen.

## Experiment 1 — pricing interaction

**Question.** Once inventory skew is on, does v3's toxicity-width rule
h*(phi) = 1/kappa + phi*sigma*sqrt(2/pi) still hold?

**Theory.** Let a = k*q, c = phi*sigma*sqrt(2/pi). Skew makes the two
quote distances asymmetric: d_ask = h - a, d_bid = h + a, with fill
probabilities p_ask = A*exp(-kappa*(h-a)) and p_bid = A*exp(-kappa*(h+a)).
The conditional expected local edge (execution edge net of that tick's
markout, given q) works out to:

    Pi(h | q) = A*exp(-kappa*h) * [ (h-c)*cosh(kappa*a) - a*sinh(kappa*a) ]

First-order condition gives:

    h*(phi, k, q) = 1/kappa + c + k*q*tanh(kappa*k*q)

so the correction is Delta h* = k*q*tanh(kappa*k*q) ≈ kappa*k^2*q^2 for
small k*q. The bid/ask symmetry cancels the first-order O(kq) term
entirely; what survives is second-order. (An earlier pass at this
derivation mistakenly canceled the correction completely by conflating
each leg's execution edge with h itself, missing that the edge is h±a,
not h — worth noting since the near-miss is instructive: the leading
term really does vanish, just not all of them.)

**Experiment 1A — fixed inventory.** sigma=.3, A=.4, kappa=1, k=.04,
q in {0, 5, 10}, phi in {0, .5, 1}, 4,000,000 samples per point. Objective
is one-step local edge (execution_edge - signed_flow*delta_S) at a fixed,
exogenous q — not a simulated path — so the empirical object matches the
conditional analytic Pi(h|q) exactly. A focused grid with common random
numbers across h, plus a quadratic fit to the peak, locates h* to
sub-grid precision (the peak is flat, so raw grid argmax isn't precise
enough on its own).

Residuals between empirical and analytic h* sat within about ±0.011
across all nine cells — consistent with Monte Carlo noise, not a
systematic miss. q=0 recovered the v3 rule exactly. q>0 produced the
predicted positive correction, symmetric in sign of q. Examples:

    phi=.5, q=5:   h*_v3=1.1197  correction=0.0395  empirical=1.1589
    phi=.5, q=10:  h*_v3=1.1197  correction=0.1520  empirical=1.2754

At q=10 the exact correction (k*q*tanh(kappa*k*q)=0.1520) and the
small-kq approximation (kappa*k^2*q^2=0.1600) differ by 0.008 — close to
the MC residual scale, so this experiment doesn't sharply distinguish the
two; it confirms the correction exists and is second-order, not which
exact functional form is right beyond that.

**Experiment 1B — endogenous inventory.** Let q evolve on real paths:
sigma=.3, n_steps=3000, n_seeds=200, k in {0, .04, .08}, phi in {0, .5,
1}, same local-edge objective, now summed over the path and fit with a
spread grid.

    k=0:     fitted h* matches v3 theory within ~.005
    k=.04:   fitted h* - v3 theory ≈ .014 to .032
    k=.08:   fitted h* - v3 theory ≈ .028 to .045

At the frozen k=.04 baseline, RMS inventory is about 3.4; the rough
second-order prediction kappa*k^2*E[q^2] ≈ .018 lands in the same range
as the observed correction.

**Takeaway.** Inventory skew does shift the optimal width, but the effect
is second-order and small at k=.04 (roughly 1.5-3% of h*). v4a does not
modify v4pre's spread rule or add an inventory-dependent width
correction.

## Experiment 2 — estimation interaction

**Question.** Skew makes ask and bid fill probabilities unequal. Does
that introduce a selection effect that contaminates the fill-based
toxicity estimator?

**Prediction.** At fixed q, let p_ask, p_bid be the two fill
probabilities. In this flow model fill probability depends on quote side
and distance but not on the size of the price move, so the filled-markout
numerator is (phi*E|delta_S|/2)*(p_ask+p_bid) against a fill probability
of (p_ask+p_bid)/2 — the asymmetry is in both numerator and denominator
and cancels:

    E[markout | fill] = phi * E|delta_S| = phi * sigma * sqrt(2/pi)

regardless of k. Skew may change *when* fills arrive, but not what each
fill reveals.

**Experiment 2A — constant-phi calibration.** sigma=.3, A=.4, kappa=1,
N=50, phi in {0, .5, 1}, k in {0, .04}, n_steps=3000, n_seeds=500. Each
phi quoted at its own h*(phi); estimator entirely offline; rolling
statistics computed at fill ticks only (no-fill carry-forward values
excluded, or repeated values would understate noise).

Same-seed paired Delta_raw_phi = raw_phi(k=.04) - raw_phi(k=0):

    phi=0:   raw_phi .0014 -> .0018   Delta=+.00036  95%CI=[-.00140,+.00211]
    phi=.5:  raw_phi .4988 -> .4995   Delta=+.00065  95%CI=[-.00104,+.00234]
    phi=1:   raw_phi 1.0008 -> 1.0004 Delta=-.00033  95%CI=[-.00150,+.00084]

All three CIs contain 0. RMS inventory fell from ~12-13 to ~3.4 across
the same comparison (skew is doing its job), while the rolling
estimator's bias/RMSE/noise were essentially unchanged between k=0 and
k=.04. The positive bias at phi=0 and negative bias at phi=1 are the
clip-boundary effect already documented in v3b — present at both k
values, not introduced by skew.

**Experiment 2B — closed loop.** Confirm the estimator still tracks once
it's feeding back into live quoting: phi schedule 0->1->0, seg_len=1500,
N=50, n_seeds=300, comparing AdaptiveMaker (k=0) against IntegratedMaker
(k=.04).

    response lag (fills):  up 23.8 vs 23.9   down 24.9 vs 25.1
    response lag (ticks):  up 171 vs 168      down 202 vs 202
    steady-state RMSE:     low1 .1097 vs .1086   high .0592 vs .0601   low2 .1067 vs .1065

Both strategies land on essentially the same numbers. Lag stays close to
the predicted N/2 fills.

**Takeaway.** Inventory skew materially changes the inventory process but
does not materially contaminate the fill-based toxicity estimator in this
flow model. v3b's estimator carries over to the integrated maker
unmodified.

## Experiment 3 — economic interaction

The first two experiments ruled out the two main mechanism-level
concerns: pricing interaction exists but is small, and estimator
contamination doesn't appear. Experiment 3 is the direct economic
question.

**Design.** Full 2x2, same-seed CRN, fresh strategy and flow instances
per seed:

    F  (Fixed):          k=0,    h=1.08 fixed
    I  (Inventory-only):  k=.04,  h=1.08 fixed
    T  (Toxicity-only):   k=0,    adaptive width
    IT (Integrated):      k=.04,  adaptive width

Environment: sigma=.3, A=.4, kappa=1, phi 0->1->0, seg_len=1500,
n_seeds=3000.

    strat   mean_pnl  std_pnl  p05_pnl  rms_inv  max|q|
    F        617.19    346.86    53.91   15.436   30.61
    I        604.81     73.62   478.64    3.475   10.10
    T        619.86    346.84    43.05   15.434   30.52
    IT       606.92     73.23   484.42    3.474   10.06

**Inventory effect.** I-F = -12.380, 95%CI [-24.106, -0.654]. IT-T =
-12.940, 95%CI [-24.675, -1.205]. RMS inventory drops ~77.5% (15.4 -> 3.47)
and terminal-PnL std drops from ~347 to ~73; p05 PnL rises from ~43-54 to
~479-484. Inventory skew trades a modest amount of mean PnL for a large
cut in inventory exposure and downside dispersion — whether that's a good
trade depends on the maker's risk objective, not a fact this experiment
settles.

**Toxicity effect.** Without inventory control: T-F = +2.672, SE=1.730,
95%CI [-0.719, +6.064] — crosses zero. With inventory control: IT-I =
+2.113, SE=0.396, 95%CI [+1.336, +2.889] — clearly resolved. The point
estimates are close (2.672 vs 2.113); what changed is the noise, not the
effect. Markout-per-tick tells the same story at the microstructure
level:

    F .01081   T .00957   (toxicity adaptation cuts exposure ~.00125/tick)
    I .01092   IT .00967  (same cut, with inventory control on)

and the adaptive width by regime is nearly identical whether inventory
control is on or off (T: 1.0326/1.2025/1.0453, IT: 1.0325/1.2027/1.0452)
— inventory control doesn't materially change what the toxicity
controller does.

**Factorial interaction.** Gamma = PnL_IT - PnL_I - PnL_T + PnL_F,
computed seed-by-seed. Gamma mean = -0.560, SE=1.739, 95%CI [-3.968,
+2.848] — contains zero. No evidence of a material positive or negative
PnL interaction; the two controls are approximately additive, not
strictly orthogonal (orthogonality is a mechanism-level claim Experiments
1-2 support; additivity is what the PnL-level data shows).

**Variance reduction (P4).** This is the most important result in v4a.
SE(IT-I) / SE(T-F) = 0.396 / 1.730 = **0.229** — a ~77% drop in paired
SE, with the effect size essentially unchanged (2.672 -> 2.113). Terminal
PnL contains a sum-of-inventory-times-price-move term; at k=0 large,
uncontrolled inventory makes that term swing widely, adding noise
unrelated to the toxicity treatment. Skew suppresses q, and with it that
noise term, which is what moves the toxicity effect from *positive but
statistically unresolved* to *positive and clearly resolved*. Inventory
control doesn't amplify the toxicity effect — it makes the effect easier
to measure by suppressing inventory-driven PnL variance. This result is
specific to the simulated environment here, not a general claim.

## Conclusion

**Pricing.** Inventory skew shifts optimal width by a small second-order
term, Delta h* = k*q*tanh(kappa*k*q) ≈ kappa*k^2*q^2. Not worth
correcting for at the frozen k=.04 baseline.

**Estimation.** The fill-based markout estimator is not materially
contaminated by inventory skew in this flow model — skew changes fill
timing, not what each fill reveals.

**Economics.** The two controllers are approximately additive at the PnL
level (gamma ≈ 0, not statistically resolved either direction).

**Risk and inference.** Inventory control gives up a small amount of mean
PnL for a large reduction in inventory and downside risk, and — the
headline result — cuts the toxicity treatment effect's standard error by
about 77%, resolving the ambiguity v3b left behind without changing the
size of the effect.

v4a studied this interaction under a deterministic 0->1->0 schedule. The
next step replaces that scripted schedule with genuinely random hidden
toxicity and studies online state estimation under it.
