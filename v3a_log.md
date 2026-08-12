# v3a — direction-informed flow

## 1. Model / Theory — direction-informed flow

v3a changes flow to single-trader-per-tick: at most one fill per step, versus
v1/v2's two independent sides.

Price is still exogenous, dS_t = sigma * eps_t. Order within a tick:

1. generate the tick's delta_S (don't apply it yet)
2. maker quotes on current S
3. a trader arrives
4. with probability phi, the trader is direction-informed: sees sign(delta_S)
   and picks the side that profits from it (buy if delta_S > 0, sell if < 0)
5. with probability 1-phi, the trader is uninformed: picks a side by coin flip
6. whichever side is picked, whether it actually fills still uses the v1 quote-
   sensitive probability, A * exp(-kappa * delta)
7. delta_S is applied, moving the price

    phi = P(trader knows the sign of the next price move)

phi = 1 does not mean the trader knows the future price, and it does not mean
the trader checks whether future fair value crosses the quote — this is a
deliberately minimal direction-informed flow model, sign only.

Because flow changed to single-trader-per-tick, phi = 0 is the uninformed
baseline but is not built to byte-for-byte recover v2's two-independent-sides
flow.

**Prediction 1 — signed markout.** For a fully informed fill, the expected
one-step move in the trader's favor is E[|delta_S|] = sigma*sqrt(2/pi). For
mixed flow:

    m(phi) ~= phi * sigma * sqrt(2/pi)

**Prediction 2 — optimal spread.** Approximating each fill's expected edge as
h - m(phi), with fill intensity still decaying as exp(-kappa*h), the resulting
optimum is

    h*(phi) ~= 1/kappa + phi * sigma * sqrt(2/pi)

This is an analytical approximation, tested directly in Experiment 3.

## 2. Experiment 1 — does direction-informed flow create measurable toxicity?

**Question.** Does raising phi make flow direction predictive of the next
price move, without mechanically changing fill volume?

**Hypothesis.** Signed markout rises with phi, tracking m(phi) ~= phi *
sigma * sqrt(2/pi). Fill count stays roughly flat across phi. Mean terminal
P&L falls as toxicity rises.

**Method.** Fixed h = 1.0, k = 0.04, sigma = 0.1, sweeping phi.

**Results.**

| phi | fills | markout | mean P&L |
|---|---|---|---|
| 0.0 | 296.8 | 0.00014 | 291.21 |
| 0.2 | 296.7 | 0.01601 | 286.40 |
| 0.5 | 296.8 | 0.03975 | 279.84 |
| 1.0 | 296.4 | 0.07978 | 267.15 |

Fill count is flat (296.4–296.9) across the full phi range. Markout rises
roughly linearly in phi. At phi = 1, sigma * sqrt(2/pi) = 0.07979 against an
observed markout of 0.07978 — matching to four significant figures.

**Conclusion.** phi changes the information content of flow without
mechanically changing fill volume. One-step signed markout accurately
measures the adverse-selection mechanism the model introduces.

## 3. Experiment 2 — can inventory control remove toxicity?

**Question.** v2's inventory skew controls inventory exposure. Does raising k
also reduce the adverse-selection cost measured by markout?

**Hypothesis.** RMS inventory falls with k. Signed markout stays roughly
constant, because k manages the maker's own position while phi determines the
information content of customer flow — different objects.

**Method.** Fixed phi = 0.5, h = 1.0, sweeping k.

**Results.**

| k | RMS inventory | markout | mean P&L |
|---|---|---|---|
| 0.000 | 10.67 | 0.03975 | 284.23 |
| 0.040 | 3.37 | 0.03975 | 279.84 |
| 0.160 | 1.75 | 0.03981 | 269.91 |
| 0.320 | 1.24 | 0.04005 | 255.61 |

RMS inventory falls by 88% across the sweep. Markout is flat at 0.0398 ±
0.0003 throughout — no trend with k. (Mean P&L continues to fall with k, as
in v2, for the same execution-side reasons; that isn't this experiment's
question.)

**Conclusion.** Inventory control is not adverse-selection control. Stronger
skew manages position size but does not make informed counterparties any less
informed.

## 4. Experiment 3 — how much spread compensation does toxicity require?

**Question.** If every toxic fill costs the maker expected edge, should the
maker demand a wider spread as phi rises?

**Hypothesis.** h*(phi) ~= 1/kappa + phi * sigma * sqrt(2/pi), so h* should
rise with phi.

**Method.** First pass used the coarse h grid from Experiment 1's design
(spacing 0.25). Raw best-h stayed at h = 1.0 for every phi tested — not
treated as a hypothesis failure, because the predicted shift at these
parameters is only about 0.08, well under the grid spacing. The grid could not
have resolved a shift of that size either way.

A fine grid was run instead: h in [0.90, 1.20] at 0.025 steps, 1000 seeds per
point, phi in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. For each phi: mean P&L at every
h, the raw argmax, and a quadratic fit through the points near the peak, whose
vertex gives a sub-grid estimate of h* (not a claim of exact recovery — the
curve is close to flat near its top, and the vertex is a smoothing device
against Monte Carlo noise at that resolution).

**Results.**

| phi | theory h* | fitted h* |
|---|---|---|
| 0.0 | 1.000 | 1.009 |
| 0.2 | 1.016 | 1.017 |
| 0.4 | 1.032 | 1.029 |
| 0.6 | 1.048 | 1.035 |
| 0.8 | 1.064 | 1.058 |
| 1.0 | 1.080 | 1.073 |

Raw argmax bounces between 1.000, 1.025, and 1.050 across phi — the P&L curve
is nearly flat near its top, and standard error at the max (~1.7–1.8 per
seed-average) is large relative to the spacing between neighboring h. The
fitted vertex is not: it rises monotonically and closely tracks theory across
the full range, with all six points within 0.013 of the analytical
prediction.

**Conclusion.** The fitted peak shifts right with phi, and the magnitude and
slope of that shift are strongly consistent with h*(phi) ~= 1/kappa +
phi*sigma*sqrt(2/pi). Higher adverse-selection toxicity requires a wider
spread to preserve expected execution profitability.

## 5. Conclusion + next step

v3a establishes direction-informed flow as a minimal adverse-selection model.
Signed one-step markout scales with phi as analytically predicted. Inventory
skew controls inventory but does not remove toxicity — they are different
problems. Optimal static spread widens as toxicity rises, with the observed
h* shift closely matching the analytical approximation.

v3a assumes toxicity is known and static. The next step is to estimate
toxicity from observed fills/markouts and adapt quoting online.
