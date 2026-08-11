# v2 — inventory-aware quoting

## 1. Model — inventory-aware quoting

v1's maker quoted symmetrically around S regardless of inventory. v2 shifts the
quote center against inventory:

    center = S - k*q
    bid = center - h
    ask = center + h

Long inventory (q > 0) pushes the center down: the ask sits closer to fair
value (fills more easily, we sell) and the bid sits further away (fills less
easily, we buy less). Short inventory does the opposite. k is the skew
strength; k = 0 recovers v1 exactly.

With v1's fill intensity, the expected inventory change per step is

    E[dq | q] = -2A * exp(-kappa*h) * sinh(kappa*k*q)

For small kappa*k*q this is approximately linear,

    E[dq | q] ~= -2*A*kappa*k*exp(-kappa*h) * q

i.e. mean reversion with a rate set by k. This is the prediction Experiment 1
tests.

Accounting note: quotes are no longer centered on S, so a fill's edge relative
to fair value is no longer a fixed h. Buying at the bid earns S - bid = h +
k*q; selling at the ask earns ask - S = h - k*q. `n_trades * h` is not
execution P&L in v2 — edge has to be computed per fill from bid/ask relative
to S.

## 2. Experiment 1 — Does inventory skew control inventory?

**Question.** Does increasing k actually control inventory, and what does that
control cost in P&L?

**Hypothesis.** Larger k should lower RMS inventory. A small positive k should
buy a large inventory reduction for a small P&L sacrifice. Pushing k further
should show diminishing returns on inventory control.

**Experiment.** k in {0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28},
h = 1.0, A = 0.4, kappa = 1.0, sigma = 0.1, 500 seeds, 2000 steps. RMS
inventory is sqrt(mean(inventory^2)) over each path, averaged across seeds.

**Data.**

| k | RMS inventory | mean P&L | mean/std |
|---|---|---|---|
| 0.00 | 14.17 | 590.49 | 7.27 |
| 0.01 | 6.14 | 588.16 | 16.10 |
| 0.02 | 4.49 | 586.82 | 19.47 |
| 0.04 | 3.22 | 583.91 | 21.74 |
| 0.08 | 2.30 | 578.42 | 23.56 |
| 0.16 | 1.63 | 567.43 | 25.32 |
| 0.32 | 1.16 | 543.37 | 26.22 |
| 0.64 | 0.83 | 488.35 | 25.71 |
| 1.28 | 0.60 | 335.84 | 22.11 |

k = 0.01 alone cuts RMS inventory by 57% (14.17 -> 6.14) at a 0.4% P&L cost
(590.49 -> 588.16). Beyond that, further RMS reduction slows while P&L cost
accelerates: k = 0.64 -> 1.28 buys only a 28% relative RMS drop (0.83 -> 0.60)
for a 31% P&L drop (488 -> 336).

**Conclusion.** Inventory skew establishes real mean reversion — RMS inventory
falls monotonically and steeply in k, consistent with the model's prediction.
The first small step in k is nearly free; further tightening shows diminishing
inventory returns at rising P&L cost. mean/std shows an interior peak around
k = 0.32 under these parameters — not claimed as a universal optimum, and not
yet explained. That's Experiment 3.

## 3. Experiment 2 — How does volatility change the value of inventory control?

**Question.** If price volatility changes, does the same skew strength remain
appropriate?

**Hypothesis.** Because S cancels out of the quote distances (S - bid = h +
k*q, ask - S = h - k*q), fill dynamics and RMS inventory should not depend on
sigma at fixed k. Higher sigma should raise the mark-to-market risk of holding
inventory, so mean/std should favor stronger skew as sigma rises.

**Experiment.** Same k grid, sigma in {0, 0.05, 0.10, 0.20, 0.40}, other
parameters unchanged.

**Data.** RMS inventory at k = 0.32 is 1.16 at every sigma tested — identical
to three decimal places across all five runs. mean P&L at k = 0 is flat across
sigma (588.8, 589.6, 590.5, 592.2, 595.6 — a 1.2% band). The best observed
mean/std region shifts with sigma:

| sigma | mean/std at k=0 | mean/std at k=0.32 | mean/std at k=0.64 |
|---|---|---|---|
| 0.00 | 25.16 | **27.11** | 26.14 |
| 0.05 | 13.14 | **26.87** | 26.04 |
| 0.10 | 7.27 | **26.22** | 25.71 |
| 0.20 | 3.74 | 24.03 | **24.44** |
| 0.40 | 1.89 | 18.78 | **20.72** |

**Conclusion.** Inventory dynamics are independent of sigma, as predicted —
RMS inventory at fixed k does not move. The best observed mean/std region does
shift toward stronger skew as sigma rises (peak near k = 0.32 for sigma <=
0.10, near k = 0.64 for sigma >= 0.20). This is an observed shift in the best
region tested, not a derived k*(sigma) relationship.

This raises the question the shift can't itself answer: at sigma = 0, where
inventory carries no mark-to-market risk at all, mean/std still peaks at a
positive k. Experiment 3 is that question.

## 4. Experiment 3 — Why is there a positive-k peak when sigma = 0?

**Question.** At sigma = 0 price never moves, so inventory has no
mark-to-market risk. Why does mean/std still peak at positive k?

The first move is not to explain this — it's to suspect the implementation.

### Sanity-check stage

Five checks, all passing:

- inventory P&L is exactly zero at sigma = 0, on every path
- fixing seed and k, changing sigma leaves fills and the inventory path
  byte-identical
- execution edge does not depend on sigma
- inventory P&L scales linearly in sigma (sigma=0.4 result equals 4x the
  sigma=0.1 result)
- terminal P&L reconstructs independently as execution + sum(q_t * dS_t),
  matching the accounting split computed from the fill log

All five hold. The positive-k peak at sigma = 0 is real model behavior, not an
implementation artifact.

### Attribution stage

At sigma = 0, terminal P&L splits exactly as

    P = A + B
    A = h * n_trades                    (flat-spread baseline)
    B = k * sum(q_pre_t * dq_t)          (inventory-skew correction)

with an independent telescoping check, B = (k/2)(q_T^2 - sum(dq_t^2)), that
agrees with the direct sum on every path.

**Why does mean P&L fall?** From k=0 to k=0.32, mean_A rises from 588.8 to
630.1 (+41.4) — skew does not reduce fill activity, it increases it (mean
single-sided fills rise from 502 to 543). But mean_B is -86.7. The correction
outweighs the extra fill income:

    dMean_P = dMean_A + mean_B = 41.4 - 86.7 = -45.3

Mean P&L falls because the maker increasingly gives up execution edge to
control inventory, not because fills disappear.

**Why does variance fall?** Decomposing dVar(P) = dVar(A) + Var(B) + 2Cov(A,B)
at k = 0.32:

| term | value | share of dVar(P) |
|---|---|---|
| dVar(A) | -40.9 | 28% |
| Var(B) | +9.9 | -7% |
| 2Cov(A,B) | -115.0 | 79% |
| dVar(P) | -146.0 | 100% |

B carries its own variance and adds to it, not away from it — this isn't
simply "B is more stable." The dominant term is the covariance: corr(A, B) is
-0.81 at k = 0.32. Paths with more fill activity (higher A) accumulate more
inventory over the path, and under skew that inventory draws a larger negative
correction (more negative B). Paths with less fill activity accumulate less
inventory and draw a smaller correction. High-A paths get pulled down more,
low-A paths get pulled down less — this negative co-movement narrows the
spread of terminal P&L across paths, even though neither leg is individually
much steadier.

Past the peak, this reverses: dVar(A) turns positive at k = 0.64 (+38.4) and
grows sharply by k = 1.28 (+182.4) — strong skew swings quotes far enough to
destabilize the fill process itself, on top of the accelerating mean loss.

### Final explanation of the sigma = 0 peak

Inventory carries no price risk at sigma = 0. The positive-k peak is not an
inventory-price-risk effect. It comes from a different mechanism: inventory
skew changes the execution process itself, and the negative correlation it
induces between baseline fill income and the skew correction compresses the
cross-path dispersion of execution P&L. mean/std rewards that compression.

Both mean and std fall monotonically as k rises. At moderate k, the relative
improvement in std outweighs the relative loss in mean, so mean/std rises. At
strong k, the execution edge given up to keep compressing dispersion grows
faster than the dispersion reduction itself — reinforced by the fill process
becoming less stable in its own right — so mean falls faster than std and
mean/std turns over. That is the interior peak.

## V2 takeaway and v3

v2 establishes inventory-aware quote skew, verifies that it produces
mean-reverting inventory, and quantifies the trade-off between inventory
control and execution profitability. It also finds that inventory skew does
more than manage inventory exposure — it reshapes the execution P&L
distribution itself, which is what drives the sigma = 0 peak.

In v2, quotes move only because the maker dislikes its own inventory; order
flow itself is still uninformed. v3 introduces informed/toxic flow, asking
whether fills carry information about future fair value and whether the maker
should update its quoting off order-flow information rather than inventory
alone.
