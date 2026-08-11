# v1 — quote-sensitive flow

v0's flow ignored the quote, so wider was always better. v1 replaces it with a
fill intensity that decays with distance from the true price,

    p_fill(delta) = min(1, A * exp(-kappa * delta))

drawn independently on each side. Quoting wider now costs volume, so a real
trade-off appears. Everything else is unchanged: symmetric quotes, random walk,
500 seeds, 2000 steps.

The maker still doesn't react to its inventory — that's v2. v1 only asks how
wide to quote.

## Optimal width, closed form

Per side per step, expected edge is delta * A * exp(-kappa * delta), maximised at

    h* = 1 / kappa

independent of A. Peak total income over the run is 2nA / (kappa * e).

## Base case (A = 0.4, kappa = 1.0)

| h | fills | mean | mean/sd |
|---|---|---|---|
| 0.25 | 1247.7 | 314.5 | 3.41 |
| 0.50 | 971.2 | 490.3 | 5.64 |
| 0.75 | 755.5 | 570.8 | 6.82 |
| **1.00** | 588.8 | **590.5** | 7.27 |
| 1.50 | 356.7 | 538.8 | **8.03** |
| 2.00 | 215.5 | 433.8 | 8.01 |
| 3.00 | 78.9 | 236.3 | 6.18 |

Mean P&L peaks at h = 1.00 = 1/kappa, and the closed form 2nA/(kappa·e) = 589
matches the 590.5 there. Note the mean/sd peak sits wider, at h ≈ 1.5.

## h* tracks 1/kappa (A fixed at 0.4)

| kappa | 1/kappa | mean-P&L peak |
|---|---|---|
| 0.5 | 2.00 | h = 2.00 |
| 1.0 | 1.00 | h = 1.00 |
| 2.0 | 0.50 | h = 0.50 |

Fill count depends only on the product kappa·h, so e.g. (kappa=0.5, h=1.0) and
(kappa=1.0, h=0.5) both give 971.2 fills on the same paths — a consequence of
the shared price stream.

## A scales height, not location (kappa fixed at 1.0)

Value at h = 1.0:

| A | fills | mean | mean/sd |
|---|---|---|---|
| 0.1 | 146.1 | 147.3 | 3.43 |
| 0.2 | 293.7 | 296.2 | 5.17 |
| 0.4 | 588.8 | 590.5 | 7.27 |
| 0.6 | 883.7 | 888.4 | 9.92 |

Mean is exactly linear in A (147 → 296 → 590 → 888, i.e. ×2, ×4, ×6) and the
peak stays at h = 1.0 throughout. mean/sd rises with A but a touch faster than
√A (2.89× from A=0.1 to 0.6, against √6 = 2.45); the √A guess ignores the
variance of spread income itself, which also grows with fill count.

## Conclusions

1. **Quote elasticity creates a finite optimum.** Once fills fall off with
   distance, wide quotes stop earning without bound. Spread capture and volume
   trade off against each other, and an interior best width exists.

2. **Price sensitivity sets the optimal spread.** h* = 1/kappa. Monte Carlo
   recovers h = 2, 1, 0.5 at kappa = 0.5, 1, 2.

3. **Flow intensity scales the opportunity, not its location.** Raising A lifts
   fills and expected P&L proportionally but leaves h* essentially fixed.

4. **Profit and risk-adjusted optima differ.** A wider spread can lower expected
   P&L yet also cuts fills and inventory swings, so mean/sd peaks wider than the
   mean-P&L spread does.

## v2

Inventory still random walks — symmetric quotes never pull it back. v2 skews
quotes by inventory, which only bites now that flow responds to the quote.
