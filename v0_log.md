# v0 — fixed-spread market maker

The maker quotes S ± h every step and ignores everything else: its inventory,
the price history, how wide it is quoting. A trader arrives with probability p
and picks a side by coin flip, so fills don't depend on the quote at all.

500 seeds, 2000 steps, S0 = 100, base case h = 0.5, σ = 0.1, p = 0.3. P&L is
split into spread income (trades × h) and the remainder, which is inventory
marked to market.

## Runs

| run | spread mean/sd | inventory mean/sd | total mean/sd | mean/sd |
|---|---|---|---|---|
| base | 300.5 / 9.9 | 6.2 / 84.2 | 306.7 / 84.7 | 3.62 |
| 8000 seeds | 300.1 / 10.2 | −1.5 / 77.3 | 298.6 / 78.0 | 3.83 |
| h = 2.0 | 1202.0 / 39.7 | 6.2 / 84.2 | 1208.2 / 92.9 | 13.01 |
| σ = 0.3 | 300.5 / 9.9 | 18.5 / 252.7 | 319.0 / 252.8 | 1.26 |
| σ = 0 | 300.5 / 9.9 | 0 / 0 | 300.5 / 9.9 | 30.27 |
| p = 0.6 | 600.2 / 10.7 | 3.6 / 113.3 | 603.7 / 113.4 | 5.33 |

Market and flow draw from separate seed streams, so rows that differ only in h
or σ face the identical fill sequence. That's why inventory is byte-identical
across the base, h = 2.0 and σ = 0.3 rows, and why σ = 0.3 lands on exactly
three times the base inventory sd rather than approximately three times.

## What the numbers say

Everything matches closed form. Spread income is p·n·h = 300. Final inventory
is a ±1 walk over p·n fills, so its sd is √600 = 24.5. Inventory P&L has sd
σ√(p/2)·n = 77.5. And

    mean/sd = h·√(2p) / σ = 3.87

which the 8000-seed run recovers to 3.83; the 500-seed runs sit within
sampling error of it.

Inventory earns nothing and carries nearly all the risk — 6 of 307 in mean,
98% of the variance. It random walks, and nothing in v0 pulls it back.

Note what is missing from that expression: n. Spread income and inventory risk
both scale linearly in time, so they cancel. Running the bot four times longer
leaves the risk-adjusted return where it started. This is not how a strategy
normally behaves — usually return grows like n and risk like √n, so Sharpe
improves with √n. Here it doesn't, because inventory has no mean reversion.

## Scaling

| change | effect | why |
|---|---|---|
| more seeds | tighter estimates | Monte Carlo convergence |
| h ↑ | P&L scales with h, risk unchanged | fills don't depend on the quote |
| σ ↑ | inventory risk ∝ σ | it's q·ΔS |
| σ = 0 | inventory risk vanishes | ΔS = 0 |
| p ↑ | income ∝ p, risk ∝ √p | more uninformed flow |

So v0 reduces to

    free spread income − random inventory risk

and since income grows like p while risk grows like √p, more random flow makes
the business unconditionally better. That holds only because every trader here
is a coin flip. Nobody knows anything the maker doesn't.

## Why v0 doesn't work

Two things are wrong, and they compound.

**Quoting wider is free.** The h = 2.0 row quadruples P&L at unchanged
inventory risk. Push h further and mean/sd keeps climbing, saturating at
√(pn/(1−p)) ≈ 29, which the σ = 0 row confirms at 30.3. A maker quoting
S ± 1000 would post the best risk-adjusted return in the book and never trade
in reality. There is no width decision here at all, because nothing punishes
being wide.

**Inventory is unbounded.** With quotes fixed at S ± h the maker has no way to
signal that it is long and wants to sell. Position drifts wherever the coin
flips take it. At the end of the base run it sits around 25 units against a
per-trade edge of 0.5 — one price tick moves the book by more than five fills
earn.

The second problem is the one worth fixing, but the first has to go first: a
skew has no effect while fills ignore the quote.

## v1

Fill intensity decays with distance from the true price, λ = A·exp(−κδ).
Quoting wider then costs volume, so an optimal width exists. Only after that
does inventory skew do anything.
