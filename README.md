# MMbot

A minimal market-making simulator, built to run experiments on. Each version
adds one mechanism and the log records what changed and why the previous
version needed it.

## Layout

    market.py       the true price
    strategy.py     how the maker quotes
    simulator.py    order flow, account, the loop
    experiments.py  sweeps

Market and flow draw from separate seed streams, so two runs differing only in
a strategy parameter face the same price path.

## Running

    pip install -r requirements.txt
    python test_invariants.py
    python experiments.py

Edit the calls at the bottom of `experiments.py` to change what gets swept.

## Versions

**v0** — quotes S ± h and ignores everything else. Fills arrive at random and
don't depend on the quote. Spread income is nearly deterministic; inventory
random walks, earns nothing, and carries 98% of the variance. Both scale
linearly in time, so risk-adjusted return doesn't improve however long the bot
runs. [Log](v0_log.md)

**v1** — fill probability decays with distance from fair value,
p = A·exp(-kappa·distance), so quoting wider costs volume. An optimal
half-spread appears at h* = 1/kappa, recovered by Monte Carlo across kappa; flow
intensity A scales P&L without moving the optimum. The risk-adjusted best sits
wider than the mean-P&L best. [Log](v1_log.md)

**v2** — the quote center shifts against inventory, center = S − k·q, which
only works now that flow responds to the quote (v0's flow couldn't feel it).
RMS inventory falls steeply and cheaply at first — one small k cuts it 57% for
a 0.4% P&L cost — with diminishing inventory returns and rising P&L cost as k
grows further. RMS inventory at fixed k is exactly independent of sigma, as
the model predicts; the best observed mean/std region shifts toward stronger
skew as sigma rises. At sigma = 0, where inventory carries no price risk at
all, mean/std still peaks at positive k — traced by exact P&L attribution to a
negative correlation skew induces between baseline fill income and its own
correction, not to any inventory-price effect. [Log](v2_log.md)

**v3a** — flow becomes single-trader-per-tick; with probability phi the trader
sees sign(delta_S) and trades in that direction, otherwise picks a side by
coin flip (fill probability unchanged either way). This decouples toxicity
from volume — fill count stays flat across phi — so P&L changes trace cleanly
to adverse selection. Signed one-step markout scales linearly in phi and
matches sigma·√(2/π) to four significant figures at phi = 1. Inventory skew
(v2) cuts RMS inventory by 88% without moving markout at all — inventory
control and adverse-selection control are different problems. A coarse h
sweep found no shift in the optimal spread; the predicted shift turned out to
be ~0.08, under the grid spacing, so a fine grid with a quadratic peak fit was
needed to see it — it tracks h*(phi) ≈ 1/κ + phi·σ·√(2/π) within 0.013 across
six phi values. [Log](v3a_log.md)

**v3b** — in progress. v3a takes toxicity as known and fixed. v3b estimates it
from observed fills and markouts and adapts quoting online.
