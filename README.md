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

**v1** — in progress. Fill intensity decays with quote distance, so width
starts to cost volume.
