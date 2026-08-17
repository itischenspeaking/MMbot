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

**v0** — baseline fixed-spread maker with random fills; establishes the P&L decomposition. Inventory carries almost all variance, so longer runs do not improve risk-adjusted return. [Research Log](v0_log.md)

**v1** — adds quote-sensitive fills, \(p=Ae^{-\kappa d}\), and the spread-volume trade-off. Monte Carlo recovers \(h^*=1/\kappa\); \(A\) scales P&L without moving the optimum. [Research Log](v1_log.md)

**v2** — adds inventory skew, \(center=S-kq\). A small \(k\) sharply reduces inventory at modest mean-P&L cost; \(\sigma\) changes P&L risk but not the inventory path. [Research Log](v2_log.md)

**v3a** — adds direction-informed toxic flow and signed one-step markout. Markout scales with toxicity, and the optimal half-spread shifts by \(\phi\sigma\sqrt{2/\pi}\) as predicted. [Research Log](v3a_log.md)

**v3b** — hides time-varying toxicity and estimates it online from rolling filled markouts. \(N=50\) tracks regime changes in roughly \(N/2\) fills; adaptive quoting works, but its P&L benefit is small relative to inventory noise at \(k=0\). [Research Log](v3b_log.md)

**v4** — combines inventory and toxicity control, then tests interaction, simplification, and generalization. The [blueprint](v4_blueprint.md) splits the work into v4pre–v4d.

- **v4pre** — integrates inventory skew and toxicity-adaptive width into one maker. Inventory exposure falls sharply while the existing toxicity response survives integration. [Research Log](v4pre_log.md)
- **v4a** — tests whether inventory and toxicity controls interfere with each other. They are approximately modular; inventory control also cuts the standard error of the toxicity treatment effect by about 77%. [Research Log](v4a_log.md)
- **v4b** — replaces scripted toxicity with a hidden two-state Markov process. The same \(N=50\) estimator remains useful, with a clear finite-memory limit when regimes switch too quickly. [Research Log](v4b_log.md)
- **v4c** — rewrites the quoting decision directly in terms of estimated adverse markout. Removing the intermediate \(\phi\) estimate and its upper cap leaves economic behavior essentially unchanged. [Research Log](v4c_log.md)
- **v4d** — freezes the final policy and evaluates it on held-out seeds and pre-specified stress environments. The layered contributions reproduce on unseen data, and the frozen policy holds up across faster switching, higher volatility, and lower fill rates without re-tuning. [Research Log](v4d_log.md)
