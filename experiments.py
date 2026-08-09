"""Experiment harness. Edit the calls at the bottom and run."""

import numpy as np

from market import RandomWalk
from simulator import PoissonFlow, run
from strategy import NaiveMaker


def sweep(n_seeds=500, n_steps=2000, sigma=0.1, half_spread=0.5, trade_prob=0.3):
    """Run many seeds, split the P&L into spread and inventory."""
    rows = []
    for s in range(n_seeds):
        r = run(
            RandomWalk(sigma=sigma),
            NaiveMaker(half_spread=half_spread),
            PoissonFlow(trade_prob=trade_prob),
            n_steps=n_steps,
            seed=s,
        )
        spread = r["n_trades"] * half_spread
        total = r["pnl"][-1]
        rows.append((spread, total - spread, total, r["inventory"][-1]))
    return np.array(rows)


def report(a, label=""):
    if label:
        print(label)
    for i, name in enumerate(["spread", "inventory", "total", "final_inv"]):
        print(f"  {name:10s} mean {a[:, i].mean():9.2f}   std {a[:, i].std():9.2f}")
    print(f"  mean/std of total: {a[:, 2].mean() / a[:, 2].std():.3f}\n")


if __name__ == "__main__":
    report(sweep(), "base")
