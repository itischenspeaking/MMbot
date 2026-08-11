"""Experiment harness. Edit the calls at the bottom and run."""

import numpy as np

from market import RandomWalk
from simulator import BernoulliFlow, QuoteSensitiveFlow, run
from strategy import NaiveMaker, InventorySkewMaker


def sweep(n_seeds=500, n_steps=2000, sigma=0.1, half_spread=0.5, trade_prob=0.3):
    """v0: random flow that ignores the quote. Split P&L into spread + inventory."""
    rows = []
    for s in range(n_seeds):
        r = run(
            RandomWalk(sigma=sigma),
            NaiveMaker(half_spread=half_spread),
            BernoulliFlow(trade_prob=trade_prob),
            n_steps=n_steps,
            seed=s,
        )
        spread = r["n_trades"] * half_spread
        total = r["pnl"][-1]
        rows.append((spread, total - spread, total, r["inventory"][-1]))
    return np.array(rows)


def sweep_v1(n_seeds=500, n_steps=2000, sigma=0.1, half_spread=0.5, A=0.4, kappa=1.0):
    """v1: quote-sensitive flow. Same split, but fills now depend on half_spread."""
    rows = []
    for s in range(n_seeds):
        r = run(
            RandomWalk(sigma=sigma),
            NaiveMaker(half_spread=half_spread),
            QuoteSensitiveFlow(A=A, kappa=kappa),
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


def width_scan(hs, A=0.4, kappa=1.0, **kw):
    """For each half_spread h, report fill rate, mean total, mean/std.

    Analytic guides (per side, per step): expected edge is h * A * exp(-kappa*h),
    maximised at h = 1/kappa. Risk-adjusted return peaks somewhat wider.
    """
    print(f"A={A}, kappa={kappa}   (edge peaks at h=1/kappa={1/kappa:.2f})")
    print(f"  {'h':>5} {'fills':>8} {'mean':>9} {'mean/std':>9}")
    for h in hs:
        a = sweep_v1(half_spread=h, A=A, kappa=kappa, **kw)
        n_fills = a[:, 0].mean() / h  # spread income = fills * h, so back it out
        print(
            f"  {h:5.2f} {n_fills:8.1f} {a[:, 2].mean():9.2f} "
            f"{a[:, 2].mean() / a[:, 2].std():9.3f}"
        )

def sweep_v2(n_seeds=500, n_steps=2000, sigma=0.1, half_spread=1.0,
             A=0.4, kappa=1.0, k=0.04):
    """v2: quote center shifts against inventory."""
    rows = []
    for s in range(n_seeds):
        r = run(
            RandomWalk(sigma=sigma),
            InventorySkewMaker(half_spread=half_spread, k=k),
            QuoteSensitiveFlow(A=A, kappa=kappa),
            n_steps=n_steps,
            seed=s,
        )
        # Execution edge computed per fill from the log, since with a skewed
        # center it no longer equals h. This is definitional and works for
        # any maker.
        execution = np.sum(
            r["buys"] * (r["S"] - r["bid"])
            + r["sells"] * (r["ask"] - r["S"])
        )
        total = r["pnl"][-1]
        inventory_pnl = total - execution
        # RMS, not std: we care how far inventory sits from zero, not how much
        # it wanders around its own mean.
        rms_inventory = np.sqrt(np.mean(r["inventory"] ** 2))
        rows.append((execution, inventory_pnl, total,
                     r["inventory"][-1], rms_inventory))
    return np.array(rows)


def k_scan(ks, half_spread=1.0, A=0.4, kappa=1.0, sigma=0.1, **kw):
    """Fix h and sweep inventory-skew strength k."""
    print(f"sigma={sigma}, h={half_spread}, A={A}, kappa={kappa}")
    print(f"  {'k':>5} {'rms_inv':>8} {'mean':>9} {'mean/std':>9}")
    for k in ks:
        a = sweep_v2(half_spread=half_spread, A=A, kappa=kappa,
                     k=k, sigma=sigma, **kw)
        mean = a[:, 2].mean()
        std = a[:, 2].std()
        rms_inv = a[:, 4].mean()
        print(f"  {k:5.3f} {rms_inv:8.2f} {mean:9.2f} {mean/std:9.3f}")


if __name__ == "__main__":
    #width_scan([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    #for k in [0.5, 1.0, 2.0]:
     #   width_scan([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], kappa=k)
      #  print()
    #for a in [0.10, 0.20, 0.40, 0.60]:
        #width_scan([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], A=a, kappa=1.0)
        #print()
    #k_scan([0.0, 0.08, 0.16, 0.32, 0.64, 1.28])
    ks = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
    sigmas = [0.0, 0.05, 0.10, 0.20, 0.40]
    for sig in sigmas:
        k_scan(ks, sigma=sig)
        print()
