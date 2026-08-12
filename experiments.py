"""Experiment harness. Edit the calls at the bottom and run."""

import numpy as np

from market import RandomWalk
from simulator import BernoulliFlow, QuoteSensitiveFlow, InformedFlow, run
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




# --------------- v3: direction-informed flow ---------------

def sweep_v3(n_seeds=500, n_steps=2000, sigma=0.1, half_spread=1.0,
             A=0.4, kappa=1.0, k=0.04, phi=0.0):
    """v3: direction-informed flow. Returns per-seed rows of
    (execution, total, rms_inv, markout, fills)."""
    rows = []
    for s in range(n_seeds):
        r = run(
            RandomWalk(sigma=sigma),
            InventorySkewMaker(half_spread=half_spread, k=k),
            InformedFlow(A=A, kappa=kappa, phi=phi),
            n_steps=n_steps,
            seed=s,
        )
        # Execution edge per fill, computed from the log (same as v2).
        execution = np.sum(
            r["buys"] * (r["S"] - r["bid"])
            + r["sells"] * (r["ask"] - r["S"])
        )
        total = r["pnl"][-1]
        rms_inv = np.sqrt(np.mean(r["inventory"] ** 2))
        # One-step customer-signed markout: positive means the customer
        # was right about direction, i.e. adverse for the maker.
        sf = r["signed_flow"]
        ds = r["delta_S"]
        mask = sf != 0
        markout = np.mean(sf[mask] * ds[mask]) if mask.any() else 0.0
        fills = r["n_trades"]
        rows.append((execution, total, rms_inv, markout, fills))
    return np.array(rows)


def phi_scan(phis, half_spread=1.0, k=0.04, **kw):
    """Experiment 1: sweep toxicity phi."""
    print(f"h={half_spread}, k={k}")
    print(f"  {'phi':>5} {'fills':>7} {'markout':>9} {'mean':>9} {'mean/std':>9}")
    for phi in phis:
        a = sweep_v3(half_spread=half_spread, k=k, phi=phi, **kw)
        fills = a[:, 4].mean()
        markout = a[:, 3].mean()
        mean = a[:, 1].mean()
        std = a[:, 1].std()
        ms = mean / std if std > 0 else float("nan")
        print(f"  {phi:5.2f} {fills:7.1f} {markout:9.5f} {mean:9.2f} {ms:9.3f}")


def k_scan_v3(ks, phi=0.5, half_spread=1.0, **kw):
    """Experiment 2: inventory control under toxicity."""
    print(f"phi={phi}, h={half_spread}")
    print(f"  {'k':>5} {'rms_inv':>8} {'markout':>9} {'mean':>9} {'mean/std':>9}")
    for k in ks:
        a = sweep_v3(half_spread=half_spread, k=k, phi=phi, **kw)
        rms_inv = a[:, 2].mean()
        markout = a[:, 3].mean()
        mean = a[:, 1].mean()
        std = a[:, 1].std()
        ms = mean / std if std > 0 else float("nan")
        print(f"  {k:5.3f} {rms_inv:8.2f} {markout:9.5f} {mean:9.2f} {ms:9.3f}")


def phi_h_grid(phis, hs, k=0.0, n_seeds=300, **kw):
    """Experiment 3: optimal spread under toxicity, no skew.
    For each phi, sweep h and find mean-P&L-maximising width."""
    print(f"k={k}")
    h_strs = [f"h={h}" for h in hs]
    print(f"  {'phi':>5}  " + "  ".join(f"{s:>8}" for s in h_strs) + "  best_h")
    for phi in phis:
        means = []
        for h in hs:
            a = sweep_v3(half_spread=h, k=k, phi=phi, n_seeds=n_seeds, **kw)
            means.append(a[:, 1].mean())
        best_idx = int(np.argmax(means))
        means_str = "  ".join(f"{m:8.1f}" for m in means)
        print(f"  {phi:5.2f}  {means_str}  {hs[best_idx]:.2f}")


def run_v3_experiments():
    print("=== Experiment 1: toxicity sweep ===")
    phi_scan([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    print()
    print("=== Experiment 2: inventory control under toxicity ===")
    k_scan_v3([0.0, 0.01, 0.04, 0.08, 0.16, 0.32])
    print()
    print("=== Experiment 3: optimal spread under toxicity ===")
    phi_h_grid([0.0, 0.1, 0.3, 0.5],
               [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])


def h_star_fine(phis, hs, k=0.0, n_seeds=1000, **kw):
    """Experiment 3b: fine grid around h* to resolve whether toxicity
    shifts the optimal spread.

    For each phi: mean P&L at every h, the argmax, a quadratic-fit vertex
    (sub-grid estimate of h*), and the standard error of the mean at the
    argmax so shifts can be judged against noise.
    """
    hs = np.asarray(hs, dtype=float)
    print(f"k={k}, n_seeds={n_seeds}")
    head = "  ".join(f"h={h:g}" for h in hs)
    print(f"  {'phi':>4}  {head}  {'argmax':>6} {'fit_h*':>7} {'se@max':>7}")

    for phi in phis:
        means, ses = [], []
        for h in hs:
            a = sweep_v3(half_spread=h, k=k, phi=phi, n_seeds=n_seeds, **kw)
            pnl = a[:, 1]
            means.append(pnl.mean())
            ses.append(pnl.std(ddof=1) / np.sqrt(len(pnl)))
        means = np.array(means)
        ses = np.array(ses)

        i_max = int(np.argmax(means))

        # Quadratic fit through all points for a sub-grid vertex estimate.
        # Guard: vertex only meaningful if curvature is negative (a peak).
        c2, c1, _ = np.polyfit(hs, means, 2)
        fit_h = -c1 / (2 * c2) if c2 < 0 else float("nan")

        row = "  ".join(f"{m:7.1f}" for m in means)
        print(f"  {phi:4.1f}  {row}  {hs[i_max]:6.3f} {fit_h:7.3f} {ses[i_max]:7.2f}")

if __name__ == "__main__":
    h_star_fine(
        phis=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        hs=[0.90, 0.95, 1.00, 1.025, 1.05, 1.075, 1.10, 1.15, 1.20],
    )
