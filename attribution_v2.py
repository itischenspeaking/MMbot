"""v2 attribution at sigma=0.

At sigma=0 there is no price movement, so terminal PnL is pure execution edge.
Yet mean/std peaks at positive k. This decomposes that PnL exactly to find out
why, rather than hand-waving about feedback.

Each fill's edge relative to fair value is h +/- k*q (q = inventory before the
fill), so per tick the execution PnL is

    h*(buy + sell)  +  k * q_t * dq_t,   dq_t = buy_t - sell_t

Summed over the path this splits terminal PnL P exactly into

    P = A + B,   A = h * N_trades,   B = k * sum(q_pre * dq)

A is what you'd earn taking a flat half-spread on every fill; B is the
correction the inventory skew applies to that edge. Two identities guard it:
the split P = A + B, and a telescoping form of B,

    B = 0.5 * k * (q_T^2 - sum(dq^2))

which gives B an independent second computation.

    python attribution_v2.py
"""

import numpy as np

from market import RandomWalk
from simulator import QuoteSensitiveFlow, run
from strategy import InventorySkewMaker

HALF_SPREAD = 1.0
A_INTENSITY = 0.4
KAPPA = 1.0
N_SEEDS = 500
N_STEPS = 2000
KS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]


def legs(r, k):
    """Return (A, B, B_telescoped, P, n_single, qT2) for one path."""
    q_pre = np.r_[0, r["inventory"][:-1]]
    dq = r["buys"].astype(int) - r["sells"].astype(int)

    A = HALF_SPREAD * r["n_trades"]
    B = k * np.sum(q_pre * dq)
    P = r["pnl"][-1]

    qT2 = r["inventory"][-1] ** 2
    n_single = np.sum(dq ** 2)
    B_tel = 0.5 * k * (qT2 - n_single)
    return A, B, B_tel, P, n_single, qT2


def collect(k):
    As, Bs, Ps, Ns, QTs = [], [], [], [], []
    for seed in range(N_SEEDS):
        r = run(
            RandomWalk(sigma=0.0),
            InventorySkewMaker(half_spread=HALF_SPREAD, k=k),
            QuoteSensitiveFlow(A=A_INTENSITY, kappa=KAPPA),
            n_steps=N_STEPS,
            seed=seed,
        )
        A, B, B_tel, P, n_single, qT2 = legs(r, k)
        # Per-path identities: fail loudly before any statistics.
        assert np.allclose(P, A + B), (k, seed, P, A + B)
        assert np.allclose(B, B_tel), (k, seed, B, B_tel)
        As.append(A); Bs.append(B); Ps.append(P); Ns.append(n_single); QTs.append(qT2)
    return map(np.array, (As, Bs, Ps, Ns, QTs))


def mean_std(x):
    s = x.std()
    return x.mean() / s if s > 0 else float("nan")


def main():
    # Pass 1: collect the raw legs for every k, checking per-path and
    # variance-decomposition identities as we go.
    stats = {}
    for k in KS:
        As, Bs, Ps, Ns, QTs = collect(k)
        cov = np.cov(As, Bs, ddof=0)[0, 1]
        var_rhs = As.var() + Bs.var() + 2 * cov
        assert np.allclose(Ps.var(), var_rhs), (k, Ps.var(), var_rhs)
        stats[k] = dict(As=As, Bs=Bs, Ps=Ps, Ns=Ns, QTs=QTs, cov=cov)

    # Detailed table: absolute level at each k.
    hdr = ("k", "mean_A", "std_A", "A_m/std", "mean_B", "std_B",
           "cov_AB", "corr_AB", "mean_P", "std_P", "P_m/std",
           "mean_Nsg", "mean_qT2")
    print("  ".join(f"{h:>8}" for h in hdr))
    for k in KS:
        s = stats[k]
        As, Bs, Ps = s["As"], s["Bs"], s["Ps"]
        if As.std() > 0 and Bs.std() > 0:
            corr = s["cov"] / (As.std() * Bs.std())
        else:
            corr = float("nan")
        row = (k, As.mean(), As.std(), mean_std(As), Bs.mean(), Bs.std(),
               s["cov"], corr, Ps.mean(), Ps.std(), mean_std(Ps),
               s["Ns"].mean(), s["QTs"].mean())
        print("  ".join(f"{v:8.3f}" for v in row))

    # Attribution table: everything relative to the k=0 baseline, which is the
    # actual research question. At k=0, B is identically 0, so
    #   dMean_P = dMean_A + mean_B
    #   dVar_P  = dVar_A + var_B + 2cov_AB
    base = stats[0.0]
    meanA0, varA0, meanP0, varP0 = (
        base["As"].mean(), base["As"].var(), base["Ps"].mean(), base["Ps"].var())

    print()
    hdr2 = ("k", "dMean_A", "mean_B", "dMean_P",
            "dVar_A", "var_B", "2cov_AB", "dVar_P")
    print("  ".join(f"{h:>9}" for h in hdr2))
    for k in KS:
        s = stats[k]
        As, Bs, Ps = s["As"], s["Bs"], s["Ps"]

        dMean_A = As.mean() - meanA0
        mean_B = Bs.mean()
        dMean_P = Ps.mean() - meanP0
        assert np.allclose(dMean_P, dMean_A + mean_B), (k, dMean_P, dMean_A + mean_B)

        dVar_A = As.var() - varA0
        var_B = Bs.var()
        two_cov = 2 * s["cov"]
        dVar_P = Ps.var() - varP0
        assert np.allclose(dVar_P, dVar_A + var_B + two_cov), (
            k, dVar_P, dVar_A + var_B + two_cov)

        row = (k, dMean_A, mean_B, dMean_P, dVar_A, var_B, two_cov, dVar_P)
        print("  ".join(f"{v:9.3f}" for v in row))


if __name__ == "__main__":
    main()
