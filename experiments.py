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


# --------------- v3b Stage 0: effect-size / power analysis ---------------
#
# Per-tick expected PnL for the k=0 single-trader model (no-clipping approx):
#
#     Pi(h; phi) = A * exp(-kappa*h) * (h - c),   c = phi*sigma*sqrt(2/pi)
#
# maximised at h*(phi) = 1/kappa + c. This is the same object v3a's
# Prediction 2 was derived from; here we integrate it over a schedule to
# size the Oracle-vs-fixed economic gap BEFORE building any v3b mechanism.

def _pi_per_tick(h, phi, A=0.4, kappa=1.0, sigma=0.1):
    """Analytical expected PnL per tick at half-spread h under toxicity phi."""
    c = phi * sigma * np.sqrt(2.0 / np.pi)
    return A * np.exp(-kappa * h) * (h - c)


def _h_star(phi, kappa=1.0, sigma=0.1):
    return 1.0 / kappa + phi * sigma * np.sqrt(2.0 / np.pi)


def _best_fixed_h(segments, A=0.4, kappa=1.0, sigma=0.1):
    """Best single fixed half-spread for a piecewise-constant schedule.

    segments: list of (length_ticks, phi). Returns (h_best, total_expected_pnl).
    Total expected PnL is sum over segments of length * Pi(h; phi_seg); we
    maximise it over h on a fine grid.

    For this exact objective the FOC gives a closed form:
    Pi_total(h) = A e^{-kappa h} (T*h - sum_i L_i c_i), so the optimum is
    h* = 1/kappa + (sum_i L_i c_i)/T, i.e. exactly time-weighted-average
    toxicity. The grid search below agrees with that; we keep it because it
    reads back the max PnL directly and needs no separate algebra.
    """
    hs = np.linspace(0.5, 2.0, 3001)
    totals = np.array([
        sum(L * _pi_per_tick(h, phi, A, kappa, sigma) for L, phi in segments)
        for h in hs
    ])
    i = int(np.argmax(totals))
    return hs[i], totals[i]


def stage0_effect_size(A=0.4, kappa=1.0, sigma=0.1,
                       phi_lo=0.0, phi_hi=1.0, seg_len=1500):
    """Full low->high->low Oracle vs Best-Fixed analytical effect size."""
    segments = [(seg_len, phi_lo), (seg_len, phi_hi), (seg_len, phi_lo)]
    n_steps = 3 * seg_len

    # Oracle: each segment quotes at that segment's own optimum.
    pnl_oracle = sum(L * _pi_per_tick(_h_star(phi, kappa, sigma), phi,
                                      A, kappa, sigma)
                     for L, phi in segments)

    # Best single fixed spread against the whole schedule.
    h_fixed, pnl_fixed = _best_fixed_h(segments, A, kappa, sigma)

    gap = pnl_oracle - pnl_fixed

    # Expected fills depend on the spread actually quoted, via p_fill =
    # A*exp(-kappa*h). The Oracle widens h in the high regime, so its
    # high-regime fill rate is genuinely lower than the low-regime one —
    # report both against each regime's own Oracle spread rather than a
    # flat 0.148/tick, so window sizing uses the real per-regime fill count.
    p_lo = A * np.exp(-kappa * _h_star(phi_lo, kappa, sigma))
    p_hi = A * np.exp(-kappa * _h_star(phi_hi, kappa, sigma))

    print(f"--- Stage 0 effect size: low->high->low ---")
    print(f"  A={A}, kappa={kappa}, sigma={sigma}, "
          f"phi_lo={phi_lo}, phi_hi={phi_hi}, seg_len={seg_len}")
    print(f"  h*(lo)={_h_star(phi_lo, kappa, sigma):.4f}  "
          f"h*(hi)={_h_star(phi_hi, kappa, sigma):.4f}  "
          f"best_fixed_h={h_fixed:.4f}")
    print(f"  E[PnL] Oracle    = {pnl_oracle:.3f}")
    print(f"  E[PnL] BestFixed = {pnl_fixed:.3f}")
    print(f"  Oracle - BestFixed = {gap:.4f}  over {n_steps} ticks")
    print(f"  Oracle fill rate: lo={p_lo:.4f}/tick  hi={p_hi:.4f}/tick")
    print(f"  Oracle fills/segment: lo={p_lo * seg_len:.1f}  "
          f"hi={p_hi * seg_len:.1f}")


def stage0_spread_mismatch_diag(n_seeds=1000, n_steps=1000, sigma=0.1,
                                A=0.4, kappa=1.0):
    """High-regime spread-mismatch diagnostic (NOT the full Oracle-vs-fixed
    effect size). Holds phi=1 constant and compares the wrong spread (the
    optimum for phi=0) against the correct one (the optimum for phi=1),
    under CRN paired differences.

    Both spreads are derived from sigma via _h_star, so this stays a valid
    diagnostic when sigma is raised — at sigma=0.1 it is 1.00 vs 1.08, at
    sigma=0.4 it is 1.00 vs ~1.32. It is the single largest static mismatch
    the model can produce at phi=1; an upper reference for how visible an
    h-error of size h*(1)-h*(0) is, not the regime-switching effect (which
    is much smaller — see stage0_effect_size). Analytical prediction printed
    alongside so the simulation can be checked against theory.
    """
    h_wrong = _h_star(0.0, kappa, sigma)    # optimum if maker assumed phi=0
    h_correct = _h_star(1.0, kappa, sigma)  # true optimum at phi=1

    dpi = (_pi_per_tick(h_correct, 1.0, A, kappa, sigma)
           - _pi_per_tick(h_wrong, 1.0, A, kappa, sigma))
    print(f"h_wrong={h_wrong:.4f}  h_correct={h_correct:.4f}")
    print(f"analytical E[diff] per tick = {dpi:.6f}  "
          f"-> over {n_steps} ticks = {dpi * n_steps:.3f}")

    a_low = sweep_v3(half_spread=h_wrong, phi=1.0, k=0.0, sigma=sigma,
                     A=A, kappa=kappa, n_steps=n_steps, n_seeds=n_seeds)
    a_high = sweep_v3(half_spread=h_correct, phi=1.0, k=0.0, sigma=sigma,
                      A=A, kappa=kappa, n_steps=n_steps, n_seeds=n_seeds)
    diff = a_high[:, 1] - a_low[:, 1]  # column 1 = total pnl, same seed order
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"mean(h_wrong)   = {a_low[:, 1].mean():.2f}")
    print(f"mean(h_correct) = {a_high[:, 1].mean():.2f}")
    print(f"paired diff: mean={diff.mean():.3f}  std={diff.std(ddof=1):.3f}  "
          f"se={se:.3f}  t={diff.mean() / se:.2f}")


if __name__ == "__main__":
    stage0_effect_size(sigma=0.3)
    print()
    stage0_spread_mismatch_diag(sigma=0.3)
