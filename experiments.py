"""Experiment harness. Edit the calls at the bottom and run."""

import numpy as np

from market import RandomWalk
from simulator import (BernoulliFlow, QuoteSensitiveFlow, InformedFlow,
                       RegimeInformedFlow, step_schedule, markov_schedule, run)
from strategy import (NaiveMaker, InventorySkewMaker, RollingToxicityEstimator,
                      AdaptiveMaker, OracleMaker, IntegratedMaker, _h_star)


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


# --------------- v3b Stage 2: offline toxicity estimation ---------------
#
# Fixed maker (h=1, k=0), primary regime schedule, sigma=0.3. The estimator
# is pure offline post-processing on the run() log — it never touches quotes.

def _v3b_primary_run(seed, sigma=0.3, seg_len=1500):
    """One run under the locked primary schedule with a fixed h=1, k=0 maker."""
    n_steps = 3 * seg_len
    schedule = step_schedule(n_steps, [seg_len, 2 * seg_len], [0.0, 1.0, 0.0])
    r = run(RandomWalk(sigma=sigma),
            InventorySkewMaker(half_spread=1.0, k=0.0),
            RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
            n_steps=n_steps, seed=seed)
    return r, schedule


def stage2_metrics(Ns=(20, 50, 100), n_seeds=300, sigma=0.3, seg_len=1500):
    """Summary table: steady-state bias/RMSE/noise and response lag per N.

    Steady state within a regime = ticks where the window holds only
    same-regime fills, i.e. from the tick at which N fills have accumulated
    since the latest boundary, to the next boundary. Transition periods are
    excluded from steady-state metrics by construction.
    """
    n_steps = 3 * seg_len
    bounds = [0, seg_len, 2 * seg_len, n_steps]
    seg_phi = [0.0, 1.0, 0.0]
    seg_name = ["low", "high", "low"]

    print(f"sigma={sigma}, seg_len={seg_len}, n_seeds={n_seeds}")
    print(f"  {'N':>4} {'regime':>7} {'mean_phi':>9} {'bias':>8} "
          f"{'rmse':>8} {'noise':>8}")

    lag_up = {N: [] for N in Ns}    # fills after 0->1 boundary until phi_hat>=0.5
    lag_down = {N: [] for N in Ns}  # fills after 1->0 boundary until phi_hat<=0.5
    # Seeds where phi_hat already sits on the wrong side of 0.5 at the boundary
    # tick are excluded from lag (they'd register a spurious ~0 lag) and
    # counted here instead, as a pre-boundary misclassification rate.
    false_up = {N: 0 for N in Ns}    # phi_hat >= 0.5 already, just before 0->1
    false_down = {N: 0 for N in Ns}  # phi_hat <= 0.5 already, just before 1->0
    n_valid = {N: 0 for N in Ns}     # seeds with a defined phi_hat at boundary

    for N in Ns:
        ss = {0: [], 1: [], 2: []}  # steady-state phi_hat samples per regime
        for seed in range(n_seeds):
            r, _ = _v3b_primary_run(seed, sigma=sigma, seg_len=seg_len)
            sf, ds = r["signed_flow"], r["delta_S"]
            phi_hat = RollingToxicityEstimator(N, sigma).run_offline(sf, ds)

            # steady-state samples: per segment, skip until N fills in-regime
            for i in range(3):
                lo, hi = bounds[i], bounds[i + 1]
                fills_seen, ss_start = 0, None
                for t in range(lo, hi):
                    if sf[t] != 0:
                        fills_seen += 1
                        if fills_seen >= N:
                            ss_start = t + 1
                            break
                if ss_start is not None:
                    seg = phi_hat[ss_start:hi]
                    seg = seg[~np.isnan(seg)]
                    if len(seg):
                        ss[i].append(seg)

            # response lag: new fills after each boundary until the 0.5 cross.
            # Skip seeds already on the wrong side at the boundary (they'd
            # log a spurious ~0 lag); count them as pre-boundary false-side.
            b = seg_len  # 0 -> 1
            pre = phi_hat[b - 1]
            if not np.isnan(pre):
                n_valid[N] += 1
                if pre >= 0.5:
                    false_up[N] += 1  # already high before the jump
                else:
                    fills_after, crossed = 0, None
                    for t in range(b, bounds[2]):
                        if sf[t] != 0:
                            fills_after += 1
                        if not np.isnan(phi_hat[t]) and phi_hat[t] >= 0.5:
                            crossed = fills_after
                            break
                    if crossed is not None:
                        lag_up[N].append(crossed)

            b = 2 * seg_len  # 1 -> 0
            pre = phi_hat[b - 1]
            if not np.isnan(pre):
                if pre <= 0.5:
                    false_down[N] += 1  # already low before the drop
                else:
                    fills_after, crossed = 0, None
                    for t in range(b, bounds[3]):
                        if sf[t] != 0:
                            fills_after += 1
                        if not np.isnan(phi_hat[t]) and phi_hat[t] <= 0.5:
                            crossed = fills_after
                            break
                    if crossed is not None:
                        lag_down[N].append(crossed)

        for i in range(3):
            if not ss[i]:
                continue
            samples = np.concatenate(ss[i])
            phi = seg_phi[i]
            mean = samples.mean()
            rmse = np.sqrt(np.mean((samples - phi) ** 2))
            print(f"  {N:>4} {seg_name[i]:>7} {mean:>9.4f} {mean - phi:>8.4f} "
                  f"{rmse:>8.4f} {samples.std():>8.4f}")

    print()
    print(f"  response lag (fills), predicted ~ N/2; "
          f"lag computed only on seeds correctly classified at the boundary")
    print(f"  {'N':>4} {'up_mean':>8} {'up_med':>8} {'down_mean':>10} "
          f"{'down_med':>9} {'false_up':>9} {'false_dn':>9}")
    for N in Ns:
        u, d = np.array(lag_up[N]), np.array(lag_down[N])
        fu = false_up[N] / n_valid[N] if n_valid[N] else float("nan")
        fd = false_down[N] / n_valid[N] if n_valid[N] else float("nan")
        print(f"  {N:>4} {u.mean():>8.1f} {np.median(u):>8.1f} "
              f"{d.mean():>10.1f} {np.median(d):>9.1f} "
              f"{fu:>9.3f} {fd:>9.3f}")


def stage2_tracking(seed=0, Ns=(20, 50, 100), sigma=0.3, seg_len=1500,
                    window=600):
    """Representative-seed numerical tracking around both transitions.
    Prints phi_true and phi_hat(N) sampled every ~40 ticks near each boundary.
    Window is wide enough (600 ticks ~= 88 fills) to show the N=100 response,
    which needs ~50 new fills (~340 ticks) to cross 0.5."""
    r, schedule = _v3b_primary_run(seed, sigma=sigma, seg_len=seg_len)
    sf, ds = r["signed_flow"], r["delta_S"]
    hats = {N: RollingToxicityEstimator(N, sigma).run_offline(sf, ds) for N in Ns}

    for b_name, b in (("0->1", seg_len), ("1->0", 2 * seg_len)):
        print(f"\n  transition {b_name} at tick {b} (seed={seed})")
        header = "  ".join(f"N={N}" for N in Ns)
        print(f"  {'tick':>6} {'phi_true':>9}  {header}")
        for t in range(b - window, b + window + 1, 40):
            if t < 0 or t >= len(sf):
                continue
            hats_str = "  ".join(
                f"{hats[N][t]:.2f}" if not np.isnan(hats[N][t]) else " nan"
                for N in Ns)
            print(f"  {t:>6} {schedule[t]:>9.1f}  {hats_str}")


# --------------- v3b Stage 3: closed-loop adaptive quoting ---------------

def stage3_compare(n_seeds=1000, sigma=0.3, seg_len=1500, N=50,
                   A=0.4, kappa=1.0):
    """CRN paired comparison: BestFixed vs Adaptive vs Oracle on the primary
    0->1->0 schedule. Same seed => same price path and same (u_side,u_fill,
    u_type) draws for all three; fills diverge only because quotes differ.
    """
    n_steps = 3 * seg_len
    schedule = step_schedule(n_steps, [seg_len, 2 * seg_len], [0.0, 1.0, 0.0])
    h_best = _best_fixed_h([(seg_len, 0.0), (seg_len, 1.0), (seg_len, 0.0)],
                           A=A, kappa=kappa, sigma=sigma)[0]

    bounds = [0, seg_len, 2 * seg_len, n_steps]
    pnl_f, pnl_a, pnl_o = [], [], []
    fills_f, fills_a, fills_o = [], [], []
    h_by_regime = {0: [], 1: [], 2: []}  # adaptive avg h per regime (diagnostic)
    phi_err = []  # adaptive phi_hat RMSE vs phi_true (post-warmup, all ticks)
    # closed-loop response lag, from the ADAPTIVE run's own fill sequence
    # (reflects the second-order feedback: wider h in high regime -> fewer
    # fills -> slower estimator updates). Lag counted only on seeds correctly
    # classified at the boundary; wrong-side seeds counted separately.
    lag_up_f, lag_down_f = [], []      # response lag in fills
    lag_up_t, lag_down_t = [], []      # response lag in ticks
    false_up = false_down = n_valid = 0

    scale = sigma * np.sqrt(2.0 / np.pi)

    for seed in range(n_seeds):
        rf = run(RandomWalk(sigma=sigma),
                 InventorySkewMaker(half_spread=h_best, k=0.0),
                 RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule),
                 n_steps=n_steps, seed=seed)
        ra = run(RandomWalk(sigma=sigma),
                 AdaptiveMaker(N=N, sigma=sigma, h_warmup=h_best, kappa=kappa),
                 RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule),
                 n_steps=n_steps, seed=seed)
        ro = run(RandomWalk(sigma=sigma),
                 OracleMaker(phi_schedule=schedule, sigma=sigma, kappa=kappa),
                 RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule),
                 n_steps=n_steps, seed=seed)

        pnl_f.append(rf["pnl"][-1])
        pnl_a.append(ra["pnl"][-1])
        pnl_o.append(ro["pnl"][-1])
        fills_f.append(rf["n_trades"])
        fills_a.append(ra["n_trades"])
        fills_o.append(ro["n_trades"])

        # adaptive avg h per regime, from logged quotes: h = (ask - bid)/2
        h_a = (ra["ask"] - ra["bid"]) / 2.0
        for i in range(3):
            h_by_regime[i].append(h_a[bounds[i]:bounds[i + 1]].mean())

        # adaptive phi_hat error: reconstruct phi_hat path offline from the
        # SAME (adaptive) log — estimator is deterministic given the fill log
        sf_a, ds_a = ra["signed_flow"], ra["delta_S"]
        phi_hat = RollingToxicityEstimator(N, sigma).run_offline(sf_a, ds_a)
        m = ~np.isnan(phi_hat)
        phi_err.append(np.sqrt(np.mean((phi_hat[m] - schedule[m]) ** 2)))

        # closed-loop response lag off the adaptive fill sequence
        b = seg_len  # 0 -> 1
        pre = phi_hat[b - 1]
        if not np.isnan(pre):
            n_valid += 1
            if pre >= 0.5:
                false_up += 1
            else:
                fa, crossed = 0, None
                for t in range(b, bounds[2]):
                    if sf_a[t] != 0:
                        fa += 1
                    if not np.isnan(phi_hat[t]) and phi_hat[t] >= 0.5:
                        crossed = (fa, t - b)
                        break
                if crossed is not None:
                    lag_up_f.append(crossed[0])
                    lag_up_t.append(crossed[1])
        b = 2 * seg_len  # 1 -> 0
        pre = phi_hat[b - 1]
        if not np.isnan(pre):
            if pre <= 0.5:
                false_down += 1
            else:
                fa, crossed = 0, None
                for t in range(b, bounds[3]):
                    if sf_a[t] != 0:
                        fa += 1
                    if not np.isnan(phi_hat[t]) and phi_hat[t] <= 0.5:
                        crossed = (fa, t - b)
                        break
                if crossed is not None:
                    lag_down_f.append(crossed[0])
                    lag_down_t.append(crossed[1])

    pnl_f = np.array(pnl_f)
    pnl_a = np.array(pnl_a)
    pnl_o = np.array(pnl_o)

    def _paired(d, label):
        mean = d.mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {label:18s} mean={mean:8.3f}  se={se:6.3f}  "
              f"95%CI=[{mean - 1.96 * se:7.3f}, {mean + 1.96 * se:7.3f}]")

    print(f"=== Stage 3: BestFixed vs Adaptive vs Oracle ===")
    print(f"  sigma={sigma}, N={N}, seg_len={seg_len}, k=0, "
          f"best_fixed_h={h_best:.4f}, n_seeds={n_seeds}")
    print(f"  mean terminal PnL: Fixed={pnl_f.mean():.2f}  "
          f"Adaptive={pnl_a.mean():.2f}  Oracle={pnl_o.mean():.2f}")
    print(f"  paired differences (same-seed):")
    _paired(pnl_a - pnl_f, "Adaptive - Fixed")
    _paired(pnl_o - pnl_a, "Oracle - Adaptive")
    _paired(pnl_o - pnl_f, "Oracle - Fixed")

    af = (pnl_a - pnl_f).mean()
    of = (pnl_o - pnl_f).mean()
    capture_noisy = af / of if of != 0 else float("nan")
    capture_theory = af / 3.7948  # Stage 0 analytical Oracle-BestFixed gap
    print(f"  capture fraction (noisy ratio) = mean(A-F)/mean(O-F) = "
          f"{capture_noisy:.3f}")
    print(f"  capture fraction / theory      = mean(A-F)/3.7948     = "
          f"{capture_theory:.3f}")

    print(f"  diagnostics:")
    print(f"    fills: Fixed={np.mean(fills_f):.1f}  "
          f"Adaptive={np.mean(fills_a):.1f}  Oracle={np.mean(fills_o):.1f}")
    print(f"    adaptive avg h by regime: "
          f"low={np.mean(h_by_regime[0]):.4f}  "
          f"high={np.mean(h_by_regime[1]):.4f}  "
          f"low={np.mean(h_by_regime[2]):.4f}  "
          f"(oracle target: low={_h_star(0.0, kappa, sigma):.4f} "
          f"high={_h_star(1.0, kappa, sigma):.4f})")
    print(f"    adaptive phi_hat RMSE (post-warmup) = {np.mean(phi_err):.4f}")
    print(f"    closed-loop response lag (adaptive fill sequence, "
          f"correctly-classified seeds only):")
    uf, df = np.array(lag_up_f), np.array(lag_down_f)
    ut, dt = np.array(lag_up_t), np.array(lag_down_t)
    print(f"      0->1: {uf.mean():.1f} fills / {ut.mean():.0f} ticks   "
          f"1->0: {df.mean():.1f} fills / {dt.mean():.0f} ticks")
    print(f"      pre-boundary false-side rate: "
          f"up={false_up / n_valid:.3f}  down={false_down / n_valid:.3f}")


# --------------- v4pre: default-parameter check (not a sweep) ---------------
#
# Confirms the frozen competition_baseline parameters run cleanly: inventory
# is well controlled relative to toxicity-only (v3b's k=0 Adaptive), spread
# widens in the toxic regime and narrows back, no PnL catastrophe.

COMPETITION_BASELINE = dict(k=0.04, N=50, sigma=0.3, h_warmup=1.08, kappa=1.0)


def v4pre_default_check(n_seeds=500, seg_len=1500, params=None):
    params = params or COMPETITION_BASELINE
    n_steps = 3 * seg_len
    schedule = step_schedule(n_steps, [seg_len, 2 * seg_len], [0.0, 1.0, 0.0])
    bounds = [0, seg_len, 2 * seg_len, n_steps]

    pnl_int, pnl_tox = [], []
    rms_int, rms_tox = [], []
    maxabs_int, maxabs_tox = [], []
    h_int_by_regime = {0: [], 1: [], 2: []}

    for seed in range(n_seeds):
        flow = RegimeInformedFlow(A=0.4, kappa=params["kappa"],
                                  phi_schedule=schedule)
        r_int = run(RandomWalk(sigma=params["sigma"]),
                    IntegratedMaker(k=params["k"], N=params["N"],
                                    sigma=params["sigma"],
                                    h_warmup=params["h_warmup"],
                                    kappa=params["kappa"], toxicity=True),
                    flow, n_steps=n_steps, seed=seed)
        # toxicity-only comparator: v3b's Adaptive, k=0
        r_tox = run(RandomWalk(sigma=params["sigma"]),
                    AdaptiveMaker(N=params["N"], sigma=params["sigma"],
                                  h_warmup=params["h_warmup"],
                                  kappa=params["kappa"], k=0.0),
                    flow, n_steps=n_steps, seed=seed)

        pnl_int.append(r_int["pnl"][-1])
        pnl_tox.append(r_tox["pnl"][-1])
        rms_int.append(np.sqrt(np.mean(r_int["inventory"] ** 2)))
        rms_tox.append(np.sqrt(np.mean(r_tox["inventory"] ** 2)))
        maxabs_int.append(np.max(np.abs(r_int["inventory"])))
        maxabs_tox.append(np.max(np.abs(r_tox["inventory"])))

        h_i = (r_int["ask"] - r_int["bid"]) / 2.0
        for i in range(3):
            h_int_by_regime[i].append(h_i[bounds[i]:bounds[i + 1]].mean())

    pnl_int, pnl_tox = np.array(pnl_int), np.array(pnl_tox)
    rms_int, rms_tox = np.array(rms_int), np.array(rms_tox)

    print(f"=== v4pre default-parameter check ===")
    print(f"  params: {params}")
    print(f"  n_seeds={n_seeds}, seg_len={seg_len}")
    print(f"  mean PnL:      Integrated={pnl_int.mean():.2f}  "
          f"Toxicity-only={pnl_tox.mean():.2f}")
    print(f"  mean RMS inv:  Integrated={rms_int.mean():.3f}  "
          f"Toxicity-only={rms_tox.mean():.3f}  "
          f"(reduction={100 * (1 - rms_int.mean() / rms_tox.mean()):.1f}%)")
    print(f"  mean max|inv|: Integrated={np.mean(maxabs_int):.2f}  "
          f"Toxicity-only={np.mean(maxabs_tox):.2f}")
    print(f"  min terminal PnL seen: Integrated={pnl_int.min():.2f}  "
          f"Toxicity-only={pnl_tox.min():.2f}")
    print(f"  Integrated avg h by regime: "
          f"low={np.mean(h_int_by_regime[0]):.4f}  "
          f"high={np.mean(h_int_by_regime[1]):.4f}  "
          f"low={np.mean(h_int_by_regime[2]):.4f}  "
          f"(targets: low={_h_star(0.0, params['kappa'], params['sigma']):.4f} "
          f"high={_h_star(1.0, params['kappa'], params['sigma']):.4f})")


# --------------- v4a Experiment 1: does k>0 break the h*(phi) rule? ---------
#
# Objective is the per-tick LOCAL EDGE (execution edge relative to fair value,
# net of that tick's markout), not terminal PnL and not execution-only:
#
#     local_edge_t = execution_edge_t - signed_flow_t * delta_S_t
#
# where a buy at bid earns (S - bid) = h + k*q, a sell at ask earns
# (ask - S) = h - k*q, and no fill earns 0. This matches the conditional
# analytic object Pi(h | q) = sum_legs (edge - c) * p_leg, so its argmax over
# h is the conditional h*(phi, k, q). Analytic prediction:
#
#     h*(phi, k, q) = 1/kappa + phi*sigma*sqrt(2/pi) + kappa*k^2*q^2 + O(k^4 q^4)
#
# (first-order-in-kq correction cancels by bid/ask symmetry; second order
# survives.)

def _local_edge_batch(S, q, h, k, phi, sigma, A, kappa, n, rng):
    """Vectorised fixed-q one-step MC: draw n independent ticks at inventory q
    against a skewed quote, return the mean local edge. Same mechanics as
    InformedFlow.fills (informed sees sign(delta_S); fill prob A*exp(-kappa*
    distance)), computed over arrays rather than one draw at a time."""
    center = S - k * q
    bid, ask = center - h, center + h
    d_bid = S - bid    # = h + k*q, distance from fair to our bid
    d_ask = ask - S    # = h - k*q
    p_bid = min(1.0, A * np.exp(-kappa * d_bid))
    p_ask = min(1.0, A * np.exp(-kappa * d_ask))

    delta_S = sigma * rng.standard_normal(n)
    u_side = rng.random(n)
    u_fill = rng.random(n)
    u_type = rng.random(n)

    informed = u_type < phi
    # side: True = customer buys (lifts ask, we sell), False = customer sells
    # (hits bid, we buy). Informed picks the profitable side from sign(delta_S);
    # uninformed flips a coin. delta_S == 0 for informed -> no trade (rare).
    side_buy = np.where(informed, delta_S > 0, u_side < 0.5)
    side_sell = np.where(informed, delta_S < 0, ~(u_side < 0.5))
    # informed with delta_S == 0 trades on neither side
    informed_flat = informed & (delta_S == 0)
    side_buy &= ~informed_flat
    side_sell &= ~informed_flat

    # fills
    filled_ask = side_buy & (u_fill < p_ask)   # we sell at ask, signed_flow +1
    filled_bid = side_sell & (u_fill < p_bid)  # we buy at bid,  signed_flow -1

    edge = np.zeros(n)
    edge[filled_ask] = d_ask
    edge[filled_bid] = d_bid
    signed = np.zeros(n)
    signed[filled_ask] = 1.0
    signed[filled_bid] = -1.0

    local_edge = edge - signed * delta_S
    return local_edge.mean()


def exp1a_fixed_q(qs=(0, 5, 10), phis=(0.0, 0.5, 1.0), ks=(0.04,),
                  sigma=0.3, A=0.4, kappa=1.0, n_samples=4000000, seed=0):
    """Exp1A — fixed-q one-step Monte Carlo. For each (k, phi, q), locate the
    empirical argmax of mean local edge and compare to the conditional
    analytic optimum. Two benchmarks are printed:

        h*_V3            = 1/kappa + phi*sigma*sqrt(2/pi)         (k=0 rule)
        correction_exact = k*q*tanh(kappa*k*q)                   (full)
        correction_small = kappa*k^2*q^2                         (small-kq)

    so h*_analytic = h*_V3 + correction_exact. At k=0.04, q=10 the exact and
    small corrections differ ~5% (kappa*k*q=0.4, tanh 0.380 vs 0.4), which a
    quadratic peak fit on a focused grid can just resolve — the empirical h*
    landing on exact rather than small validates the tanh structure.

    The local-edge peak in h is flat (curvature ~kappa*p/2), so raw grid
    argmax jitters; we sample a focused grid around the prediction and fit a
    parabola to locate the vertex to sub-grid precision (v3a's method)."""
    c_of = lambda phi: phi * sigma * np.sqrt(2.0 / np.pi)
    S0 = 100.0

    print(f"=== Exp1A: fixed-q one-step MC (local-edge objective) ===")
    print(f"  sigma={sigma}, A={A}, kappa={kappa}, n_samples={n_samples}, "
          f"quadratic peak fit")
    print(f"  {'q':>4} {'phi':>4} {'k':>5} {'h*_V3':>7} {'corr_exact':>10} "
          f"{'corr_small':>10} {'emp_h*':>7} {'emp-analytic':>12}")
    for k in ks:
        for phi in phis:
            for q in qs:
                h_v3 = 1.0 / kappa + c_of(phi)
                corr_exact = k * q * np.tanh(kappa * k * q)
                corr_small = kappa * k * k * q * q
                pred = h_v3 + corr_exact
                # narrow focused grid: +/-0.06 in 0.01 steps. Narrow enough
                # that the true e^{-kh}(h-c) curve is near-parabolic here, so
                # the quadratic vertex has negligible deterministic bias
                # (a wide window carries ~0.005 vertex bias, material vs the
                # 0.008 exact-vs-small gap at q=10).
                hs = np.round(pred + np.arange(-0.06, 0.061, 0.01), 4)
                edges = np.empty(len(hs))
                for i, h in enumerate(hs):
                    # CRN across h: identical draws for every h in this cell,
                    # so the peak SHAPE is noise-cancelled and the vertex is
                    # located precisely. Seed does NOT depend on i.
                    rng = np.random.default_rng(
                        [seed, int(k * 1000), int(phi * 10), q])
                    edges[i] = _local_edge_batch(S0, q, h, k, phi, sigma, A,
                                                 kappa, n_samples, rng)
                a, b, _ = np.polyfit(hs, edges, 2)  # vertex = -b/2a
                h_star = -b / (2 * a) if a < 0 else hs[int(np.argmax(edges))]
                print(f"  {q:>4} {phi:>4} {k:>5} {h_v3:>7.4f} "
                      f"{corr_exact:>10.4f} {corr_small:>10.4f} "
                      f"{h_star:>7.4f} {h_star - pred:>12.4f}")


def exp1b_endogenous(ks=(0.0, 0.04, 0.08), phis=(0.0, 0.5, 1.0),
                     hs=None, sigma=0.3, A=0.4, kappa=1.0,
                     n_steps=3000, n_seeds=200):
    """Exp1B — full endogenous-q paths, same local-edge-per-tick objective.
    Measures whether the V3 rule h*(phi) stays a good POLICY-level spread
    even though q now evolves. Constant-phi environment (no schedule), fixed
    half-spread swept; local edge summed over the path, averaged over seeds."""
    if hs is None:
        hs = np.round(np.arange(0.80, 1.601, 0.02), 3)
    c_of = lambda phi: phi * sigma * np.sqrt(2.0 / np.pi)

    print(f"\n=== Exp1B: endogenous-q paths (local-edge objective) ===")
    print(f"  sigma={sigma}, n_steps={n_steps}, n_seeds={n_seeds}")
    print(f"  {'k':>5} {'phi':>4} {'h*_grid':>8} {'h*_fit':>7} {'h*(phi,0)':>9} "
          f"{'fit-V3':>7} {'mean_rms_q':>10}")
    for k in ks:
        for phi in phis:
            mean_edges = np.empty(len(hs))
            rms_qs_all = np.empty(len(hs))
            for j, h in enumerate(hs):
                edges, rms_qs = [], []
                for seed in range(n_seeds):
                    r = run(RandomWalk(sigma=sigma),
                            InventorySkewMaker(half_spread=h, k=k),
                            InformedFlow(A=A, kappa=kappa, phi=phi),
                            n_steps=n_steps, seed=seed)
                    exec_edge = (r["sells"] * (r["ask"] - r["S"])
                                 + r["buys"] * (r["S"] - r["bid"]))
                    local_edge = exec_edge - r["signed_flow"] * r["delta_S"]
                    edges.append(local_edge.sum())
                    rms_qs.append(np.sqrt(np.mean(r["inventory"] ** 2)))
                mean_edges[j] = np.mean(edges)
                rms_qs_all[j] = np.mean(rms_qs)
            j_best = int(np.argmax(mean_edges))
            h_grid = hs[j_best]
            # local quadratic fit around the grid argmax (+/-2 points)
            lo, hi = max(0, j_best - 2), min(len(hs), j_best + 3)
            if hi - lo >= 3:
                a, b, _ = np.polyfit(hs[lo:hi], mean_edges[lo:hi], 2)
                h_fit = -b / (2 * a) if a < 0 else h_grid
            else:
                h_fit = h_grid
            analytic = 1.0 / kappa + c_of(phi)
            print(f"  {k:>5} {phi:>4} {h_grid:>8.3f} {h_fit:>7.3f} "
                  f"{analytic:>9.4f} {h_fit - analytic:>7.3f} "
                  f"{rms_qs_all[j_best]:>10.3f}")


# --------------- v4a Experiment 2: does k>0 contaminate the estimator? ------
#
# One question only: with inventory skew on, bid/ask fill probabilities differ,
# so does the fill-based toxicity estimator still recover phi? Estimator is
# offline post-processing throughout Exp2A (no closed-loop quoting).
#
# Prediction: E[markout | fill] = phi*sigma*sqrt(2/pi) regardless of k, because
# the bid/ask asymmetry cancels once conditioned on a fill. So the paired
# Delta raw_phi = raw_phi(k=.04) - raw_phi(k=0) should be ~0, and the rolling
# estimator's bias/RMSE/noise should not materially worsen.

def exp2a_constant_phi(phis=(0.0, 0.5, 1.0), ks=(0.0, 0.04), sigma=0.3,
                       A=0.4, kappa=1.0, N=50, n_steps=3000, n_seeds=500):
    scale = sigma * np.sqrt(2.0 / np.pi)
    print(f"=== Exp2A: constant-phi calibration (offline estimator) ===")
    print(f"  sigma={sigma}, A={A}, kappa={kappa}, N={N}, "
          f"n_steps={n_steps}, n_seeds={n_seeds}")
    print(f"  {'phi':>4} {'k':>5} {'raw_phi':>8} {'roll_mean':>9} {'bias':>8} "
          f"{'rmse':>8} {'noise':>8} {'rms_q':>7} {'fills':>7}")

    # store per-seed raw_phi for the paired comparison
    raw_phi_by = {}  # (phi, k) -> array over seeds
    for phi in phis:
        h = 1.0 / kappa + phi * scale
        for k in ks:
            raw_phi_seeds = np.empty(n_seeds)
            roll_means, roll_all, rms_qs, fills_ct = [], [], [], []
            for seed in range(n_seeds):
                r = run(RandomWalk(sigma=sigma),
                        InventorySkewMaker(half_spread=h, k=k),
                        InformedFlow(A=A, kappa=kappa, phi=phi),
                        n_steps=n_steps, seed=seed)
                sf, ds = r["signed_flow"], r["delta_S"]
                mask = sf != 0
                markouts = sf[mask] * ds[mask]
                raw_phi_seeds[seed] = (markouts.mean() / scale
                                       if len(markouts) else np.nan)
                # offline rolling estimator. Metrics are computed at FILL
                # ticks only: run_offline carries the estimate forward across
                # no-fill ticks, and those repeated values would otherwise
                # inflate the apparent smoothness and understate the noise.
                phi_hat = RollingToxicityEstimator(N, sigma).run_offline(sf, ds)
                fill_hat = phi_hat[mask]                    # fill ticks only
                settled = fill_hat[~np.isnan(fill_hat)]     # post warm-up
                if len(settled):
                    roll_all.append(settled)
                    roll_means.append(settled.mean())
                rms_qs.append(np.sqrt(np.mean(r["inventory"] ** 2)))
                fills_ct.append(int(mask.sum()))
            raw_phi_by[(phi, k)] = raw_phi_seeds

            samples = np.concatenate(roll_all)
            mean_hat = samples.mean()
            rmse = np.sqrt(np.mean((samples - phi) ** 2))
            print(f"  {phi:>4} {k:>5} {np.nanmean(raw_phi_seeds):>8.4f} "
                  f"{mean_hat:>9.4f} {mean_hat - phi:>8.4f} {rmse:>8.4f} "
                  f"{samples.std():>8.4f} {np.mean(rms_qs):>7.3f} "
                  f"{np.mean(fills_ct):>7.1f}")

    # same-seed paired comparison: raw_phi(k=.04) - raw_phi(k=0)
    if 0.0 in ks and 0.04 in ks:
        print(f"  paired Delta raw_phi = raw_phi(k=.04) - raw_phi(k=0):")
        for phi in phis:
            d = raw_phi_by[(phi, 0.04)] - raw_phi_by[(phi, 0.0)]
            d = d[~np.isnan(d)]
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"    phi={phi}: mean={d.mean():+.5f}  se={se:.5f}  "
                  f"95%CI=[{d.mean() - 1.96 * se:+.5f}, "
                  f"{d.mean() + 1.96 * se:+.5f}]")


def exp2b_closed_loop(sigma=0.3, seg_len=1500, N=50, A=0.4, kappa=1.0,
                      h_warmup=1.08, n_seeds=300):
    """Closed-loop check: does skew change the estimator's response once it
    feeds back into quoting? Compare AdaptiveMaker(k=0) vs IntegratedMaker
    (k=.04), same schedule, same seed. No PnL — that's Exp3."""
    n_steps = 3 * seg_len
    schedule = step_schedule(n_steps, [seg_len, 2 * seg_len], [0.0, 1.0, 0.0])
    bounds = [0, seg_len, 2 * seg_len, n_steps]
    seg_phi = [0.0, 1.0, 0.0]

    def _makers(seed):
        a = run(RandomWalk(sigma=sigma),
                AdaptiveMaker(N=N, sigma=sigma, h_warmup=h_warmup,
                              kappa=kappa, k=0.0),
                RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule),
                n_steps=n_steps, seed=seed)
        i = run(RandomWalk(sigma=sigma),
                IntegratedMaker(k=0.04, N=N, sigma=sigma, h_warmup=h_warmup,
                                kappa=kappa, toxicity=True),
                RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule),
                n_steps=n_steps, seed=seed)
        return a, i

    def _lags_and_rmse(r):
        sf, ds = r["signed_flow"], r["delta_S"]
        phi_hat = RollingToxicityEstimator(N, sigma).run_offline(sf, ds)
        # response lag off the fill sequence, wrong-side-at-boundary excluded
        lags = {}
        for name, b, hi, cross in (("up", seg_len, bounds[2], "ge"),
                                    ("down", 2 * seg_len, bounds[3], "le")):
            pre = phi_hat[b - 1]
            if np.isnan(pre):
                lags[name] = (np.nan, np.nan, False)
                continue
            wrong = (pre >= 0.5) if cross == "ge" else (pre <= 0.5)
            if wrong:
                lags[name] = (np.nan, np.nan, False)
                continue
            fa, res = 0, None
            for t in range(b, hi):
                if sf[t] != 0:
                    fa += 1
                hit = (phi_hat[t] >= 0.5) if cross == "ge" else (phi_hat[t] <= 0.5)
                if not np.isnan(phi_hat[t]) and hit:
                    res = (fa, t - b)
                    break
            lags[name] = (res[0], res[1], True) if res else (np.nan, np.nan, False)
        # steady-state RMSE by regime, from FILL ticks only after the
        # N-in-regime-fill washout (transitions excluded). No-fill carry-
        # forward values are dropped so noise isn't understated.
        rmse = {}
        for idx in range(3):
            lo, hig = bounds[idx], bounds[idx + 1]
            seen, start = 0, None
            for t in range(lo, hig):
                if sf[t] != 0:
                    seen += 1
                    if seen >= N:
                        start = t + 1
                        break
            if start is not None:
                seg_mask = sf[start:hig] != 0
                seg = phi_hat[start:hig][seg_mask]
                seg = seg[~np.isnan(seg)]
                rmse[idx] = (np.sqrt(np.mean((seg - seg_phi[idx]) ** 2))
                             if len(seg) else np.nan)
            else:
                rmse[idx] = np.nan
        return lags, rmse, int((sf != 0).sum())

    agg = {"a": {"up_f": [], "up_t": [], "dn_f": [], "dn_t": [],
                 "rmse": {0: [], 1: [], 2: []}, "fills": []},
           "i": {"up_f": [], "up_t": [], "dn_f": [], "dn_t": [],
                 "rmse": {0: [], 1: [], 2: []}, "fills": []}}
    for seed in range(n_seeds):
        a, i = _makers(seed)
        for key, r in (("a", a), ("i", i)):
            lags, rmse, fills = _lags_and_rmse(r)
            if lags["up"][2]:
                agg[key]["up_f"].append(lags["up"][0])
                agg[key]["up_t"].append(lags["up"][1])
            if lags["down"][2]:
                agg[key]["dn_f"].append(lags["down"][0])
                agg[key]["dn_t"].append(lags["down"][1])
            for idx in range(3):
                if not np.isnan(rmse[idx]):
                    agg[key]["rmse"][idx].append(rmse[idx])
            agg[key]["fills"].append(fills)

    print(f"\n=== Exp2B: closed-loop estimator check (no PnL) ===")
    print(f"  sigma={sigma}, N={N}, seg_len={seg_len}, n_seeds={n_seeds}")
    print(f"  {'strategy':>16} {'up_fill':>8} {'up_tick':>8} {'dn_fill':>8} "
          f"{'dn_tick':>8} {'rmse_lo':>8} {'rmse_hi':>8} {'rmse_lo2':>9} "
          f"{'fills':>7}")
    for key, label in (("a", "Adaptive k=0"), ("i", "Integrated k=.04")):
        d = agg[key]
        print(f"  {label:>16} {np.mean(d['up_f']):>8.1f} "
              f"{np.mean(d['up_t']):>8.0f} {np.mean(d['dn_f']):>8.1f} "
              f"{np.mean(d['dn_t']):>8.0f} "
              f"{np.mean(d['rmse'][0]):>8.4f} {np.mean(d['rmse'][1]):>8.4f} "
              f"{np.mean(d['rmse'][2]):>9.4f} {np.mean(d['fills']):>7.1f}")


# --------------- v4a Experiment 3: inventory x toxicity economics -----------
#
# Frozen 2x2 controller design (no parameter changes, no new mechanism):
#
#                   Toxicity OFF     Toxicity ON
#   Inventory OFF   F (Fixed)        T (Toxicity-only)
#   Inventory ON    I (Inventory)    IT (Integrated)
#
# All four run same-seed CRN. gamma and treatment effects are computed
# seed-by-seed (not from aggregate means) so paired variance is preserved.

def exp3_interaction_economics(n_seeds=3000, sigma=0.3, seg_len=1500, N=50,
                               A=0.4, kappa=1.0, k=0.04, h_fixed=1.08):
    n_steps = 3 * seg_len
    schedule = step_schedule(n_steps, [seg_len, 2 * seg_len], [0.0, 1.0, 0.0])
    bounds = [0, seg_len, 2 * seg_len, n_steps]

    def _flow():
        return RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule)

    def _makers():
        return {
            "F": InventorySkewMaker(half_spread=h_fixed, k=0.0),
            "I": InventorySkewMaker(half_spread=h_fixed, k=k),
            "T": AdaptiveMaker(N=N, sigma=sigma, h_warmup=h_fixed,
                               kappa=kappa, k=0.0),
            "IT": IntegratedMaker(k=k, N=N, sigma=sigma, h_warmup=h_fixed,
                                  kappa=kappa, toxicity=True),
        }

    keys = ("F", "I", "T", "IT")
    pnl = {kk: np.empty(n_seeds) for kk in keys}
    rms_inv = {kk: np.empty(n_seeds) for kk in keys}
    maxabs = {kk: np.empty(n_seeds) for kk in keys}
    fills = {kk: np.empty(n_seeds) for kk in keys}
    mpt = {kk: np.empty(n_seeds) for kk in keys}       # markout per tick
    mean_h = {kk: np.empty(n_seeds) for kk in keys}
    h_reg = {kk: {0: np.empty(n_seeds), 1: np.empty(n_seeds),
                  2: np.empty(n_seeds)} for kk in keys}

    for seed in range(n_seeds):
        makers = _makers()   # fresh strategy state each seed
        for kk in keys:
            r = run(RandomWalk(sigma=sigma), makers[kk], _flow(),
                    n_steps=n_steps, seed=seed)
            pnl[kk][seed] = r["pnl"][-1]
            rms_inv[kk][seed] = np.sqrt(np.mean(r["inventory"] ** 2))
            maxabs[kk][seed] = np.max(np.abs(r["inventory"]))
            fills[kk][seed] = r["n_trades"]
            markout = r["signed_flow"] * r["delta_S"]
            mpt[kk][seed] = markout.sum() / n_steps
            hs = (r["ask"] - r["bid"]) / 2.0
            mean_h[kk][seed] = hs.mean()
            for i in range(3):
                h_reg[kk][i][seed] = hs[bounds[i]:bounds[i + 1]].mean()

    # ---- summary table ----
    print(f"=== Exp3: inventory x toxicity economics (2x2, CRN) ===")
    print(f"  sigma={sigma}, N={N}, k={k}, h_fixed={h_fixed}, "
          f"seg_len={seg_len}, n_seeds={n_seeds}")
    print(f"  {'strat':>5} {'mean_pnl':>9} {'std_pnl':>8} {'p05_pnl':>8} "
          f"{'rms_inv':>8} {'maxabs':>7} {'fills':>7} {'mkt/tick':>9} "
          f"{'mean_h':>7} {'h_low1':>7} {'h_high':>7} {'h_low2':>7}")
    for kk in keys:
        print(f"  {kk:>5} {pnl[kk].mean():>9.2f} {pnl[kk].std(ddof=1):>8.2f} "
              f"{np.percentile(pnl[kk], 5):>8.2f} {rms_inv[kk].mean():>8.3f} "
              f"{maxabs[kk].mean():>7.2f} {fills[kk].mean():>7.1f} "
              f"{mpt[kk].mean():>9.5f} {mean_h[kk].mean():>7.4f} "
              f"{h_reg[kk][0].mean():>7.4f} {h_reg[kk][1].mean():>7.4f} "
              f"{h_reg[kk][2].mean():>7.4f}")

    # ---- paired PnL comparisons (seed-by-seed) ----
    def _paired(d, label):
        m = d.mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"    {label:9} mean={m:+8.3f}  se={se:6.3f}  "
              f"95%CI=[{m - 1.96 * se:+8.3f}, {m + 1.96 * se:+8.3f}]")

    print(f"  paired PnL differences (same-seed):")
    _paired(pnl["I"] - pnl["F"], "I - F")
    _paired(pnl["T"] - pnl["F"], "T - F")
    _paired(pnl["IT"] - pnl["F"], "IT - F")
    _paired(pnl["IT"] - pnl["T"], "IT - T")
    _paired(pnl["IT"] - pnl["I"], "IT - I")

    # ---- factorial interaction ----
    gamma = pnl["IT"] - pnl["I"] - pnl["T"] + pnl["F"]
    gm = gamma.mean()
    gse = gamma.std(ddof=1) / np.sqrt(n_seeds)
    print(f"  factorial interaction gamma = IT - I - T + F (per seed):")
    print(f"    mean={gm:+.3f}  se={gse:.3f}  "
          f"95%CI=[{gm - 1.96 * gse:+.3f}, {gm + 1.96 * gse:+.3f}]  "
          f"(>0 complementary, <0 conflict, ~0 additive)")

    # ---- variance-reduction test (pre-specified P4) ----
    d_no_inv = pnl["T"] - pnl["F"]     # toxicity effect without inventory ctrl
    d_inv = pnl["IT"] - pnl["I"]       # toxicity effect with inventory ctrl
    se_no = d_no_inv.std(ddof=1) / np.sqrt(n_seeds)
    se_in = d_inv.std(ddof=1) / np.sqrt(n_seeds)
    print(f"  variance-reduction test (toxicity treatment effect):")
    print(f"    d_no_inventory (T - F):  mean={d_no_inv.mean():+.3f}  "
          f"se={se_no:.3f}")
    print(f"    d_inventory   (IT - I):  mean={d_inv.mean():+.3f}  "
          f"se={se_in:.3f}")
    print(f"    se_ratio SE(IT-I)/SE(T-F) = {se_in / se_no:.3f}  "
          f"(pre-specified prediction: < 1)")


# --------------- v4b: hidden stochastic (Markov) toxicity -------------------
#
# Environment: two-state Markov phi_t in {0,1}, switch prob p per tick (mean
# regime length ~1/p). Same fill mechanics, same N=50 rolling estimator as
# v3b/v4a. No estimator upgrade — the question is whether the existing tool
# still tracks when toxicity is genuinely hidden and stochastic.

P_PRIMARY = 0.002   # mean regime length ~500 ticks (comfortably > the ~170-tick
                    # N=50 response window, but switches ~9x over 4500 ticks)


def _switch_response_lags(phi_hat, schedule, sf, N):
    """Response behavior at true regime switches. For each eligible switch,
    count fills until phi_hat crosses 0.5 toward the new state, searching only
    up to the NEXT true switch.

    Eligibility (v3b convention): the estimator must be on the OLD-state side
    just before the switch — 0->1 requires pre < 0.5, 1->0 requires pre > 0.5.
    Switches already on the new side, or with NaN pre, are not eligible.

    Crucially, eligible transitions that do NOT cross before the next switch
    are counted as UNRESOLVED (they enter the denominator) rather than dropped
    — dropping them would hide exactly the short-regime failures that matter
    most under stochastic switching (survivorship bias).

    Returns (lags_list over resolved transitions, n_resolved, n_eligible).
    The caller pools lags across seeds so every resolved transition carries
    equal weight (a seed with 8 resolved switches shouldn't count the same as
    a seed with 1)."""
    switches = np.flatnonzero(np.diff(schedule) != 0) + 1
    lags = []
    n_eligible = 0
    for i, b in enumerate(switches):
        new_state = schedule[b]
        pre = phi_hat[b - 1]
        if np.isnan(pre):
            continue
        if new_state == 1.0:
            if pre >= 0.5:
                continue
            hit = lambda x: x >= 0.5
        else:
            if pre <= 0.5:
                continue
            hit = lambda x: x <= 0.5
        n_eligible += 1
        nxt = switches[switches > b]
        stop = nxt[0] if len(nxt) else len(schedule)
        fa, res = 0, None
        for t in range(b, stop):
            if sf[t] != 0:
                fa += 1
            if not np.isnan(phi_hat[t]) and hit(phi_hat[t]):
                res = fa
                break
        if res is not None:
            lags.append(res)          # resolved: record lag
        # else: unresolved — counted in n_eligible, no fabricated lag
    return lags, len(lags), n_eligible


def exp2b_markov_tracking(p=P_PRIMARY, sigma=0.3, N=50, A=0.4, kappa=1.0,
                          n_steps=4500, n_seeds=300):
    """Exp2 — can the existing N=50 rolling estimator track a hidden Markov
    toxicity path at k=0? Reports tracking RMSE and mean switch response lag.
    Offline estimator on a fixed-quote maker (h=1), so estimation is decoupled
    from quoting feedback."""
    rmses, fills_ct = [], []
    rmse_lo, rmse_hi = [], []
    lags, n_res_tot, n_elig_tot = [], 0, 0
    for seed in range(n_seeds):
        schedule = markov_schedule(n_steps, p, seed=seed)
        r = run(RandomWalk(sigma=sigma),
                InventorySkewMaker(half_spread=1.0, k=0.0),
                RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule),
                n_steps=n_steps, seed=seed)
        sf, ds = r["signed_flow"], r["delta_S"]
        phi_hat = RollingToxicityEstimator(N, sigma).run_offline(sf, ds)
        m = ~np.isnan(phi_hat)
        rmses.append(np.sqrt(np.mean((phi_hat[m] - schedule[m]) ** 2)))
        lo = m & (schedule == 0.0)
        hi = m & (schedule == 1.0)
        if lo.any():
            rmse_lo.append(np.sqrt(np.mean((phi_hat[lo] - 0.0) ** 2)))
        if hi.any():
            rmse_hi.append(np.sqrt(np.mean((phi_hat[hi] - 1.0) ** 2)))
        mean_lag, n_res, n_elig = _switch_response_lags(phi_hat, schedule, sf, N)
        lags.extend(mean_lag)          # pool all resolved transitions
        n_res_tot += n_res
        n_elig_tot += n_elig
        fills_ct.append(int((sf != 0).sum()))

    res_rate = n_res_tot / n_elig_tot if n_elig_tot else float("nan")
    print(f"=== Exp2b: N=50 rolling estimator on hidden Markov toxicity (k=0) ===")
    print(f"  sigma={sigma}, p={p} (mean regime ~{1/p:.0f} ticks), "
          f"N={N}, n_steps={n_steps}, n_seeds={n_seeds}")
    print(f"  tracking RMSE (all)   = {np.mean(rmses):.4f}")
    print(f"  tracking RMSE (low)   = {np.nanmean(rmse_lo):.4f}")
    print(f"  tracking RMSE (high)  = {np.nanmean(rmse_hi):.4f}")
    print(f"  switch resp lag       = {np.mean(lags):.1f} fills, resolved "
          f"(pooled over transitions; ~N/2 = {N/2:.0f} predicted)")
    print(f"  resolution rate       = {res_rate:.3f} "
          f"({n_res_tot}/{n_elig_tot} eligible switches crossed before next)")
    print(f"  mean fills            = {np.mean(fills_ct):.1f}")


def exp3b_markov_integrated(p=P_PRIMARY, sigma=0.3, N=50, A=0.4, kappa=1.0,
                            h_warmup=1.08, n_steps=4500, n_seeds=300):
    """Exp3 — put the SAME estimator into the full k=.04 IntegratedMaker under
    hidden Markov toxicity. Compare Toxicity-only (k=0) vs Integrated (k=.04),
    same-seed CRN (same latent path, same price path). Estimator side + economic
    side, both kept simple."""
    def _metrics(r, schedule):
        sf, ds = r["signed_flow"], r["delta_S"]
        phi_hat = RollingToxicityEstimator(N, sigma).run_offline(sf, ds)
        m = ~np.isnan(phi_hat)
        rmse = np.sqrt(np.mean((phi_hat[m] - schedule[m]) ** 2))
        seed_lags, n_res, n_elig = _switch_response_lags(phi_hat, schedule, sf, N)
        pnl = r["pnl"][-1]
        rms_inv = np.sqrt(np.mean(r["inventory"] ** 2))
        fills = int((sf != 0).sum())
        return rmse, seed_lags, n_res, n_elig, pnl, rms_inv, fills

    keys = ("T", "IT")
    rmse = {k: [] for k in keys}
    lag = {k: [] for k in keys}
    n_res = {k: 0 for k in keys}
    n_elig = {k: 0 for k in keys}
    pnl = {k: np.empty(n_seeds) for k in keys}
    rms_inv = {k: [] for k in keys}
    fills = {k: [] for k in keys}

    for seed in range(n_seeds):
        schedule = markov_schedule(n_steps, p, seed=seed)
        makers = {
            "T": AdaptiveMaker(N=N, sigma=sigma, h_warmup=h_warmup,
                               kappa=kappa, k=0.0),
            "IT": IntegratedMaker(k=0.04, N=N, sigma=sigma, h_warmup=h_warmup,
                                  kappa=kappa, toxicity=True),
        }
        for kk in keys:
            r = run(RandomWalk(sigma=sigma), makers[kk],
                    RegimeInformedFlow(A=A, kappa=kappa, phi_schedule=schedule),
                    n_steps=n_steps, seed=seed)
            rm, seed_lags, nr, ne, pn, ri, fl = _metrics(r, schedule)
            rmse[kk].append(rm)
            lag[kk].extend(seed_lags)      # pool resolved transitions
            n_res[kk] += nr
            n_elig[kk] += ne
            pnl[kk][seed] = pn
            rms_inv[kk].append(ri)
            fills[kk].append(fl)

    print(f"\n=== Exp3b: same estimator inside full IntegratedMaker (Markov) ===")
    print(f"  sigma={sigma}, p={p} (mean regime ~{1/p:.0f} ticks), N={N}, "
          f"k=.04, n_seeds={n_seeds}")
    print(f"  {'strategy':>16} {'phi_rmse':>9} {'lag_res':>8} {'res_rate':>9} "
          f"{'mean_pnl':>9} {'rms_inv':>8} {'fills':>7}")
    for kk, label in (("T", "Toxicity-only k=0"), ("IT", "Integrated k=.04")):
        rr = n_res[kk] / n_elig[kk] if n_elig[kk] else float("nan")
        pooled_lag = np.mean(lag[kk]) if lag[kk] else float("nan")
        print(f"  {label:>16} {np.mean(rmse[kk]):>9.4f} "
              f"{pooled_lag:>8.1f} {rr:>9.3f} {pnl[kk].mean():>9.2f} "
              f"{np.mean(rms_inv[kk]):>8.3f} {np.mean(fills[kk]):>7.1f}")
    d = pnl["IT"] - pnl["T"]
    se = d.std(ddof=1) / np.sqrt(n_seeds)
    print(f"  paired IT - T: mean={d.mean():+.3f}  se={se:.3f}  "
          f"95%CI=[{d.mean()-1.96*se:+.3f}, {d.mean()+1.96*se:+.3f}]")


if __name__ == "__main__":
    exp2b_markov_tracking()
    exp3b_markov_integrated()
