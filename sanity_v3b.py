"""v3b Stage 1 sanity checks — dynamic hidden toxicity mechanism only.

Primary v3b schedule (locked in Stage 0):
    sigma=0.3, phi 0->1->0, seg_len=1500, n_steps=4500, k=0.

    1. constant phi_schedule byte-identical to InformedFlow
    2. price path independent of phi_schedule
    3. fill volume stable across regimes (low segment ~= high segment)
    4. per-regime signed markout tracks phi_segment * sigma * sqrt(2/pi)
    5. schedule boundary ticks land in the right regime (off-by-one guard)

    python sanity_v3b.py
"""

import numpy as np

from market import RandomWalk
from simulator import InformedFlow, RegimeInformedFlow, step_schedule, run
from strategy import InventorySkewMaker, RollingToxicityEstimator

SIGMA = 0.3  # v3b primary (Stage 0)
SIGMA_SQRT_2_OVER_PI = SIGMA * np.sqrt(2.0 / np.pi)

# Primary low->high->low schedule, reused across the tests below.
SEG_LEN = 1500
N_STEPS = 3 * SEG_LEN
BREAKPOINTS = [SEG_LEN, 2 * SEG_LEN]
PHIS = [0.0, 1.0, 0.0]


def _maker():
    return InventorySkewMaker(half_spread=1.0, k=0.0)  # k=0: v3b main-experiment maker


def _primary_schedule():
    return step_schedule(N_STEPS, BREAKPOINTS, PHIS)


def test_constant_schedule_byte_identical_to_informedflow():
    n_steps = 500
    for phi in (0.0, 0.3, 0.7, 1.0):
        schedule = np.full(n_steps, phi)
        a = run(RandomWalk(sigma=SIGMA), _maker(),
                InformedFlow(A=0.4, kappa=1.0, phi=phi),
                n_steps=n_steps, seed=11)
        b = run(RandomWalk(sigma=SIGMA), _maker(),
                RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
                n_steps=n_steps, seed=11)
        for key in ("S", "bid", "ask", "inventory", "cash", "pnl",
                    "buys", "sells", "delta_S", "signed_flow"):
            assert np.array_equal(a[key], b[key]), (phi, key)
        assert a["n_trades"] == b["n_trades"], phi


def test_price_path_independent_of_schedule():
    n_steps = 900
    schedules = [
        np.zeros(n_steps),
        np.ones(n_steps),
        step_schedule(n_steps, [300, 600], [0.0, 1.0, 0.2]),
    ]
    base = run(RandomWalk(sigma=SIGMA), _maker(),
               RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedules[0]),
               n_steps=n_steps, seed=5)
    for sched in schedules[1:]:
        r = run(RandomWalk(sigma=SIGMA), _maker(),
                RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=sched),
                n_steps=n_steps, seed=5)
        assert np.array_equal(base["S"], r["S"])
        assert np.array_equal(base["delta_S"], r["delta_S"])


def test_fill_volume_stable_across_regimes():
    """Fill rate depends on h, not phi (v3a result) — so with k=0 and fixed
    h, the low-phi and high-phi segments should see statistically the same
    fill count. Uses the primary 0->1->0 schedule (seg_len=1500)."""
    schedule = _primary_schedule()

    seg_fills = np.zeros(3)
    n_seeds = 100
    for seed in range(n_seeds):
        r = run(RandomWalk(sigma=SIGMA), _maker(),
                RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
                n_steps=N_STEPS, seed=seed)
        filled = r["buys"] | r["sells"]
        seg_fills[0] += filled[0:SEG_LEN].sum()
        seg_fills[1] += filled[SEG_LEN:2 * SEG_LEN].sum()
        seg_fills[2] += filled[2 * SEG_LEN:3 * SEG_LEN].sum()
    seg_fills /= n_seeds

    lo, hi = seg_fills.min(), seg_fills.max()
    assert (hi - lo) / lo < 0.05, seg_fills


def test_segment_markout_matches_theory():
    """Per-regime signed markout, primary 0->1->0 schedule. Expect
    ~0, ~sigma*sqrt(2/pi)=0.2394, ~0 at sigma=0.3."""
    schedule = _primary_schedule()

    n_seeds = 100
    seg_sf = [[] for _ in range(3)]
    seg_ds = [[] for _ in range(3)]
    bounds = [0, SEG_LEN, 2 * SEG_LEN, 3 * SEG_LEN]
    for seed in range(n_seeds):
        r = run(RandomWalk(sigma=SIGMA), _maker(),
                RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
                n_steps=N_STEPS, seed=seed)
        for i in range(3):
            lo, hi = bounds[i], bounds[i + 1]
            seg_sf[i].append(r["signed_flow"][lo:hi])
            seg_ds[i].append(r["delta_S"][lo:hi])

    for i, phi in enumerate(PHIS):
        sf = np.concatenate(seg_sf[i])
        ds = np.concatenate(seg_ds[i])
        mask = sf != 0
        markout = np.mean(sf[mask] * ds[mask])
        expected = phi * SIGMA_SQRT_2_OVER_PI
        if expected == 0.0:
            assert abs(markout) < 0.02, (i, phi, markout)
        else:
            assert abs(markout - expected) / expected < 0.10, (i, phi, markout, expected)


def test_schedule_boundary_ticks():
    """Off-by-one guard on the primary 0->1->0 schedule. The hidden phi_t
    logged by run() must place each boundary tick in the intended regime:
    1499->low, 1500->high, 2999->high, 3000->low. Uses n_trades-free logging
    (phi_true is recorded every tick regardless of fills)."""
    schedule = _primary_schedule()
    # Check the schedule array directly...
    assert schedule[1499] == 0.0
    assert schedule[1500] == 1.0
    assert schedule[2999] == 1.0
    assert schedule[3000] == 0.0
    # ...and that run() logs phi_true identically (no indexing drift in loop).
    r = run(RandomWalk(sigma=SIGMA), _maker(),
            RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
            n_steps=N_STEPS, seed=0)
    pt = r["phi_true"]
    assert pt[1499] == 0.0 and pt[1500] == 1.0
    assert pt[2999] == 1.0 and pt[3000] == 0.0
    assert np.array_equal(pt, schedule)


# --- Stage 2 estimator sanity ---

def test_estimator_nan_before_N_fills():
    """phi_hat is NaN until exactly N fills have entered the window."""
    N = 10
    est = RollingToxicityEstimator(N, SIGMA)
    fills_seen = 0
    rng = np.random.default_rng(0)
    for _ in range(500):
        sf = int(rng.choice([-1, 0, 1], p=[0.07, 0.86, 0.07]))
        ds = rng.standard_normal() * SIGMA
        phi = est.update(sf, ds)
        if sf != 0:
            fills_seen += 1
        if fills_seen < N:
            assert np.isnan(phi), (fills_seen, phi)
        else:
            assert not np.isnan(phi), (fills_seen, phi)


def test_estimator_nofill_ticks_dont_update():
    """A no-fill tick must not change the window or the estimate — not even
    its delta_S (fed a huge 999 here to prove it is ignored)."""
    est = RollingToxicityEstimator(5, SIGMA)
    for _ in range(5):
        est.update(1, 0.2)  # five fills to warm up
    phi_before = est.update(1, 0.2)
    snapshot = list(est.markouts)
    phi_after = est.update(0, 999.0)  # no-fill: delta_S must be ignored
    assert phi_after == phi_before
    assert list(est.markouts) == snapshot


def test_estimator_recovers_constant_phi():
    """Long constant-phi runs: phi_hat settles near the true phi."""
    for phi, tol in [(0.0, 0.05), (0.5, 0.08), (1.0, 0.05)]:
        schedule = np.full(6000, phi)
        r = run(RandomWalk(sigma=SIGMA), _maker(),
                RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
                n_steps=6000, seed=1)
        phi_hat = RollingToxicityEstimator(100, SIGMA).run_offline(
            r["signed_flow"], r["delta_S"])
        settled = phi_hat[~np.isnan(phi_hat)]
        assert abs(settled.mean() - phi) < tol, (phi, settled.mean())


def test_estimator_offline_independent_of_N():
    """Estimation is pure post-processing: the simulated price/fill path does
    not depend on N. Different N give different estimates from one same log,
    and running the sim again yields the identical log."""
    sched = _primary_schedule()
    r = run(RandomWalk(sigma=SIGMA), _maker(),
            RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=sched),
            n_steps=N_STEPS, seed=3)
    sf, ds = r["signed_flow"], r["delta_S"]
    h20 = RollingToxicityEstimator(20, SIGMA).run_offline(sf, ds)
    h100 = RollingToxicityEstimator(100, SIGMA).run_offline(sf, ds)
    assert not np.array_equal(np.nan_to_num(h20), np.nan_to_num(h100))
    r2 = run(RandomWalk(sigma=SIGMA), _maker(),
             RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=sched),
             n_steps=N_STEPS, seed=3)
    assert np.array_equal(sf, r2["signed_flow"])
    assert np.array_equal(ds, r2["delta_S"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
