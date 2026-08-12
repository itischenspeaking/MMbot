"""v3b Stage 1 sanity checks — dynamic hidden toxicity mechanism only.

    1. constant phi_schedule byte-identical to InformedFlow
    2. price path independent of phi_schedule
    3. fill volume stable across regimes (low segment ~= high segment)
    4. per-regime signed markout tracks phi_segment * sigma * sqrt(2/pi)

    python sanity_v3b.py
"""

import numpy as np

from market import RandomWalk
from simulator import InformedFlow, RegimeInformedFlow, step_schedule, run
from strategy import InventorySkewMaker

SIGMA = 0.1
SIGMA_SQRT_2_OVER_PI = SIGMA * np.sqrt(2.0 / np.pi)


def _maker():
    return InventorySkewMaker(half_spread=1.0, k=0.0)  # k=0: v3b main-experiment maker


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
    h, a low-phi segment and a high-phi segment should see statistically
    the same fill count. Compare low-high-low segments against each other."""
    seg_len = 300
    n_steps = 3 * seg_len
    schedule = step_schedule(n_steps, [seg_len, 2 * seg_len], [0.0, 1.0, 0.0])

    seg_fills = np.zeros(3)
    n_seeds = 150
    for seed in range(n_seeds):
        r = run(RandomWalk(sigma=SIGMA), _maker(),
                RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
                n_steps=n_steps, seed=seed)
        filled = r["buys"] | r["sells"]
        seg_fills[0] += filled[0:seg_len].sum()
        seg_fills[1] += filled[seg_len:2 * seg_len].sum()
        seg_fills[2] += filled[2 * seg_len:3 * seg_len].sum()
    seg_fills /= n_seeds

    lo, hi = seg_fills.min(), seg_fills.max()
    assert (hi - lo) / lo < 0.10, seg_fills


def test_segment_markout_matches_theory():
    seg_len = 400
    n_steps = 3 * seg_len
    phis = (0.0, 1.0, 0.3)
    schedule = step_schedule(n_steps, [seg_len, 2 * seg_len], list(phis))

    n_seeds = 150
    seg_sf = [[] for _ in range(3)]
    seg_ds = [[] for _ in range(3)]
    bounds = [0, seg_len, 2 * seg_len, 3 * seg_len]
    for seed in range(n_seeds):
        r = run(RandomWalk(sigma=SIGMA), _maker(),
                RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=schedule),
                n_steps=n_steps, seed=seed)
        for i in range(3):
            lo, hi = bounds[i], bounds[i + 1]
            seg_sf[i].append(r["signed_flow"][lo:hi])
            seg_ds[i].append(r["delta_S"][lo:hi])

    for i, phi in enumerate(phis):
        sf = np.concatenate(seg_sf[i])
        ds = np.concatenate(seg_ds[i])
        mask = sf != 0
        markout = np.mean(sf[mask] * ds[mask])
        expected = phi * SIGMA_SQRT_2_OVER_PI
        if expected == 0.0:
            assert abs(markout) < 0.01, (phi, markout)
        else:
            assert abs(markout - expected) / expected < 0.20, (phi, markout, expected)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
