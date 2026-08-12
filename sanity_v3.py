"""v3 sanity checks.

    1. phi=0 recovers v2 byte-identically
    2. changing phi leaves price path untouched
    3. informed fills produce positive customer-signed markout
    4. markout increases with phi

    python sanity_v3.py
"""

import numpy as np

from market import RandomWalk
from simulator import InformedFlow, QuoteSensitiveFlow, run
from strategy import InventorySkewMaker


def _run_v2(seed, **kw):
    return run(RandomWalk(**kw.pop("market_kw", {})),
               InventorySkewMaker(half_spread=1.0, k=0.04),
               QuoteSensitiveFlow(A=0.4, kappa=1.0),
               n_steps=500, seed=seed)


def _run_v3(seed, phi=0.0, **kw):
    return run(RandomWalk(**kw.pop("market_kw", {})),
               InventorySkewMaker(half_spread=1.0, k=0.04),
               InformedFlow(A=0.4, kappa=1.0, phi=phi),
               n_steps=500, seed=seed)


def test_phi0_statistically_symmetric():
    """phi=0 can't be byte-identical to v2 (one attempt per step vs two
    independent sides), so check the structure instead: buys and sells
    balanced, fill rate near the one-sided theory value, max one fill
    per step."""
    n_buys, n_sells, n_steps_total = 0, 0, 0
    for seed in range(50):
        r = _run_v3(seed, phi=0.0)
        n_buys += r["buys"].sum()
        n_sells += r["sells"].sum()
        n_steps_total += len(r["buys"])
        # one trader per step: never two fills in one tick
        assert not np.any(r["buys"] & r["sells"])
    # symmetric sides: buys ~= sells (binomial, generous 3% tolerance)
    total = n_buys + n_sells
    assert abs(n_buys - n_sells) / total < 0.03, (n_buys, n_sells)
    # fill rate ~= A * exp(-kappa*h) * (coin picks a side) — with h=1,
    # k=0.04 skew wobbles this slightly; check within 10%
    expected = 0.4 * np.exp(-1.0)
    observed = total / n_steps_total
    assert abs(observed - expected) / expected < 0.10, (observed, expected)


def test_price_path_independent_of_phi():
    """Price is exogenous; changing phi changes fills but not S."""
    a = _run_v3(seed=7, phi=0.0)
    for phi in (0.3, 0.7, 1.0):
        b = _run_v3(seed=7, phi=phi)
        assert np.array_equal(a["S"], b["S"]), phi
        assert np.array_equal(a["delta_S"], b["delta_S"]), phi


def test_fill_count_stable_across_phi():
    """The bug being fixed: phi must not mechanically change trade volume.
    Informed and uninformed traders share one arrival and one fill mechanism,
    so mean fills should be flat in phi (within Monte Carlo noise)."""
    counts = []
    for phi in (0.0, 0.5, 1.0):
        total = sum(_run_v3(seed, phi=phi)["n_trades"] for seed in range(100))
        counts.append(total / 100)
    lo, hi = min(counts), max(counts)
    assert (hi - lo) / lo < 0.05, counts


def _avg_markout(phi, n_seeds=200, n_steps=500):
    """Mean of signed_flow * delta_S over fills, averaged across seeds."""
    markouts = []
    for seed in range(n_seeds):
        r = _run_v3(seed, phi=phi)
        sf = r["signed_flow"]
        ds = r["delta_S"]
        mask = sf != 0
        if mask.any():
            markouts.append(np.mean(sf[mask] * ds[mask]))
    return np.mean(markouts)


def test_informed_markout_positive():
    """Customer-signed markout should be positive when phi > 0 —
    informed customers are right about direction on average."""
    m = _avg_markout(0.5)
    assert m > 0, f"Expected positive markout at phi=0.5, got {m:.6f}"


def test_markout_increases_with_phi():
    """More toxicity => more adverse selection => higher markout."""
    m0 = _avg_markout(0.0)
    m3 = _avg_markout(0.3)
    m7 = _avg_markout(0.7)
    assert m3 > m0, f"markout at phi=0.3 ({m3:.6f}) <= phi=0.0 ({m0:.6f})"
    assert m7 > m3, f"markout at phi=0.7 ({m7:.6f}) <= phi=0.3 ({m3:.6f})"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
