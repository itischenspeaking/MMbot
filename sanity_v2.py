"""v2 sanity checks.

The 2D (k, sigma) sweep showed mean/std peaking at positive k even when
sigma=0, where there is no price movement and so — supposedly — no inventory
risk. Before reading anything into that, verify that sigma affects no trading
dynamics except inventory mark-to-market PnL. Sigma does move the absolute
S / bid / ask, but it must not change quote distances, fills, the inventory
path, or execution edge. If any of these fail, it's a bug, not a result.

    python sanity_v2.py
"""

import numpy as np

from market import RandomWalk
from simulator import QuoteSensitiveFlow, run
from strategy import InventorySkewMaker


def _run(sigma, k, seed, n_steps=2000):
    return run(
        RandomWalk(sigma=sigma),
        InventorySkewMaker(half_spread=1.0, k=k),
        QuoteSensitiveFlow(A=0.4, kappa=1.0),
        n_steps=n_steps,
        seed=seed,
    )


def _decompose(r):
    execution = np.sum(
        r["buys"] * (r["S"] - r["bid"]) + r["sells"] * (r["ask"] - r["S"])
    )
    total = r["pnl"][-1]
    return execution, total - execution


def test_sigma0_inventory_pnl_is_zero():
    """No price movement => inventory can't make or lose money."""
    for seed in range(20):
        r = _run(sigma=0.0, k=0.16, seed=seed)
        _, inv_pnl = _decompose(r)
        assert np.allclose(inv_pnl, 0.0), (seed, inv_pnl)


def test_sigma_leaves_fills_and_inventory_unchanged():
    """S cancels out of the quote distance, and market/flow RNG are separate,
    so changing sigma must not move a single fill."""
    base = _run(sigma=0.0, k=0.16, seed=7)
    for sigma in (0.1, 0.4):
        r = _run(sigma=sigma, k=0.16, seed=7)
        for key in ("buys", "sells", "inventory"):
            assert np.array_equal(base[key], r[key]), (sigma, key)


def test_execution_edge_independent_of_sigma():
    """Fills are identical across sigma, and each fill's edge (h +/- k*q)
    has no sigma in it, so the execution total can't depend on sigma."""
    exec0, _ = _decompose(_run(sigma=0.0, k=0.16, seed=7))
    for sigma in (0.1, 0.4):
        execs, _ = _decompose(_run(sigma=sigma, k=0.16, seed=7))
        assert np.allclose(exec0, execs), (sigma, exec0, execs)


def test_inventory_pnl_scales_linearly_in_sigma():
    """Same seed => same epsilon path; same inventory path (verified above);
    so inventory PnL = sum q*dS scales with sigma."""
    for seed in range(10):
        _, inv01 = _decompose(_run(sigma=0.1, k=0.16, seed=seed))
        _, inv04 = _decompose(_run(sigma=0.4, k=0.16, seed=seed))
        assert np.allclose(inv04, 4.0 * inv01), (seed, inv01, inv04)


def test_pnl_decomposition_identity():
    """Independent witness for the residual split.

    _decompose defines inventory PnL as total - execution, so it can't catch a
    systematic error in that subtraction. Recompute the inventory leg straight
    from the price path as sum q_t * (S_{t+1} - S_t) — the inventory held after
    trading at t, times the move that follows — and require the two to agree,
    and their sum to reconstruct terminal PnL.
    """
    for sigma in (0.0, 0.1, 0.4):
        for seed in range(10):
            r = _run(sigma=sigma, k=0.16, seed=seed)
            execution, inv_residual = _decompose(r)
            inv_direct = np.sum(r["inventory"][:-1] * np.diff(r["S"]))
            assert np.allclose(inv_residual, inv_direct), (
                sigma, seed, inv_residual, inv_direct)
            assert np.allclose(r["pnl"][-1], execution + inv_direct), (sigma, seed)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
