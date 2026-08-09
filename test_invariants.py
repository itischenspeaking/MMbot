"""Invariants that must hold whatever gets added later.

    python test_invariants.py
"""

import numpy as np

from market import RandomWalk
from simulator import Account, BernoulliFlow, run
from strategy import NaiveMaker


def base(**kw):
    return run(
        RandomWalk(sigma=kw.pop("sigma", 0.1)),
        NaiveMaker(half_spread=kw.pop("half_spread", 0.5)),
        BernoulliFlow(trade_prob=kw.pop("trade_prob", 0.3)),
        **kw,
    )


def test_reproducible():
    a, b = base(n_steps=500, seed=7), base(n_steps=500, seed=7)
    for k in ("S", "pnl", "inventory"):
        assert np.array_equal(a[k], b[k]), k


def test_market_path_independent_of_flow():
    """The whole point of splitting the streams."""
    a = base(n_steps=500, seed=7, trade_prob=0.3)
    b = base(n_steps=500, seed=7, trade_prob=0.9)
    assert np.array_equal(a["S"], b["S"])


def test_no_flow_no_pnl():
    r = base(n_steps=500, seed=7, trade_prob=0.0)
    assert r["n_trades"] == 0
    assert np.all(r["pnl"] == 0.0)
    assert np.all(r["inventory"] == 0)


def test_quotes_straddle_mid():
    r = base(n_steps=500, seed=7, half_spread=0.75)
    assert np.allclose((r["bid"] + r["ask"]) / 2, r["S"])
    assert np.allclose(r["ask"] - r["bid"], 1.5)


def test_accounting():
    """Rebuild cash and inventory from the fill log."""
    r = base(n_steps=500, seed=7)
    inv = np.cumsum(r["buys"].astype(int) - r["sells"].astype(int))
    cash = np.cumsum(-(r["buys"].astype(float) * r["bid"]) + r["sells"] * r["ask"])
    assert np.array_equal(inv, r["inventory"])
    assert np.allclose(cash, r["cash"])
    assert np.allclose(r["cash"] + r["inventory"] * r["S"], r["pnl"])
    assert r["n_trades"] == int(r["buys"].sum() + r["sells"].sum())


def test_clock_length():
    """n rows, n - 1 price moves."""
    n = 50
    r = base(n_steps=n, seed=7, trade_prob=0.0)
    assert len(r["S"]) == n
    assert np.count_nonzero(np.diff(r["S"])) == n - 1


def test_bad_params_raise():
    for fn in (
        lambda: RandomWalk(sigma=-1),
        lambda: NaiveMaker(half_spread=0),
        lambda: BernoulliFlow(trade_prob=1.5),
        lambda: base(n_steps=0),
    ):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
