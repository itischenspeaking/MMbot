"""v4pre sanity — IntegratedMaker (v2 inventory skew + v3b adaptive width).

    1. k=0                       -> byte-identical to v3b's AdaptiveMaker
    2. toxicity disabled         -> byte-identical to v2's InventorySkewMaker
    3. k=0, toxicity disabled    -> byte-identical to a static maker
    4. IntegratedMaker never reads phi_true (structural)
    5. markout at tick t affects only the quote at tick t+1 (no look-ahead)
    6. same seed, different k    -> S and delta_S byte-identical (CRN)

    python sanity_v4.py
"""

import numpy as np

from market import RandomWalk
from simulator import RegimeInformedFlow, step_schedule, run
from strategy import (NaiveMaker, InventorySkewMaker, AdaptiveMaker,
                      IntegratedMaker, RollingToxicityEstimator, _h_star)
from simulator import InformedFlow

SIGMA = 0.3
N = 50
H_WARMUP = 1.08
KAPPA = 1.0
SEG_LEN = 1500
N_STEPS = 3 * SEG_LEN


def _schedule():
    return step_schedule(N_STEPS, [SEG_LEN, 2 * SEG_LEN], [0.0, 1.0, 0.0])


def _flow(sched):
    return RegimeInformedFlow(A=0.4, kappa=KAPPA, phi_schedule=sched)


def test_k0_recovers_v3b_adaptive():
    """k=0 must reduce IntegratedMaker to v3b's AdaptiveMaker exactly."""
    sched = _schedule()
    ri = run(RandomWalk(sigma=SIGMA),
             IntegratedMaker(k=0.0, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                             kappa=KAPPA, toxicity=True),
             _flow(sched), n_steps=N_STEPS, seed=0)
    ra = run(RandomWalk(sigma=SIGMA),
             AdaptiveMaker(N=N, sigma=SIGMA, h_warmup=H_WARMUP, kappa=KAPPA, k=0.0),
             _flow(sched), n_steps=N_STEPS, seed=0)
    for key in ("S", "bid", "ask", "inventory", "cash", "pnl",
                "buys", "sells", "delta_S", "signed_flow"):
        assert np.array_equal(ri[key], ra[key]), key
    assert ri["n_trades"] == ra["n_trades"]


def test_toxicity_disabled_recovers_v2():
    """toxicity=False, k>0 must reduce to v2's InventorySkewMaker exactly:
    fixed half-spread = h_warmup, center skewed by k."""
    sched = _schedule()
    k = 0.04
    ri = run(RandomWalk(sigma=SIGMA),
             IntegratedMaker(k=k, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                             kappa=KAPPA, toxicity=False),
             _flow(sched), n_steps=N_STEPS, seed=1)
    rv2 = run(RandomWalk(sigma=SIGMA),
              InventorySkewMaker(half_spread=H_WARMUP, k=k),
              _flow(sched), n_steps=N_STEPS, seed=1)
    for key in ("S", "bid", "ask", "inventory", "cash", "pnl",
                "buys", "sells", "delta_S", "signed_flow"):
        assert np.array_equal(ri[key], rv2[key]), key
    assert ri["n_trades"] == rv2["n_trades"]


def test_k0_toxicity_disabled_recovers_static():
    """k=0 and toxicity=False must reduce to a plain static maker: quotes
    S +/- h_warmup with no skew, no adaptation."""
    sched = _schedule()
    ri = run(RandomWalk(sigma=SIGMA),
             IntegratedMaker(k=0.0, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                             kappa=KAPPA, toxicity=False),
             _flow(sched), n_steps=N_STEPS, seed=2)
    rs = run(RandomWalk(sigma=SIGMA),
             NaiveMaker(half_spread=H_WARMUP),
             _flow(sched), n_steps=N_STEPS, seed=2)
    for key in ("S", "bid", "ask", "inventory", "cash", "pnl",
                "buys", "sells", "delta_S", "signed_flow"):
        assert np.array_equal(ri[key], rs[key]), key
    assert ri["n_trades"] == rs["n_trades"]


def test_never_reads_phi_true():
    """Structural check: IntegratedMaker has no channel to the hidden
    schedule. It must not expose set_t (the hook run() uses to feed Oracle
    the true phi) and must not hold a phi_schedule attribute."""
    maker = IntegratedMaker(k=0.04, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                            kappa=KAPPA, toxicity=True)
    assert not hasattr(maker, "set_t")
    assert not hasattr(maker, "phi_schedule")
    assert not hasattr(maker, "phi_true")
    # and a full run doesn't error or silently rely on such an attribute
    sched = _schedule()
    run(RandomWalk(sigma=SIGMA), maker, _flow(sched), n_steps=N_STEPS, seed=3)


def test_no_lookahead():
    """The quote at tick t must use phi_hat from fills strictly before t.
    Post-warmup, realized h must equal h_star(phi_hat observed up to t-1).
    Same check as v3b's AdaptiveMaker test, applied to IntegratedMaker."""
    sched = _schedule()
    ri = run(RandomWalk(sigma=SIGMA),
             IntegratedMaker(k=0.04, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                             kappa=KAPPA, toxicity=True),
             _flow(sched), n_steps=N_STEPS, seed=4)
    h_realized = (ri["ask"] - ri["bid"]) / 2.0
    phi_after_t = RollingToxicityEstimator(N, SIGMA).run_offline(
        ri["signed_flow"], ri["delta_S"])
    for t in range(1, N_STEPS):
        pt = phi_after_t[t - 1]
        if np.isnan(pt):
            continue
        assert abs(h_realized[t] - _h_star(pt, KAPPA, SIGMA)) < 1e-9, t


def test_crn_k_does_not_perturb_price_path():
    """Same seed, different k: S and delta_S must be byte-identical. Guards
    against strategy-dependent randomness leaking into the exogenous market
    path during integration — the market draws from its own RNG stream,
    untouched by the strategy's k."""
    sched = _schedule()
    common = dict(n_steps=N_STEPS, seed=5)
    r_k0 = run(RandomWalk(sigma=SIGMA),
               IntegratedMaker(k=0.0, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                               kappa=KAPPA, toxicity=True),
               _flow(sched), **common)
    r_k1 = run(RandomWalk(sigma=SIGMA),
               IntegratedMaker(k=0.04, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                               kappa=KAPPA, toxicity=True),
               _flow(sched), **common)
    r_k2 = run(RandomWalk(sigma=SIGMA),
               IntegratedMaker(k=0.10, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                               kappa=KAPPA, toxicity=True),
               _flow(sched), **common)
    assert np.array_equal(r_k0["S"], r_k1["S"])
    assert np.array_equal(r_k0["S"], r_k2["S"])
    assert np.array_equal(r_k0["delta_S"], r_k1["delta_S"])
    assert np.array_equal(r_k0["delta_S"], r_k2["delta_S"])
    # fills should actually diverge across k (else this check is trivial)
    assert not np.array_equal(r_k0["signed_flow"], r_k1["signed_flow"])


def test_exp1a_batch_matches_informedflow():
    """Exp1A's vectorised local-edge computation must reproduce InformedFlow's
    per-tick fill logic exactly, not a re-invented approximation. Drive both
    with identical draws (same delta_S, u_side, u_fill, u_type) at fixed q and
    check EVERY tick's (edge, signed_flow) agrees element-wise. Guards against
    'validating a world slightly different from the simulator'."""
    S, q, h, k = 100.0, 7, 1.1, 0.04
    phi, sigma, A, kappa = 0.5, 0.3, 0.4, 1.0
    center = S - k * q
    bid, ask = center - h, center + h
    d_bid, d_ask = S - bid, ask - S
    p_bid = min(1.0, A * np.exp(-kappa * d_bid))
    p_ask = min(1.0, A * np.exp(-kappa * d_ask))
    flow = InformedFlow(A=A, kappa=kappa, phi=phi)

    n = 20000
    # draw in the exact order _local_edge_batch uses
    rng = np.random.default_rng(123)
    delta_S = sigma * rng.standard_normal(n)
    u_side = rng.random(n)
    u_fill = rng.random(n)
    u_type = rng.random(n)

    # vectorised per-tick local edge (mirror of _local_edge_batch internals)
    informed = u_type < phi
    side_buy = np.where(informed, delta_S > 0, u_side < 0.5)
    side_sell = np.where(informed, delta_S < 0, ~(u_side < 0.5))
    informed_flat = informed & (delta_S == 0)
    side_buy &= ~informed_flat
    side_sell &= ~informed_flat
    filled_ask = side_buy & (u_fill < p_ask)
    filled_bid = side_sell & (u_fill < p_bid)
    vec_edge = np.zeros(n)
    vec_edge[filled_ask] = d_ask
    vec_edge[filled_bid] = d_bid
    vec_signed = np.zeros(n)
    vec_signed[filled_ask] = 1.0
    vec_signed[filled_bid] = -1.0

    # reference: InformedFlow.fills tick-by-tick on the SAME draws
    class _ScriptedRNG:
        def __init__(self, pair): self.pair = pair
        def random(self, k_=None):
            return np.array(self.pair) if k_ == 2 else self.pair[0]

    class _ScriptedInf:
        def __init__(self, v): self.v = v
        def random(self): return self.v

    for i in range(n):
        hit_bid, lift_ask = flow.fills(
            S, bid, ask, _ScriptedRNG((u_side[i], u_fill[i])),
            delta_S=delta_S[i], informed_rng=_ScriptedInf(u_type[i]))
        ref_signed = 1.0 if lift_ask else (-1.0 if hit_bid else 0.0)
        ref_edge = d_ask if lift_ask else (d_bid if hit_bid else 0.0)
        assert vec_signed[i] == ref_signed, (i, vec_signed[i], ref_signed)
        assert abs(vec_edge[i] - ref_edge) < 1e-12, (i, vec_edge[i], ref_edge)

    # and the batch mean matches the per-tick local edge mean
    from experiments import _local_edge_batch
    rng2 = np.random.default_rng(123)
    got = _local_edge_batch(S, q, h, k, phi, sigma, A, kappa, n, rng2)
    vec_local = vec_edge - vec_signed * delta_S
    assert abs(got - vec_local.mean()) < 1e-12, (got, vec_local.mean())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
