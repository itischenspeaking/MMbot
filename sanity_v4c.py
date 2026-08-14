"""v4c Exp1 sanity — decision equivalence.

The phi-based rule and the capped direct-markout rule are algebraically the
same: h = 1/kappa + clip(m_hat, 0, sigma*sqrt(2/pi)). This checks they agree
in simulation.

Note: NOT byte-identical. The phi path computes (clip(m_hat/scale,0,1))*scale,
a divide-then-multiply whose round-trip rounding differs from clip(m_hat,0,
scale) at the ~1e-16 level for interior m_hat. So the check is numerical
equivalence (allclose), not array_equal — the ~1e-16 gap is floating-point
round-trip, not an economic difference. IntegratedMaker is the frozen
baseline and is left untouched; the equivalence is verified, not forced
bit-exact.

    1. DirectMarkoutMaker(cap=True) == IntegratedMaker, numerically
    2. cap=False diverges from cap=True only when m_hat > sigma*sqrt(2/pi)

    python sanity_v4c.py
"""

import numpy as np

from market import RandomWalk
from simulator import RegimeInformedFlow, markov_schedule, run
from strategy import IntegratedMaker, DirectMarkoutMaker

SIGMA = 0.3
N = 50
H_WARMUP = 1.08
KAPPA = 1.0
P = 0.002
N_STEPS = 4500


def _sched(seed=0):
    return markov_schedule(N_STEPS, P, seed=seed)


def test_direct_capped_equals_phi_policy():
    """cap=True direct-markout maker reproduces the phi-based IntegratedMaker
    to floating-point tolerance (algebraic identity, ~1e-16 round-trip gap)."""
    for seed in (0, 1, 2):
        sched = _sched(seed)
        ri = run(RandomWalk(sigma=SIGMA),
                 IntegratedMaker(k=0.04, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                                 kappa=KAPPA, toxicity=True),
                 RegimeInformedFlow(A=0.4, kappa=KAPPA, phi_schedule=sched),
                 n_steps=N_STEPS, seed=seed)
        rd = run(RandomWalk(sigma=SIGMA),
                 DirectMarkoutMaker(k=0.04, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                                    kappa=KAPPA, cap=True),
                 RegimeInformedFlow(A=0.4, kappa=KAPPA, phi_schedule=sched),
                 n_steps=N_STEPS, seed=seed)
        for key in ("S", "delta_S"):
            assert np.array_equal(ri[key], rd[key]), (seed, key)  # exogenous, exact
        for key in ("bid", "ask", "inventory", "cash", "pnl"):
            assert np.allclose(ri[key], rd[key], atol=1e-9), (seed, key)
        # fills are decided by (u_fill < p_fill(distance)); a ~1e-16 quote
        # difference cannot flip a fill, so these must be exactly equal
        assert np.array_equal(ri["signed_flow"], rd["signed_flow"]), seed
        assert ri["n_trades"] == rd["n_trades"], seed


def test_uncapped_diverges_only_above_cap():
    """The uncapped rule differs from the capped rule only above the cap. Two
    full runs can't be compared tick-by-tick (their fills diverge once quotes
    differ, so their estimator states drift apart), so check the DECISION RULE
    directly: for the same m_hat, capped and uncapped h agree at or below the
    cap and the uncapped is wider above it. Then confirm a real run actually
    reaches the regime where they differ."""
    scale = SIGMA * np.sqrt(2.0 / np.pi)

    def h_capped(m):
        return 1.0 / KAPPA + min(max(m, 0.0), scale)

    def h_uncapped(m):
        return 1.0 / KAPPA + max(m, 0.0)

    for m in np.linspace(-0.2, 0.6, 400):
        if m <= scale:
            assert abs(h_capped(m) - h_uncapped(m)) < 1e-12, m
        else:
            assert h_uncapped(m) > h_capped(m) + 1e-9, m

    # and a real high-toxicity run must push the windowed markout above the cap
    # often enough that the two policies genuinely differ in practice
    sched = _sched(3)
    r = run(RandomWalk(sigma=SIGMA),
            DirectMarkoutMaker(k=0.04, N=N, sigma=SIGMA, h_warmup=H_WARMUP,
                               kappa=KAPPA, cap=False),
            RegimeInformedFlow(A=0.4, kappa=KAPPA, phi_schedule=sched),
            n_steps=N_STEPS, seed=3)
    # reconstruct the windowed markout the maker acted on and count cap breaches
    from strategy import RollingToxicityEstimator
    est = RollingToxicityEstimator(N, SIGMA)
    breaches = 0
    for sf_t, ds_t in zip(r["signed_flow"], r["delta_S"]):
        est.update(sf_t, ds_t)
        if not np.isnan(est._m_hat) and est._m_hat > scale:
            breaches += 1
    assert breaches > 0, "no cap breaches — cap vs uncapped would be identical"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
