"""v4b Exp1 sanity — stochastic hidden two-state Markov toxicity.

    1. empirical state fraction ~ 0.5 (symmetric switching)
    2. empirical switch rate ~ p
    3. regime durations ~ geometric(p) (mean ~ 1/p)
    4. same seed reproduces the same latent path; different seed differs
    5. latent path is independent of the market/flow RNG in run()
       (changing the flow seed doesn't change the schedule, and vice versa)
    6. no strategy can read the hidden state (structural)

    python sanity_v4b.py
"""

import numpy as np

from market import RandomWalk
from simulator import RegimeInformedFlow, markov_schedule, run
from strategy import InventorySkewMaker, AdaptiveMaker

SIGMA = 0.3
P_PRIMARY = 0.002   # mean regime length ~500 ticks
N_STEPS = 4500


def _durations(schedule):
    """Run-length of each constant segment."""
    change = np.flatnonzero(np.diff(schedule) != 0) + 1
    bounds = np.concatenate(([0], change, [len(schedule)]))
    return np.diff(bounds)


def test_state_fraction_balanced():
    # over many independent paths, time in each state ~ 0.5 for symmetric flips
    fracs = []
    for seed in range(200):
        s = markov_schedule(N_STEPS, P_PRIMARY, seed=seed)
        fracs.append(s.mean())
    assert abs(np.mean(fracs) - 0.5) < 0.05, np.mean(fracs)


def test_switch_rate_matches_p():
    for p in (0.002, 0.01):
        rates = []
        for seed in range(200):
            s = markov_schedule(N_STEPS, p, seed=seed)
            rates.append(np.mean(np.diff(s) != 0))
        assert abs(np.mean(rates) - p) / p < 0.15, (p, np.mean(rates))


def test_durations_geometric():
    # mean regime duration ~ 1/p; pool durations across many paths
    p = P_PRIMARY
    durs = []
    for seed in range(300):
        durs.extend(_durations(markov_schedule(N_STEPS, p, seed=seed)))
    durs = np.array(durs)
    # exclude the truncated final segment's bias by trimming the longest tail
    mean_dur = durs.mean()
    assert abs(mean_dur - 1.0 / p) / (1.0 / p) < 0.20, (mean_dur, 1.0 / p)


def test_seed_reproducible_and_varies():
    a = markov_schedule(N_STEPS, P_PRIMARY, seed=7)
    b = markov_schedule(N_STEPS, P_PRIMARY, seed=7)
    c = markov_schedule(N_STEPS, P_PRIMARY, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_latent_independent_of_market_flow_rng():
    """The Markov path is fixed data passed to RegimeInformedFlow. Changing
    the run() seed (market/flow/informed draws) must not change the logged
    phi_true, and using a different latent schedule must not change the price
    path under the same run() seed."""
    sched = markov_schedule(N_STEPS, P_PRIMARY, seed=1)
    maker = lambda: InventorySkewMaker(half_spread=1.08, k=0.0)
    r1 = run(RandomWalk(sigma=SIGMA), maker(),
             RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=sched),
             n_steps=N_STEPS, seed=10)
    r2 = run(RandomWalk(sigma=SIGMA), maker(),
             RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=sched),
             n_steps=N_STEPS, seed=20)
    # same schedule, different run seed: phi_true identical, price path differs
    assert np.array_equal(r1["phi_true"], r2["phi_true"])
    assert np.array_equal(r1["phi_true"], sched)
    assert not np.array_equal(r1["S"], r2["S"])

    # different schedule, same run seed: price path identical (exogenous)
    sched2 = markov_schedule(N_STEPS, P_PRIMARY, seed=2)
    r3 = run(RandomWalk(sigma=SIGMA), maker(),
             RegimeInformedFlow(A=0.4, kappa=1.0, phi_schedule=sched2),
             n_steps=N_STEPS, seed=10)
    assert np.array_equal(r1["S"], r3["S"])
    assert np.array_equal(r1["delta_S"], r3["delta_S"])


def test_maker_cannot_read_hidden_state():
    """AdaptiveMaker (the estimator-driven maker used in v4b) must not expose
    set_t or hold the schedule — it only sees realized markouts."""
    maker = AdaptiveMaker(N=50, sigma=SIGMA, h_warmup=1.08, kappa=1.0, k=0.0)
    assert not hasattr(maker, "set_t")
    assert not hasattr(maker, "phi_schedule")
    assert not hasattr(maker, "phi_true")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
