# v4pre — integrated baseline

v2 controls inventory (center skew). v3b controls toxicity (adaptive
width). v4pre wires them into one maker and checks nothing breaks. No new
theory. Scope and rationale: [v4 blueprint](v4_blueprint.md).

## Architecture

    center_t = S_t - k*q_t

    h_t = h_warmup                                    warm-up (< N fills)
    h_t = 1/kappa + phi_hat_t * sigma * sqrt(2/pi)     after warm-up (lambda=1)

An earlier draft set the post-warm-up base spread to h_warmup=1.08 (which
already bakes in the time-averaged toxicity premium) and added phi_hat on
top — double-counting. Fixed: h_warmup is the warm-up fallback only; once
phi_hat is defined the base is 1/kappa=1.00.

Frozen parameters:

    k = 0.04, N = 50, sigma = 0.3, kappa = 1, h_warmup = 1.08, lambda = 1

`IntegratedMaker` (strategy.py) is v3b's `AdaptiveMaker` with the center
shifted by -k*q. No new mechanism, no new estimator.

## Experiment 1 — integration regressions

**Purpose.** Confirm merging inventory skew and toxicity adaptation didn't
break either module's original behavior, and introduced no look-ahead or
RNG bug.

**Checks** (`sanity_v4.py`, deterministic/same-seed, no Monte Carlo):

1. k=0 -> byte-identical to v3b's AdaptiveMaker.
2. toxicity disabled -> byte-identical to v2's inventory-skew maker.
3. k=0 and toxicity disabled -> byte-identical to a static maker.
4. maker never reads phi_true.
5. markout at tick t affects only the quote at t+1.
6. same seed, different k: S and delta_S byte-identical; fills may differ.

**Result.** 6/6 passed. k=0 recovers v3b exactly, causal timing holds, and
changing k does not perturb the underlying price path.

**Conclusion.** IntegratedMaker's implementation is clean. Safe to use as
the unified baseline.

## Experiment 2 — frozen baseline sanity

**Purpose.** Confirm the frozen baseline keeps both inventory control and
toxicity adaptation active together, with no obvious integration failure.

**Predictions.** Relative to Toxicity-only (AdaptiveMaker, k=0): (1) RMS and
max inventory drop substantially; (2) spread still widens then narrows
across the 0->1->0 regime; (3) Integrated is not required to beat
Toxicity-only on mean PnL.

**Method.** sigma=0.3, phi: 0->1->0, seg_len=1500 (4500 ticks total),
n_seeds=500. Compare AdaptiveMaker(k=0, N=50) against IntegratedMaker(k=.04,
N=50).

**Results.**

| | Integrated | Toxicity-only |
|---|---|---|
| Mean PnL | 607.70 | 630.91 |
| RMS inventory | 3.477 | 15.933 |
| Mean max\|inventory\| | 10.04 | 31.06 |

RMS inventory reduction: 78.2%. Integrated spread by regime — low1=1.0327,
high=1.2035, low2=1.0443 — still widens in the toxic regime and narrows
back afterward.

**Conclusion.** Integration succeeded: inventory exposure fell sharply,
toxicity-width adaptation stayed active, no integration failure. The lower
mean PnL (607.70 vs 630.91) is not interpreted here — v4pre isn't designed
to attribute the paired economic cost of inventory control. That, along
with interaction effects and estimator contamination, is deferred to v4a.

**v4pre baseline frozen.** Interaction economics continue in
[v4a](v4a_log.md).
