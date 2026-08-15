"""How the maker quotes.

Every strategy exposes the same method:

    quote(S, inventory) -> (bid, ask)

`inventory` is passed even to strategies that ignore it, so that later
versions can use it without changing the simulator.
"""

from collections import deque

import numpy as np


class NaiveMaker:
    """Quote S +/- h. Ignores inventory, ignores everything."""

    def __init__(self, half_spread=0.5):
        if half_spread <= 0:
            raise ValueError("half_spread must be positive")
        self.half_spread = half_spread

    def quote(self, S, inventory):
        return S - self.half_spread, S + self.half_spread


class InventorySkewMaker:
    """Shift the quote center against inventory to pull the position flat.

        center = S - k * q
        bid, ask = center - h, center + h

    With v1's quote-sensitive flow this creates negative feedback: long (q>0)
    lowers the center, so the ask sits nearer S (fills easier, we sell) and the
    bid sits further (fills harder, we buy less). Both push q back toward zero.

    k = 0 recovers NaiveMaker exactly.

      half_spread  distance from center to each quote
      k            price shift per unit of inventory
    """

    def __init__(self, half_spread=1.0, k=0.04):
        if half_spread <= 0:
            raise ValueError("half_spread must be positive")
        if k < 0:
            raise ValueError("k must be non-negative")
        self.half_spread = half_spread
        self.k = k

    def quote(self, S, inventory):
        center = S - self.k * inventory
        return center - self.half_spread, center + self.half_spread


# --------------------------------------------------------------------------
# v3b Stage 2: offline toxicity estimation (does not quote).
# --------------------------------------------------------------------------

class RollingToxicityEstimator:
    """Estimate hidden toxicity phi from realized signed markouts.

    Offline (Stage 2) usage: feed it the per-tick (signed_flow, delta_S) log
    from run(), in tick order, and read back a phi_hat path aligned to phi_true.

    Per filled trade the observed markout is signed_flow_t * delta_S_t. Only
    fills enter the window (no-fill ticks carry the previous estimate forward,
    so the output stays tick-aligned). Once N fills have been seen, the estimate
    inverts the v3a calibration m(phi) = phi*sigma*sqrt(2/pi):

        m_hat   = mean(last N filled-trade markouts)
        phi_hat = clip(m_hat / (sigma*sqrt(2/pi)), 0, 1)

    Before N fills have accumulated, phi_hat is NaN (no estimating from a
    partial window). The estimator knows sigma; it never sees phi_true.

    No look-ahead: markout_t depends on delta_S_t, which in the v3b tick order
    is the move applied *after* the fill at t is decided and logged. Feeding
    the log in tick order therefore uses only information realized up to t.

      N      window size, in number of FILLS (not ticks)
      sigma  known volatility, used only to invert the calibration
    """

    def __init__(self, N, sigma):
        if N < 1:
            raise ValueError("N must be at least 1")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.N = N
        self.sigma = sigma
        self._scale = sigma * np.sqrt(2.0 / np.pi)  # m(phi=1)
        self.markouts = deque(maxlen=N)
        self._phi_hat = np.nan  # current estimate, carried across no-fill ticks
        self._m_hat = np.nan    # raw windowed markout mean (v4c direct policy)

    def update(self, signed_flow, delta_S):
        """Advance one tick. Only fills (signed_flow != 0) update the window.
        Returns the current phi_hat (NaN until N fills seen)."""
        if signed_flow != 0:
            self.markouts.append(signed_flow * delta_S)
            if len(self.markouts) == self.N:
                m_hat = sum(self.markouts) / self.N
                self._m_hat = m_hat
                self._phi_hat = min(1.0, max(0.0, m_hat / self._scale))
        return self._phi_hat

    def run_offline(self, signed_flow, delta_S):
        """Vectorised convenience: given full per-tick log arrays, return the
        tick-aligned phi_hat path. Pure post-processing, no simulation."""
        signed_flow = np.asarray(signed_flow)
        delta_S = np.asarray(delta_S)
        out = np.empty(len(signed_flow))
        self.markouts.clear()
        self._phi_hat = np.nan
        self._m_hat = np.nan
        for t in range(len(signed_flow)):
            out[t] = self.update(signed_flow[t], delta_S[t])
        return out


# --------------------------------------------------------------------------
# v3b Stage 3: closed-loop adaptive quoting.
# --------------------------------------------------------------------------

def _h_star(phi, kappa, sigma):
    """v3a-verified optimal half-spread: 1/kappa + phi*sigma*sqrt(2/pi)."""
    return 1.0 / kappa + phi * sigma * np.sqrt(2.0 / np.pi)


class AdaptiveMaker:
    """Quote a spread that adapts to estimated toxicity (v3b Stage 3).

    Closed loop: each tick quotes on the phi_hat available from fills observed
    up to the PREVIOUS tick, then observe(markout) folds in the current tick's
    realized markout for use next tick. It never sees phi_true.

        h_t = 1/kappa + phi_hat_t * sigma * sqrt(2/pi)

    Warm-up: until the estimator has N fills, phi_hat is NaN and the maker
    quotes at h_warmup (the BestFixed spread). So before it has enough data
    the Adaptive maker is identical to the Fixed baseline — the capture
    fraction then measures learning value alone, not the coincidence that the
    schedule happens to start in the low regime.

    k is kept in the signature for parity but the v3b main experiment fixes
    k=0 (no inventory skew); center = S.

      N          estimator window, in fills
      sigma      known volatility (for both estimator and h formula)
      h_warmup   spread used before N fills accumulate (BestFixed h)
      kappa      known flow decay
    """

    def __init__(self, N, sigma, h_warmup, kappa=1.0, k=0.0):
        self.est = RollingToxicityEstimator(N, sigma)
        self.sigma = sigma
        self.h_warmup = h_warmup
        self.kappa = kappa
        self.k = k

    def quote(self, S, inventory):
        phi_hat = self.est._phi_hat  # estimate from fills up to previous tick
        if np.isnan(phi_hat):
            h = self.h_warmup
        else:
            h = _h_star(phi_hat, self.kappa, self.sigma)
        center = S - self.k * inventory
        return center - h, center + h

    def observe(self, signed_flow, delta_S):
        """Fold this tick's realized markout into the estimator, for use from
        the next tick onward. Called by run() after delta_S has been applied."""
        self.est.update(signed_flow, delta_S)


class OracleMaker:
    """Upper benchmark: quotes at the true-phi optimum every tick (v3b Stage 3).

    Holds the hidden schedule and reads phi_t via a tick index set by run()'s
    set_t hook. This is NOT a realistic strategy — no maker sees phi_true — it
    exists only to bound how much a perfect toxicity signal is worth.

        h_t = 1/kappa + phi_true_t * sigma * sqrt(2/pi)
    """

    def __init__(self, phi_schedule, sigma, kappa=1.0, k=0.0):
        self.phi_schedule = np.asarray(phi_schedule, dtype=float)
        self.sigma = sigma
        self.kappa = kappa
        self.k = k
        self._t = 0

    def set_t(self, t):
        self._t = t

    def quote(self, S, inventory):
        phi = self.phi_schedule[self._t]
        h = _h_star(phi, self.kappa, self.sigma)
        center = S - self.k * inventory
        return center - h, center + h


# --------------------------------------------------------------------------
# v4pre: integrated baseline — v2's inventory skew + v3b's adaptive width,
# wired into one maker. No new theory.
# --------------------------------------------------------------------------

class IntegratedMaker:
    """v2 center-skew + v3b toxicity-adaptive width, combined.

        center_t = S_t - k * q_t

        warm-up (fewer than N fills observed):
            h_t = h_warmup

        after warm-up:
            h_t = 1/kappa + phi_hat_t * sigma * sqrt(2/pi)     (lambda = 1)

    h_warmup is a fallback used only before the estimator has N fills — once
    phi_hat is defined the adaptive base is 1/kappa, not h_warmup. (An
    earlier draft mistakenly kept using h_warmup as the base and added
    phi_hat on top, double-counting the toxicity premium; fixed here.)

    toxicity=False disables the adaptive width entirely: h_t = h_warmup for
    every tick, the estimator is never updated, and the maker behaves as a
    pure inventory-skew quoter. This exists only so k=0, toxicity=False
    recovers a static maker exactly, and k>0, toxicity=False recovers v2's
    InventorySkewMaker exactly — see sanity_v4.py regressions 2 and 3.

      k          price shift per unit of inventory (v2)
      N          estimator window, in fills (v3b)
      sigma      known volatility
      h_warmup   spread used before N fills, or always if toxicity=False
      kappa      known flow decay
      toxicity   if False, adaptive width is disabled (diagnostic/regression use)
    """

    def __init__(self, k, N, sigma, h_warmup, kappa=1.0, toxicity=True):
        self.k = k
        self.sigma = sigma
        self.h_warmup = h_warmup
        self.kappa = kappa
        self.toxicity = toxicity
        self.est = RollingToxicityEstimator(N, sigma) if toxicity else None

    def quote(self, S, inventory):
        if self.toxicity:
            phi_hat = self.est._phi_hat  # from fills up to the previous tick
            h = self.h_warmup if np.isnan(phi_hat) else _h_star(
                phi_hat, self.kappa, self.sigma)
        else:
            h = self.h_warmup
        center = S - self.k * inventory
        return center - h, center + h

    def observe(self, signed_flow, delta_S):
        """Fold this tick's realized markout into the estimator, for use from
        the next tick onward. No-op when toxicity=False."""
        if self.toxicity:
            self.est.update(signed_flow, delta_S)


class DirectMarkoutMaker:
    """v4c: quote directly on estimated adverse markout, skipping phi.

    The phi-based rule h = 1/kappa + phi_hat*sigma*sqrt(2/pi) with
    phi_hat = clip(m_hat/(sigma*sqrt(2/pi)), 0, 1) is algebraically

        h = 1/kappa + clip(m_hat, 0, sigma*sqrt(2/pi))

    so sigma cancels except through the clip bounds. This maker acts on
    m_hat directly. The `cap` switch isolates the one structural leftover
    from the latent-phi parameterization — the phi<=1 upper cap:

        cap=True :  h = 1/kappa + clip(m_hat, 0, sigma*sqrt(2/pi))
                    (byte-identical to IntegratedMaker; regression, Exp1)
        cap=False:  h = 1/kappa + max(m_hat, 0)
                    (drops the inherited upper cap; Exp2)

    Same center skew, warm-up, estimator, and timing as IntegratedMaker.

      k, N, sigma, h_warmup, kappa : as IntegratedMaker
      cap : keep the phi<=1 upper cap (True) or drop it (False)
      lam : toxicity premium strength (v4d). h = 1/kappa + lam*premium.
            lam=1 recovers the v4c policy exactly.
    """

    def __init__(self, k, N, sigma, h_warmup, kappa=1.0, cap=True, lam=1.0):
        self.k = k
        self.sigma = sigma
        self.h_warmup = h_warmup
        self.kappa = kappa
        self.cap = cap
        self.lam = lam                                 # toxicity premium strength (v4d)
        self._cap_val = sigma * np.sqrt(2.0 / np.pi)  # m(phi=1)
        self.est = RollingToxicityEstimator(N, sigma)

    def quote(self, S, inventory):
        m_hat = self.est._m_hat  # from fills up to the previous tick
        if np.isnan(m_hat):
            h = self.h_warmup
        else:
            premium = max(m_hat, 0.0)
            if self.cap:
                premium = min(premium, self._cap_val)
            h = 1.0 / self.kappa + self.lam * premium
        center = S - self.k * inventory
        return center - h, center + h

    def observe(self, signed_flow, delta_S):
        self.est.update(signed_flow, delta_S)
