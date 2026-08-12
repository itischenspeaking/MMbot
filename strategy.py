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

    def update(self, signed_flow, delta_S):
        """Advance one tick. Only fills (signed_flow != 0) update the window.
        Returns the current phi_hat (NaN until N fills seen)."""
        if signed_flow != 0:
            self.markouts.append(signed_flow * delta_S)
            if len(self.markouts) == self.N:
                m_hat = sum(self.markouts) / self.N
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
        for t in range(len(signed_flow)):
            out[t] = self.update(signed_flow[t], delta_S[t])
        return out
