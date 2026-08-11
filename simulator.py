"""Exchange, account, and the loop that drives them.

Ordering within a step, which is what keeps the log aligned:

    1. read S_t
    2. maker quotes on S_t
    3. flow decides fills against those quotes
    4. record everything at time t
    5. price moves to S_{t+1}, except after the last tick

So row t holds S_t, the quotes made on S_t, and the position held after
trading at time t. There are n_steps rows and n_steps - 1 price moves;
the walk never takes a step nobody observes.

Market and flow draw from separate streams spawned off one seed. Changing
a flow parameter therefore leaves the price path untouched, which is what
makes A/B comparisons cheap. Flow implementations must consume a fixed
number of draws per step for that to hold across parameter changes too.
"""

import numpy as np


class BernoulliFlow:
    """At most one trader per step, arriving with fixed probability.

    Not a Poisson process — arrivals are Bernoulli on a discrete grid.

    Fill probability does not depend on where the quotes are, which is
    wrong and is the point of v1.
    """

    def __init__(self, trade_prob=0.3):
        if not 0.0 <= trade_prob <= 1.0:
            raise ValueError("trade_prob must be in [0, 1]")
        self.trade_prob = trade_prob

    def fills(self, S, bid, ask, rng):
        """Return (hit_bid, lift_ask) — did we buy, did we sell.

        Both draws are taken unconditionally so the stream stays aligned
        when trade_prob changes.
        """
        u_arrive, u_side = rng.random(), rng.random()
        if u_arrive >= self.trade_prob:
            return False, False
        return (True, False) if u_side < 0.5 else (False, True)


class QuoteSensitiveFlow:
    """Fill probability decays with how far the quote sits from fair value.

    p_fill(delta) = min(1, A * exp(-kappa * delta))

    where delta is the distance from the true price S to that side's quote.
    Quoting wider now costs volume, which is the whole point of v1: it makes
    an optimal spread exist. The two sides are drawn independently, so in one
    step we can fill both, one, or neither — this is separate buy and sell
    customer flow, not a single trader picking a side (that was v0).

      A     one-sided fill prob when quoting exactly at S (delta = 0)
      kappa how fast that prob decays as the quote moves away; larger = steeper
    """

    def __init__(self, A, kappa):
        # These are experiment knobs, so validate them here, once. The prices
        # passed to fills() come from our own market/strategy and are trusted;
        # we don't re-check them on the hot path.
        if not 0.0 < A <= 1.0:
            raise ValueError("A must be in (0, 1]")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        self.A = A
        self.kappa = kappa
        self._log_A = np.log(A)  # precomputed so the hot path avoids a log()

    def _p_fill(self, delta):
        # Work in log space: A*exp(-kappa*delta) overflows to inf when delta
        # is very negative (a quote through fair value, which v2's skew can
        # produce). log_p >= 0 means the raw prob is >= 1, so cap at 1.
        log_p = self._log_A - self.kappa * delta
        return 1.0 if log_p >= 0.0 else np.exp(log_p)

    def fills(self, S, bid, ask, rng):
        # Exactly two draws every call, unconditionally, so that changing A or
        # kappa leaves the price path and draw alignment untouched across runs.
        u_bid, u_ask = rng.random(2)
        hit = u_bid < self._p_fill(S - bid)   # someone sells to us at our bid
        lift = u_ask < self._p_fill(ask - S)  # someone buys from us at our ask
        return hit, lift


class Account:
    def __init__(self):
        self.cash = 0.0
        self.inventory = 0
        self.n_trades = 0

    def buy(self, price):
        self.cash -= price
        self.inventory += 1
        self.n_trades += 1

    def sell(self, price):
        self.cash += price
        self.inventory -= 1
        self.n_trades += 1

    def pnl(self, S):
        return self.cash + self.inventory * S


def run(market, strategy, flow, n_steps=2000, seed=0):
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    market_seed, flow_seed = np.random.SeedSequence(seed).spawn(2)
    market_rng = np.random.default_rng(market_seed)
    flow_rng = np.random.default_rng(flow_seed)

    market.reset()
    acct = Account()

    cols = ("S", "bid", "ask", "inventory", "cash", "pnl", "buys", "sells")
    out = {k: [] for k in cols}

    for t in range(n_steps):
        S = market.S
        bid, ask = strategy.quote(S, acct.inventory)
        hit_bid, lift_ask = flow.fills(S, bid, ask, flow_rng)

        if hit_bid:
            acct.buy(bid)
        if lift_ask:
            acct.sell(ask)

        out["S"].append(S)
        out["bid"].append(bid)
        out["ask"].append(ask)
        out["inventory"].append(acct.inventory)
        out["cash"].append(acct.cash)
        out["pnl"].append(acct.pnl(S))
        out["buys"].append(hit_bid)
        out["sells"].append(lift_ask)

        if t < n_steps - 1:
            market.step(market_rng)

    res = {k: np.array(v) for k, v in out.items()}
    res["n_trades"] = acct.n_trades
    return res
