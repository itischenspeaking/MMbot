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


class InformedFlow:
    """One trader per step. Type first, then side, then fill.

    With probability phi the trader is direction-informed: it sees
    sign(delta_S) and picks the side that profits from the coming move
    (buys before a rise, sells before a fall). Otherwise it is uninformed
    and picks a side by coin flip. Either way, whether the chosen side
    actually fills uses the same A*exp(-kappa*delta) probability.

    phi therefore changes only the direction distribution of attempts,
    not the arrival process — fills per step stay comparable across phi,
    which is what makes the toxicity attribution clean.

    Note this is one attempt per step (max 1 fill), unlike
    QuoteSensitiveFlow's two independent sides (max 2). phi=0 is
    statistically equivalent to a halved-intensity symmetric flow, not
    byte-identical to v2.

    Draw budget per step (unconditional):
      flow_rng:     2  (side coin, fill uniform)
      informed_rng: 1  (informed/uninformed coin)
    """

    def __init__(self, A, kappa, phi):
        if not 0.0 < A <= 1.0:
            raise ValueError("A must be in (0, 1]")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if not 0.0 <= phi <= 1.0:
            raise ValueError("phi must be in [0, 1]")
        self.A = A
        self.kappa = kappa
        self.phi = phi
        self._log_A = np.log(A)

    def _p_fill(self, delta):
        log_p = self._log_A - self.kappa * delta
        return 1.0 if log_p >= 0.0 else np.exp(log_p)

    def fills(self, S, bid, ask, rng, delta_S=0.0, informed_rng=None):
        # Fixed draw budget regardless of phi or delta_S: u_side is consumed
        # even when the informed branch ignores it, so draw alignment holds
        # across parameter changes.
        u_side, u_fill = rng.random(2)
        u_type = informed_rng.random()

        if u_type < self.phi:
            # Informed: side dictated by the coming move. delta_S == 0
            # (last tick) gives the informed trader nothing to act on.
            if delta_S > 0:
                side = "buy"
            elif delta_S < 0:
                side = "sell"
            else:
                return False, False
        else:
            side = "buy" if u_side < 0.5 else "sell"

        if side == "buy":
            # Customer buys: lifts our ask.
            return False, u_fill < self._p_fill(ask - S)
        else:
            # Customer sells: hits our bid.
            return u_fill < self._p_fill(S - bid), False


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
    """Run the simulation loop.

    v3 ordering within a step:

        1. read S_t
        2. maker quotes on S_t
        3. pre-generate delta_S (but don't move the price yet)
        4. flow decides fills — InformedFlow can see sign(delta_S)
        5. record everything at time t
        6. apply delta_S to get S_{t+1}, except after the last tick

    SeedSequence.spawn(3) produces the same first two streams as spawn(2),
    so v0/v1/v2 flows get byte-identical price paths and fill sequences.
    The third stream is consumed only by InformedFlow.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    seeds = np.random.SeedSequence(seed).spawn(3)
    market_rng = np.random.default_rng(seeds[0])
    flow_rng = np.random.default_rng(seeds[1])

    # Third stream only for informed flow; other flows never see it.
    is_informed = hasattr(flow, 'phi')
    informed_rng = np.random.default_rng(seeds[2]) if is_informed else None

    market.reset()
    acct = Account()

    cols = ("S", "bid", "ask", "inventory", "cash", "pnl",
            "buys", "sells", "delta_S", "signed_flow")
    out = {k: [] for k in cols}

    for t in range(n_steps):
        S = market.S
        bid, ask = strategy.quote(S, acct.inventory)

        # Pre-generate the price move. Informed flow can see its sign,
        # but the price doesn't move until after fills and recording.
        delta_S = market.generate_step(market_rng) if t < n_steps - 1 else 0.0

        if is_informed:
            hit_bid, lift_ask = flow.fills(S, bid, ask, flow_rng,
                                           delta_S=delta_S,
                                           informed_rng=informed_rng)
        else:
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
        out["delta_S"].append(delta_S)
        out["signed_flow"].append(int(lift_ask) - int(hit_bid))

        if t < n_steps - 1:
            market.apply_step(delta_S)

    res = {k: np.array(v) for k, v in out.items()}
    res["n_trades"] = acct.n_trades
    return res
