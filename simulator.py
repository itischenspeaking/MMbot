"""Exchange, account, and the loop that drives them.

Ordering within a step, which is what keeps the log aligned:

    1. read S_t
    2. maker quotes on S_t
    3. exchange decides fills against those quotes
    4. record everything at time t
    5. price moves to S_{t+1}

So row t holds S_t, the quotes made on S_t, and the position held after
trading at time t. Nothing in a row comes from a different clock tick.
"""

import numpy as np


class PoissonFlow:
    """A trader shows up with fixed probability and picks a side at random.

    Fill probability does not depend on where the quotes are, which is
    wrong and is the point of v1.
    """

    def __init__(self, trade_prob=0.3):
        self.trade_prob = trade_prob

    def fills(self, S, bid, ask, rng):
        """Return (hit_bid, lift_ask) — did we buy, did we sell."""
        if rng.random() >= self.trade_prob:
            return False, False
        return (True, False) if rng.random() < 0.5 else (False, True)


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
    rng = np.random.default_rng(seed)
    market.reset()
    acct = Account()

    cols = ("S", "bid", "ask", "inventory", "cash", "pnl", "buys", "sells")
    out = {k: [] for k in cols}

    for _ in range(n_steps):
        S = market.S
        bid, ask = strategy.quote(S, acct.inventory)
        hit_bid, lift_ask = flow.fills(S, bid, ask, rng)

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

        market.step(rng)

    res = {k: np.array(v) for k, v in out.items()}
    res["n_trades"] = acct.n_trades
    return res
