"""Tiny Market Maker, v0.

The dumbest thing that could possibly work.

  - a true price that random walks
  - a market maker that quotes S +/- h, always, regardless of anything
  - traders who arrive at random and buy or sell with equal probability

Known holes in v0, deliberately left in:
  - the maker sees the true price exactly
  - the spread does not affect how often we get filled
  - traders are pure noise; nobody knows anything we don't
  - inventory is unbounded

Each of those becomes a later version.
"""

import numpy as np


def run(
    n_steps=2000,
    sigma=0.1,          # price move per step
    half_spread=0.5,    # we quote S - h and S + h
    trade_prob=0.3,     # chance a trader shows up in a given step
    S0=100.0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    S = S0
    cash = 0.0
    inventory = 0
    n_trades = 0

    out = {k: [] for k in ("S", "bid", "ask", "inventory", "cash", "pnl")}

    for _ in range(n_steps):
        bid = S - half_spread
        ask = S + half_spread

        if rng.random() < trade_prob:
            n_trades += 1
            if rng.random() < 0.5:
                # trader lifts our ask: we sell one unit
                cash += ask
                inventory -= 1
            else:
                # trader hits our bid: we buy one unit
                cash -= bid
                inventory += 1

        S += sigma * rng.standard_normal()

        out["S"].append(S)
        out["bid"].append(bid)
        out["ask"].append(ask)
        out["inventory"].append(inventory)
        out["cash"].append(cash)
        out["pnl"].append(cash + inventory * S)   # mark to market

    res = {k: np.array(v) for k, v in out.items()}
    res["n_trades"] = n_trades
    return res


if __name__ == "__main__":
    r = run(seed=0)
    print(f"trades      {r['n_trades']}")
    print(f"final inv   {r['inventory'][-1]}")
    print(f"final pnl   {r['pnl'][-1]:.2f}")
