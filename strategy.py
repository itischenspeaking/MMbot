"""How the maker quotes.

Every strategy exposes the same method:

    quote(S, inventory) -> (bid, ask)

`inventory` is passed even to strategies that ignore it, so that later
versions can use it without changing the simulator.
"""


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
