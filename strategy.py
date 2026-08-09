"""How the maker quotes.

Every strategy exposes the same method:

    quote(S, inventory) -> (bid, ask)

`inventory` is passed even to strategies that ignore it, so that later
versions can use it without changing the simulator.
"""


class NaiveMaker:
    """Quote S +/- h. Ignores inventory, ignores everything."""

    def __init__(self, half_spread=0.5):
        self.half_spread = half_spread

    def quote(self, S, inventory):
        return S - self.half_spread, S + self.half_spread
