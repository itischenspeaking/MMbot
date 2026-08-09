# MMbot

A market-making bot built up in versions, each one fixing a specific
failure of the last.

## Versions

**v0** — quotes S ± h regardless of anything. Fills arrive at random and
don't depend on the quote.

Over 500 runs of 2000 steps (σ=0.1, h=0.5, fill prob 0.3):

    spread pnl      mean  299.4    std  10.6
    inventory pnl   mean   -1.3    std  78.6
    total           mean  298.1    std  79.3

Inventory contributes no expected return and 98% of the variance. It
random walks — nothing in v0 pulls it back to zero.

Both spread income and inventory risk scale linearly in n, so mean/std is
3.76 and stays there however long the bot runs. Inventory control isn't an
optimisation; without it the business doesn't compound.

## Running

    pip install -r requirements.txt
    python V0.py
