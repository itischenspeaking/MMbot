import numpy as np
from V0 import run

h = 2
rows = []
for s in range(500):
    r = run(seed=s, half_spread=2.0) # r = run(seed=s, sigma=0.1, n_steps=500, half_spred =0.5, trade_prob=0.3), if change half_spread, the h=0.5 in this file also need to be changed
    spread = r["n_trades"] * h          
    total = r["pnl"][-1]
    rows.append((spread, total - spread, total, r["inventory"][-1]))

a = np.array(rows)
for i, name in enumerate(["spread", "inventory", "total", "final_inv"]):
    print(f"{name:10s} mean {a[:, i].mean():9.2f}   std {a[:, i].std():9.2f}")

print(f"\nmean/std of total: {a[:, 2].mean() / a[:, 2].std():.3f}")
