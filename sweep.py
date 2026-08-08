import numpy as np
from V0 import run

h = 0.5
rows = []
for s in range(500):
    r = run(seed=s)
    spread = r["n_trades"] * h          # 每笔成交赚 h
    total = r["pnl"][-1]
    rows.append((spread, total - spread, total, r["inventory"][-1]))

a = np.array(rows)
for i, name in enumerate(["spread", "inventory", "total", "final_inv"]):
    print(f"{name:10s} mean {a[:, i].mean():9.2f}   std {a[:, i].std():9.2f}")

print(f"\nmean/std of total: {a[:, 2].mean() / a[:, 2].std():.3f}")
