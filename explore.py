import matplotlib.pyplot as plt
from V0 import run

r = run(seed=0)

fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(r["inventory"]); ax[0].set_ylabel("inventory"); ax[0].axhline(0, c="k", lw=0.8)
ax[1].plot(r["pnl"]);       ax[1].set_ylabel("pnl");       ax[1].axhline(0, c="k", lw=0.8)
plt.tight_layout()
plt.savefig("figures/v0_inventory.png", dpi=130) 
plt.show()
