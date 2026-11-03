import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
df = pd.read_csv(f"results/trials_seed{seed}.csv")

fig, ax = plt.subplots(figsize=(7, 4))
for method, g in df.groupby("method"):
    ax.step(g["trial"] + 1, g["best_so_far"], where="post", label=method)
ax.set_xlabel("Trial (evaluation budget)")
ax.set_ylabel("Best feasible CV F1 so far")
ax.set_title(f"Best-so-far vs trials (seed {seed})")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"figures/best_so_far_seed{seed}.png", dpi=150)
print("Saved figures/best_so_far_seed%d.png" % seed)
