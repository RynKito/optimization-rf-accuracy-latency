import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import LATENCY_LIMIT_MS

files = sorted(glob.glob("results/trials_seed[0-9]*.csv"))
files = [f for f in files if "pilot" not in f and not f.endswith("seed42.csv")]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print("Seeds:", sorted(df["seed"].unique()))

rows = []
for (m, s), g in df.groupby(["method", "seed"]):
    feas = g[g["feasible"]]
    rows.append({"method": m, "seed": s,
                 "best_feasible_f1": feas["cv_f1"].max() if len(feas) else 0.0,
                 "n_feasible": int(g["feasible"].sum()),
                 "runtime_s": g["elapsed_s"].max()})
per = pd.DataFrame(rows)
per.to_csv("results/summary_per_seed.csv", index=False)
summ = per.groupby("method").agg(
    f1_mean=("best_feasible_f1", "mean"), f1_std=("best_feasible_f1", "std"),
    feas_mean=("n_feasible", "mean"), runtime_mean=("runtime_s", "mean")).round(4)
summ.to_csv("results/summary.csv")
print(summ.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
for m, g in df.groupby("method"):
    piv = g.pivot(index="trial", columns="seed", values="best_feasible_f1")
    mu, sd = piv.mean(axis=1), piv.std(axis=1)
    x = piv.index + 1
    ax.plot(x, mu, label=m)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
ax.set_xlabel("Trial"); ax.set_ylabel("Best feasible CV F1 so far")
ax.set_title("Convergence (mean +/- std over seeds)")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig("figures/convergence_mean.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
for m, g in df.groupby("method"):
    ax.scatter(g["latency_median_ms"], g["cv_f1"], s=12, alpha=0.4, label=m)
ax.axvline(LATENCY_LIMIT_MS, color="k", ls="--", label="limit")
ax.set_xlabel("Latency median (ms)"); ax.set_ylabel("CV F1")
ax.set_title("F1 vs latency, all trials")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig("figures/f1_vs_latency.png", dpi=150)
print("Saved figures")
