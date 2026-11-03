import glob
import pandas as pd

files = [f for f in sorted(glob.glob("results/trials_seed[0-9]*.csv")) if not f.endswith("seed42.csv")]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

rows = []
for limit in [4.0, 5.0, 6.0]:
    for (m, s), g in df.groupby(["method", "seed"]):
        ok = g[g["latency_median_ms"] <= limit]
        rows.append({"limit_ms": limit, "method": m, "seed": s,
                     "best_f1": ok["cv_f1"].max() if len(ok) else 0.0,
                     "n_feasible": len(ok)})
per = pd.DataFrame(rows)
out = per.groupby(["limit_ms", "method"]).agg(
    f1_mean=("best_f1", "mean"), f1_std=("best_f1", "std"),
    feas_mean=("n_feasible", "mean")).round(4)
out.to_csv("results/sensitivity.csv")
print(out.to_string())
