import json
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_data, evaluate, LATENCY_LIMIT_MS

optuna.logging.set_verbosity(optuna.logging.WARNING)
BUDGET = 36
SEEDS = range(1, 6)
COLS = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"]


def nondominated(df):
    """Giu nghiem khong bi dominate: max cv_f1, min latency_median_ms."""
    v = df[["cv_f1", "latency_median_ms"]].values
    keep = []
    for f, l in v:
        dom = ((v[:, 0] >= f) & (v[:, 1] <= l) & ((v[:, 0] > f) | (v[:, 1] < l))).any()
        keep.append(not dom)
    return df[keep]


def run(X, y, seed):
    rows = []

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        r = evaluate(p, X, y, seed=seed)
        rows.append({"seed": seed, "trial": trial.number, **p, **r})
        return r["cv_f1"], r["latency_median_ms"]

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=seed, population_size=12))
    study.optimize(objective, n_trials=BUDGET)
    return rows


if __name__ == "__main__":
    X_dev, X_test, y_dev, y_test = load_data()  # test khoa
    all_rows = []
    for s in SEEDS:
        print("NSGA-II seed", s)
        all_rows += run(X_dev, y_dev, s)
    df = pd.DataFrame(all_rows)
    df.to_csv("results/pareto_trials.csv", index=False)

    front = nondominated(df).sort_values("latency_median_ms")
    front.to_csv("results/pareto_front.csv", index=False)
    print("Pareto front size:", len(front))
    print(front[["cv_f1", "latency_median_ms", "latency_p95_ms", "n_nodes"] + COLS].round(4).to_string())

    # Quy tac chon dat truoc: max CV F1 trong cac nghiem co latency median <= limit
    ok = df[df["latency_median_ms"] <= LATENCY_LIMIT_MS].sort_values(
        ["cv_f1", "latency_median_ms"], ascending=[False, True])
    best = ok.iloc[0]
    cand = {k: (None if pd.isna(best[k]) else (best[k].item() if hasattr(best[k], "item") else best[k]))
            for k in COLS}
    cand["max_features"] = None if pd.isna(best["max_features"]) or best["max_features"] == "None" else best["max_features"]
    print("Candidate (rule: max CV F1 s.t. latency <= %.1f ms):" % LATENCY_LIMIT_MS)
    print(best[["cv_f1", "latency_median_ms", "latency_p95_ms", "n_nodes"] + COLS].to_string())
    with open("results/final_candidate.json", "w") as f:
        json.dump({"params": cand, "cv_f1": float(best["cv_f1"]),
                   "latency_median_ms": float(best["latency_median_ms"])}, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["latency_median_ms"], df["cv_f1"], s=12, c="lightgray", label="all trials")
    ax.step(front["latency_median_ms"], front["cv_f1"], where="post", c="tab:red", marker="o",
            ms=4, label="Pareto front")
    ax.axvline(LATENCY_LIMIT_MS, color="k", ls="--", label="limit")
    ax.set_xlabel("Latency median (ms)"); ax.set_ylabel("CV F1")
    ax.set_title("Pareto front: F1 vs latency (NSGA-II, 5 seeds pooled)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig("figures/pareto_front.png", dpi=150)
    print("Saved figures/pareto_front.png")
