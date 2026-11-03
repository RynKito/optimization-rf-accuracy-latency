import sys, time, itertools
import numpy as np
import pandas as pd
import optuna
from common import load_data, evaluate, SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)
BUDGET = 36

GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20],
    "min_samples_leaf": [1, 4],
    "max_features": ["sqrt", None],
}


def score(params, X, y, seed):
    """Objective co phat rang buoc: infeasible -> 0."""
    res = evaluate(params, X, y, seed=seed)
    res["objective"] = res["cv_f1"] if res["feasible"] else 0.0
    return res


def log_row(method, seed, trial, params, res, t0):
    return {"method": method, "seed": seed, "trial": trial, **params,
            **res, "elapsed_s": time.perf_counter() - t0}


def run_grid(X, y, seed):
    keys = list(GRID)
    rows, t0 = [], time.perf_counter()
    for i, vals in enumerate(itertools.product(*GRID.values())):
        params = dict(zip(keys, vals))
        rows.append(log_row("grid", seed, i, params, score(params, X, y, seed), t0))
    return rows


def sample_random(rng):
    return {
        "n_estimators": int(rng.integers(100, 801)),
        "max_depth": int(rng.integers(3, 21)),
        "min_samples_split": int(rng.integers(2, 21)),
        "min_samples_leaf": int(rng.integers(1, 11)),
        "max_features": [None, "sqrt", "log2"][int(rng.integers(0, 3))],
    }


def run_random(X, y, seed):
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.perf_counter()
    for i in range(BUDGET):
        params = sample_random(rng)
        rows.append(log_row("random", seed, i, params, score(params, X, y, seed), t0))
    return rows


def run_optuna(X, y, seed):
    rows, t0 = [], time.perf_counter()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        res = score(params, X, y, seed)
        rows.append(log_row("optuna_tpe", seed, trial.number, params, res, t0))
        return res["objective"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=BUDGET)
    return rows


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
    X_dev, X_test, y_dev, y_test = load_data()  # test khoa, khong dung o day
    all_rows = []
    for name, fn in [("grid", run_grid), ("random", run_random), ("optuna_tpe", run_optuna)]:
        print("Running", name, "...")
        all_rows += fn(X_dev, y_dev, seed)
    df = pd.DataFrame(all_rows)
    df["best_so_far"] = df.groupby("method")["objective"].cummax()
    df.to_csv(f"results/trials_seed{seed}.csv", index=False)
    print(df.groupby("method")[["objective", "cv_f1", "latency_median_ms"]].agg(["max", "mean"]).round(4).to_string())
    print("Feasible trials:", df.groupby("method")["feasible"].sum().to_dict())
