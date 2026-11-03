import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

SEED = 42
LATENCY_LIMIT_MS = 5.0  # se chot lai 1 lan sau khi do baseline on dinh


def load_data():
    """80% development (X_dev), 20% test khoa. Test chi dung 1 lan o cuoi."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def get_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def measure_latency_ms(model, X, n=100, warmup=10):
    """Latency du doan 1 mau (ms): warm-up roi tra ve (median, p95)."""
    for i in range(warmup):
        model.predict(X[i:i + 1])
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        model.predict(X[i:i + 1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def count_nodes(model):
    """Complexity proxy: tong so node cua moi cay."""
    return int(sum(t.tree_.node_count for t in model.estimators_))


def evaluate(params, X_dev, y_dev, seed=SEED):
    """F1 trung binh 5-fold CV tren X_dev + latency + complexity."""
    model = RandomForestClassifier(**params, n_jobs=1, random_state=seed)
    f1 = cross_val_score(model, X_dev, y_dev, cv=get_cv(), scoring="f1").mean()
    model.fit(X_dev, y_dev)
    med, p95 = measure_latency_ms(model, X_dev)
    return {"cv_f1": float(f1), "latency_median_ms": med, "latency_p95_ms": p95,
            "n_nodes": count_nodes(model), "feasible": med <= LATENCY_LIMIT_MS}
