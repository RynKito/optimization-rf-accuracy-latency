import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

SEED = 42
LATENCY_LIMIT_MS = 5.0  # sẽ chốt lại sau khi xem kết quả baseline


def load_data():
    """Split 80/20 có phân tầng. Test set CHỈ dùng 1 lần ở cuối."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def get_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def measure_latency_ms(model, X, n=100, repeats=3):
    """Latency dự đoán 1 mẫu (ms), lấy trung vị của nhiều lần đo."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(n):
            model.predict(X[i:i + 1])
        times.append((time.perf_counter() - start) / n * 1000)
    return float(np.median(times))


def evaluate(params, X_train, y_train, seed=SEED):
    """Objective: trả về F1 trung bình 5-fold CV và latency (ms/mẫu)."""
    model = RandomForestClassifier(**params, n_jobs=1, random_state=seed)
    f1 = cross_val_score(model, X_train, y_train, cv=get_cv(), scoring="f1").mean()
    model.fit(X_train, y_train)
    latency = measure_latency_ms(model, X_train)
    return {"cv_f1": float(f1), "latency_ms": latency,
            "feasible": latency <= LATENCY_LIMIT_MS}