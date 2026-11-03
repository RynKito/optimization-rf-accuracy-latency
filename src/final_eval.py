import os, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score
from common import load_data, SEED, measure_latency_ms, count_nodes

OUT = "results/final_test.json"
if os.path.exists(OUT):
    raise SystemExit("final_test.json already exists: test set was already evaluated. Do not re-run to tune.")

X_dev, X_test, y_dev, y_test = load_data()
cand = json.load(open("results/final_candidate.json"))["params"]

models = {
    "default_rf": RandomForestClassifier(n_jobs=1, random_state=SEED),
    "selected_config": RandomForestClassifier(**cand, n_jobs=1, random_state=SEED),
}
res = {}
for name, m in models.items():
    m.fit(X_dev, y_dev)                      # refit tren toan bo development set
    pred = m.predict(X_test)                 # test dung 1 lan
    res[name] = {"test_f1": float(f1_score(y_test, pred)),
                 "test_accuracy": float(accuracy_score(y_test, pred)),
                 "test_recall_malignant_as_class0": float(recall_score(y_test, pred, pos_label=0)),
                 "n_nodes": count_nodes(m), "lat_med": [], "lat_p95": []}

# Do latency xen ke 5 vong de giam anh huong nhieu may
for _ in range(5):
    for name, m in models.items():
        med, p95 = measure_latency_ms(m, X_test)
        res[name]["lat_med"].append(med); res[name]["lat_p95"].append(p95)
for name in res:
    res[name]["latency_median_ms"] = float(np.median(res[name].pop("lat_med")))
    res[name]["latency_p95_ms"] = float(np.median(res[name].pop("lat_p95")))
res["selected_params"] = cand

json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
