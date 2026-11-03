import pandas as pd
from common import load_data, evaluate, SEED

X_train, X_test, y_train, y_test = load_data()
print("Train:", X_train.shape, "| Test:", X_test.shape)

# Baseline 1: Random Forest mặc định (scikit-learn defaults)
params = {}
rows = []
for run in range(3):  # đo 3 lần để thấy độ nhiễu của latency
    res = evaluate(params, X_train, y_train, seed=SEED)
    rows.append({"method": "default_rf", "run": run, **res})

df = pd.DataFrame(rows)
df.to_csv("results/baseline.csv", index=False)
print(df)