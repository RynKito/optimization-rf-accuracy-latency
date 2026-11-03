import pandas as pd
from common import load_data, evaluate, SEED

X_dev, X_test, y_dev, y_test = load_data()
print("Dev:", X_dev.shape, "| Test (locked):", X_test.shape)

rows = []
for run in range(5):  # do 5 lan de thay do nhieu cua latency
    res = evaluate({}, X_dev, y_dev, seed=SEED)
    rows.append({"method": "default_rf", "run": run, **res})

df = pd.DataFrame(rows)
df.to_csv("results/baseline.csv", index=False)
print(df.to_string())
