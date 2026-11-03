# Accuracy-Latency Optimization of Random Forest Hyperparameters

Hyperparameter optimization of a Random Forest classifier on the Breast Cancer Wisconsin dataset, comparing Default, Random Search and Optuna (TPE) under a prediction-latency constraint.

> Status: Week 1 (formulation and project setup). Results will be added in later weeks.

## 1. Problem

A Random Forest classifier detects malignant vs. benign tumors. Its hyperparameters (number of trees, depth, ...) affect both accuracy and prediction time: bigger forests are usually more accurate but slower.

**Question:** which hyperparameter configuration gives the highest cross-validated F1-score while keeping prediction latency within a deployment limit?

## 2. Optimization Formulation

**Decision variables** x = (n_estimators, max_depth, min_samples_leaf, max_features)

| Variable | Type | Search range |
|---|---|---|
| n_estimators | integer | [10, 300] |
| max_depth | integer | [2, 20] |
| min_samples_leaf | integer | [1, 10] |
| max_features | float (fraction of features) | [0.1, 1.0] |

**Objective:** maximize the mean 5-fold cross-validated F1-score

    max  f(x) = F1_CV(x)

**Constraints:**

    g(x) = latency_ms_per_sample(x) - 5 <= 0
    x in the search ranges above

Infeasible configurations (g(x) > 0) are rejected by a penalty: f(x) is set to 0.

**Assumptions:** fixed random seed for the model and CV folds; latency is measured on the same machine with the same procedure for all methods; the test set is never used for tuning.

## 3. Dataset

- Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer`
- 569 samples, 30 numeric features, binary target
- Source: UCI Machine Learning Repository (Street, Wolberg and Mangasarian, 1993)
- Split: stratified 80% train+validation (5-fold CV) / 20% held-out test

## 4. Method

| Method | Role | Budget |
|---|---|---|
| Default Random Forest (scikit-learn defaults) | Baseline 1 | 1 config |
| Random Search | Baseline 2 | 40 trials |
| Optuna TPE | Main method | 40 trials |

All methods use the same data, folds, metric and budget. Stochastic methods are repeated with 10 seeds and reported as mean ± std. The test set is evaluated once, only for the final chosen configuration.

## 5. Key Results

_To be added in Week 4._

## 6. How to Run

```bash
git clone https://github.com/RynKito/optimization-rf-accuracy-latency.git
cd optimization-rf-accuracy-latency
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python version: 3.12.6. Main notebook: _to be added_.

## 7. Repository Structure

```
├─ README.md
├─ requirements.txt
├─ notebooks/   # experiments
├─ src/         # reusable code
├─ results/     # CSV/JSON logs
├─ figures/     # plots
├─ report/      # PDF report
└─ data/        # data notes
```

## 8. Limitations

_To be added in Week 5._

## 9. AI assistance

AI tools (Claude) were used to help plan the project and draft code and documentation. All formulations, code and results are checked and understood by the author.
