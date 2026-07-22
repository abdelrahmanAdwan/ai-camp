# Breast Cancer Wisconsin — Classification Project

An end-to-end machine-learning classification project on the **Breast Cancer
Wisconsin (Diagnostic)** dataset (569 records, 30 numeric features, binary
target: *malignant* / *benign*).

> **Positive class = malignant (0).** In cancer screening the costly error is a
> false negative (missing a malignant tumour), so **recall on the malignant
> class** is the priority metric throughout.

## Pipeline (run the notebooks in order)

| # | Notebook | Stage | Key outputs |
|---|----------|-------|-------------|
| 1 | `01_data_understanding.ipynb` | Data understanding | `INITIAL_ANALYSIS_REPORT.md`, EDA figures |
| 2 | `02_preprocessing.ipynb` | Cleaning & preprocessing | `processed/train_scaled.csv`, `processed/test_scaled.csv`, fitted scaler/pipeline |
| 3 | `03_individual_models.ipynb` | Train 5 models | `reports/individual_model_metrics.csv`, 5 fitted models, confusion matrices |
| 4 | `04_model_comparison.ipynb` | Compare & select | `reports/COMPARISON_REPORT.md`, comparison charts |
| 5 | `05_model_optimization.ipynb` | Tune best 2 + finalize | `reports/OPTIMIZATION_REPORT.md`, `models/final_model.joblib` |

## Results (held-out test set, 114 rows)

Baseline models:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes   | 0.930 | 0.905 | 0.905 | 0.905 |
| Decision Tree | 0.930 | 0.886 | 0.929 | 0.907 |
| Random Forest | 0.947 | 0.929 | 0.929 | 0.929 |
| **SVM**       | **0.965** | 0.932 | **0.976** | **0.954** |
| XGBoost       | 0.956 | 0.974 | 0.905 | 0.938 |

After tuning (GridSearchCV / RandomizedSearchCV, 5-fold stratified CV on SVM +
XGBoost):

- **XGBoost improved**: F1 0.938 → **0.951**, recall 0.905 → 0.929.
- **SVM** test F1 dipped slightly (0.954 → 0.943) despite a higher CV score —
  small-test-set variance.

**Final saved model:** **SVM (baseline)** — Accuracy 0.965, Precision 0.932,
Recall **0.976**, F1 0.954 (`models/final_model.joblib`, bundled with its
scaler).

### Final-model decision (recorded)

The **baseline SVM** is set as the final model. Selection is driven by
**malignant recall**, the priority metric for this medical screening task:
missing a malignant tumour (a false negative) is the most harmful error, so the
model that catches the most malignancies is preferred.

- **SVM (baseline)** — recall **0.976**, F1 0.954, accuracy 0.965 → **chosen**.
- Optimized **XGBoost** — recall 0.929, F1 0.951 (higher precision 0.975) →
  best *tuned* model, kept available at `models/xgboost.joblib`.

Tuning was still valuable: it measurably improved XGBoost (F1 0.938 → 0.951,
recall 0.905 → 0.929) and confirmed via cross-validation that SVM was already
near-optimal (its tuned test F1 did not exceed the baseline). Given the
recall-first clinical priority, the **baseline SVM is deliberately retained** as
the final model. The set-up is reproducible via `set_final_model_svm.py`.

## Folder layout

```
Breast_Cancer_ML_Project/
├── 01_data_understanding.ipynb ... 05_model_optimization.ipynb
├── breast_cancer_wisconsin.csv        # raw dataset (569 × 32)
├── processed/                         # cleaned, scaled, split data + fitted scaler/pipeline
├── models/                            # 5 baseline models + final_model.joblib
├── reports/                           # metrics CSV + markdown reports
├── figures/                           # all charts (EDA, comparison, tuning, confusion matrices)
├── INITIAL_ANALYSIS_REPORT.md
└── build_*.py / export_and_analyze.py # scripts that generate the notebooks
```

## Environment

Python 3.11 with: `pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost,
joblib, nbformat, nbconvert`.

## Using the final model on new data

```python
import joblib, pandas as pd
b = joblib.load("models/final_model.joblib")
X_new = pd.DataFrame(...)[b["feature_cols"]]      # raw (unscaled) feature rows
pred = b["model"].predict(b["scaler"].transform(X_new))   # 0 = malignant, 1 = benign
```
