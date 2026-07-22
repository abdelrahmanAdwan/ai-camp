"""
Set the baseline SVM as the final model (final_model.joblib).

Rationale: SVM has the highest malignant RECALL (0.9762), which is the priority
metric for this medical screening task (minimise false negatives).

Reproduces the exact Stage-2/3 split + training-only scaling (random_state=42),
loads the already-fitted baseline SVM, verifies its metrics, and saves the same
bundle format used before (model + scaler + feature_cols + metadata).
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

HERE = Path(__file__).resolve().parent
RANDOM_STATE = 42
POS_LABEL = 0  # malignant

df = pd.read_csv(HERE / "breast_cancer_wisconsin.csv")
feature_cols = [c for c in df.columns if c not in ("target", "diagnosis")]
X, y = df[feature_cols], df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
scaler = StandardScaler().fit(X_train)
X_test_s = scaler.transform(X_test)

# Load the baseline SVM fitted in Stage 3 (identical deterministic pipeline)
svm = joblib.load(HERE / "models" / "svm.joblib")

y_pred = svm.predict(X_test_s)
metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, pos_label=POS_LABEL),
    "Recall":    recall_score(y_test, y_pred, pos_label=POS_LABEL),
    "F1-score":  f1_score(y_test, y_pred, pos_label=POS_LABEL),
}
print("Baseline SVM test metrics:", {k: round(v, 4) for k, v in metrics.items()})

bundle = {
    "model": svm,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "positive_class": "malignant (0)",
    "model_name": "SVM (baseline)",
    "best_params": {"kernel": "rbf", "C": 1.0, "gamma": "scale",
                    "class_weight": "balanced"},
    "test_metrics": metrics,
    "selection_reason": ("Highest malignant recall (0.9762) — prioritised for "
                         "the medical screening use case to minimise false "
                         "negatives."),
}
joblib.dump(bundle, HERE / "models" / "final_model.joblib")
print("Saved -> models/final_model.joblib  (final model = SVM baseline)")

# sanity reload
loaded = joblib.load(HERE / "models" / "final_model.joblib")
print("Reloaded final model:", loaded["model_name"])
