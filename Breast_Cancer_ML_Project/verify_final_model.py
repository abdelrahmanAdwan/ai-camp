"""Independent verification of the saved final model."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

HERE = Path(__file__).resolve().parent
POS = 0  # malignant

b = joblib.load(HERE / "models" / "final_model.joblib")
print("=== Bundle contents ===")
for k in b:
    if k in ("model", "scaler"):
        print(f"  {k:16s}: {type(b[k]).__name__}")
    else:
        print(f"  {k:16s}: {b[k]}")

# Rebuild the same held-out test set
df = pd.read_csv(HERE / "breast_cancer_wisconsin.csv")
cols = b["feature_cols"]
X, y = df[cols], df["target"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# Predict through the bundled scaler (raw -> scaled -> predict)
y_pred = b["model"].predict(b["scaler"].transform(X_test))

live = {
    "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "Precision": round(precision_score(y_test, y_pred, pos_label=POS), 4),
    "Recall":    round(recall_score(y_test, y_pred, pos_label=POS), 4),
    "F1-score":  round(f1_score(y_test, y_pred, pos_label=POS), 4),
}
print("\n=== Live re-evaluation on held-out test set ===")
print("  ", live)
print("=== Metrics stored in bundle ===")
print("  ", {k: round(v, 4) for k, v in b["test_metrics"].items()})

match = all(abs(live[k] - round(b["test_metrics"][k], 4)) < 1e-4 for k in live)
print("\nStored vs live metrics match:", match)

print("\n=== Confusion matrix (rows=true, cols=pred; [malignant, benign]) ===")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
tn_fp_fn_tp = confusion_matrix(y_test, y_pred, labels=[0, 1])
fn = tn_fp_fn_tp[0, 1]  # true malignant predicted benign
print(f"False negatives (missed malignancies): {fn}")

print("\n=== classification_report ===")
print(classification_report(y_test, y_pred,
                            target_names=["malignant (0)", "benign (1)"], digits=4))

assert b["model_name"] == "SVM (baseline)", "Final model is not the baseline SVM!"
assert match, "Stored metrics do not match live evaluation!"
assert abs(live["Recall"] - 0.9762) < 1e-3, "Recall mismatch!"
print("VERIFICATION PASSED: final model = SVM (baseline), recall = 0.9762")
