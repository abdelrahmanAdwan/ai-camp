"""
Export the Breast Cancer Wisconsin (Diagnostic) dataset and run the initial
data-understanding analysis. Produces:
  - breast_cancer_wisconsin.csv   (the dataset, target column = 'diagnosis')
  - analysis_summary.json         (machine-readable summary for the report)

Runs on pandas + scikit-learn only (no plotting libs required).
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

HERE = Path(__file__).resolve().parent


def build_dataframe() -> pd.DataFrame:
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    # sklearn encodes target as 0 = malignant, 1 = benign.
    # Add a human-readable label column for reporting/visuals.
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    return df, ds


def analyze(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c not in ("target", "diagnosis")]

    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    class_counts = df["diagnosis"].value_counts().to_dict()
    total = int(len(df))
    class_pct = {k: round(100 * v / total, 2) for k, v in class_counts.items()}

    desc = df[feature_cols].describe().T
    numeric_summary = {
        col: {
            "mean": round(float(desc.loc[col, "mean"]), 4),
            "std": round(float(desc.loc[col, "std"]), 4),
            "min": round(float(desc.loc[col, "min"]), 4),
            "max": round(float(desc.loc[col, "max"]), 4),
        }
        for col in feature_cols
    }

    summary = {
        "n_rows": total,
        "n_columns_total": int(df.shape[1]),
        "n_feature_columns": len(feature_cols),
        "target_variable": "diagnosis (0=malignant, 1=benign)",
        "task_type": "binary classification",
        "dtype_counts": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().to_dict().items()},
        "column_dtypes": dtypes,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isnull().sum().sum()),
        "columns_with_missing": missing,
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "class_percentages": class_pct,
        "imbalance_ratio_majority_to_minority": round(
            max(class_counts.values()) / min(class_counts.values()), 3
        ),
        "numeric_feature_summary": numeric_summary,
    }
    return summary


def main():
    df, _ = build_dataframe()
    csv_path = HERE / "breast_cancer_wisconsin.csv"
    df.to_csv(csv_path, index=False)

    summary = analyze(df)
    (HERE / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Wrote {csv_path.name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Target classes: {summary['class_counts']}")
    print(f"Missing values total: {summary['missing_values_total']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")


if __name__ == "__main__":
    main()
