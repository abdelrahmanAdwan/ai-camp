"""
Generate `02_preprocessing.ipynb` for the Breast Cancer Wisconsin assignment.

Stage 2 (Preprocessing) — builds directly on the Stage-1 findings:
  - complete data (no imputation), all-numeric features, wide scales,
    heavy multicollinearity, moderate class imbalance.
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# Data Preprocessing
## Breast Cancer Wisconsin (Diagnostic) Dataset — Stage 2

This notebook prepares the dataset for modelling, acting on what the
**data-understanding** stage (`01_data_understanding.ipynb`) established:

| Finding (Stage 1) | Consequence for preprocessing |
|---|---|
| 569 rows, 0 missing, 0 duplicates | **No imputation / de-duplication** required |
| All 30 features numeric (`float64`) | **No categorical encoding** required |
| Features on very different scales | **Feature scaling (standardisation)** is essential |
| Moderate class imbalance (~63/37) | **Stratified** split & cross-validation |
| Strong multicollinearity (|r| > 0.9 pairs) | Explore **PCA / feature selection** |

**Golden rule applied throughout:** every transformation is **fit on the
training set only** and then applied to the test set, to prevent
**data leakage**.
""")

md("## 1. Setup & Imports")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

print("Libraries loaded. random_state =", RANDOM_STATE)
""")

md("## 2. Load the Dataset")

code(r"""
from pathlib import Path

csv_path = Path("breast_cancer_wisconsin.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})

print("Shape:", df.shape)
df.head()
""")

md(r"""
## 3. Re-verify Data Quality (quick gate)

Preprocessing should never assume the upstream stage — we re-check the two
quality gates before transforming anything.
""")

code(r"""
print("Missing values total :", int(df.isnull().sum().sum()))
print("Duplicate rows       :", int(df.duplicated().sum()))
assert df.isnull().sum().sum() == 0, "Unexpected missing values!"
assert df.duplicated().sum() == 0, "Unexpected duplicate rows!"
print("Quality gate passed -> no imputation / de-duplication needed.")
""")

md("## 4. Separate Features (X) and Target (y)")

code(r"""
feature_cols = [c for c in df.columns if c not in ("target", "diagnosis")]

X = df[feature_cols].copy()          # 30 numeric features
y = df["target"].copy()              # 0 = malignant, 1 = benign

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nTarget distribution:")
print(y.value_counts().rename({0: "malignant (0)", 1: "benign (1)"}))
""")

md(r"""
## 5. Stratified Train / Test Split

We hold out **20%** for testing. `stratify=y` preserves the ~63/37 class
ratio in both splits — important given the moderate imbalance. The split is
done **before** scaling so the scaler never sees test data.
""")

code(r"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print("\nClass proportions preserved (stratified):")
prop = pd.DataFrame({
    "train": y_train.value_counts(normalize=True).round(3),
    "test":  y_test.value_counts(normalize=True).round(3),
    "full":  y.value_counts(normalize=True).round(3),
}).rename(index={0: "malignant", 1: "benign"})
print(prop)
""")

md(r"""
## 6. Feature Scaling (Standardisation)

Stage 1 showed features spanning very different ranges (`area` in the
thousands vs `smoothness` < 1). We apply **`StandardScaler`** (z-score:
mean 0, std 1). Crucially the scaler is **fit on the training data only**,
then used to transform both train and test.
""")

code(r"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on TRAIN
X_test_scaled  = scaler.transform(X_test)        # transform only on TEST

# Back to DataFrames for readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_cols, index=X_test.index)

print("After scaling — training set (should be ~0 mean, ~1 std):")
check = pd.DataFrame({
    "mean": X_train_scaled.mean().round(3),
    "std":  X_train_scaled.std().round(3),
}).head(6)
print(check)
""")

md(r"""
**Leakage check.** The *test* set will not be perfectly mean-0 / std-1, and
that is correct — it is scaled using the **training** statistics, exactly as a
deployed model would treat unseen data.
""")

code(r"""
print("Test set after scaling (NOT exactly 0/1 — uses TRAIN stats, as intended):")
print(pd.DataFrame({
    "mean": X_test_scaled.mean().round(3),
    "std":  X_test_scaled.std().round(3),
}).head(6))
""")

md("### 6.1 Visualise the effect of scaling")

code(r"""
demo = ["mean area", "mean smoothness", "worst area", "mean texture"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, feat in zip(axes.ravel(), demo):
    ax.hist(X_train[feat], bins=30, alpha=0.6, label="raw", color="#C44E52")
    ax.set_title(f"{feat} — before scaling")
    ax.set_ylabel("count")
plt.suptitle("Raw feature ranges differ by orders of magnitude", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("figures/06_before_scaling.png", dpi=130, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, feat in zip(axes.ravel(), demo):
    ax.hist(X_train_scaled[feat], bins=30, alpha=0.6, label="scaled", color="#4C72B0")
    ax.set_title(f"{feat} — after standardisation")
    ax.set_ylabel("count")
    ax.set_xlim(-4, 6)
plt.suptitle("After StandardScaler: comparable, centred scales", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("figures/07_after_scaling.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
## 7. Multicollinearity — Optional Dimensionality Reduction (PCA)

Stage 1 found many feature pairs with |correlation| > 0.9. **PCA** is one way
to compress that redundancy into a smaller set of uncorrelated components.
PCA **requires scaled input**, which we now have. We fit PCA on the *training*
data only.

> This is exploratory: whether to actually use PCA is a modelling choice. Here
> we quantify how many components capture most of the variance.
""")

code(r"""
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

for thresh in (0.90, 0.95, 0.99):
    k = int(np.argmax(cum_var >= thresh) + 1)
    print(f"{int(thresh*100)}% of variance is captured by {k} components (of 30)")
""")

code(r"""
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#4C72B0")
for thresh, c in [(0.90, "#4C9F70"), (0.95, "#DD8452"), (0.99, "#C44E52")]:
    plt.axhline(thresh, ls="--", color=c, lw=1, label=f"{int(thresh*100)}% variance")
plt.title("PCA — Cumulative Explained Variance (fit on training set)")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.legend()
plt.tight_layout()
plt.savefig("figures/08_pca_scree.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
**Observation.** The 30 correlated features can be compressed dramatically —
roughly **10 components explain ~95%** of the variance. This confirms the
redundancy seen in Stage 1. We keep the full scaled feature set as the primary
output, and expose a PCA option for the modelling stage to compare against.
""")

md(r"""
## 8. Assemble a Reusable Preprocessing Pipeline

For deployment we wrap the steps in an sklearn `Pipeline` so the exact same
transformations are applied to any future data with a single `.transform()`.
The pipeline is fit on the **training** data only.
""")

code(r"""
preprocess = Pipeline(steps=[
    ("scaler", StandardScaler()),
    # PCA is left OUT of the default pipeline (kept as an option for modelling);
    # uncomment to enable a 10-component reduction:
    # ("pca", PCA(n_components=10, random_state=RANDOM_STATE)),
])

preprocess.fit(X_train)
_ = preprocess.transform(X_train)   # sanity check
print("Pipeline fitted:", preprocess)
""")

md("## 9. Persist the Processed Artifacts")

code(r"""
from pathlib import Path
out_dir = Path("processed")
out_dir.mkdir(exist_ok=True)

# 1) Scaled train/test feature matrices + labels as CSV (model-ready)
X_train_scaled.assign(target=y_train.values).to_csv(out_dir / "train_scaled.csv", index=False)
X_test_scaled.assign(target=y_test.values).to_csv(out_dir / "test_scaled.csv", index=False)

# 2) The fitted preprocessing pipeline and scaler (for inference on new data)
joblib.dump(preprocess, out_dir / "preprocess_pipeline.joblib")
joblib.dump(scaler, out_dir / "standard_scaler.joblib")

print("Saved to ./processed/:")
for p in sorted(out_dir.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size/1024:6.1f} KB")
""")

md(r"""
## 10. Summary — Preprocessing Complete

**What was done**

1. **Quality gate re-verified** — 0 missing, 0 duplicates → no imputation or
   de-duplication needed.
2. **X / y separation** — 30 numeric features vs `target` (0/1).
3. **Stratified 80/20 split** — class ratio preserved in train and test.
4. **Standardisation** — `StandardScaler` **fit on training only**, applied to
   both sets (no leakage). No categorical encoding needed (all-numeric).
5. **Multicollinearity explored** — PCA shows ~10 components capture ~95% of
   variance; offered as an optional pipeline step.
6. **Reusable pipeline + artifacts saved** to `./processed/` for the modelling
   stage.

**Outputs (`./processed/`)**

| File | Purpose |
|---|---|
| `train_scaled.csv` / `test_scaled.csv` | Model-ready scaled feature matrices + target |
| `standard_scaler.joblib` | Fitted scaler (train statistics) |
| `preprocess_pipeline.joblib` | Full fitted preprocessing pipeline for new data |

**Next stage:** model training & evaluation — train classifiers on
`train_scaled.csv`, use **stratified cross-validation**, and evaluate with
**precision / recall / F1 / ROC-AUC** (prioritising recall on the malignant
class), never accuracy alone.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out = HERE / "02_preprocessing.ipynb"
nbf.write(nb, out)
print(f"Wrote {out.name} with {len(cells)} cells")
