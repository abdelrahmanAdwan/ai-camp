"""
Generate the Jupyter notebook `01_data_understanding.ipynb` for the
Breast Cancer Wisconsin data-understanding assignment.

Writing the .ipynb as JSON via nbformat keeps the notebook reproducible and
lets us version it as source.
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
# Dataset Selection & Data Understanding
## Breast Cancer Wisconsin (Diagnostic) Dataset

**Assignment:** Select a suitable classification dataset and perform an initial
analysis to understand its structure and characteristics *before* starting the
machine-learning process.

**Dataset:** Breast Cancer Wisconsin (Diagnostic) — from the UCI Machine
Learning Repository, bundled in scikit-learn.

**Why this dataset**
- **569 records** (≥ 500 requirement satisfied) and **30 numeric features**.
- A genuine, well-documented **binary classification** problem: predict whether
  a tumour is **malignant** or **benign** from digitised images of a fine-needle
  aspirate (FNA).
- **Not** the Iris dataset (explicitly excluded by the brief).
- Real-world, medically meaningful, and clean enough to reason about while still
  showing class imbalance and strong feature correlations worth discussing.

### Objectives
1. Understand the dataset's structure.
2. Identify the target variable.
3. Explore the data (types, distributions, relationships).
4. Assess data quality (missing values, duplicates, class balance).
5. Prepare for the preprocessing stage.
""")

md("## 1. Setup & Imports")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

print("pandas", pd.__version__, "| numpy", np.__version__)
""")

md(r"""
## 2. Load the Dataset

We load from the exported CSV so the notebook is fully reproducible from the
dataset file shipped with this assignment. If the CSV is missing, we fall back
to loading it directly from scikit-learn.
""")

code(r"""
from pathlib import Path

csv_path = Path("breast_cancer_wisconsin.csv")

if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"Loaded from CSV: {csv_path}")
else:
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    print("Loaded from scikit-learn")

df.head()
""")

md(r"""
**Column notes**

- The 30 feature columns describe cell-nucleus characteristics. Each of ten base
  measurements (*radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension*) appears three times:
  the **mean**, the **standard error (`error`)**, and the **worst / largest
  (`worst`)** value across the nuclei in the image.
- `target` is the scikit-learn integer encoding: **0 = malignant, 1 = benign**.
- `diagnosis` is the human-readable label we added for analysis and plotting.
""")

md("## 3. Dataset Structure — Rows, Columns & Data Types")

code(r"""
n_rows, n_cols = df.shape
print(f"Number of rows (records):   {n_rows}")
print(f"Number of columns (total):  {n_cols}")

feature_cols = [c for c in df.columns if c not in ("target", "diagnosis")]
print(f"Feature columns:            {len(feature_cols)}")
print(f"Label/target columns:       {['target', 'diagnosis']}")
""")

code(r"""
# Full structural overview: dtypes + non-null counts + memory
df.info()
""")

code(r"""
# How many columns of each data type?
df.dtypes.value_counts()
""")

md(r"""
**Observation.** All 30 features are continuous **float64** measurements. The
only non-numeric column is the `diagnosis` label we added (object/string).
`target` is an integer encoding of that same label. There are **no categorical
feature columns** to encode — the modelling work will centre on *scaling*, not
encoding.
""")

md("## 4. Identify the Target Variable")

code(r"""
print("Target variable: 'diagnosis'  (encoded as 'target': 0 = malignant, 1 = benign)")
print("Task type      : Binary classification")
print()
print("Unique target values:")
print(df["diagnosis"].value_counts())
""")

md(r"""
The target is **`diagnosis`** — a two-class label (`malignant` / `benign`).
This is a **supervised binary classification** problem. `target` (0/1) is the
model-ready numeric form of the same variable; we keep both but will train on
`target` and read `diagnosis` for interpretation.
""")

md("## 5. Data Quality — Missing Values & Duplicates")

code(r"""
missing = df.isnull().sum()
print("Total missing values across the whole dataset:", int(missing.sum()))
print()
cols_with_missing = missing[missing > 0]
if cols_with_missing.empty:
    print("No column contains missing values.")
else:
    print("Columns with missing values:")
    print(cols_with_missing)
""")

code(r"""
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
""")

md(r"""
**Observation.** The dataset is complete — **0 missing values** and **0
duplicate rows**. This is unusual for real-world data and means the preprocessing
stage will not need imputation or de-duplication. We still document the check
because *verifying* the absence of missing data is itself part of data
understanding.
""")

md("## 6. Descriptive Statistics")

code(r"""
df[feature_cols].describe().T
""")

md(r"""
**Observation — features are on very different scales.** For example `area`
features run into the thousands while `smoothness` and `fractal dimension` sit
below 1. Distance- and gradient-based models (kNN, SVM, logistic regression,
neural nets) will therefore require **feature scaling / standardisation** during
preprocessing.
""")

md("## 7. Class Distribution (Target Balance)")

code(r"""
counts = df["diagnosis"].value_counts()
pct = (counts / len(df) * 100).round(2)
balance = pd.DataFrame({"count": counts, "percent": pct})
print(balance)
print()
ratio = counts.max() / counts.min()
print(f"Imbalance ratio (majority : minority) = {ratio:.2f} : 1")
""")

code(r"""
fig, ax = plt.subplots(figsize=(6, 4))
order = ["benign", "malignant"]
sns.countplot(data=df, x="diagnosis", order=order,
              hue="diagnosis", palette=["#4C9F70", "#C44E52"], legend=False, ax=ax)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Class Distribution — Diagnosis")
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Number of records")
plt.tight_layout()
plt.savefig("figures/01_class_distribution.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
**Observation.** The classes are **moderately imbalanced**: ~63% benign vs ~37%
malignant (≈ 1.68 : 1). Not severe, but enough that we should:
- prefer **stratified** train/test splits and cross-validation, and
- report **precision, recall and F1** (not just accuracy), since in a cancer
  setting a false negative — missing a malignant tumour — is the costly error.
""")

md("## 8. Feature Distributions")

code(r"""
# Focus on the 10 'mean' features for readable histograms
mean_features = [c for c in feature_cols if c.startswith("mean ")]

axes = df[mean_features].hist(figsize=(14, 10), bins=30, color="#4C72B0",
                              edgecolor="white")
plt.suptitle("Distributions of the 10 'mean' Features", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("figures/02_feature_histograms.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
**Observation.** Several features (`mean area`, `mean concavity`, `mean concave
points`) are **right-skewed**, hinting that a log or power transform *may* help
some linear models. Others are roughly bell-shaped.
""")

md("## 9. How Well Do Features Separate the Two Classes?")

code(r"""
# Boxplots of a few discriminative 'mean' features, split by diagnosis
show = ["mean radius", "mean perimeter", "mean area",
        "mean concavity", "mean concave points", "mean texture"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, feat in zip(axes.ravel(), show):
    sns.boxplot(data=df, x="diagnosis", y=feat, order=order,
                hue="diagnosis", palette=["#4C9F70", "#C44E52"],
                legend=False, ax=ax)
    ax.set_title(feat)
    ax.set_xlabel("")
plt.suptitle("Feature values by Diagnosis (malignant tends higher)", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("figures/03_boxplots_by_class.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
**Observation.** Malignant tumours consistently show **larger** radius,
perimeter, area, concavity and concave-point values. These size/shape features
look **strongly predictive**, which is encouraging for downstream modelling.
""")

md("## 10. Feature Correlations")

code(r"""
corr = df[feature_cols].corr()

plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.6}, linewidths=0.3)
plt.title("Correlation Matrix of the 30 Features", fontsize=14)
plt.tight_layout()
plt.savefig("figures/04_correlation_heatmap.png", dpi=130, bbox_inches="tight")
plt.show()
""")

code(r"""
# Highly correlated feature pairs (|r| > 0.9) — candidates for redundancy
corr_abs = corr.abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
high_pairs = (
    upper.stack()
    .reset_index()
    .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"})
)
high_pairs = high_pairs[high_pairs["abs_corr"] > 0.9].sort_values(
    "abs_corr", ascending=False
)
print(f"{len(high_pairs)} feature pairs with |correlation| > 0.9")
high_pairs.head(15)
""")

md(r"""
**Observation.** Many features are **highly correlated** — unsurprising, since
`radius`, `perimeter` and `area` are geometrically linked, and each base
measurement repeats as *mean / error / worst*. This **multicollinearity**
suggests dimensionality reduction (PCA) or feature selection could be explored
in preprocessing, and warns against interpreting individual coefficients of a
linear model too literally.
""")

md("## 11. Correlation of Each Feature with the Target")

code(r"""
target_corr = (
    df[feature_cols + ["target"]].corr()["target"]
    .drop("target")
    .sort_values()
)

plt.figure(figsize=(9, 10))
colors = ["#C44E52" if v < 0 else "#4C9F70" for v in target_corr.values]
plt.barh(target_corr.index, target_corr.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature correlation with target (1=benign, 0=malignant)")
plt.xlabel("Pearson correlation with 'target'")
plt.tight_layout()
plt.savefig("figures/05_target_correlation.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
**Observation.** Because `target` encodes benign=1, features that are **large in
malignant tumours** correlate **negatively** with the target. The strongest
signals (`worst concave points`, `worst perimeter`, `worst radius`, `mean
concave points`) are exactly the size/shape features the boxplots highlighted.
""")

md(r"""
## 12. Summary & Readiness for Preprocessing

### What we learned
| Aspect | Finding |
|---|---|
| Records × columns | **569 rows × 32 columns** (30 features + `target` + `diagnosis`) |
| Feature types | All 30 features are continuous **float64**; no categorical features |
| Target variable | **`diagnosis`** (`malignant` / `benign`), encoded as `target` 0/1 |
| Task | **Binary classification** |
| Missing values | **None** (0) |
| Duplicate rows | **None** (0) |
| Class balance | Benign ~63% vs Malignant ~37% (**≈1.68:1**, moderate imbalance) |
| Feature scales | Very different (area in thousands vs smoothness < 1) → **scaling needed** |
| Correlations | Strong **multicollinearity**; size/shape features most predictive |

### Implications for the preprocessing stage
1. **No imputation / de-duplication needed** — data is complete and unique.
2. **Feature scaling is essential** (StandardScaler) given the wide value ranges.
3. **No categorical encoding needed** — features are already numeric; only the
   target label needs its 0/1 encoding (already provided as `target`).
4. **Use a stratified train/test split** and stratified cross-validation to
   preserve the class ratio.
5. **Evaluate with precision / recall / F1 / ROC-AUC**, not accuracy alone —
   false negatives (missed malignancies) are the expensive error.
6. **Consider dimensionality reduction / feature selection** (PCA or dropping
   near-duplicate features) to address the heavy multicollinearity.

The dataset is well understood and ready for the preprocessing and modelling
stages.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out = HERE / "01_data_understanding.ipynb"
nbf.write(nb, out)
print(f"Wrote {out.name} with {len(cells)} cells")
