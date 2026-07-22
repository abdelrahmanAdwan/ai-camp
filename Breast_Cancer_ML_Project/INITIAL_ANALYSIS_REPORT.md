# Initial Analysis Report
## Dataset Selection & Data Understanding

**Task:** Select a suitable classification dataset and perform an initial
analysis to understand its structure and characteristics before starting the
machine-learning process.

**Author:** _(your name)_
**Date:** 2026-07-19

---

## 1. Dataset Selection

**Chosen dataset:** **Breast Cancer Wisconsin (Diagnostic)**

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository (bundled in scikit-learn `load_breast_cancer`) |
| Original study | Wolberg, Street & Mangasarian, University of Wisconsin |
| Domain | Medical diagnostics (oncology) |
| Records | **569** (satisfies the ≥ 500 requirement) |
| Features | 30 numeric |
| Target | `diagnosis` — malignant / benign |
| Problem type | **Binary classification** |
| Is it Iris? | **No** (Iris is explicitly excluded) |

**Why this dataset was selected**

- Meets the size requirement (569 ≥ 500) and is a genuine, well-documented
  classification problem.
- Real-world and meaningful: each record is computed from a digitised image of a
  fine-needle aspirate (FNA) of a breast mass, describing the cell nuclei
  present.
- Clean enough to reason about clearly, yet still exhibits **class imbalance**
  and strong **feature multicollinearity** — both worth discussing in a data
  understanding exercise.
- Fully reproducible offline (no external download or API key needed).

**Feature construction.** Ten base measurements are computed for each cell
nucleus — *radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry, fractal dimension*. Each appears three times in the
dataset: the **mean**, the **standard error** (`error`), and the **worst**
(mean of the three largest values). 10 × 3 = **30 features**.

---

## 2. Dataset Structure

| Property | Value |
|---|---|
| Rows (records) | **569** |
| Columns (total) | **32** (30 features + `target` + `diagnosis`) |
| Feature columns | **30** |
| Label columns | `target` (0/1 integer), `diagnosis` (text label) |

### Data types

| dtype | Count | Notes |
|---|---|---|
| `float64` | 30 | All feature columns — continuous measurements |
| `int32` | 1 | `target` (0 = malignant, 1 = benign) |
| `object` (str) | 1 | `diagnosis` (human-readable label added for analysis) |

**Takeaway:** every predictive feature is already **numeric and continuous**.
There are **no categorical features to encode** — the modelling effort will
centre on *scaling*, not encoding.

---

## 3. Target Variable

- **Target:** `diagnosis` — categorical with two classes: **malignant** /
  **benign**.
- **Model-ready encoding:** `target` — `0 = malignant`, `1 = benign` (provided
  by scikit-learn).
- **Task type:** supervised **binary classification**.

---

## 4. Data Quality

| Check | Result |
|---|---|
| Total missing values | **0** |
| Columns with missing values | **None** |
| Duplicate rows | **0** |

The dataset is **complete and free of duplicates**. No imputation or
de-duplication will be required in preprocessing. (The absence of missing data
was verified, not assumed — verification is part of the analysis.)

---

## 5. Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Benign | 357 | 62.74 % |
| Malignant | 212 | 37.26 % |
| **Total** | **569** | 100 % |

**Imbalance ratio (majority : minority) = 1.68 : 1** — a **moderate** imbalance.

**Implications:**
- Use a **stratified** train/test split and stratified cross-validation to keep
  the class ratio consistent across folds.
- Report **precision, recall, F1 and ROC-AUC**, not accuracy alone. In a medical
  context a **false negative** (a missed malignant tumour) is the most costly
  error, so recall on the malignant class matters most.

---

## 6. Descriptive Statistics — Key Observations

- Features live on **very different scales**. Examples:
  - `mean area` ≈ 655 (range 143.5 – 2501), `worst area` up to 4254.
  - `mean smoothness` ≈ 0.096, `mean fractal dimension` ≈ 0.063 (all < 1).
- This wide range means distance- and gradient-based algorithms (kNN, SVM,
  logistic regression, neural networks) will need **feature standardisation**
  (e.g. `StandardScaler`).
- Several features (`mean area`, `mean concavity`, `mean concave points`) are
  **right-skewed**; a transform may help some linear models.

---

## 7. Relationships Between Features

- **Strong multicollinearity.** Many feature pairs have |correlation| > 0.9 —
  expected because `radius`, `perimeter` and `area` are geometrically linked and
  each base measurement repeats as mean/error/worst.
- **Most predictive features** (largest |correlation| with the target) are the
  size/shape descriptors: `worst concave points`, `worst perimeter`,
  `worst radius`, `mean concave points`, `worst area`. Malignant tumours show
  systematically **larger** values on these.
- Multicollinearity suggests **PCA or feature selection** could be explored, and
  cautions against over-interpreting individual linear-model coefficients.

---

## 8. Visualizations Produced

Saved in `figures/` and embedded in the notebook:

1. `01_class_distribution.png` — bar chart of benign vs malignant counts.
2. `02_feature_histograms.png` — distributions of the ten `mean` features.
3. `03_boxplots_by_class.png` — key features split by diagnosis (class separation).
4. `04_correlation_heatmap.png` — 30×30 feature correlation matrix.
5. `05_target_correlation.png` — each feature's correlation with the target.

---

## 9. Readiness for Preprocessing — Action List

1. **No imputation / de-duplication** needed (data is complete and unique).
2. **Apply feature scaling** (StandardScaler) — essential given the value ranges.
3. **No categorical encoding** needed — features are numeric; target encoding
   (`target` 0/1) is already provided.
4. **Stratified split & CV** to preserve the ~63/37 class balance.
5. **Evaluate with precision / recall / F1 / ROC-AUC**, prioritising recall on
   the malignant class.
6. **Consider dimensionality reduction / feature selection** (PCA or dropping
   near-duplicate features) to handle multicollinearity.

**Conclusion:** the dataset is well understood, high quality, and ready to move
into the preprocessing and modelling stages.

---

## Deliverables in this folder

| File | Description |
|---|---|
| `01_data_understanding.ipynb` | Jupyter notebook — full initial analysis + visualizations |
| `breast_cancer_wisconsin.csv` | The dataset (569 × 32) |
| `INITIAL_ANALYSIS_REPORT.md` | This report |
| `figures/*.png` | Generated visualizations |
| `analysis_summary.json` | Machine-readable analysis summary |
| `export_and_analyze.py` / `build_notebook.py` | Scripts that produced the artifacts |
