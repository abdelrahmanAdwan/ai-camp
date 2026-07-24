"""
Breast Cancer Wisconsin — ML Web Application (Streamlit)
========================================================
An interactive interface over the classification pipeline. Pages:
  1. Home
  2. Dataset Upload
  3. Model Selection
  4. Model Training
  5. Prediction
  6. Performance Dashboard

Run:  streamlit run app.py
"""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, ConfusionMatrixDisplay)
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# --------------------------------------------------------------------------- #
# Config & constants
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "breast_cancer_wisconsin.csv"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"
POS_LABEL = 0                       # malignant = positive class
LABEL_NAMES = {0: "malignant", 1: "benign"}
RANDOM_STATE = 42

st.set_page_config(page_title="Breast Cancer ML App", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_bundled_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df["diagnosis"] = df["target"].map(LABEL_NAMES)
    return df


@st.cache_data(show_spinner=False)
def feature_columns():
    df = load_bundled_data()
    return [c for c in df.columns if c not in ("target", "diagnosis")]


@st.cache_data(show_spinner=False)
def load_baseline_metrics():
    p = REPORTS_DIR / "individual_model_metrics.csv"
    if p.exists():
        return pd.read_csv(p, index_col="Model")
    return None


@st.cache_resource(show_spinner=False)
def load_final_model():
    p = MODELS_DIR / "final_model.joblib"
    if p.exists():
        return joblib.load(p)
    return None


def build_model(name):
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE,
                                      class_weight="balanced")
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                      class_weight="balanced", n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE,
                   class_weight="balanced")
    if name == "XGBoost" and HAS_XGB:
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric="logloss", random_state=RANDOM_STATE,
                             n_jobs=-1)
    raise ValueError(f"Unknown / unavailable model: {name}")


MODEL_INFO = {
    "Naive Bayes":   "Probabilistic, assumes feature independence. Fast baseline; "
                     "handicapped by the strong feature correlations in this data.",
    "Decision Tree": "Single interpretable tree. Prone to overfitting / instability.",
    "Random Forest": "Ensemble of many trees. Strong, robust, low-variance.",
    "SVM":           "Margin-maximising classifier (RBF kernel). Excellent on this "
                     "scaled, high-dimensional data — highest malignant recall.",
    "XGBoost":       "Gradient-boosted trees. Highest precision here; top tuned model.",
}


def evaluate(model, X_test_s, y_test):
    y_pred = model.predict(X_test_s)
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
        "F1-score":  f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0),
    }, y_pred


def get_active_data():
    """Return the dataset currently in use (uploaded or bundled)."""
    if "data" not in st.session_state:
        st.session_state["data"] = load_bundled_data()
        st.session_state["data_name"] = "breast_cancer_wisconsin.csv (bundled)"
    return st.session_state["data"]


def _show_prediction(pred, proba, actual):
    """Render a single-sample prediction result."""
    label = LABEL_NAMES[pred]
    if pred == 0:   # malignant
        st.error(f"### 🔴 Prediction: **MALIGNANT**")
    else:
        st.success(f"### 🟢 Prediction: **BENIGN**")

    if proba is not None:
        p_mal = float(proba[0][0])   # column 0 == class 0 == malignant
        p_ben = float(proba[0][1])
        c1, c2 = st.columns(2)
        c1.metric("P(malignant)", f"{p_mal:.1%}")
        c2.metric("P(benign)", f"{p_ben:.1%}")
        st.progress(p_mal, text=f"Malignant probability: {p_mal:.1%}")

    if actual is not None:
        ok = (actual == pred)
        st.caption(f"Actual label: **{LABEL_NAMES[actual]}** — "
                   f"{'✅ correct' if ok else '❌ incorrect'}")


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 Breast Cancer ML")
st.sidebar.caption("Wisconsin Diagnostic — classification app")
PAGE = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📤 Dataset Upload", "⚙️ Model Selection",
     "🏋️ Model Training", "🔮 Prediction", "📊 Performance Dashboard"],
)
st.sidebar.divider()
st.sidebar.caption(f"Active dataset:\n\n**{st.session_state.get('data_name', 'bundled')}**")
if st.session_state.get("model_choice"):
    st.sidebar.caption(f"Selected model: **{st.session_state['model_choice']}**")
if st.session_state.get("trained"):
    st.sidebar.success(f"Trained: {st.session_state['trained']['name']}")


# =========================================================================== #
# 1. HOME
# =========================================================================== #
if PAGE.startswith("🏠"):
    st.title("Breast Cancer Wisconsin — ML Classification App")
    st.markdown(
        "Interactively upload data, train classifiers, make predictions, and "
        "inspect performance for the **malignant vs benign** diagnosis task.")

    df = load_bundled_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df)}")
    c2.metric("Features", f"{len(feature_columns())}")
    n_mal = int((df['target'] == 0).sum()); n_ben = int((df['target'] == 1).sum())
    c3.metric("Malignant", f"{n_mal}")
    c4.metric("Benign", f"{n_ben}")

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("About the dataset")
        st.markdown(
            "- **Source:** UCI / scikit-learn *Breast Cancer Wisconsin (Diagnostic)*\n"
            "- **Task:** binary classification (0 = malignant, 1 = benign)\n"
            "- **Features:** 30 numeric measurements of cell nuclei "
            "(mean / standard-error / worst of 10 base measures)\n"
            "- **Priority metric:** *recall on malignant* — minimise missed cancers.")
    with right:
        st.subheader("Final model")
        bundle = load_final_model()
        if bundle:
            m = bundle.get("test_metrics", {})
            st.markdown(f"**{bundle.get('model_name', 'model')}**")
            st.markdown(
                f"- Accuracy: **{m.get('Accuracy', float('nan')):.3f}**\n"
                f"- Precision: **{m.get('Precision', float('nan')):.3f}**\n"
                f"- Recall: **{m.get('Recall', float('nan')):.3f}**\n"
                f"- F1: **{m.get('F1-score', float('nan')):.3f}**")
            if bundle.get("selection_reason"):
                st.caption(bundle["selection_reason"])
        else:
            st.info("No saved final model found in `models/`.")

    st.divider()
    st.subheader("How to use this app")
    st.markdown(
        "1. **Dataset Upload** — use the bundled data or upload your own CSV.\n"
        "2. **Model Selection** — pick one of five algorithms.\n"
        "3. **Model Training** — train it and see the metrics.\n"
        "4. **Prediction** — score new samples (manual, from a row, or batch CSV).\n"
        "5. **Performance Dashboard** — compare models and inspect results.")


# =========================================================================== #
# 2. DATASET UPLOAD
# =========================================================================== #
elif PAGE.startswith("📤"):
    st.title("📤 Dataset Upload")
    st.markdown("Upload a CSV, or use the bundled Breast Cancer Wisconsin dataset.")

    up = st.file_uploader("Upload a CSV file", type=["csv"])
    col1, col2 = st.columns(2)
    if col1.button("Use bundled dataset", use_container_width=True):
        st.session_state["data"] = load_bundled_data()
        st.session_state["data_name"] = "breast_cancer_wisconsin.csv (bundled)"
        st.success("Loaded the bundled dataset.")
    if up is not None:
        try:
            df_up = pd.read_csv(up)
            st.session_state["data"] = df_up
            st.session_state["data_name"] = up.name
            st.success(f"Loaded **{up.name}** — {df_up.shape[0]} rows × {df_up.shape[1]} cols")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    df = get_active_data()
    st.divider()
    st.subheader(f"Preview — {st.session_state.get('data_name')}")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))

    st.subheader("Column types")
    dtypes = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    st.dataframe(dtypes, use_container_width=True, height=250)

    # Class distribution if a target/diagnosis column is present
    tgt_col = "diagnosis" if "diagnosis" in df.columns else (
        "target" if "target" in df.columns else None)
    if tgt_col:
        st.subheader("Class distribution")
        vc = df[tgt_col].value_counts()
        st.bar_chart(vc)
    else:
        st.info("No `target` / `diagnosis` column detected — this dataset can be "
                "used for prediction but not for training here.")

    # Validate feature compatibility
    missing_feats = [c for c in feature_columns() if c not in df.columns]
    if missing_feats:
        st.warning(f"{len(missing_feats)} expected feature columns are missing — "
                   "training/prediction expects the 30 Wisconsin features.")
    else:
        st.success("All 30 expected feature columns are present. ✅")


# =========================================================================== #
# 3. MODEL SELECTION
# =========================================================================== #
elif PAGE.startswith("⚙️"):
    st.title("⚙️ Model Selection")
    st.markdown("Choose the algorithm to train.")

    options = ["Naive Bayes", "Decision Tree", "Random Forest", "SVM"]
    if HAS_XGB:
        options.append("XGBoost")

    default_idx = options.index(st.session_state["model_choice"]) \
        if st.session_state.get("model_choice") in options else 3  # SVM default
    choice = st.radio("Algorithm", options, index=default_idx)
    st.session_state["model_choice"] = choice
    st.info(MODEL_INFO.get(choice, ""))

    st.divider()
    st.subheader("Training options")
    c1, c2 = st.columns(2)
    st.session_state["test_size"] = c1.slider("Test set size", 0.1, 0.4, 0.20, 0.05)
    st.session_state["stratify"] = c2.checkbox("Stratified split (recommended)", value=True)
    st.caption("Numerical features are standardised (scaler fit on the training "
               "split only, to avoid data leakage).")
    st.success(f"Selected model: **{choice}** — proceed to **Model Training**.")


# =========================================================================== #
# 4. MODEL TRAINING
# =========================================================================== #
elif PAGE.startswith("🏋️"):
    st.title("🏋️ Model Training")
    df = get_active_data()
    feats = feature_columns()
    choice = st.session_state.get("model_choice", "SVM")

    missing_feats = [c for c in feats if c not in df.columns]
    if "target" not in df.columns or missing_feats:
        st.error("The active dataset is not trainable here — it must contain the "
                 "30 Wisconsin feature columns and a `target` column. Use the "
                 "bundled dataset in **Dataset Upload**.")
        st.stop()

    st.markdown(f"Model to train: **{choice}**  |  Dataset: "
                f"**{st.session_state.get('data_name')}**")
    test_size = st.session_state.get("test_size", 0.20)
    stratify = st.session_state.get("stratify", True)

    if st.button("🚀 Train model", type="primary"):
        X, y = df[feats], df["target"]
        strat = y if stratify else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        with st.spinner(f"Training {choice}…"):
            model = build_model(choice)
            model.fit(X_tr_s, y_tr)
        metrics, y_pred = evaluate(model, X_te_s, y_te)

        st.session_state["trained"] = {
            "name": choice, "model": model, "scaler": scaler,
            "X_test_s": X_te_s, "y_test": y_te.reset_index(drop=True),
            "y_pred": y_pred, "metrics": metrics, "features": feats,
        }
        st.success(f"✅ {choice} trained successfully.")

    trained = st.session_state.get("trained")
    if trained:
        st.divider()
        st.subheader(f"Results — {trained['name']}")
        m = trained["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['Accuracy']:.3f}")
        c2.metric("Precision", f"{m['Precision']:.3f}")
        c3.metric("Recall", f"{m['Recall']:.3f}")
        c4.metric("F1-score", f"{m['F1-score']:.3f}")
        st.caption("Precision / Recall / F1 are for the **malignant (0)** class.")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(trained["y_test"], trained["y_pred"], labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4, 3.4))
            ConfusionMatrixDisplay(cm, display_labels=["malig", "benign"]).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format="d")
            ax.grid(False)
            st.pyplot(fig)
        with colB:
            st.markdown("**Classification report**")
            rep = classification_report(
                trained["y_test"], trained["y_pred"],
                target_names=["malignant (0)", "benign (1)"], output_dict=True)
            st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)
        st.info("Head to **Prediction** to score new samples with this model.")
    else:
        st.info("Click **Train model** to fit the selected algorithm.")


# =========================================================================== #
# 5. PREDICTION
# =========================================================================== #
elif PAGE.startswith("🔮"):
    st.title("🔮 Prediction")
    feats = feature_columns()
    df = load_bundled_data()

    # Choose which model to use
    sources = []
    if st.session_state.get("trained"):
        sources.append(f"Trained this session ({st.session_state['trained']['name']})")
    if load_final_model():
        sources.append("Saved final model (SVM)")
    if not sources:
        st.error("No model available. Train one in **Model Training** first.")
        st.stop()

    src = st.radio("Model to use", sources, horizontal=True)
    if src.startswith("Trained"):
        t = st.session_state["trained"]
        model, scaler, model_name = t["model"], t["scaler"], t["name"]
    else:
        b = load_final_model()
        model, scaler, model_name = b["model"], b["scaler"], b["model_name"]
    st.caption(f"Active model: **{model_name}**")

    st.divider()
    mode = st.radio("Input method",
                    ["Load a sample row", "Manual entry", "Batch CSV upload"],
                    horizontal=True)

    def predict_rows(X_df):
        Xs = scaler.transform(X_df[feats])
        preds = model.predict(Xs)
        proba = model.predict_proba(Xs) if hasattr(model, "predict_proba") else None
        return preds, proba

    # ---- Mode A: sample row from the dataset -------------------------------
    if mode == "Load a sample row":
        idx = st.number_input("Row index from bundled dataset", 0, len(df) - 1, 0)
        row = df.iloc[[idx]]
        with st.expander("Show feature values"):
            st.dataframe(row[feats].T, use_container_width=True)
        if st.button("Predict", type="primary"):
            preds, proba = predict_rows(row)
            pred = int(preds[0])
            actual = int(row["target"].iloc[0]) if "target" in row else None
            _show_prediction(pred, proba, actual)

    # ---- Mode B: manual entry ----------------------------------------------
    elif mode == "Manual entry":
        st.caption("Adjust the 30 features (pre-filled with dataset medians).")
        medians = df[feats].median()
        groups = {"Mean features": [f for f in feats if f.startswith("mean ")],
                  "Standard-error features": [f for f in feats if f.endswith(" error")],
                  "Worst features": [f for f in feats if f.startswith("worst ")]}
        values = {}
        for gname, gfeats in groups.items():
            with st.expander(gname, expanded=(gname == "Mean features")):
                cols = st.columns(2)
                for i, f in enumerate(gfeats):
                    lo, hi = float(df[f].min()), float(df[f].max())
                    values[f] = cols[i % 2].number_input(
                        f, min_value=float(lo), max_value=float(hi * 1.5),
                        value=float(medians[f]), format="%.4f")
        if st.button("Predict", type="primary"):
            X_one = pd.DataFrame([values])[feats]
            preds, proba = predict_rows(X_one)
            _show_prediction(int(preds[0]), proba, None)

    # ---- Mode C: batch CSV --------------------------------------------------
    else:
        st.caption("Upload a CSV containing the 30 feature columns; each row is scored.")
        up = st.file_uploader("CSV to predict", type=["csv"], key="batch")
        if up is not None:
            try:
                bdf = pd.read_csv(up)
                miss = [c for c in feats if c not in bdf.columns]
                if miss:
                    st.error(f"Missing {len(miss)} required feature columns, e.g. {miss[:3]}")
                else:
                    preds, proba = predict_rows(bdf)
                    out = bdf.copy()
                    out["prediction"] = [LABEL_NAMES[int(p)] for p in preds]
                    if proba is not None:
                        out["P(malignant)"] = proba[:, 0].round(4)
                    st.success(f"Scored {len(out)} rows.")
                    st.dataframe(out, use_container_width=True, height=350)
                    st.download_button(
                        "⬇️ Download predictions",
                        out.to_csv(index=False).encode("utf-8"),
                        "predictions.csv", "text/csv")
                    st.bar_chart(out["prediction"].value_counts())
            except Exception as e:
                st.error(f"Failed: {e}")


# =========================================================================== #
# 6. PERFORMANCE DASHBOARD
# =========================================================================== #
elif PAGE.startswith("📊"):
    st.title("📊 Performance Dashboard")

    baseline = load_baseline_metrics()
    tab1, tab2, tab3 = st.tabs(
        ["Model comparison", "Trained model", "Saved final model"])

    # ---- Tab 1: baseline comparison ----------------------------------------
    with tab1:
        if baseline is None:
            st.info("No `reports/individual_model_metrics.csv` found. "
                    "Run notebook 03 to generate baseline metrics.")
        else:
            st.subheader("Baseline model metrics (test set)")
            st.dataframe(baseline.round(4).style.highlight_max(axis=0, color="#c6efce"),
                         use_container_width=True)
            metric = st.selectbox("Chart a metric",
                                  ["Accuracy", "Precision", "Recall", "F1-score"],
                                  index=2)
            s = baseline[metric].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.bar(s.index, s.values, color="#4C72B0")
            ax.set_ylim(min(0.85, s.min() - 0.03), 1.005)
            ax.set_title(f"{metric} by model")
            for i, v in enumerate(s.values):
                ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9)
            plt.xticks(rotation=15)
            st.pyplot(fig)
            st.caption("Metrics use malignant (0) as the positive class.")

    # ---- Tab 2: this-session trained model ---------------------------------
    with tab2:
        t = st.session_state.get("trained")
        if not t:
            st.info("No model trained this session. Train one in **Model Training**.")
        else:
            st.subheader(f"{t['name']} — trained this session")
            m = t["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['Accuracy']:.3f}")
            c2.metric("Precision", f"{m['Precision']:.3f}")
            c3.metric("Recall", f"{m['Recall']:.3f}")
            c4.metric("F1-score", f"{m['F1-score']:.3f}")

            colA, colB = st.columns(2)
            with colA:
                cm = confusion_matrix(t["y_test"], t["y_pred"], labels=[0, 1])
                fig, ax = plt.subplots(figsize=(4, 3.4))
                ConfusionMatrixDisplay(cm, display_labels=["malig", "benign"]).plot(
                    ax=ax, cmap="Greens", colorbar=False, values_format="d")
                ax.grid(False); ax.set_title("Confusion matrix")
                st.pyplot(fig)
            with colB:
                # ROC curve if probabilities available
                if hasattr(t["model"], "predict_proba"):
                    proba_mal = t["model"].predict_proba(t["X_test_s"])[:, 0]
                    y_true_mal = (t["y_test"] == 0).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_mal, proba_mal)
                    fig, ax = plt.subplots(figsize=(4, 3.4))
                    ax.plot(fpr, tpr, color="#C44E52",
                            label=f"AUC = {auc(fpr, tpr):.3f}")
                    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
                    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC (malignant)"); ax.legend(loc="lower right")
                    st.pyplot(fig)

    # ---- Tab 3: saved final model ------------------------------------------
    with tab3:
        b = load_final_model()
        if not b:
            st.info("No saved final model found.")
        else:
            st.subheader(f"Saved final model — {b['model_name']}")
            m = b.get("test_metrics", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m.get('Accuracy', 0):.3f}")
            c2.metric("Precision", f"{m.get('Precision', 0):.3f}")
            c3.metric("Recall", f"{m.get('Recall', 0):.3f}")
            c4.metric("F1-score", f"{m.get('F1-score', 0):.3f}")
            st.markdown(f"**Best params:** `{b.get('best_params', {})}`")
            if b.get("selection_reason"):
                st.success(b["selection_reason"])
