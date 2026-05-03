"""
streamlit_app.py
────────────────
Real-time Fraud Detection Dashboard

Features
────────
• Single-transaction scoring with XGBoost / LightGBM
• SHAP waterfall explanation for every prediction
• Live model performance metrics (from saved report JSON)
• Feature drift (PSI) indicators
• Batch CSV upload for bulk scoring

Run:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Allow imports from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: .5rem;
    }
    .fraud-badge {
        background: #e74c3c;
        color: white;
        border-radius: 8px;
        padding: .3rem .8rem;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .legit-badge {
        background: #27ae60;
        color: white;
        border-radius: 8px;
        padding: .3rem .8rem;
        font-weight: 700;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_assets():
    try:
        from src.models import load_models
        iso, xgb_m, lgbm_m, scaler = load_models()
        return iso, xgb_m, lgbm_m, scaler, None
    except Exception as e:
        return None, None, None, None, str(e)


iso, xgb_m, lgbm_m, scaler, load_error = load_assets()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/fraud.png", width=72)
    st.title("⚙️ Settings")
    model_choice = st.selectbox("Active model", ["XGBoost", "LightGBM"])
    threshold    = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    st.markdown(
        "**Dataset:** Credit Card Fraud  \n"
        "284,807 transactions · 492 fraud  \n"
        "[Kaggle →](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)"
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_score, tab_batch, tab_metrics, tab_drift = st.tabs(
    ["🔍 Score Transaction", "📋 Batch Upload", "📊 Model Metrics", "📡 Drift Monitor"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Single Transaction Scoring
# ══════════════════════════════════════════════════════════════════════════════

with tab_score:
    st.header("Real-Time Transaction Scorer")

    if load_error:
        st.error(
            f"❌ Models not loaded: {load_error}\n\n"
            "Run `python -m src.pipeline` first to train and save models."
        )
    else:
        st.info(
            "📌 Features V1–V28 are PCA-transformed components from the dataset. "
            "Use realistic values (typically in range -5 to 5).  \n"
            "Amount and Time are raw values."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Amount ($)", min_value=0.0, value=149.62, step=0.01)
            time_s = st.number_input("Time (seconds elapsed)", min_value=0, value=406, step=1)

        st.markdown("**PCA Features (V1–V28)**")
        cols = st.columns(4)
        v_vals = {}
        defaults = {
            "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
            "V6": 0.46,  "V7": 0.24,  "V8": 0.10, "V9": 0.36, "V10": 0.09,
            "V11": -0.55,"V12": -0.62,"V13": -0.99,"V14": -0.31,"V15": 1.47,
            "V16": -0.47,"V17": 0.21, "V18": 0.03,"V19": 0.40, "V20": 0.25,
            "V21": -0.02,"V22": 0.28, "V23": -0.11,"V24": 0.07,"V25": 0.13,
            "V26": -0.19,"V27": 0.13, "V28": -0.02,
        }
        for i, feat in enumerate([f"V{j}" for j in range(1, 29)]):
            with cols[i % 4]:
                v_vals[feat] = st.number_input(feat, value=defaults[feat], format="%.4f", key=feat)

        if st.button("🚀 Score Transaction", type="primary", use_container_width=True):
            # Build feature dict
            raw = {"Amount": amount, "Time": time_s, **v_vals}
            df_raw = pd.DataFrame([raw])

            # Engineer features
            import numpy as np
            df_raw["Amount_Log"]    = np.log1p(df_raw["Amount"])
            mu, sigma = 88.35, 250.12
            df_raw["Amount_ZScore"] = (df_raw["Amount"] - mu) / sigma
            df_raw["Hour"]          = (df_raw["Time"] % 86_400) // 3_600
            df_raw.drop(columns=["Time", "Amount"], inplace=True)

            X_scaled = scaler.transform(df_raw)

            # Score
            active_model = xgb_m if model_choice == "XGBoost" else lgbm_m
            fraud_prob   = float(active_model.predict_proba(X_scaled)[0, 1])
            prediction   = "FRAUD" if fraud_prob >= threshold else "LEGITIMATE"

            # Results
            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Fraud Probability", f"{fraud_prob:.2%}")
            res_col2.metric("Threshold", f"{threshold:.2%}")
            res_col3.metric(
                "Verdict",
                prediction,
                delta="⚠ High Risk" if fraud_prob > 0.8 else ("🟡 Medium" if fraud_prob > threshold else "✅ Low Risk"),
            )

            badge = (
                '<span class="fraud-badge">🚨 FRAUD DETECTED</span>'
                if prediction == "FRAUD"
                else '<span class="legit-badge">✅ LEGITIMATE</span>'
            )
            st.markdown(badge, unsafe_allow_html=True)

            # SHAP explanation
            st.markdown("### 🔎 SHAP Explanation")
            try:
                import shap, matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                explainer   = shap.TreeExplainer(active_model)
                shap_values = explainer(X_scaled)

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values.values[0, :, 1]
                            if shap_values.values.ndim == 3
                            else shap_values.values[0],
                        base_values=shap_values.base_values[0]
                            if np.ndim(shap_values.base_values) == 1
                            else shap_values.base_values[0, 1],
                        data=X_scaled[0],
                        feature_names=df_raw.columns.tolist(),
                    ),
                    show=False,
                )
                st.pyplot(plt.gcf())
                plt.close("all")
            except Exception as e:
                st.warning(f"SHAP plot unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Batch Upload
# ══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.header("📋 Batch Transaction Scoring")
    st.markdown(
        "Upload a CSV with the same column format as the Kaggle dataset "
        "(V1–V28, Amount, Time). The app scores every row and lets you "
        "download the flagged results."
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded and not load_error:
        raw_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(raw_df):,} rows · {raw_df.shape[1]} columns")

        # Engineer
        df_proc = raw_df.copy()
        df_proc["Amount_Log"]    = np.log1p(df_proc["Amount"])
        mu, sigma = 88.35, 250.12
        df_proc["Amount_ZScore"] = (df_proc["Amount"] - mu) / sigma
        df_proc["Hour"]          = (df_proc["Time"] % 86_400) // 3_600
        drop_cols = [c for c in ["Time", "Amount", "Class"] if c in df_proc.columns]
        df_proc.drop(columns=drop_cols, inplace=True)

        X_batch = scaler.transform(df_proc)
        active  = xgb_m if model_choice == "XGBoost" else lgbm_m
        probas  = active.predict_proba(X_batch)[:, 1]

        raw_df["fraud_score"]  = probas
        raw_df["prediction"]   = np.where(probas >= threshold, "FRAUD", "LEGITIMATE")

        fraud_count = (raw_df["prediction"] == "FRAUD").sum()
        st.metric("Flagged as Fraud", f"{fraud_count:,} / {len(raw_df):,}")

        st.dataframe(
            raw_df[["fraud_score", "prediction"] + [c for c in raw_df.columns
                                                     if c not in ("fraud_score","prediction")]]
                .sort_values("fraud_score", ascending=False)
                .head(500),
            use_container_width=True,
        )

        csv_out = raw_df.to_csv(index=False).encode()
        st.download_button("⬇ Download Scored CSV", csv_out, "scored_transactions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Model Metrics
# ══════════════════════════════════════════════════════════════════════════════

with tab_metrics:
    st.header("📊 Model Performance Metrics")

    report_path = ROOT / "reports" / "model_comparison.json"
    if report_path.exists():
        with open(report_path) as f:
            comparison = json.load(f)

        for model_name, metrics in comparison.items():
            with st.expander(f"**{model_name}**", expanded=True):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.4f}")
                c2.metric("PR-AUC",    f"{metrics.get('pr_auc', 0):.4f}")
                c3.metric("KS-Stat",   f"{metrics.get('ks', 0):.4f}")
                c4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                c5.metric("Recall",    f"{metrics.get('recall', 0):.4f}")

        # Show saved plots
        st.markdown("### 📈 ROC & Precision-Recall Curves")
        plot_cols = st.columns(2)
        idx = 0
        for png in sorted((ROOT / "reports").glob("roc_*.png")):
            with plot_cols[idx % 2]:
                st.image(str(png), use_column_width=True)
            idx += 1

        st.markdown("### Precision-Recall Curves")
        idx = 0
        pr_cols = st.columns(2)
        for png in sorted((ROOT / "reports").glob("pr_*.png")):
            with pr_cols[idx % 2]:
                st.image(str(png), use_column_width=True)
            idx += 1

        st.markdown("### Confusion Matrices")
        idx = 0
        cm_cols = st.columns(3)
        for png in sorted((ROOT / "reports").glob("cm_*.png")):
            with cm_cols[idx % 3]:
                st.image(str(png), use_column_width=True)
            idx += 1
    else:
        st.warning(
            "No metrics report found. Run `python -m src.pipeline` to generate one."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Drift Monitor
# ══════════════════════════════════════════════════════════════════════════════

with tab_drift:
    st.header("📡 Feature Drift Monitor (PSI)")
    st.markdown(
        "Population Stability Index (PSI) measures how much each feature's "
        "distribution has shifted between the training baseline and a current window.  \n\n"
        "| PSI | Status |  \n"
        "|-----|--------|  \n"
        "| < 0.10 | ✅ Stable |  \n"
        "| 0.10 – 0.20 | ⚠️ Monitor |  \n"
        "| > 0.20 | 🚨 Retrain |  \n"
    )

    psi_png = ROOT / "reports" / "psi_dashboard.png"
    if psi_png.exists():
        st.image(str(psi_png), use_column_width=True)
    else:
        st.info("Run the full pipeline first to generate the PSI dashboard.")

    if not load_error:
        if st.button("🔄 Re-run Drift Analysis on Test Set"):
            from src.features import build_train_test
            from src.monitoring import simulate_drift, plot_psi_dashboard

            with st.spinner("Running drift analysis …"):
                _, _, X_test, _, _, y_test, feat_names = build_train_test(apply_smote=False)
                drift_results = simulate_drift(X_test, feat_names, n_windows=4)
                if drift_results:
                    path = plot_psi_dashboard(drift_results[0], "PSI — W2 vs W1")
                    st.image(str(path), use_column_width=True)

                    for i, dr in enumerate(drift_results):
                        st.markdown(f"**Window W{i+2} vs W1**")
                        st.dataframe(dr, use_container_width=True)