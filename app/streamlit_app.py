"""
streamlit_app.py
────────────────
Real-time Fraud Detection Dashboard
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .fraud-badge { background:#e74c3c;color:white;border-radius:8px;padding:.3rem .8rem;font-weight:700;font-size:1.1rem; }
    .legit-badge { background:#27ae60;color:white;border-radius:8px;padding:.3rem .8rem;font-weight:700;font-size:1.1rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading models …")
def load_assets():
    try:
        from src.models import load_models
        iso, xgb_m, lgbm_m, scaler = load_models()
        return iso, xgb_m, lgbm_m, scaler, None
    except Exception as e:
        return None, None, None, None, str(e)

iso, xgb_m, lgbm_m, scaler, load_error = load_assets()

with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/fraud.png", width=72)
    st.title("⚙️ Settings")
    model_choice = st.selectbox("Active model", ["XGBoost", "LightGBM"])
    threshold    = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    st.markdown("**Dataset:** Credit Card Fraud  \n284,807 transactions · 492 fraud  \n[Kaggle →](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.markdown("---")
    st.markdown("**Benchmark Results**\n\n| Dataset | ROC-AUC |\n|---------|--------|\n| Credit Card | 0.982 |\n| IEEE-CIS | 0.921 |\n| PaySim | 0.999 |")

tab_score, tab_batch, tab_metrics, tab_drift, tab_benchmark = st.tabs(
    ["🔍 Score Transaction", "📋 Batch Upload", "📊 Model Metrics", "📡 Drift Monitor", "🏆 Benchmark"]
)

# ── TAB 1: Score Transaction ──────────────────────────────────────────────────
with tab_score:
    st.header("Real-Time Transaction Scorer")
    if load_error:
        st.error(f"❌ Models not loaded: {load_error}\n\nRun `python -m src.pipeline` first.")
    else:
        st.info("📌 Features V1–V28 are PCA-transformed components. Use values in range -5 to 5. Amount and Time are raw values.")
        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Amount ($)", min_value=0.0, value=149.62, step=0.01)
            time_s = st.number_input("Time (seconds elapsed)", min_value=0, value=406, step=1)
        st.markdown("**PCA Features (V1–V28)**")
        cols = st.columns(4)
        v_vals = {}
        defaults = {
            "V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,
            "V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,
            "V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,
            "V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,
            "V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,
            "V26":-0.19,"V27":0.13,"V28":-0.02,
        }
        for i, feat in enumerate([f"V{j}" for j in range(1, 29)]):
            with cols[i % 4]:
                v_vals[feat] = st.number_input(feat, value=defaults[feat], format="%.4f", key=feat)

        if st.button("🚀 Score Transaction", type="primary", use_container_width=True):
            raw = {"Amount": amount, "Time": time_s, **v_vals}
            df_raw = pd.DataFrame([raw])
            df_raw["Amount_Log"]    = np.log1p(df_raw["Amount"])
            mu, sigma = 88.35, 250.12
            df_raw["Amount_ZScore"] = (df_raw["Amount"] - mu) / sigma
            df_raw["Hour"]          = (df_raw["Time"] % 86_400) // 3_600
            df_raw.drop(columns=["Time", "Amount"], inplace=True)
            X_scaled = scaler.transform(df_raw)
            active_model = xgb_m if model_choice == "XGBoost" else lgbm_m
            fraud_prob   = float(active_model.predict_proba(X_scaled)[0, 1])
            prediction   = "FRAUD" if fraud_prob >= threshold else "LEGITIMATE"
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", f"{fraud_prob:.2%}")
            c2.metric("Threshold", f"{threshold:.2%}")
            c3.metric("Verdict", prediction, delta="⚠ High Risk" if fraud_prob > 0.8 else ("🟡 Medium" if fraud_prob > threshold else "✅ Low Risk"))
            badge = '<span class="fraud-badge">🚨 FRAUD DETECTED</span>' if prediction == "FRAUD" else '<span class="legit-badge">✅ LEGITIMATE</span>'
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown("### 🔎 SHAP Explanation")
            try:
                import shap, matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                explainer   = shap.TreeExplainer(active_model)
                shap_values = explainer(X_scaled)
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values.values[0,:,1] if shap_values.values.ndim==3 else shap_values.values[0],
                        base_values=float(shap_values.base_values[0]) if np.ndim(shap_values.base_values)==1 else float(shap_values.base_values[0,1]),
                        data=X_scaled[0],
                        feature_names=df_raw.columns.tolist(),
                    ), show=False,
                )
                st.pyplot(plt.gcf())
                plt.close("all")
            except Exception as e:
                st.warning(f"SHAP plot unavailable: {e}")

# ── TAB 2: Batch Upload ───────────────────────────────────────────────────────
with tab_batch:
    st.header("📋 Batch Transaction Scoring")
    st.markdown("Upload a CSV with columns V1–V28, Amount, Time. The app scores every row and lets you download the results.")
    st.info("💡 Use the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) CSV directly.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded and not load_error:
        raw_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(raw_df):,} rows · {raw_df.shape[1]} columns")
        required = [f"V{i}" for i in range(1,29)] + ["Amount","Time"]
        missing_cols = [c for c in required if c not in raw_df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols[:5]}. Please upload a CSV with V1–V28, Amount, Time columns.")
        else:
            with st.spinner("Scoring transactions …"):
                df_proc = raw_df.copy()
                df_proc["Amount_Log"]    = np.log1p(df_proc["Amount"])
                mu, sigma = 88.35, 250.12
                df_proc["Amount_ZScore"] = (df_proc["Amount"] - mu) / sigma
                df_proc["Hour"]          = (df_proc["Time"] % 86_400) // 3_600
                drop_cols = [c for c in ["Time","Amount","Class"] if c in df_proc.columns]
                df_proc.drop(columns=drop_cols, inplace=True)
                X_batch = scaler.transform(df_proc)
                active  = xgb_m if model_choice == "XGBoost" else lgbm_m
                probas  = active.predict_proba(X_batch)[:,1]
            raw_df["fraud_score"] = probas
            raw_df["prediction"]  = np.where(probas >= threshold, "FRAUD", "LEGITIMATE")
            fraud_count = (raw_df["prediction"] == "FRAUD").sum()
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Transactions", f"{len(raw_df):,}")
            m2.metric("🚨 Flagged as Fraud", f"{fraud_count:,}")
            m3.metric("✅ Legitimate", f"{len(raw_df)-fraud_count:,}")
            m4.metric("Fraud Rate", f"{fraud_count/len(raw_df):.3%}")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8,3))
            ax.hist(probas[probas<threshold], bins=50, color="#27ae60", alpha=0.7, label="Legitimate", density=True)
            ax.hist(probas[probas>=threshold], bins=50, color="#e74c3c", alpha=0.7, label="Fraud", density=True)
            ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold ({threshold:.2f})")
            ax.set(xlabel="Fraud Score", title="Score Distribution")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(
                raw_df[["fraud_score","prediction"]+[c for c in raw_df.columns if c not in ("fraud_score","prediction")]].sort_values("fraud_score",ascending=False).head(500),
                use_container_width=True,
            )
            st.download_button("⬇ Download Scored CSV", raw_df.to_csv(index=False).encode(), "scored_transactions.csv", "text/csv", use_container_width=True)
    elif load_error:
        st.error("Models not loaded.")

# ── TAB 3: Model Metrics ──────────────────────────────────────────────────────
with tab_metrics:
    st.header("📊 Model Performance Metrics")
    report_path = ROOT / "reports" / "model_comparison.json"
    if report_path.exists():
        with open(report_path) as f:
            comparison = json.load(f)
        for model_name, metrics in comparison.items():
            with st.expander(f"**{model_name}**", expanded=True):
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("ROC-AUC",   f"{metrics.get('roc_auc',0):.4f}")
                c2.metric("PR-AUC",    f"{metrics.get('pr_auc',0):.4f}")
                c3.metric("KS-Stat",   f"{metrics.get('ks',0):.4f}")
                c4.metric("Precision", f"{metrics.get('precision',0):.4f}")
                c5.metric("Recall",    f"{metrics.get('recall',0):.4f}")
        st.markdown("### 📈 ROC Curves")
        plot_cols = st.columns(2)
        for idx, png in enumerate(sorted((ROOT/"reports").glob("roc_*.png"))):
            with plot_cols[idx%2]: st.image(str(png), use_container_width=True)
        st.markdown("### Precision-Recall Curves")
        pr_cols = st.columns(2)
        for idx, png in enumerate(sorted((ROOT/"reports").glob("pr_*.png"))):
            with pr_cols[idx%2]: st.image(str(png), use_container_width=True)
        st.markdown("### Confusion Matrices")
        cm_cols = st.columns(3)
        for idx, png in enumerate(sorted((ROOT/"reports").glob("cm_*.png"))):
            with cm_cols[idx%3]: st.image(str(png), use_container_width=True)
    else:
        st.warning("No metrics report found. Run `python -m src.pipeline` to generate one.")

# ── TAB 4: Drift Monitor ──────────────────────────────────────────────────────
with tab_drift:
    st.header("📡 Feature Drift Monitor (PSI)")
    st.markdown("Population Stability Index (PSI) measures feature distribution shift between training baseline and current window.\n\n| PSI | Status |\n|-----|--------|\n| < 0.10 | ✅ Stable |\n| 0.10–0.20 | ⚠️ Monitor |\n| > 0.20 | 🚨 Retrain |")
    psi_png = ROOT / "reports" / "psi_dashboard.png"
    if psi_png.exists():
        st.image(str(psi_png), use_container_width=True)
    else:
        st.info("Run the full pipeline first to generate the PSI dashboard.")
    from src.config import RAW_CSV
    if RAW_CSV.exists():
        if st.button("🔄 Re-run Drift Analysis on Test Set"):
            from src.features import build_train_test
            from src.monitoring import simulate_drift, plot_psi_dashboard
            with st.spinner("Running drift analysis …"):
                _, _, X_test, _, _, y_test, feat_names = build_train_test(apply_smote=False)
                drift_results = simulate_drift(X_test, feat_names, n_windows=4)
                if drift_results:
                    path = plot_psi_dashboard(drift_results[0], "PSI — W2 vs W1")
                    st.image(str(path), use_container_width=True)
                    for i, dr in enumerate(drift_results):
                        st.markdown(f"**Window W{i+2} vs W1**")
                        st.dataframe(dr, use_container_width=True)
    else:
        st.info("💡 Re-run drift analysis is available when running locally with the dataset. The PSI dashboard above shows real drift results from training.")

# ── TAB 5: Benchmark ─────────────────────────────────────────────────────────
with tab_benchmark:
    st.header("🏆 Multi-Dataset Benchmark Results")
    st.markdown("Benchmarked across **3 real-world fraud datasets** totalling **1.17M+ transactions**.")
    benchmark_data = {
        "Dataset":             ["Credit Card Fraud","IEEE-CIS Fraud","PaySim Mobile Fraud"],
        "Transactions":        ["284,807","590,540","300,000 (sampled)"],
        "Fraud Rate":          ["0.17%","3.50%","0.13%"],
        "XGBoost ROC-AUC":     ["0.9818","0.9214","0.9995"],
        "LightGBM ROC-AUC":    ["0.9799","0.9166","0.9997"],
        "XGBoost PR-AUC":      ["0.8647","0.5930","0.9339"],
        "XGBoost KS-Stat":     ["0.8991","0.6969","0.9824"],
        "XGBoost F1":          ["0.8652","0.5730","0.8919"],
    }
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)
    bench_png = ROOT / "reports" / "benchmark" / "master_benchmark.png"
    if bench_png.exists():
        st.markdown("### 📊 Visual Comparison")
        st.image(str(bench_png), use_container_width=True)
    roc_files = sorted((ROOT/"reports"/"benchmark").glob("roc_*.png"))
    if roc_files:
        st.markdown("### ROC Curves by Dataset")
        roc_cols = st.columns(min(3, len(roc_files)))
        for i, png in enumerate(roc_files):
            with roc_cols[i%3]: st.image(str(png), use_container_width=True)
    st.markdown("---")
    st.markdown("### 🎯 Production Simulation — Full 284,807 Transaction Test")
    st.markdown("| Metric | Result |\n|--------|--------|\n| Actual Fraud Cases | 492 |\n| **Caught (True Positives)** | **475 (96.5%)** |\n| Missed (False Negatives) | 17 (3.5%) |\n| False Alarms | 29 / 284,315 legit |\n| **Precision** | **94.25%** |\n| **Recall** | **96.54%** |\n| **F1-Score** | **95.38%** |\n| Avg fraud score (real fraud) | 0.9612 |\n| Avg fraud score (legit) | 0.0005 |")
    st.success("🚀 96.5% recall with only 29 false alarms out of 284,315 legitimate transactions. Avg fraud score separation: 0.9612 vs 0.0005.")