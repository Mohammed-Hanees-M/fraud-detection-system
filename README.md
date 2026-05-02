# 🔍 Fraud Detection & Anomaly Scoring System

> End-to-end ML pipeline for real-time credit card fraud detection —  
> 284K+ transactions · Isolation Forest + XGBoost + LightGBM · SHAP · PSI Drift Monitoring · Streamlit Dashboard

---

## 📸 Project Overview

| Component | Detail |
|-----------|--------|
| **Dataset** | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 txns, 492 fraud (0.17%) |
| **Models** | Isolation Forest (unsupervised) · XGBoost · LightGBM |
| **Imbalance** | SMOTE (10% oversample ratio) |
| **Explainability** | SHAP waterfall, summary, dependence plots |
| **Monitoring** | Population Stability Index (PSI) across time windows |
| **Dashboard** | Streamlit — live scoring, SHAP explanation, PSI dashboard |

---

## 🗂️ Project Structure

```
fraud-detection-system/
├── data/                        # Dataset (gitignored)
├── models/                      # Trained model artefacts (gitignored)
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_drift_monitoring.ipynb  # SHAP + PSI
├── src/
│   ├── config.py                # Central config (paths, params)
│   ├── features.py              # Feature engineering + SMOTE
│   ├── models.py                # Train / persist models
│   ├── evaluation.py            # Metrics, curves, plots
│   ├── monitoring.py            # PSI drift detection
│   └── pipeline.py              # End-to-end orchestrator
├── app/
│   └── streamlit_app.py         # Dashboard
├── reports/                     # Saved plots & JSON report
├── scripts/
│   └── download_data.py         # Kaggle dataset downloader
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (5 steps)

### 1. Clone & create environment
```bash
git clone <your-repo-url>
cd fraud-detection-system

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
# Option A — Kaggle CLI (recommended):
python scripts/download_data.py

# Option B — Manual:
# 1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 2. Download creditcard.csv → place in data/creditcard.csv
```

### 4. Run the full training pipeline
```bash
python -m src.pipeline
```
This trains all three models, evaluates them, generates all plots in `reports/`, and saves a `model_comparison.json`.

### 5. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```
Open http://localhost:8501 in your browser.

---

## 📓 Notebooks (run in order)

```bash
jupyter notebook
```

| Notebook | What it covers |
|----------|---------------|
| `01_EDA.ipynb` | Class imbalance, distributions, correlations |
| `02_feature_engineering.ipynb` | SMOTE, new features, mutual information |
| `03_modelling.ipynb` | Training all three models |
| `04_evaluation.ipynb` | ROC, PR, KS, confusion matrices |
| `05_drift_monitoring.ipynb` | SHAP + PSI drift across time windows |

---

## 📊 Results

| Model | ROC-AUC | PR-AUC | KS-Stat |
|-------|---------|--------|---------|
| Isolation Forest | ~0.95 | ~0.35 | ~0.75 |
| **XGBoost** | **~0.98** | **~0.85** | **~0.88** |
| **LightGBM** | **~0.98** | **~0.84** | **~0.87** |

> Actual numbers will vary slightly based on Optuna's random search.

---

## 🧠 Key Technical Decisions

| Decision | Reason |
|----------|--------|
| SMOTE (not undersampling) | Preserves majority class information |
| PR-AUC over ROC-AUC | More meaningful for severe class imbalance |
| KS-Statistic | Standard fintech risk model evaluation metric |
| Optuna > GridSearch | Bayesian optimisation → better results in fewer trials |
| SHAP TreeExplainer | Exact Shapley values for tree models, computationally efficient |
| PSI for drift | Industry standard for monitoring model input stability |

---

## 💼 Resume Bullet

> Built a production-style real-time fraud detection system on 284K+ transactions using Isolation Forest, XGBoost, and LightGBM; achieved ROC-AUC 0.98+ with SMOTE for class imbalance; deployed SHAP-explainable Streamlit dashboard with live PSI drift monitoring.

---

## 📜 License
MIT
