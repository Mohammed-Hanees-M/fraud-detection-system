"""
config.py
─────────
Central project configuration — paths, model params, thresholds.
All other modules import from here; change values in ONE place.
"""

from pathlib import Path

# ── Directory layout ───────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
REPORTS_DIR  = ROOT_DIR / "reports"
MODELS_DIR   = ROOT_DIR / "models"

for _d in (DATA_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
RAW_CSV      = DATA_DIR / "creditcard.csv"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Feature engineering ────────────────────────────────────────────────────────
AMOUNT_LOG_FEATURE   = "Amount_Log"
AMOUNT_ZSCORE_FEATURE= "Amount_ZScore"
HOUR_FEATURE         = "Hour"

# ── Train / test split ─────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10          # fraction of train set used for validation

# ── SMOTE ──────────────────────────────────────────────────────────────────────
SMOTE_STRATEGY = 0.1        # minority : majority ratio after oversampling

# ── Model artefact names ───────────────────────────────────────────────────────
ISO_FOREST_PATH = MODELS_DIR / "isolation_forest.joblib"
XGB_PATH        = MODELS_DIR / "xgboost_model.joblib"
LGBM_PATH       = MODELS_DIR / "lightgbm_model.joblib"
SCALER_PATH     = MODELS_DIR / "scaler.joblib"

# ── Evaluation thresholds ──────────────────────────────────────────────────────
CLASSIFICATION_THRESHOLD = 0.5     # default; tuned per model in evaluation

# ── Drift monitoring ───────────────────────────────────────────────────────────
PSI_BUCKETS          = 10
PSI_WARNING_THRESHOLD= 0.10
PSI_ALERT_THRESHOLD  = 0.20
DRIFT_WINDOW_DAYS    = 7            # synthetic time-window width in the demo

# ── Streamlit ──────────────────────────────────────────────────────────────────
APP_TITLE = "Fraud Detection & Anomaly Scoring System"
